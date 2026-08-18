# ══════════════════════════════════════════════════════════════════════════════
# ATTRIBUTION PATCHING (AtP) vs q_j -- comparaison à famille causale égale,
# tâche du marqueur.
#
# CONTEXTE
# --------
# bench_eps_semantic_objective_marker.jl a montré que q_j (Théorème thm:ablation),
# même amorcé avec l'objectif comportemental EXACT de P3 (dl = logit(v(sigma(k)))
# - logit(v(k))), désigne layer_1_mha_output_out comme branche #1, alors que le
# circuit trouvé INDÉPENDAMMENT par patching d'activation (P3, greedy_patch_search!
# + backward_prune!, marker_task_causal_analysis.jl) est layer_1_mlp_out.
#
# Mais q_j est PROUVABLEMENT invariant à la sortie avant y par construction
# (Définition def:gate, branch_order_theorem.tex lignes 140-148) : c'est une
# mesure de routage du gradient dans le graphe de dérivation arrière, pas un
# effet causal sur y. Comparer q_j à P3 (patching en avant de la VALEUR) compare
# donc potentiellement deux familles de méthodes différentes.
#
# Ce script compare plutôt q_j à Attribution Patching (AtP), une méthode de
# gradient elle-même EXPLICITEMENT conçue pour approximer l'effet causal du
# patching d'activation par un développement de Taylor au premier ordre :
#
#     AtP_j ≈ < grad_{a_j}(métrique) |_{run receveur/clean}, a_j^donneur - a_j^receveur >
#
# où a_j est l'activation de SORTIE en avant de la branche j (le nœud
# layer_l_mha_output_out / layer_l_mlp_out lui-même -- le même nœud que celui
# que P3 patche), le gradient est le gradient ordinaire de rétropropagation de
# la métrique (dl, SANS normalisation, même contraste que q_j et P3) par
# rapport à cette activation, évalué au run receveur -- UNE SEULE passe
# arrière donne le gradient des 8 branches simultanément (pas B+1 passes) --
# et a_j^donneur - a_j^receveur est la vraie différence d'activation entre le
# run donneur et le run receveur (mêmes deux runs que P3, même 3 paires, même
# checkpoint, réutilisés tels quels).
#
# CE QUI EST DÉJÀ ÉTABLI (cité, PAS recalculé ici) :
#   - Circuit P3 (patching d'activation, consensus sur 3 paires, r médian =
#     1.043 en validation indépendante) : layer_1_mlp_out
#     (marker_task_causal_analysis.jl / .log)
#   - q_j sous l'objectif sémantique dl à :embed_sum, mêmes 3 paires :
#     bench_eps_semantic_objective_marker_results.txt (valeurs signées par
#     paire + agrégat |q_j| moyen, recopiés ci-dessous tels quels).
#
# RISQUE D'IMPLÉMENTATION SPÉCIFIQUE À CE SCRIPT (vérifié explicitement plus
# bas, pas supposé) : pour lire `.gradient` sur un nœud INTERMÉDIAIRE après
# `backward_graph!`, il faut `is_param=true` (seul levier de rétention,
# src/backward.jl:704-711 -- sinon le nœud est mis à `nothing` par le
# nettoyage de fin de passe). MAIS `invalidate_all!` (src/graph_api.jl:325-326)
# SAUTE les nœuds `is_param=true` (`nd.is_param || (nd.valid = false)`) --
# donc si les 8 branches restent `is_param=true` au moment du run donneur
# suivant, leur valeur RESTERAIT bloquée sur celle du run receveur (jamais
# recalculée, `demand!` la trouverait `valid=true`). Ce script bascule donc
# `is_param` à `true` juste avant `backward_graph!`, copie immédiatement le
# gradient, puis le repasse à `false` AVANT le run donneur -- vérifié par une
# porte dédiée (G_RESET) ci-dessous plutôt que supposé correct.
#
# CPU uniquement, checkpoint réutilisé tel quel (AUCUN réentraînement), une
# seule passe arrière par paire (+ 2 passes avant receveur/donneur, déjà
# nécessaires pour capturer les activations) -- moins cher que l'identité
# d'ablation exacte (qui fait B+1 passes arrière par paire).
#
# Usage : julia --project=. notebook/bench_eps_atp_comparison_marker.jl
# Écrit : notebook/bench_eps_atp_comparison_marker_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, LinearAlgebra

const OUT  = joinpath(@__DIR__, "bench_eps_atp_comparison_marker_results.txt")
const CKPT = joinpath(@__DIR__, "marker_ckpt", "marker_task")
(isfile(CKPT * ".json") && isfile(CKPT * ".bin")) ||
    error("Checkpoint introuvable : $CKPT -- ce script ne réentraîne JAMAIS.")

# ── Construction du graphe (convention du projet : MARKER_STEPS=1/BATCH=2 pour
#    ne construire QUE le graphe et les fonctions utilitaires, puis on charge
#    le vrai checkpoint par-dessus) ────────────────────────────────────────────
ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))
# -> V, N_PAIRS, SEQ_LEN, MARKER_A, MARKER_B, SIGMA, N_LAYERS, DIM,
#    sample_marker_sequence (le g/ns/dev CUDA d'inclusion ne sont PAS réutilisés).

const N_HEADS = 4
const N_CTX   = 2 * N_PAIRS

const DEV_CPU = NeuroDSL.Backend.CPUDevice()
const NS = :marker_atp
G = NeuroDSL.NeuroGraph(namespace=NS, device=DEV_CPU)
NeuroDSL.load_graph!(G, NS, CKPT; overwrite=true)

function run_forward!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
end

# ── P1 (gate) sur le checkpoint fraîchement chargé ────────────────────────────
function check_p1(g, ns; n_eval=200, seed=999)
    rng = MersenneTwister(seed)
    okA = totA = okB = totB = 0
    for _ in 1:n_eval
        tokens, _, fmt, _, v = sample_marker_sequence(rng)
        pred = argmax(vec(run_forward!(g, ns, tokens)))
        if fmt == :A; totA += 1; okA += (pred == v) ? 1 : 0
        else;         totB += 1; okB += (pred == v) ? 1 : 0
        end
    end
    return (; acc_A = okA / totA, acc_B = okB / totB)
end
p1 = check_p1(G, NS)
p1_ok = p1.acc_A >= 0.95 && p1.acc_B >= 0.95
p1_ok || error("P1 non confirmé sur le checkpoint rechargé -- arrêt (p1=$p1)")

# ── Contraste P3, copié tel quel (même filtre, même fonction, même graine) ────
function sample_contrast(rng)
    while true
        k  = rand(rng, 1:V)
        sk = SIGMA[k]
        others = collect(setdiff(1:V, (k, sk)))
        rest = length(others) >= N_PAIRS - 2 ? shuffle(rng, others)[1:N_PAIRS-2] : Int[]
        ks = vcat([k, sk], rest)
        vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS)
            push!(ctx, ks[i]); push!(ctx, vs[i])
        end
        return (; ctx, k, sk, vk = vs[1], vsk = vs[2])
    end
end
donor_tokens(c)    = vcat(c.ctx, [MARKER_A, c.sk])
receiver_tokens(c) = vcat(c.ctx, [MARKER_B, c.sk])
delta_logit(out, c) = Float64(out[1, c.vsk] - out[1, c.vk])

function draw_search_pair!(g, ns, rng; max_tries=200)
    for _ in 1:max_tries
        c = sample_contrast(rng)
        d_out = run_forward!(g, ns, donor_tokens(c))
        r_out = run_forward!(g, ns, receiver_tokens(c))
        dd, dr = delta_logit(d_out, c), delta_logit(r_out, c)
        (argmax(vec(d_out)) == c.vsk && argmax(vec(r_out)) == c.vk && dd > 0 > dr) && return c
    end
    error("Impossible de tirer une paire de contraste exploitable en $max_tries essais")
end

# Même graine que marker_task_causal_analysis.jl / bench_eps_semantic_objective_marker.jl
# (rng_p3 = MersenneTwister(31415)) -- forward bit-identique (aucun réencâblage
# ici, contrairement au script sémantique), donc les 3 MÊMES paires sont retirées.
const N_PAIRS_ANALYSE = 3
rng_p3 = MersenneTwister(31415)
PAIRS = [draw_search_pair!(G, NS, rng_p3) for _ in 1:N_PAIRS_ANALYSE]

# ── Nœud custom dl_obj = logit(v(sigma(k))) - logit(v(k)), sans normalisation ─
# (identique à bench_eps_semantic_objective_marker.jl -- même contraste, même
# objectif brut, pour que la comparaison à q_j soit à métrique strictement égale.)
const VKI  = Ref(0)
const VSKI = Ref(0)
if !haskey(NeuroDSL.CUSTOM_OPS, :dl_logit_diff)
    NeuroDSL.register_op!(:dl_logit_diff,
        (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> begin
            fl = inputs[1]
            out_buf[1] = fl[1, VSKI[]] - fl[1, VKI[]]
            out_buf
        end)
    NeuroDSL.CUSTOM_SHAPE_RULES[:dl_logit_diff] = (inputs, attrs) -> (1,)
    NeuroDSL.GRAD_RULES[:dl_logit_diff] = (dev, dy, ctx, inputs) -> begin
        fl = inputs[1]
        gr = zeros(Float32, size(fl))
        gr[1, VSKI[]] = dy[1]
        gr[1, VKI[]]  = -dy[1]
        (NeuroDSL.Backend.to_device(dev, gr),)
    end
end
NeuroDSL.addrule!(G, NeuroDSL.GraphRule(:dl_obj, [:final_logits], :dl_logit_diff; namespace=NS))

# ── Les 8 branches (2 par couche x 4 couches), même ordre que les autres scripts ─
BRANCHES = Symbol[]
for l in 1:N_LAYERS
    push!(BRANCHES, Symbol("layer_$(l)_mha_output_out"))
    push!(BRANCHES, Symbol("layer_$(l)_mlp_out"))
end
@assert length(BRANCHES) == 8 "attendu B=8 pour 4 couches x 2 branches, obtenu $(length(BRANCHES))"

const GROUND_TRUTH = :layer_1_mlp_out   # P3, cité, pas recalculé

# ── q_j déjà mesuré (bench_eps_semantic_objective_marker_results.txt), CITÉ
#    tel quel, PAS recalculé -- mêmes 3 paires, même objectif dl, site :embed_sum ─
const QJ_SIGNED = Dict{Symbol,Vector{Float64}}(
    :layer_1_mha_output_out => [0.996546, 0.994625, 0.996547],
    :layer_1_mlp_out        => [0.190164, 0.692506, 0.199659],
    :layer_2_mha_output_out => [0.061667, 0.208872, 0.751129],
    :layer_2_mlp_out        => [0.575130, 0.248087, 0.147844],
    :layer_3_mha_output_out => [0.326623, 0.098118, 0.039648],
    :layer_3_mlp_out        => [0.120114, 0.046088, 0.076712],
    :layer_4_mha_output_out => [0.198806, 0.031643, 0.004198],
    :layer_4_mlp_out        => [0.321153, 0.564539, 0.013338],
)
const QJ_AGG_MEAN_ABS = Dict{Symbol,Float64}(b => sum(abs.(v)) / length(v) for (b, v) in QJ_SIGNED)

# ── Corrélation de rang de Spearman (sans dépendance externe) ─────────────────
function spearman(xs::Vector{Float64}, ys::Vector{Float64})
    n = length(xs)
    function ranks(v)
        ord = sortperm(v)
        r = zeros(Float64, n)
        for (i, idx) in enumerate(ord); r[idx] = i; end
        return r
    end
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx .- mx) .* (ry .- my))
    sx, sy = sqrt(sum((rx .- mx) .^ 2)), sqrt(sum((ry .- my) .^ 2))
    return cov / (sx * sy)
end

open(OUT, "w") do io
    emit(s="") = (println(io, s); println(s); flush(io))
    emitf(fmt, args...) = emit(Printf.format(Printf.Format(fmt), args...))

    emit("ATTRIBUTION PATCHING (AtP) vs q_j -- comparaison à famille causale égale")
    emit("tâche du marqueur. AtP_j = <grad_{a_j}(dl)|_{receveur}, a_j^donneur - a_j^receveur>,")
    emit("UNE seule passe arrière par paire (pas B+1). Checkpoint réutilisé tel quel,")
    emit("AUCUN réentraînement. Mêmes 3 paires que P3/q_j (graine MersenneTwister(31415)).")
    emit()
    emitf("P1 (checkpoint) : acc_A=%.4f  acc_B=%.4f", p1.acc_A, p1.acc_B)
    emit()
    emit("Branches (B=8, ordre en profondeur) : " * join(string.(BRANCHES), ", "))
    emitf("Circuit P3 (cité, patching d'activation, marker_task_causal_analysis.log) : %s",
          string(GROUND_TRUTH))
    emit()

    # ── Portes de correction ──────────────────────────────────────────────
    emit("═"^78); emit("PORTES DE CORRECTION"); emit("═"^78)

    c1 = PAIRS[1]
    VKI[] = c1.vk; VSKI[] = c1.vsk
    r_out1 = run_forward!(G, NS, receiver_tokens(c1))
    graph_dl  = Float64(Array(NeuroDSL.demand!(G, :dl_obj; namespace=NS))[1])
    manual_dl = delta_logit(r_out1, c1)
    val_ok = abs(graph_dl - manual_dl) < 1f-4
    emitf("VAL dl_obj (nœud graphe) == delta_logit manuel (paire 1, receveur) : graphe=%.6f  manuel=%.6f  écart=%.3e  %s",
          graph_dl, manual_dl, abs(graph_dl - manual_dl), val_ok ? "OK" : "ÉCHEC")

    # G_RESET : le cycle is_param=true (rétention gradient) -> false (avant le
    # run donneur) ne doit RIEN corrompre -- un run receveur repris après doit
    # redonner EXACTEMENT le même logits que r_out1.
    for b in BRANCHES; G.nodes[NS][b].is_param = true; end
    NeuroDSL.demand!(G, :dl_obj; namespace=NS)
    NeuroDSL.backward_graph!(G, :dl_obj; namespace=NS)
    _ = Dict(b => copy(Array(G.nodes[NS][b].gradient)) for b in BRANCHES)  # juste pour exercer le chemin
    for b in BRANCHES; G.nodes[NS][b].is_param = false; end
    _ = run_forward!(G, NS, donor_tokens(c1))   # run intermédiaire, comme dans la vraie boucle
    r_out1_again = run_forward!(G, NS, receiver_tokens(c1))
    g_reset_ok = (r_out1 == r_out1_again)
    emitf("G_RESET is_param cycle (true pour rétention grad, false avant run donneur) sans corruption : %s  (max|diff|=%.3e)",
          g_reset_ok ? "OK" : "ÉCHEC", maximum(abs.(r_out1 .- r_out1_again)))

    gates_ok = val_ok && g_reset_ok
    emit(gates_ok ? "\n✓ Les 2 portes passent -- les AtP_j ci-dessous sont lisibles." :
                    "\n✗ AU MOINS UNE PORTE A ÉCHOUÉ -- arrêt, les AtP_j ne seraient pas fiables.")
    gates_ok || return

    # ── AtP_j pour chaque paire ──────────────────────────────────────────────
    emit()
    emit("═"^78)
    emit("AtP_j PAR PAIRE (grad à la sortie de branche, run receveur ; delta = ")
    emit("activation donneur - activation receveur, tenseur complet du nœud)")
    emit("═"^78)

    ALL_ATP = Dict{Symbol,Vector{Float64}}(b => Float64[] for b in BRANCHES)
    LIN_CHECK = NamedTuple[]
    for (ip, c) in enumerate(PAIRS)
        VKI[] = c.vk; VSKI[] = c.vsk

        # Run receveur (clean) : forward + capture activation de branche + 1 passe arrière.
        r_out = run_forward!(G, NS, receiver_tokens(c))
        recv_act = Dict(b => copy(Array(G.nodes[NS][b].value)) for b in BRANCHES)
        dr = delta_logit(r_out, c)

        for b in BRANCHES; G.nodes[NS][b].is_param = true; end
        NeuroDSL.demand!(G, :dl_obj; namespace=NS)
        NeuroDSL.backward_graph!(G, :dl_obj; namespace=NS)
        grad_b = Dict(b => copy(Array(G.nodes[NS][b].gradient)) for b in BRANCHES)
        for b in BRANCHES; G.nodes[NS][b].is_param = false; end

        # Run donneur : forward + capture activation de branche (même noeud).
        d_out = run_forward!(G, NS, donor_tokens(c))
        donor_act = Dict(b => copy(Array(G.nodes[NS][b].value)) for b in BRANCHES)
        dd = delta_logit(d_out, c)

        ATP = Dict{Symbol,Float64}()
        for b in BRANCHES
            g_ = Float64.(grad_b[b]); a_diff = Float64.(donor_act[b]) .- Float64.(recv_act[b])
            ATP[b] = sum(g_ .* a_diff)
            push!(ALL_ATP[b], ATP[b])
        end
        push!(LIN_CHECK, (; ip, dd, dr, sum_atp = sum(values(ATP))))

        emit()
        emitf("Paire %d : k=%d sigma(k)=%d v(k)=%d v(sigma(k))=%d  dl_donneur=%.3f  dl_receveur=%.3f",
              ip, c.k, c.sk, c.vk, c.vsk, dd, dr)
        ranked = sort(BRANCHES; by = b -> -abs(ATP[b]))
        for (rk, b) in enumerate(ranked)
            emitf("  #%d  %-24s AtP_j = %+.6f   (q_j signé cité = %+.6f)",
                  rk, string(b), ATP[b], QJ_SIGNED[b][ip])
        end
        emitf("  Diagnostic Taylor 1er ordre : sum(AtP_j) = %+.6f   vs   dl_donneur - dl_receveur = %+.6f  (écart=%.6f)",
              sum(values(ATP)), dd - dr, abs(sum(values(ATP)) - (dd - dr)))
        rho_pair = spearman([ATP[b] for b in BRANCHES], [QJ_SIGNED[b][ip] for b in BRANCHES])
        emitf("  Spearman(AtP signé, q_j signé cité) pour cette paire : %.4f", rho_pair)
    end

    # ── Agrégat sur les 3 paires ────────────────────────────────────────────
    emit()
    emit("═"^78)
    emit("AGRÉGAT SUR LES 3 PAIRES")
    emit("═"^78)
    meanabs_atp = Dict(b => sum(abs.(ALL_ATP[b])) / length(ALL_ATP[b]) for b in BRANCHES)
    ranked_atp = sort(BRANCHES; by = b -> -meanabs_atp[b])
    ranked_qj  = sort(BRANCHES; by = b -> -QJ_AGG_MEAN_ABS[b])
    emit("  Classement AtP (|AtP_j| moyen)             Classement q_j (cité, |q_j| moyen)")
    for rk in 1:8
        ba, bq = ranked_atp[rk], ranked_qj[rk]
        emitf("  #%d  %-24s %.6f      #%d  %-24s %.6f",
              rk, string(ba), meanabs_atp[ba], rk, string(bq), QJ_AGG_MEAN_ABS[bq])
    end

    rho_agg = spearman([meanabs_atp[b] for b in BRANCHES], [QJ_AGG_MEAN_ABS[b] for b in BRANCHES])
    emit()
    emitf("Spearman(classement AtP agrégé, classement q_j agrégé cité), 8 branches : %.4f", rho_agg)

    top_atp = ranked_atp[1]
    match_top1 = top_atp == GROUND_TRUTH
    rank_truth_in_atp = findfirst(==(GROUND_TRUTH), ranked_atp)
    rank_truth_in_qj  = findfirst(==(GROUND_TRUTH), ranked_qj)
    emit()
    emit("═"^78)
    emit("VALIDATION DE LA MÉTHODE AtP ELLE-MÊME (sous-vérification, pas supposée)")
    emit("═"^78)
    emitf("  Branche #1 par |AtP_j| moyen : %s", string(top_atp))
    emitf("  Circuit P3 (indépendant)     : %s", string(GROUND_TRUTH))
    emitf("  Rang de %s dans le classement AtP : #%d / 8", string(GROUND_TRUTH), rank_truth_in_atp)
    emitf("  Rang de %s dans le classement q_j (rappel) : #%d / 8", string(GROUND_TRUTH), rank_truth_in_qj)
    emit(match_top1 ?
        "  => AtP RETROUVE le circuit P3 en position #1 : la méthode elle-même se valide sur ce banc d'essai." :
        "  => AtP NE RETROUVE PAS le circuit P3 en position #1 sur ce banc d'essai.")

    emit()
    emit("═"^78)
    emit("VERDICT")
    emit("═"^78)
    if match_top1 && rho_agg < 0.3
        emit("AtP retrouve le circuit P3 (validation de la méthode), mais son classement des 8")
        emit("branches est FAIBLEMENT corrélé (Spearman < 0.3) à celui de q_j : même comparé à une")
        emit("méthode de gradient de la MÊME famille causale que le patching, q_j ne reproduit pas")
        emit("l'ordre d'importance causale-sur-y. Le désaccord q_j vs P3 déjà observé n'est donc pas")
        emit("un artefact de familles de méthodes incompatibles -- q_j diverge aussi d'AtP.")
    elseif match_top1 && rho_agg >= 0.3
        emit("AtP retrouve le circuit P3 (validation de la méthode) ET son classement des 8 branches")
        emit("est corrélé positivement à celui de q_j (Spearman >= 0.3) : dans ce cas précis, malgré")
        emit("l'invariance par construction de q_j à y (Définition def:gate), son classement recoupe")
        emit("partiellement celui d'une méthode causale-sur-y établie. À nuancer par le classement")
        emit("détaillé ci-dessus (accord de rang, pas forcément de top-1) plutôt que généralisé.")
    else
        emit("AtP NE retrouve PAS le circuit P3 en position #1 sur ce banc d'essai précis (méthode")
        emit("non validée ici malgré sa garantie théorique de 1er ordre) -- la comparaison à q_j est")
        emit("donc à lire avec cette réserve : voir le classement détaillé et la corrélation de rang")
        emit("ci-dessus pour juger de l'accord/désaccord AtP vs q_j indépendamment de ce sous-test.")
    end
end

println("\nÉcrit : ", OUT)
