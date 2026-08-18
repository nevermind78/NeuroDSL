# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIF SÉMANTIQUE POUR L'IDENTITÉ D'ABLATION EXACTE (thm:ablation) --
# est-ce que la MÊME machinerie B+1-passes, appliquée à un contraste
# comportemental de tâche au lieu de l'objectif générique "ordre de branche
# moyen", localise quelque chose qui recoupe le circuit trouvé
# INDÉPENDAMMENT par patching d'activation (P3/P4, marker_task_causal_analysis.jl) ?
#
# CONTEXTE
# --------
# Toutes les mesures précédentes de q_j (bench_eps_branch_order.jl,
# bench_eps_exact_ablation_qwen.jl, bench_eps_fresh_ground_truth_check.jl)
# amorcent la passe arrière avec un objectif générique, architectural :
# la perte d'entraînement (:loss / :ce_loss) ou un readout synthétique sans
# contenu sémantique. Ici on amorce à la place avec le contraste EXACT que
# P3 utilise pour chercher le circuit causal de la "tâche du marqueur" :
#     dl = logit(v(sigma(k))) - logit(v(k))   à la position finale
# (lecture littérale moins lecture correcte), mais SANS la normalisation
# min-max (dl - dl_receveur)/(dl_donneur - dl_receveur) que P3 applique --
# ici dl BRUT sert directement d'objectif scalaire amorcé en sortie.
#
# CE QUI EST DÉJÀ ÉTABLI (ne PAS re-vérifier ici, redondant) :
#   - l'identité q_j = 1 - <g^(j->0),g>/||g||^2 est EXACTE pour ce mécanisme
#     de porte (gain=identité en avant, eps*dy en arrière), vérifié à 1e-7
#     près sur Qwen2.5-1.5B (bench_eps_exact_ablation_qwen.jl, portes P1/P2)
#     et à précision machine sur une instance synthétique (fresh_ground_truth).
#   - CE qui EST nouveau ici et DOIT être vérifié : (a) le réencâblage des 8
#     branches sur CE graphe (tâche du marqueur, 4 couches) est correct et
#     transparent en avant, (b) le nouveau nœud custom `dl_obj` calcule
#     exactement le même scalaire que la fonction Julia `delta_logit` déjà
#     validée par P3, (c) le nœud custom est correctement câblé pour la passe
#     arrière (comparaison indépendante contre un graphe SANS aucune porte
#     eps). Ces trois portes sont testées explicitement plus bas.
#
# SITE DE LECTURE CHOISI : le gradient de dl_obj PAR RAPPORT À :embed_sum
# (racine du réseau, en amont des 8 branches). C'est un site FIXE, commun à
# tous les q_j (contrairement à utiliser le gradient capturé À la branche
# elle-même comme site : ce choix est dégénéré, q_jj = 1 par construction
# même mécanisme de capture -- porte P1 "faible" documentée dans
# bench_eps_exact_ablation_qwen.jl). :embed_sum est en amont de TOUTES les
# branches, donc aucune n'est "hors cône" par construction : les 8 q_j sont
# a priori tous susceptibles d'être non nuls, ce qui est le régime le plus
# informatif pour classer les branches entre elles.
#
# CHECKPOINT : réutilisé tel quel (aucun réentraînement). Convention du
# projet (marker_task_adiag.jl, verify_curvature_bound.jl, etc.) :
# `include(marker_task_experiment.jl)` avec MARKER_STEPS=1/MARKER_BATCH=2
# pour ne construire QUE le graphe et les fonctions utilitaires (le mini
# "entraînement" d'un pas est un bruit de fond négligeable sur CUDA, jamais
# utilisé), puis on reconstruit un graphe CPU frais et `load_graph!` y
# écrase les poids avec le checkpoint réel (notebook/marker_ckpt/marker_task,
# P1 = 0.973/0.991 -- log : marker_task_causal_analysis.log).
#
# GROUND TRUTH INDÉPENDANTE (déjà mesurée, PAS re-dérivée ici) :
# marker_task_causal_analysis.log, section P3 : sur 3 paires de recherche
# indépendantes, "layer_1_mlp_out" est le site individuel le PLUS fort dans
# les 3 paires (r = 1.125 / 1.009 / 1.144) ET le SEUL site qui survit
# `backward_prune!` dans les 3 paires -> circuit consensus = [:layer_1_mlp_out]
# (P3 VALIDÉ : r médian = 1.043 sur 20 paires de validation indépendantes,
# bascule argmax = 0.85). P4 (ablation de poids) n'a PAS validé au niveau des
# poids (littéral_B = 0.42, sous le seuil 0.9), donc la lecture correcte est
# distribuée au niveau des poids même si l'ablation d'ACTIVATION localise
# proprement au niveau du site.
#
# USAGE : julia --project=. notebook/bench_eps_semantic_objective_marker.jl
# Écrit : notebook/bench_eps_semantic_objective_marker_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, LinearAlgebra

const OUT  = joinpath(@__DIR__, "bench_eps_semantic_objective_marker_results.txt")
const CKPT = joinpath(@__DIR__, "marker_ckpt", "marker_task")
(isfile(CKPT * ".json") && isfile(CKPT * ".bin")) ||
    error("Checkpoint introuvable : $CKPT -- ce script ne réentraîne JAMAIS.")

# ── Construction du graphe + fonctions utilitaires (aucun réentraînement réel :
#    1 pas x batch 2, convention du projet -- voir marker_task_adiag.jl) ──────
ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))
# -> V, N_PAIRS, SEQ_LEN, MARKER_A, MARKER_B, SIGMA, N_LAYERS, DIM,
#    sample_marker_sequence, build_marker_graph (le `g`/`ns`/`dev` du CUDA
#    d'inclusion ne sont PAS réutilisés ci-dessous -- on reconstruit du CPU).

const N_HEADS = 4
const N_CTX   = 2 * N_PAIRS

# ── Deux graphes CPU frais, chargés depuis le MÊME checkpoint ────────────────
#    NS_MAIN : réencâblé avec les 8 portes eps (analyse).
#    NS_PLAIN : intact, aucune porte -- sert de témoin indépendant pour les
#    portes de correction G1/G2 (rien de commun avec le mécanisme eps).
const DEV_CPU = NeuroDSL.Backend.CPUDevice()
const NS_MAIN  = :marker_semantic_main
const NS_PLAIN = :marker_semantic_plain

G_MAIN  = NeuroDSL.NeuroGraph(namespace=NS_MAIN,  device=DEV_CPU)
G_PLAIN = NeuroDSL.NeuroGraph(namespace=NS_PLAIN, device=DEV_CPU)
NeuroDSL.load_graph!(G_MAIN,  NS_MAIN,  CKPT; overwrite=true)
NeuroDSL.load_graph!(G_PLAIN, NS_PLAIN, CKPT; overwrite=true)

function run_forward!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
end

# ── P1 (gate) sur les DEUX graphes fraîchement chargés ────────────────────────
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

p1_main  = check_p1(G_MAIN,  NS_MAIN)
p1_plain = check_p1(G_PLAIN, NS_PLAIN)
p1_ok = p1_main.acc_A >= 0.95 && p1_main.acc_B >= 0.95 &&
        p1_plain.acc_A >= 0.95 && p1_plain.acc_B >= 0.95
p1_ok || error("P1 non confirmé sur le checkpoint rechargé -- arrêt (main=$p1_main, plain=$p1_plain)")

# ── Contraste P3, copié tel quel (mêmes filtres, même fonction) ──────────────
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

# Même graine que marker_task_causal_analysis.jl (rng_p3 = MersenneTwister(31415)) :
# comme le forward est bit-identique après réencâblage (porte G1 ci-dessous),
# les 3 MÊMES paires que dans le log P3 sont retirées -- comparaison directe,
# pas seulement "même distribution".
const N_PAIRS_ANALYSE = 3
rng_p3 = MersenneTwister(31415)
PAIRS = [draw_search_pair!(G_PLAIN, NS_PLAIN, rng_p3) for _ in 1:N_PAIRS_ANALYSE]

# ── Nœud custom dl_obj = logit(v(sigma(k))) - logit(v(k)), sans normalisation ─
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
NeuroDSL.addrule!(G_MAIN,  NeuroDSL.GraphRule(:dl_obj, [:final_logits], :dl_logit_diff; namespace=NS_MAIN))
NeuroDSL.addrule!(G_PLAIN, NeuroDSL.GraphRule(:dl_obj, [:final_logits], :dl_logit_diff; namespace=NS_PLAIN))

# :embed_sum retenu après backward (nœud intermédiaire normalement rendu au
# pool -- is_param=true est le SEUL levier de rétention, src/backward.jl:704-711
# et 894-901 : aucun autre effet de bord ici puisqu'on n'appelle jamais
# `NeuroDSL.params(g)` ni d'optimiseur sur ces deux graphes d'analyse).
G_MAIN.nodes[NS_MAIN][:embed_sum].is_param   = true
G_PLAIN.nodes[NS_PLAIN][:embed_sum].is_param = true

function embed_grad!(g, ns, tokens, c)
    VKI[] = c.vk; VSKI[] = c.vsk
    run_forward!(g, ns, tokens)
    NeuroDSL.demand!(g, :dl_obj; namespace=ns)
    NeuroDSL.backward_graph!(g, :dl_obj; namespace=ns)
    return copy(Array(g.nodes[ns][:embed_sum].gradient))
end

# ── Les 8 branches (2 par couche x 4 couches) et le réencâblage eps ──────────
const EPSV = Dict{Symbol,Float32}()
function ensure_eps_op!(branch::Symbol)
    op = Symbol("epsop_", branch)
    EPSV[branch] = 1.0f0
    haskey(NeuroDSL.CUSTOM_OPS, op) && return op
    NeuroDSL.register_op!(op,
        (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> (out_buf .= inputs[1]))
    NeuroDSL.CUSTOM_SHAPE_RULES[op] = (inputs, attrs) -> size(inputs[1])
    NeuroDSL.GRAD_RULES[op] = (dev, dy, ctx, inputs) -> begin
        e = EPSV[branch]
        (e .* dy,)
    end
    return op
end

BRANCHES = Symbol[]
for l in 1:N_LAYERS
    push!(BRANCHES, Symbol("layer_$(l)_mha_output_out"))
    push!(BRANCHES, Symbol("layer_$(l)_mlp_out"))
end
@assert length(BRANCHES) == 8 "attendu B=8 pour 4 couches x 2 branches, obtenu $(length(BRANCHES))"

for l in 1:N_LAYERS
    for (join_sym, branch_sym) in ((Symbol("layer_$(l)_res1"), Symbol("layer_$(l)_mha_output_out")),
                                    (Symbol("layer_$(l)_out"),  Symbol("layer_$(l)_mlp_out")))
        r = G_MAIN.rules[NS_MAIN][join_sym]
        r.inputs[2] == branch_sym ||
            error("$join_sym : branche attendue en 2e position, trouvé $(r.inputs[2])")
        eps_sym = Symbol("eps_", branch_sym)
        NeuroDSL.addrule!(G_MAIN, NeuroDSL.GraphRule(eps_sym, [branch_sym],
                                                      ensure_eps_op!(branch_sym); namespace=NS_MAIN))
        NeuroDSL.addrule!(G_MAIN, NeuroDSL.GraphRule(join_sym, [r.inputs[1], eps_sym], r.op;
                                                      attrs=r.attrs, namespace=NS_MAIN, atom_type=r.atom_type))
        NeuroDSL._invalidate_downstream!(G_MAIN, join_sym, NS_MAIN)
    end
end

open(OUT, "w") do io
    emit(s="") = (println(io, s); println(s); flush(io))
    emitf(fmt, args...) = emit(Printf.format(Printf.Format(fmt), args...))

    emit("OBJECTIF SÉMANTIQUE POUR L'IDENTITÉ D'ABLATION EXACTE -- tâche du marqueur")
    emit("Amorçage de la passe arrière avec dl = logit(v(sigma(k))) - logit(v(k))")
    emit("(contraste P3, SANS normalisation), au lieu de l'objectif générique habituel.")
    emit("Checkpoint réutilisé tel quel (P1 vérifié ci-dessous, AUCUN réentraînement).")
    emit()
    emitf("P1 (checkpoint, graphe réencâblé)  : acc_A=%.4f  acc_B=%.4f", p1_main.acc_A, p1_main.acc_B)
    emitf("P1 (checkpoint, graphe témoin)     : acc_A=%.4f  acc_B=%.4f", p1_plain.acc_A, p1_plain.acc_B)
    emit()
    emit("Branches (B=8, ordre en profondeur) : " * join(string.(BRANCHES), ", "))
    emit()

    # ── Portes de correction ──────────────────────────────────────────────
    emit("═"^78); emit("PORTES DE CORRECTION"); emit("═"^78)

    c1 = PAIRS[1]
    d_out_plain = run_forward!(G_PLAIN, NS_PLAIN, receiver_tokens(c1))
    d_out_main  = run_forward!(G_MAIN,  NS_MAIN,  receiver_tokens(c1))
    g1 = (d_out_plain == d_out_main)
    emitf("G1  forward bit-identique (réencâblé eps=1 vs témoin sans porte) : %s  (max|diff|=%.3e)",
          g1 ? "OK" : "ÉCHEC", maximum(abs.(d_out_plain .- d_out_main)))

    VKI[] = c1.vk; VSKI[] = c1.vsk
    graph_dl   = Float64(Array(NeuroDSL.demand!(G_PLAIN, :dl_obj; namespace=NS_PLAIN))[1])
    manual_dl  = delta_logit(d_out_plain, c1)
    val_ok = abs(graph_dl - manual_dl) < 1f-4
    emitf("VAL dl_obj (nœud graphe) == delta_logit manuel : graphe=%.6f  manuel=%.6f  écart=%.3e  %s",
          graph_dl, manual_dl, abs(graph_dl - manual_dl), val_ok ? "OK" : "ÉCHEC")

    for b in BRANCHES; EPSV[b] = 1.0f0; end
    g_base_plain = embed_grad!(G_PLAIN, NS_PLAIN, receiver_tokens(c1), c1)
    g_base_main  = embed_grad!(G_MAIN,  NS_MAIN,  receiver_tokens(c1), c1)
    g2 = (g_base_plain == g_base_main)
    emitf("G2  gradient embed_sum bit-identique (eps=1 vs témoin) : %s  (max|diff|=%.3e)",
          g2 ? "OK" : "ÉCHEC", maximum(abs.(g_base_plain .- g_base_main)))

    gates_ok = g1 && val_ok && g2
    emit(gates_ok ? "\n✓ Les 3 portes passent -- les q_j ci-dessous sont lisibles." :
                    "\n✗ AU MOINS UNE PORTE A ÉCHOUÉ -- arrêt, les q_j ne seraient pas fiables.")
    gates_ok || return

    # ── q_j pour chaque paire, objectif = dl(recepteur), site = embed_sum ──
    emit()
    emit("═"^78)
    emit("q_j PAR PAIRE (objectif dl amorcé sur l'entrée RECEVEUR = lecture correcte attendue,")
    emit("site de lecture = gradient à :embed_sum, en amont des 8 branches)")
    emit("═"^78)

    ALLQ = Dict{Symbol,Vector{Float64}}(b => Float64[] for b in BRANCHES)
    for (ip, c) in enumerate(PAIRS)
        for b in BRANCHES; EPSV[b] = 1.0f0; end
        gbase = embed_grad!(G_MAIN, NS_MAIN, receiver_tokens(c), c)
        n2g = Float64(sum(abs2, gbase))
        emit()
        emitf("Paire %d : k=%d sigma(k)=%d v(k)=%d v(sigma(k))=%d  ||embed_grad||^2=%.6e",
              ip, c.k, c.sk, c.vk, c.vsk, n2g)
        Q = Dict{Symbol,Float64}()
        for bj in BRANCHES
            EPSV[bj] = 0.0f0
            g0 = embed_grad!(G_MAIN, NS_MAIN, receiver_tokens(c), c)
            EPSV[bj] = 1.0f0
            dotg0g = Float64(sum(g0 .* gbase))
            Q[bj] = 1.0 - dotg0g / n2g
            push!(ALLQ[bj], Q[bj])
        end
        ranked = sort(BRANCHES; by = b -> -abs(Q[b]))
        for (rk, b) in enumerate(ranked)
            emitf("  #%d  %-24s q_j = %+.6f", rk, string(b), Q[b])
        end
        emitf("  somme des q_j (diagnostic, pas borné a priori) : %+.6f", sum(values(Q)))
    end

    # ── Agrégat sur les 3 paires ────────────────────────────────────────────
    emit()
    emit("═"^78)
    emit("AGRÉGAT SUR LES 3 PAIRES")
    emit("═"^78)
    meanabs = Dict(b => sum(abs.(ALLQ[b])) / length(ALLQ[b]) for b in BRANCHES)
    ranked_agg = sort(BRANCHES; by = b -> -meanabs[b])
    for (rk, b) in enumerate(ranked_agg)
        emitf("  #%d  %-24s  |q_j| moyen = %.6f   valeurs = %s",
              rk, string(b), meanabs[b], string(round.(ALLQ[b], digits=4)))
    end

    top_branch = ranked_agg[1]
    ground_truth = :layer_1_mlp_out
    emit()
    emit("═"^78)
    emit("COMPARAISON AVEC LE CIRCUIT INDÉPENDANT (P3, marker_task_causal_analysis.log)")
    emit("═"^78)
    emitf("  Circuit P3 (patching d'activation, consensus sur 3 paires) : %s", string(ground_truth))
    emitf("  Branche #1 par |q_j| moyen (objectif sémantique, cette expérience) : %s", string(top_branch))
    match_top1 = top_branch == ground_truth
    rank_of_truth = findfirst(==(ground_truth), ranked_agg)
    emitf("  Rang de %s dans le classement |q_j| : #%d / 8", string(ground_truth), rank_of_truth)
    emit()
    if match_top1
        emit("=> ACCORD : la branche la plus causale sous l'objectif sémantique NOUVEAU coïncide")
        emit("   avec le circuit trouvé indépendamment par patching d'activation (P3). Le même")
        emit("   mécanisme d'ablation exacte, amorcé sur un contraste comportemental au lieu de")
        emit("   l'ordre de branche générique, localise le MÊME site -- pas seulement un site à")
        emit("   grande norme.")
    else
        emit("=> DÉSACCORD : la branche la plus causale sous l'objectif sémantique NE coïncide PAS")
        emit("   avec le circuit P3. C'est un résultat négatif réel, pas un échec d'implémentation")
        emit("   (les 3 portes de correction ci-dessus passent) : la méthode d'ablation exacte,")
        emit("   amorcée sur ce contraste précis et lue à :embed_sum, met en avant un site DIFFÉRENT")
        emit("   du circuit d'activation P3. Les deux méthodes ne mesurent pas rigoureusement la")
        emit("   même chose (site de lecture différent, ablation en arrière du gradient contre")
        emit("   patching en avant de la valeur), donc le désaccord est informatif sur la LIMITE de")
        emit("   la généralisation \"objectif sémantique + q_j\" -> \"circuit d'activation\", pas une")
        emit("   réfutation de l'un ou l'autre résultat pris isolément.")
    end
end

println("\nÉcrit : ", OUT)
