# ══════════════════════════════════════════════════════════════════════════════
# CONTRÔLE : L'OBJECTIF GÉNÉRIQUE (perte d'entraînement, :loss) DOMINE-T-IL
# AUSSI PAR layer_1_mha_output_out À :embed_sum ?
#
# Sépare "artefact du SITE de lecture" de "artefact de L'OBJECTIF employé".
# bench_eps_semantic_objective_marker.jl a trouvé layer_1_mha_output_out
# dominant (|q_j|~0.996) à :embed_sum sous l'objectif SÉMANTIQUE dl =
# logit(v(sigma(k))) - logit(v(k)). Ici : MÊME checkpoint, MÊMES 3 paires
# (même graine rng_p3=31415), MÊME site de lecture :embed_sum, MÊME mécanisme
# d'ablation à B+1 passes -- mais objectif = :loss (la perte d'entraînement
# native du graphe, cross-entropy à la position finale contre le VRAI label
# v(k), sur l'entrée RECEVEUR), l'objectif "générique, architectural" utilisé
# partout ailleurs dans ce projet (bench_eps_branch_order.jl,
# bench_eps_exact_ablation_qwen.jl -- cf. l'en-tête de bench_eps_semantic_
# objective_marker.jl, lignes 8-14, qui nomme explicitement :loss/:ce_loss
# comme la référence "générique" par opposition au contraste sémantique dl).
#
# PRÉDICTION TESTÉE
#   Si layer_1_mha_output_out domine AUSSI sous :loss à :embed_sum : la
#   domination est un artefact du SITE (indépendant de l'objectif), séparation
#   nette et décisive avec le résultat de bench_eps_semantic_objective_marker_
#   readsite_sweep.jl (qui isole la même chose côté site).
#   Si un AUTRE branchement domine sous :loss : l'objectif sémantique jouait un
#   rôle, la conclusion est plus nuancée qu'un pur artefact de site.
#
# Mêmes portes de correction (G1 forward bit-identique, G2 gradient bit-
# identique à :embed_sum) + une porte de cohérence VAL (loss du graphe ==
# cross-entropy manuelle calculée depuis les logits). Checkpoint réutilisé tel
# quel, AUCUN réentraînement.
#
# USAGE : julia --project=. notebook/bench_eps_semantic_objective_marker_generic_control.jl
# Écrit : notebook/bench_eps_semantic_objective_marker_generic_control_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, LinearAlgebra

const OUT  = joinpath(@__DIR__, "bench_eps_semantic_objective_marker_generic_control_results.txt")
const CKPT = joinpath(@__DIR__, "marker_ckpt", "marker_task")
(isfile(CKPT * ".json") && isfile(CKPT * ".bin")) ||
    error("Checkpoint introuvable : $CKPT -- ce script ne réentraîne JAMAIS.")

ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))

const N_HEADS = 4
const N_CTX   = 2 * N_PAIRS

const DEV_CPU = NeuroDSL.Backend.CPUDevice()
const NS_MAIN  = :marker_generic_main
const NS_PLAIN = :marker_generic_plain

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

# ── Mêmes 3 paires de contraste que le script sémantique original (même
#    graine rng_p3=31415) -- on ne s'en sert ici QUE pour fixer les mêmes
#    entrées RECEVEUR (contexte + [m_B, sigma(k)]) et le VRAI label v(k), afin
#    que la comparaison porte sur l'OBJECTIF, rien d'autre.
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

const N_PAIRS_ANALYSE = 3
rng_p3 = MersenneTwister(31415)
PAIRS = [draw_search_pair!(G_PLAIN, NS_PLAIN, rng_p3) for _ in 1:N_PAIRS_ANALYSE]

# ── :embed_sum retenu, MÊME site que le script sémantique original ──────────
G_MAIN.nodes[NS_MAIN][:embed_sum].is_param   = true
G_PLAIN.nodes[NS_PLAIN][:embed_sum].is_param = true

function set_label!(g, ns, tokens, v_true)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :final_label, [v_true]; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
end

function embed_grad!(g, ns, tokens, v_true)
    set_label!(g, ns, tokens, v_true)
    NeuroDSL.demand!(g, :loss; namespace=ns)
    NeuroDSL.backward_graph!(g, :loss; namespace=ns)
    return copy(Array(g.nodes[ns][:embed_sum].gradient))
end

# ── Les 8 branches et le réencâblage eps (identique à l'original) ────────────
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

    emit("CONTRÔLE -- OBJECTIF GÉNÉRIQUE (:loss, perte d'entraînement) au site")
    emit(":embed_sum, sur le MÊME checkpoint et les MÊMES 3 paires que")
    emit("bench_eps_semantic_objective_marker.jl. Aucun réentraînement.")
    emit()
    emitf("P1 (checkpoint, graphe réencâblé)  : acc_A=%.4f  acc_B=%.4f", p1_main.acc_A, p1_main.acc_B)
    emitf("P1 (checkpoint, graphe témoin)     : acc_A=%.4f  acc_B=%.4f", p1_plain.acc_A, p1_plain.acc_B)
    emit()
    emit("Branches (B=8, ordre en profondeur) : " * join(string.(BRANCHES), ", "))
    emit()

    emit("═"^78); emit("PORTES DE CORRECTION"); emit("═"^78)

    c1 = PAIRS[1]
    d_out_plain = run_forward!(G_PLAIN, NS_PLAIN, receiver_tokens(c1))
    d_out_main  = run_forward!(G_MAIN,  NS_MAIN,  receiver_tokens(c1))
    g1 = (d_out_plain == d_out_main)
    emitf("G1  forward bit-identique (réencâblé eps=1 vs témoin sans porte) : %s  (max|diff|=%.3e)",
          g1 ? "OK" : "ÉCHEC", maximum(abs.(d_out_plain .- d_out_main)))

    # VAL : loss du graphe == cross-entropy manuelle depuis les logits déjà
    # calculés (stabilité numérique par max-shift, comme le kernel du moteur).
    set_label!(G_PLAIN, NS_PLAIN, receiver_tokens(c1), c1.vk)
    graph_loss = Float64(Array(NeuroDSL.demand!(G_PLAIN, :loss; namespace=NS_PLAIN))[1])
    logits_row = vec(d_out_plain[1, :])
    mx = maximum(logits_row)
    manual_loss = -( (logits_row[c1.vk] - mx) - log(sum(exp.(logits_row .- mx))) )
    val_ok = abs(graph_loss - manual_loss) < 1f-4
    emitf("VAL :loss (nœud graphe) == cross-entropy manuelle : graphe=%.6f  manuel=%.6f  écart=%.3e  %s",
          graph_loss, manual_loss, abs(graph_loss - manual_loss), val_ok ? "OK" : "ÉCHEC")

    for b in BRANCHES; EPSV[b] = 1.0f0; end
    g_base_plain = embed_grad!(G_PLAIN, NS_PLAIN, receiver_tokens(c1), c1.vk)
    g_base_main  = embed_grad!(G_MAIN,  NS_MAIN,  receiver_tokens(c1), c1.vk)
    g2 = (g_base_plain == g_base_main)
    emitf("G2  gradient embed_sum bit-identique (eps=1 vs témoin) : %s  (max|diff|=%.3e)",
          g2 ? "OK" : "ÉCHEC", maximum(abs.(g_base_plain .- g_base_main)))

    gates_ok = g1 && val_ok && g2
    emit(gates_ok ? "\n✓ Les 3 portes passent -- les q_j ci-dessous sont lisibles." :
                    "\n✗ AU MOINS UNE PORTE A ÉCHOUÉ -- arrêt, les q_j ne seraient pas fiables.")
    gates_ok || return

    emit()
    emit("═"^78)
    emit("q_j PAR PAIRE (objectif :loss amorcé sur l'entrée RECEVEUR contre le VRAI")
    emit("label v(k), site de lecture = gradient à :embed_sum)")
    emit("═"^78)

    ALLQ = Dict{Symbol,Vector{Float64}}(b => Float64[] for b in BRANCHES)
    for (ip, c) in enumerate(PAIRS)
        for b in BRANCHES; EPSV[b] = 1.0f0; end
        gbase = embed_grad!(G_MAIN, NS_MAIN, receiver_tokens(c), c.vk)
        n2g = Float64(sum(abs2, gbase))
        emit()
        emitf("Paire %d : k=%d sigma(k)=%d v(k)=%d v(sigma(k))=%d  ||embed_grad||^2=%.6e",
              ip, c.k, c.sk, c.vk, c.vsk, n2g)
        Q = Dict{Symbol,Float64}()
        for bj in BRANCHES
            EPSV[bj] = 0.0f0
            g0 = embed_grad!(G_MAIN, NS_MAIN, receiver_tokens(c), c.vk)
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
    emit()
    emit("═"^78)
    emit("COMPARAISON AVEC LE RÉSULTAT SÉMANTIQUE (bench_eps_semantic_objective_marker.jl)")
    emit("═"^78)
    emitf("  Branche #1 sous l'objectif SÉMANTIQUE (dl, même site) : layer_1_mha_output_out")
    emitf("  Branche #1 sous l'objectif GÉNÉRIQUE (:loss, ce script) : %s", string(top_branch))
    same_top = top_branch == :layer_1_mha_output_out
    emit()
    if same_top
        emit("=> layer_1_mha_output_out DOMINE AUSSI sous l'objectif générique. La")
        emit("   domination à :embed_sum est donc INDÉPENDANTE du choix d'objectif")
        emit("   (sémantique vs générique) : c'est un effet du SITE de lecture, pas du")
        emit("   contenu sémantique de dl. Séparation nette avec le test de site")
        emit("   (bench_eps_semantic_objective_marker_readsite_sweep.jl).")
    else
        emit("=> Un AUTRE branchement domine sous l'objectif générique -- la domination")
        emit("   de layer_1_mha_output_out sous dl n'est PAS reproduite par un objectif")
        emit("   générique au même site : le résultat sémantique original n'est pas un")
        emit("   pur artefact de site, l'objectif joue un rôle mesurable.")
    end
end

println("\nÉcrit : ", OUT)
