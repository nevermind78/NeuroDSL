# ══════════════════════════════════════════════════════════════════════════════
# TEST DE L'ARTEFACT DE PROXIMITÉ -- déplacer le site de lecture loin de
# :embed_sum, sur le MÊME objectif sémantique dl = logit(v(sigma(k))) -
# logit(v(k)) (contraste P3, marker task), pour savoir si la domination quasi-
# totale de `layer_1_mha_output_out` (|q_j|~0.996, bench_eps_semantic_objective
# _marker.jl) est un signal réel ou un artefact de proximité entre :embed_sum
# et cette branche (immédiatement en aval, rien entre les deux).
#
# CONTRAINTE STRUCTURELLE DE L'IDENTITÉ (À NE PAS IGNORER)
# ----------------------------------------------------------
# q_j = 1 - <g^(j->0),g>/||g||^2 n'est NON NUL que pour les branches EN AVAL du
# site de lecture (porte P2 de bench_eps_exact_ablation_qwen.jl : une branche
# EN AMONT du site lu n'est sur aucun chemin site->perte, q_j=0 EXACTEMENT).
# :embed_sum est le SEUL nœud en amont des 8 branches SIMULTANÉMENT -- c'est
# pour ça qu'il a été choisi à l'origine, pas par hasard. Un site "symétrique"
# placé juste après la couche 4 (ce que la demande initiale suggérait) serait
# en AVAL de TOUTES les branches : les 8 q_j y seraient EXACTEMENT nuls par
# construction (aucun chemin), un résultat dégénéré et non informatif, pas un
# test de la domination par proximité. Vérifié par calcul direct de la règle
# de gradient avant d'écrire une seule ligne de ce script.
#
# LE TEST VALIDE, À LA PLACE
# ---------------------------
# On déplace le site de lecture vers l'aval PAS À PAS, en restant du côté où
# il reste informatif, avec deux sites choisis pour leur rôle structurel :
#   RS1 = :embed_sum       (référence, déjà mesuré) -- voisin immédiat de
#                           layer_1_mha_output_out (co-entrée de SA jonction)
#   RS2 = :layer_1_res1    -- voisin immédiat de layer_1_mlp_out (MÊME rôle
#                           structurel que RS1/layer_1_mha : co-entrée de la
#                           jonction layer_1_out = add(layer_1_res1, eps_mlp)).
#                           layer_1_mha_output_out y est EXCLU (q=0 exact, en
#                           amont du site). Coïncide avec le circuit P3 --
#                           n'isole PAS proximité de signal à lui seul.
#   RS3 = :layer_2_res1    -- voisin immédiat de layer_2_mha_output_out, MÊME
#                           rôle structurel encore, une couche plus loin.
#                           layer_1_mha_output_out ET layer_1_mlp_out (circuit
#                           P3) y sont TOUS DEUX EXCLUS (q=0 exact -- en amont).
#                           layer_2_mha_output_out N'EST PAS le circuit P3 (il
#                           était classé #3 dans l'agrégat original, |q|=0.34).
#                           C'est le test DÉCISIF : si layer_2_mha_output_out
#                           domine ici comme layer_1_mha_output_out dominait à
#                           RS1, la domination est un pur artefact de position
#                           (voisin immédiat du site = dominant, indépendamment
#                           de tout rôle causal), démontré sur une branche qui
#                           n'a AUCUNE prétention à être le circuit réel.
#
# Les 3 sites sont mesurés en UNE SEULE passe arrière par (paire, branche
# ablatée) -- is_param=true sur les 3 nœuds simultanément, aucun coût
# supplémentaire par rapport à mesurer un seul site.
#
# Mêmes 3 paires (même graine rng_p3=31415, même contraste dl, SANS
# normalisation), même checkpoint, mêmes portes de correction (G1 forward
# bit-identique, VAL dl_obj==delta_logit manuel, G2 gradient bit-identique --
# une instance par site de lecture). Aucun réentraînement.
#
# USAGE : julia --project=. notebook/bench_eps_semantic_objective_marker_readsite_sweep.jl
# Écrit : notebook/bench_eps_semantic_objective_marker_readsite_sweep_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, LinearAlgebra

const OUT  = joinpath(@__DIR__, "bench_eps_semantic_objective_marker_readsite_sweep_results.txt")
const CKPT = joinpath(@__DIR__, "marker_ckpt", "marker_task")
(isfile(CKPT * ".json") && isfile(CKPT * ".bin")) ||
    error("Checkpoint introuvable : $CKPT -- ce script ne réentraîne JAMAIS.")

ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))

const N_HEADS = 4
const N_CTX   = 2 * N_PAIRS

const DEV_CPU = NeuroDSL.Backend.CPUDevice()
const NS_MAIN  = :marker_rssweep_main
const NS_PLAIN = :marker_rssweep_plain

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

# Même graine que marker_task_causal_analysis.jl et bench_eps_semantic_objective
# _marker.jl : les 3 MÊMES paires (comparaison directe avec le résultat déjà
# obtenu à :embed_sum, pas seulement même distribution).
const N_PAIRS_ANALYSE = 3
rng_p3 = MersenneTwister(31415)
PAIRS = [draw_search_pair!(G_PLAIN, NS_PLAIN, rng_p3) for _ in 1:N_PAIRS_ANALYSE]

# ── Nœud custom dl_obj, identique au script d'origine ────────────────────────
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

# ── Les 3 sites de lecture, retenus SIMULTANÉMENT sur les DEUX graphes ───────
const READSITES = [:embed_sum, :layer_1_res1, :layer_2_res1]
const NEAREST_BRANCH = Dict(
    :embed_sum    => :layer_1_mha_output_out,
    :layer_1_res1 => :layer_1_mlp_out,
    :layer_2_res1 => :layer_2_mha_output_out,
)
for s in READSITES
    G_MAIN.nodes[NS_MAIN][s].is_param   = true
    G_PLAIN.nodes[NS_PLAIN][s].is_param = true
end

function sites_grad!(g, ns, tokens, c)
    VKI[] = c.vk; VSKI[] = c.vsk
    run_forward!(g, ns, tokens)
    NeuroDSL.demand!(g, :dl_obj; namespace=ns)
    NeuroDSL.backward_graph!(g, :dl_obj; namespace=ns)
    return Dict(s => copy(Array(g.nodes[ns][s].gradient)) for s in READSITES)
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

    emit("TEST DE L'ARTEFACT DE PROXIMITÉ -- balayage du site de lecture")
    emit("Même objectif dl = logit(v(sigma(k))) - logit(v(k)) (contraste P3, SANS")
    emit("normalisation), mêmes 3 paires, même checkpoint (AUCUN réentraînement).")
    emit()
    emitf("P1 (checkpoint, graphe réencâblé)  : acc_A=%.4f  acc_B=%.4f", p1_main.acc_A, p1_main.acc_B)
    emitf("P1 (checkpoint, graphe témoin)     : acc_A=%.4f  acc_B=%.4f", p1_plain.acc_A, p1_plain.acc_B)
    emit()
    emit("Branches (B=8, ordre en profondeur) : " * join(string.(BRANCHES), ", "))
    emit()
    emit("Sites de lecture testés, et branche STRUCTURELLEMENT voisine de chacun")
    emit("(co-entrée de la MÊME jonction résiduelle que la branche, par construction --")
    emit("pas une mesure, une propriété du graphe) :")
    for s in READSITES
        emitf("  %-14s -> voisine attendue : %s", string(s), string(NEAREST_BRANCH[s]))
    end
    emit()
    emit("Rappel de la contrainte structurelle de l'identité q_j : q_j = 0 EXACTEMENT")
    emit("pour toute branche EN AMONT du site lu (porte P2 du script Qwen). Donc :")
    emit("  à :embed_sum       : les 8 branches sont mesurables (aucune en amont).")
    emit("  à :layer_1_res1    : layer_1_mha_output_out est EXCLUE (q=0 exact).")
    emit("  à :layer_2_res1    : layer_1_mha_output_out ET layer_1_mlp_out (circuit P3)")
    emit("                       sont EXCLUES (q=0 exact) -- prédiction vérifiable.")
    emit()

    # ── Portes de correction (une fois pour toutes, communes aux 3 sites) ──
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
    gbp = sites_grad!(G_PLAIN, NS_PLAIN, receiver_tokens(c1), c1)
    gbm = sites_grad!(G_MAIN,  NS_MAIN,  receiver_tokens(c1), c1)
    g2_per_site = Dict(s => (gbp[s] == gbm[s]) for s in READSITES)
    for s in READSITES
        emitf("G2[%-14s] gradient bit-identique (eps=1 vs témoin) : %s  (max|diff|=%.3e)",
              string(s), g2_per_site[s] ? "OK" : "ÉCHEC", maximum(abs.(gbp[s] .- gbm[s])))
    end

    gates_ok = g1 && val_ok && all(values(g2_per_site))
    emit(gates_ok ? "\n✓ Toutes les portes passent -- les q_j ci-dessous sont lisibles." :
                    "\n✗ AU MOINS UNE PORTE A ÉCHOUÉ -- arrêt, les q_j ne seraient pas fiables.")
    gates_ok || return

    # ── q_j pour chaque paire, pour les 3 sites simultanément ──────────────
    ALLQ = Dict(s => Dict{Symbol,Vector{Float64}}(b => Float64[] for b in BRANCHES) for s in READSITES)
    p2_violations = Dict(s => 0 for s in READSITES)

    for (ip, c) in enumerate(PAIRS)
        for b in BRANCHES; EPSV[b] = 1.0f0; end
        gbase = sites_grad!(G_MAIN, NS_MAIN, receiver_tokens(c), c)
        n2g = Dict(s => Float64(sum(abs2, gbase[s])) for s in READSITES)

        Qs = Dict(s => Dict{Symbol,Float64}() for s in READSITES)
        for bj in BRANCHES
            EPSV[bj] = 0.0f0
            g0 = sites_grad!(G_MAIN, NS_MAIN, receiver_tokens(c), c)
            EPSV[bj] = 1.0f0
            for s in READSITES
                dotg0g = Float64(sum(g0[s] .* gbase[s]))
                q = n2g[s] > 0 ? 1.0 - dotg0g / n2g[s] : 0.0
                Qs[s][bj] = q
                push!(ALLQ[s][bj], q)
            end
        end

        emit()
        emitf("Paire %d : k=%d sigma(k)=%d v(k)=%d v(sigma(k))=%d", ip, c.k, c.sk, c.vk, c.vsk)
        for s in READSITES
            emitf("  -- site = %s  (||grad||^2=%.4e) --", string(s), n2g[s])
            ranked = sort(BRANCHES; by = b -> -abs(Qs[s][b]))
            for (rk, b) in enumerate(ranked[1:3])
                emitf("    #%d  %-24s q_j = %+.6f", rk, string(b), Qs[s][b])
            end
        end
    end

    branch_pos = Dict(b => i for (i, b) in enumerate(BRANCHES))
    site_upstream_bound = Dict(  # dernière branche EXCLUE (en amont) par site
        :embed_sum    => 0,                                    # aucune exclue
        :layer_1_res1 => branch_pos[:layer_1_mha_output_out],  # exclut #1
        :layer_2_res1 => branch_pos[:layer_1_mlp_out],         # exclut #1,#2
    )

    emit()
    emit("═"^78)
    emit("AGRÉGAT SUR LES 3 PAIRES, PAR SITE DE LECTURE")
    emit("═"^78)
    for s in READSITES
        emit()
        emitf("── site = %s  (voisine structurelle attendue : %s) ──",
              string(s), string(NEAREST_BRANCH[s]))
        meanabs = Dict(b => sum(abs.(ALLQ[s][b])) / length(ALLQ[s][b]) for b in BRANCHES)
        ranked_agg = sort(BRANCHES; by = b -> -meanabs[b])
        for (rk, b) in enumerate(ranked_agg)
            excluded = branch_pos[b] <= site_upstream_bound[s]
            emitf("  #%d  %-24s  |q_j| moyen = %.6f   valeurs = %s  %s",
                  rk, string(b), meanabs[b], string(round.(ALLQ[s][b], digits=4)),
                  excluded ? "(EN AMONT -- devrait être ~0)" : "")
        end
        top = ranked_agg[1]
        match = top == NEAREST_BRANCH[s]
        emitf("\n  Branche #1 mesurée : %s  --  voisine structurelle prédite : %s  --  %s",
              string(top), string(NEAREST_BRANCH[s]), match ? "COÏNCIDENCE" : "PAS DE COÏNCIDENCE")
        gt_rank = findfirst(==(:layer_1_mlp_out), ranked_agg)
        gt_q = meanabs[:layer_1_mlp_out]
        emitf("  Circuit P3 (layer_1_mlp_out) : rang #%d/8, |q_j| moyen = %.6f", gt_rank, gt_q)
    end

    emit()
    emit("═"^78)
    emit("VERDICT")
    emit("═"^78)
    q_rs1_top = sort(BRANCHES; by = b -> -sum(abs.(ALLQ[:embed_sum][b]))/3)[1]
    q_rs2_top = sort(BRANCHES; by = b -> -sum(abs.(ALLQ[:layer_1_res1][b]))/3)[1]
    q_rs3_top = sort(BRANCHES; by = b -> -sum(abs.(ALLQ[:layer_2_res1][b]))/3)[1]
    rs1_match = q_rs1_top == NEAREST_BRANCH[:embed_sum]
    rs2_match = q_rs2_top == NEAREST_BRANCH[:layer_1_res1]
    rs3_match = q_rs3_top == NEAREST_BRANCH[:layer_2_res1]
    emitf("  :embed_sum    -> #1 = %-24s (voisine attendue %s)   %s",
          string(q_rs1_top), string(NEAREST_BRANCH[:embed_sum]), rs1_match ? "COÏNCIDE" : "NON")
    emitf("  :layer_1_res1 -> #1 = %-24s (voisine attendue %s)   %s",
          string(q_rs2_top), string(NEAREST_BRANCH[:layer_1_res1]), rs2_match ? "COÏNCIDE" : "NON")
    emitf("  :layer_2_res1 -> #1 = %-24s (voisine attendue %s)   %s",
          string(q_rs3_top), string(NEAREST_BRANCH[:layer_2_res1]), rs3_match ? "COÏNCIDE" : "NON")
    emit()
    if rs1_match && rs2_match && rs3_match
        emit("=> Sur les 3 sites, la branche #1 mesurée est SYSTÉMATIQUEMENT la voisine")
        emit("   structurelle du site de lecture (co-entrée de sa jonction), y compris au")
        emit("   site :layer_2_res1 où cette voisine (layer_2_mha_output_out) n'a AUCUN lien")
        emit("   avec le circuit P3 (qui y est d'ailleurs EXCLU, q=0 par construction).")
        emit("   => DOMINATION PAR PROXIMITÉ DE LECTURE CONFIRMÉE : le résultat original à")
        emit("   :embed_sum est un artefact de placement du site, pas un signal sémantique.")
    else
        emit("=> La branche #1 mesurée ne coïncide PAS systématiquement avec la voisine")
        emit("   structurelle prédite -- la domination n'est PAS purement positionnelle.")
    end
end

println("\nÉcrit : ", OUT)
