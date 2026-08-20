# ══════════════════════════════════════════════════════════════════════════════
# DOMINANCE AU SITE RACINE -- Qwen2.5-1.5B-Instruct, lu à :tok_out
#
# CE QUE CE SCRIPT TESTE
# ------------------------
# Sur le modèle jouet (marker task), lire le gradient au nœud :embed_sum
# (la SEULE co-entrée de la toute première jonction résiduelle, en amont de
# TOUTES les branches) fait ressortir une domination quasi-totale de la
# branche immédiatement en aval (layer_1_mha_output_out, |q_j| ~ 0.996,
# bench_eps_semantic_objective_marker.jl) -- domination ensuite testée pour
# proximité (bench_eps_semantic_objective_marker_readsite_sweep.jl : le même
# schéma ne se reproduit PAS systématiquement à d'autres sites, donc pas un
# pur artefact positionnel).
#
# Ce script rejoue EXACTEMENT le même protocole (lecture du gradient à un
# nœud littéral is_param=true, ablation eps in {0,1} de chaque branche,
# q_j = 1 - <g0,gbase>/||gbase||^2) sur le VRAI modèle entraîné, au VRAI nœud
# racine.
#
# NŒUD RACINE RÉEL (vérifié dans le JSON du checkpoint, pas supposé) :
#   layer_1_res1 = add(tok_out, layer_1_mha_output_out)
# tok_out = sortie de l'embedding de token (op=embedding, inputs=[tok_E,
# token_ids]). Qwen n'a PAS d'embedding de position appris à sommer (RoPE
# remplace ce rôle, voir notebook/load_qwen2.jl) -- tok_out EST le nœud où
# "les embeddings de token sont sommés/normalisés avant la couche 1" : il
# n'y a rien à sommer de plus, c'est directement l'entrée résiduelle. C'est
# l'exact analogue structurel de :embed_sum (co-entrée de la même jonction
# layer_1_res1 = add(x, branche)).
#
# PORTÉE, BUDGET
# --------------
# CPU exclu (modèle 1.5B) -- GPU requis. 2-3 prompts réels (courts/moyens/
# longs, mêmes indices que la vérification JVP : 1, 18, 12), PAS les 22.
# 56 branches x (2-3 prompts + 1 gabarit) passes arrière -- nettement moins
# cher que la matrice complète 56 sites x 56 branches x 22 prompts déjà
# calculée avec succès (bench_eps_exact_ablation_qwen_multiprompt.jl).
#
# USAGE : julia --project=. notebook/bench_eps_qwen_root_site_dominance.jl
# ÉCRIT  notebook/bench_eps_qwen_root_site_dominance_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Printf, JSON, LinearAlgebra

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const PROMPTS_F = joinpath(@__DIR__, "qwen_sweep_prompts.json")
const OUT       = joinpath(@__DIR__, "bench_eps_qwen_root_site_dominance_results.txt")

const PROMPT_IDX = [1, 18, 12]   # court (5 tok), moyen (9 tok), long (14 tok) -- PAS le prompt 6

const EPSV     = Dict{Symbol,Float32}()
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

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())

open(OUT, "w") do io
    emit(s="") = (println(io, s); println(s); flush(io))
    emitf(fmt, args...) = emit(Printf.format(Printf.Format(fmt), args...))

    emit("DOMINANCE AU SITE RACINE (:tok_out) -- Qwen2.5-1.5B-Instruct, modèle réel")
    emit("Réplique le protocole embed_sum du marker task (is_param=true sur le nœud")
    emit("littéral, q_j = 1 - <g0,gbase>/||gbase||^2, ablation eps in {0,1} par branche).")
    emitf("Date : %s", strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit()

    t_load0 = time()
    dev = NeuroDSL.Backend.CUDADevice()
    ns, L = :qwen2, 28
    logits, loss = :lm_head_out, :ce_loss

    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    emit("Chargement du checkpoint natif...")
    NeuroDSL.load_graph!(g, ns, CKPT)
    reclaim()
    emitf("Chargé en %.1fs.", time() - t_load0)

    # ── vérification structurelle : layer_1_res1 = add(tok_out, layer_1_mha_output_out) ──
    r1 = g.rules[ns][:layer_1_res1]
    struct_ok = r1.op == :add && r1.inputs[1] == :tok_out && r1.inputs[2] == :layer_1_mha_output_out
    emitf("Vérification structurelle : layer_1_res1 = %s(%s, %s)  attendu add(tok_out, layer_1_mha_output_out)  %s",
          string(r1.op), string(r1.inputs[1]), string(r1.inputs[2]), struct_ok ? "OK" : "ÉCHEC")
    struct_ok || (emit("✗ Structure inattendue -- arrêt."); return)

    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss, [logits, :labels], :cross_entropy; namespace=ns))

    branches = Symbol[]
    for i in 1:L
        push!(branches, Symbol("layer_", i, "_mha_output_out"))
        push!(branches, Symbol("layer_", i, "_mlp_out"))
    end
    B = length(branches)
    emitf("Branches : B=%d (28 couches x 2)", B)

    function set_input!(ids::Vector{Int})
        n = length(ids)
        NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, vcat(ids[2:end], ids[end]); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
    end

    prompts = JSON.parsefile(PROMPTS_F)

    # ─── portes de correction, sur le prompt de gabarit (prompt 1) ──────────
    emit()
    emit("═"^78); emit("PORTES DE CORRECTION (prompt de gabarit)"); emit("═"^78)

    ids1 = Int.(prompts[PROMPT_IDX[1]]["token_ids"]) .+ 1
    set_input!(ids1)
    v_ref = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))
    g.nodes[ns][:tok_out].is_param = true
    NeuroDSL.demand!(g, loss; namespace=ns)
    NeuroDSL.backward_graph!(g, loss; namespace=ns)
    w_ref = copy(Array(g.nodes[ns][:tok_out].gradient))

    # ── réencâblage eps sur les 56 jonctions résiduelles ────────────────────
    for i in 1:L
        for (join_sym, branch_sym) in ((Symbol("layer_", i, "_res1"), Symbol("layer_", i, "_mha_output_out")),
                                        (Symbol("layer_", i, "_out"),  Symbol("layer_", i, "_mlp_out")))
            r = g.rules[ns][join_sym]
            r.inputs[2] == branch_sym || error("$join_sym : branche inattendue")
            eps_sym = Symbol("eps_", branch_sym)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(eps_sym, [branch_sym], ensure_eps_op!(branch_sym); namespace=ns))
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(join_sym, [r.inputs[1], eps_sym], r.op;
                                                    attrs=r.attrs, namespace=ns, atom_type=r.atom_type))
            NeuroDSL._invalidate_downstream!(g, join_sym, ns)
        end
    end

    for b in branches; EPSV[b] = 1.0f0; end
    g1ok = (copy(Array(NeuroDSL.demand!(g, logits; namespace=ns))) == v_ref)
    NeuroDSL.demand!(g, loss; namespace=ns)
    NeuroDSL.backward_graph!(g, loss; namespace=ns)
    w1 = copy(Array(g.nodes[ns][:tok_out].gradient))
    g2ok = (w1 == w_ref)
    emitf("G1 forward bit-identique (réencâblé eps=1 vs gabarit)      : %s  (max|diff|=%.3e)",
          g1ok ? "OK" : "ÉCHEC", maximum(abs.(Array(NeuroDSL.demand!(g, logits; namespace=ns)) .- v_ref)))
    emitf("G2 gradient à :tok_out bit-identique (eps=1 vs gabarit)     : %s  (max|diff|=%.3e)",
          g2ok ? "OK" : "ÉCHEC", maximum(abs.(w1 .- w_ref)))

    # sanity : ablater UNE branche doit changer le gradient (l'ablation a un effet réel)
    EPSV[branches[1]] = 0.0f0
    NeuroDSL.demand!(g, loss; namespace=ns)
    NeuroDSL.backward_graph!(g, loss; namespace=ns)
    w_pert = copy(Array(g.nodes[ns][:tok_out].gradient))
    EPSV[branches[1]] = 1.0f0
    g3ok = maximum(abs.(w_pert .- w_ref)) > 1f-8
    emitf("G3 ablation branche#1 change le gradient (sanity, doit être >>0) : %s  (max|diff|=%.3e)",
          g3ok ? "OK" : "ÉCHEC", maximum(abs.(w_pert .- w_ref)))

    gates_ok = g1ok && g2ok && g3ok
    emit(gates_ok ? "\n✓ Toutes les portes passent -- les q_j ci-dessous sont lisibles." :
                    "\n✗ AU MOINS UNE PORTE A ÉCHOUÉ -- arrêt.")
    gates_ok || return

    # ── q_j au site :tok_out, pour chaque prompt ─────────────────────────────
    ALLQ = Dict{Int,Dict{Symbol,Float64}}()
    t_meas0 = time()

    for pidx in PROMPT_IDX
        p = prompts[pidx]
        ids = Int.(p["token_ids"]) .+ 1
        set_input!(ids)

        # [correctif] :tok_out est is_param=true pour RETENIR son gradient
        # (voir portes ci-dessus), mais sa VALEUR a la forme (n_tokens, dim)
        # -- backward_graph! réutilise le buffer existant par fill!(0) pour
        # tout nœud is_param (optimisation pensée pour des poids à forme
        # FIXE, src/backward.jl:716-731). Entre deux prompts de longueur
        # différente, l'ancien buffer (mauvaise forme) survit et
        # accum_grad! plante en DimensionMismatch à la première accumulation
        # -- confirmé empiriquement (prompt 1 n=5 -> prompt 18 n=9). Forcer
        # la réallocation à chaque changement de prompt.
        g.nodes[ns][:tok_out].gradient = nothing

        for b in branches; EPSV[b] = 1.0f0; end
        NeuroDSL.demand!(g, loss; namespace=ns)
        NeuroDSL.backward_graph!(g, loss; namespace=ns)
        gbase = copy(Array(g.nodes[ns][:tok_out].gradient))
        n2g = Float64(sum(abs2, gbase))

        Q = Dict{Symbol,Float64}()
        for bj in branches
            EPSV[bj] = 0.0f0
            NeuroDSL.demand!(g, loss; namespace=ns)
            NeuroDSL.backward_graph!(g, loss; namespace=ns)
            g0 = Array(g.nodes[ns][:tok_out].gradient)
            EPSV[bj] = 1.0f0
            dotg0g = Float64(sum(g0 .* gbase))
            Q[bj] = n2g > 0 ? 1.0 - dotg0g / n2g : 0.0
        end
        ALLQ[pidx] = Q
        reclaim()

        emit()
        emitf("--- prompt %d (n=%d tokens) : \"%s\" --- ||grad_tok_out||^2=%.4e",
              pidx, length(ids), p["prompt"], n2g)
        ranked = sort(branches; by = b -> -abs(Q[b]))
        for (rk, b) in enumerate(ranked[1:8])
            emitf("  #%d  %-24s  q_j = %+.6f", rk, string(b), Q[b])
        end
        top1, top2 = Q[ranked[1]], Q[ranked[2]]
        ratio = abs(top2) > 1e-12 ? abs(top1) / abs(top2) : Inf
        nearest_match = ranked[1] == :layer_1_mha_output_out
        emitf("  #1/#2 ratio |q|=%.3f   #1 == voisine structurelle (layer_1_mha_output_out) : %s",
              ratio, nearest_match ? "OUI" : "NON")
    end
    emitf("\nTemps de mesure (3 prompts x 56 branches) : %.1fs", time() - t_meas0)

    emit()
    emit("═"^78); emit("VERDICT"); emit("═"^78)
    for pidx in PROMPT_IDX
        Q = ALLQ[pidx]
        ranked = sort(branches; by = b -> -abs(Q[b]))
        top1 = ranked[1]
        emitf("  prompt %-3d : #1 = %-24s  |q_j|=%.4f   voisine structurelle %s",
              pidx, string(top1), abs(Q[top1]), top1 == :layer_1_mha_output_out ? "COÏNCIDE" : "NON")
    end
    all_dominant = all(begin
        Q = ALLQ[pidx]; ranked = sort(branches; by = b -> -abs(Q[b]))
        ranked[1] == :layer_1_mha_output_out && abs(Q[ranked[1]]) > 0.9
    end for pidx in PROMPT_IDX)
    emit()
    if all_dominant
        emit("=> Sur les 3 prompts, layer_1_mha_output_out (voisine structurelle de :tok_out)")
        emit("   domine avec |q_j| > 0.9 -- MÊME SIGNATURE que :embed_sum sur le modèle jouet.")
    else
        emit("=> Le schéma de domination quasi-totale du modèle jouet ne se reproduit PAS")
        emit("   à l'identique sur le modèle réel (voisine structurelle non systématiquement")
        emit("   #1, ou dominance <0.9) -- à interpréter honnêtement, pas comme un échec du")
        emit("   protocole (les portes G1/G2/G3 ci-dessus, elles, sont passées).")
    end
end

println("\nÉcrit : ", OUT)
