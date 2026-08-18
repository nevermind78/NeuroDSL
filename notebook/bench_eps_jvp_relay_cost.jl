# ══════════════════════════════════════════════════════════════════════════════
# RELAIS JVP : COÛT RÉEL D'UNE ABLATION "PAR RELAIS" CONTRE UNE ABLATION DIRECTE
# ══════════════════════════════════════════════════════════════════════════════
# NOUVEAUTÉ DE CE SCRIPT (pas encore testée nulle part dans ce projet) : combiner
# deux mécanismes déjà prouvés SÉPARÉMENT --
#   (a) le Corollaire jvp (session du jour, branch_order_theorem.tex) : le
#       vecteur de transport h_s = J_Phi(x_s) g^(s) se calcule par UNE différence
#       finie centrée FORWARD (2 passes), sans backward, et prédit la ligne
#       entière {q_j^(s)}_{j in cone(t)} à partir des quantités déjà mesurées AU
#       SITE t (beta_j^(t), g^(t)) ;
#   (b) `prune_frozen=true` (src/backward.jl, déjà validé sur un jouet 8 couches,
#       notebook/bench_prune_frozen_results.txt : max_err=0.000e+00, gain 69-76%
#       de temps d'horloge) : si aucun paramètre entraînable ne se trouve en
#       amont d'un site, le backward SAUTE structurellement ce préfixe.
#
# L'IDÉE TESTÉE : pour lire le profil {q_j^(s)}_{j in cone(t)} à un site PEU
# PROFOND s (cône B(s) grand, donc backward coûteux car needs_bwd doit remonter
# jusqu'à s), au lieu de faire (|cone(t)|+1) passes backward COMPLÈTES à s
# (protocole direct, Théorème ablation appliqué tel quel), on fait :
#   - UNE SEULE passe backward complète pour g^(s) (direction du relais) ;
#   - (|cone(t)|+1) passes backward ÉLAGUÉES à t SEULEMENT (is_param concentré
#     à partir de la couche de t -- rien en amont de t n'est calculé) ;
#   - UN SEUL relais JVP (2 passes forward, coût du segment s->t seulement).
# Si le coût d'une passe élaguée à t est proportionnel à B(t) (petit, car t est
# profond) plutôt qu'à B(s) (grand), c'est un vrai gain de coût, pas seulement
# une vérification -- contrairement à ce que le papier dit du Corollaire jvp
# ("a different kind of check on the same equation", jamais présenté comme une
# économie de coût).
#
# CE QUI PEUT ÉCHOUER (donc ce n'est pas tautologique) :
#   - la précision de q_hat vs q_direct (déjà connue à ~1e-5 pour UNE paire de
#     sites -- ici testée sur 3 paires supplémentaires, profondeurs différentes) ;
#   - le VRAI gain de temps d'horloge (le mécanisme prune_frozen a été validé
#     sur un jouet 8 couches/27 nœuds -- rien ne garantit qu'il tienne à l'échelle
#     Qwen 28 couches/GQA, ni que le coût forward partagé (identique dans les
#     deux bras, car demand!(loss) est appelé à chaque passe) ne dilue pas le
#     gain au point de l'annuler).
#
# USAGE : julia --project=. notebook/bench_eps_jvp_relay_cost.jl
# ÉCRIT  notebook/bench_eps_jvp_relay_cost_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, Statistics, LinearAlgebra
using CUDA

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const PROMPTS_F = joinpath(@__DIR__, "qwen_sweep_prompts.json")
const OUT       = joinpath(@__DIR__, "bench_eps_jvp_relay_cost_results.txt")

const EPS_RELAY = 1.0f-2                 # meilleur eps mesuré aujourd'hui (bench_eps_jvp_transport_check)
const T_LAYERS  = [16, 22, 26]           # ancres profondes : B(t) ~ 26, 14, ~6 (cf. tab:profile)
const N_TIMING  = 8
const N_WARMUP  = 3

const EPSV     = Dict{Symbol,Float32}()
const CAPTURED = Dict{Symbol,Array{Float32}}()

function ensure_eps_op!(branch::Symbol)
    op = Symbol("epsop_", branch)
    EPSV[branch] = 1.0f0
    haskey(NeuroDSL.CUSTOM_OPS, op) && return op
    NeuroDSL.register_op!(op,
        (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> (out_buf .= inputs[1]))
    NeuroDSL.CUSTOM_SHAPE_RULES[op] = (inputs, attrs) -> size(inputs[1])
    NeuroDSL.GRAD_RULES[op] = (dev, dy, ctx, inputs) -> begin
        e = EPSV[branch]
        gr = e .* dy
        CAPTURED[branch] = copy(Array(gr))
        return (gr,)
    end
    return op
end

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))

    emit("RELAIS JVP -- COÛT RÉEL (GPU chronométré) : ablation directe au site peu")
    emit("profond s VS ablation élaguée (prune_frozen) au site profond t + relais")
    emit("JVP (Corollaire jvp), sur le modèle entraîné réel (Qwen2.5-1.5B-Instruct).")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))

    dev = NeuroDSL.Backend.CUDADevice()
    ns, L = :qwen2, 28
    logits, loss = :lm_head_out, :ce_loss

    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    emit("\nChargement du checkpoint natif...")
    NeuroDSL.load_graph!(g, ns, CKPT)
    reclaim()

    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss, [logits, :labels], :cross_entropy;
                                            namespace=ns))

    branches = Symbol[]
    for i in 1:L
        push!(branches, Symbol("layer_", i, "_mha_output_out"))
        push!(branches, Symbol("layer_", i, "_mlp_out"))
    end
    pos = Dict(b => k for (k, b) in enumerate(branches))
    S_SYM = branches[1]
    Bs = length(branches)

    # ─── parseur "couche" + regroupement des feuilles paramètres par couche ───
    function layer_of(sym::Symbol)
        m = match(r"^layer_(\d+)_", String(sym))
        m === nothing ? nothing : parse(Int, m.captures[1])
    end
    param_syms_by_layer = Dict{Int,Vector{Symbol}}()
    for (s_, nd) in g.nodes[ns]
        li = layer_of(s_)
        if li !== nothing && !haskey(g.rules[ns], s_)
            push!(get!(param_syms_by_layer, li, Symbol[]), s_)
        end
    end
    emit(@sprintf("\nFeuilles paramètres trouvées : %d couches, %d..%d feuilles/couche",
                  length(param_syms_by_layer),
                  minimum(length.(values(param_syms_by_layer))),
                  maximum(length.(values(param_syms_by_layer)))))
    all_param_syms = vcat(values(param_syms_by_layer)...)
    function set_trainable_from!(from_layer::Int)
        for s_ in all_param_syms
            li = layer_of(s_)
            g.nodes[ns][s_].is_param = (li >= from_layer)
        end
    end

    # ─── eps-ops sur toutes les branches (mêmes conventions qu'aujourd'hui) ───
    BRANCH_TO_JOIN = Dict{Symbol,Symbol}()
    for i in 1:L
        for (join_sym, branch_sym) in ((Symbol("layer_", i, "_res1"),
                                        Symbol("layer_", i, "_mha_output_out")),
                                       (Symbol("layer_", i, "_out"),
                                        Symbol("layer_", i, "_mlp_out")))
            r = g.rules[ns][join_sym]
            r.inputs[2] == branch_sym || error("$join_sym : branche inattendue")
            eps_sym = Symbol("eps_", branch_sym)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(eps_sym, [branch_sym],
                                                    ensure_eps_op!(branch_sym); namespace=ns))
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(join_sym, [r.inputs[1], eps_sym], r.op;
                                                    attrs=r.attrs, namespace=ns,
                                                    atom_type=r.atom_type))
            NeuroDSL._invalidate_downstream!(g, join_sym, ns)
            BRANCH_TO_JOIN[branch_sym] = join_sym
        end
    end
    for b in branches; EPSV[b] = 1.0f0; end

    function set_input!(ids::Vector{Int})
        n = length(ids)
        NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, vcat(ids[2:end], ids[end]);
                      atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
    end
    function backward!()
        empty!(CAPTURED)
        NeuroDSL.demand!(g, loss; namespace=ns)
        NeuroDSL.backward_graph!(g, loss; namespace=ns, prune_frozen=true)
        return CAPTURED
    end

    prompts = JSON.parsefile(PROMPTS_F)
    order = sortperm([length(p["token_ids"]) for p in prompts]; rev=true)
    ids1 = Int.(prompts[order[1]]["token_ids"]) .+ 1
    set_input!(ids1)
    emit(@sprintf("Prompt : \"%s\" (n=%d tokens)", prompts[order[1]]["prompt"][1:min(40,end)], length(ids1)))

    S_FWD_SYM = BRANCH_TO_JOIN[S_SYM]

    # ═══ g^(s) plein réseau -- UNE SEULE FOIS, partagée par toutes les ancres t ═══
    set_trainable_from!(1)
    for b in branches; EPSV[b] = 1.0f0; end
    base_s = backward!()
    g_s = Float64.(vec(base_s[S_SYM]))
    x_s_orig = copy(NeuroDSL.demand!(g, S_FWD_SYM; namespace=ns))
    u_cpu = Float32.(reshape(g_s ./ norm(g_s), size(x_s_orig)))
    u_dev = NeuroDSL.Backend.to_device(dev, u_cpu)

    # ─── timing : UNE passe backward complète (is_param depuis couche 1) ───
    function time_pass!(from_layer::Int, probe_branch::Symbol; n=N_TIMING, warmup=N_WARMUP)
        set_trainable_from!(from_layer)
        for _ in 1:warmup
            EPSV[probe_branch] = 0.0f0; backward!(); EPSV[probe_branch] = 1.0f0
        end
        ts = Float64[]
        for _ in 1:n
            EPSV[probe_branch] = 0.0f0
            t0 = time_ns(); backward!(); CUDA.synchronize(); push!(ts, (time_ns()-t0)/1e6)
            EPSV[probe_branch] = 1.0f0
        end
        return median(ts)
    end
    ms_full = time_pass!(1, branches[2])
    emit(@sprintf("\nPasse backward COMPLÈTE (is_param depuis couche 1, réseau entier, prune_frozen=true) :"))
    emit(@sprintf("  médiane sur %d = %.3f ms", N_TIMING, ms_full))

    for T_LAYER in T_LAYERS
        T_SYM = Symbol("layer_", T_LAYER, "_mha_output_out")
        T_SITE_POS = pos[T_SYM]
        cone_t = [b for b in branches if pos[b] >= T_SITE_POS]
        Bt = length(cone_t)
        T_FWD_SYM = BRANCH_TO_JOIN[T_SYM]

        emit("\n" * "="^92)
        emit(@sprintf("ANCRE t = %s (pos=%d, B(t)=%d)   vs   s = %s (B(s)=%d)",
                      T_SYM, T_SITE_POS, Bt, S_SYM, Bs))

        # ─── porte de correction : g_t identique, élagué vs plein (max_err) ───
        set_trainable_from!(1)
        for b in branches; EPSV[b] = 1.0f0; end
        g_t_full = Float64.(vec(backward!()[T_SYM]))
        set_trainable_from!(T_LAYER)
        for b in branches; EPSV[b] = 1.0f0; end
        g_t_pruned = Float64.(vec(backward!()[T_SYM]))
        max_err = maximum(abs.(g_t_full .- g_t_pruned))
        emit(@sprintf("  PORTE  g^(t) élagué == g^(t) plein réseau : max|diff| = %.3e %s",
                      max_err, max_err < 1f-4 ? "OK" : "❌ ÉCHEC"))
        max_err < 1f-4 || error("❌ prune_frozen casse la valeur du gradient à $T_SYM -- arrêt")

        # ─── DIRECT : q_j^(s) pour j in cone_t, is_param depuis couche 1 (réseau entier) ───
        set_trainable_from!(1)
        for b in branches; EPSV[b] = 1.0f0; end
        base = backward!()
        g_s2 = Float64.(vec(base[S_SYM]))
        qs_direct = Vector{Float64}(undef, Bt)
        for (row, bj) in enumerate(cone_t)
            EPSV[bj] = 0.0f0
            abl_s = Float64.(vec(backward!()[S_SYM]))
            EPSV[bj] = 1.0f0
            qs_direct[row] = 1.0 - dot(abl_s, g_s2)/dot(g_s2, g_s2)
            (row % 8 == 0) && reclaim()
        end

        # ─── RELAIS : g^(t), beta_j^(t), is_param depuis couche T_LAYER (élagué) ───
        set_trainable_from!(T_LAYER)
        for b in branches; EPSV[b] = 1.0f0; end
        base_t = backward!()
        g_t = Float64.(vec(base_t[T_SYM]))
        D = length(g_t)
        Bmat = Matrix{Float64}(undef, Bt, D)
        for (row, bj) in enumerate(cone_t)
            EPSV[bj] = 0.0f0
            abl_t = Float64.(vec(backward!()[T_SYM]))
            EPSV[bj] = 1.0f0
            Bmat[row, :] = g_t .- abl_t
            (row % 8 == 0) && reclaim()
        end

        # ─── relais JVP : h_s par différence finie centrée (forward seul, segment s->t) ───
        orig_rule_s = g.rules[ns][S_FWD_SYM]
        delete!(g.rules[ns], S_FWD_SYM)
        x_plus  = x_s_orig .+ EPS_RELAY .* u_dev
        x_minus = x_s_orig .- EPS_RELAY .* u_dev
        NeuroDSL.set!(g, S_FWD_SYM, x_plus; namespace=ns)
        t_plus = Float64.(vec(Array(NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns))))
        NeuroDSL.set!(g, S_FWD_SYM, x_minus; namespace=ns)
        t_minus = Float64.(vec(Array(NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns))))
        hs_hat = norm(g_s) .* (t_plus .- t_minus) ./ (2.0*EPS_RELAY)
        denom = dot(g_t, hs_hat)
        qhat = (Bmat * hs_hat) ./ denom
        g.rules[ns][S_FWD_SYM] = orig_rule_s
        NeuroDSL.set!(g, S_FWD_SYM, x_s_orig; namespace=ns)

        errs = abs.(qhat .- qs_direct)
        emit(@sprintf("  Exactitude relais : erreur médiane=%.4e  max=%.4e  (q_j^(s) médian=%.4e, %d branches)",
                      median(errs), maximum(errs), median(abs.(qs_direct)), Bt))

        # ─── timing : UNE passe ÉLAGUÉE à l'ancre t ───
        probe_t = length(cone_t) > 1 ? cone_t[2] : cone_t[1]
        ms_pruned = time_pass!(T_LAYER, probe_t)

        # ─── timing : relais JVP (2 passes forward, segment s->t) ───
        delete!(g.rules[ns], S_FWD_SYM)
        for _ in 1:N_WARMUP
            NeuroDSL.set!(g, S_FWD_SYM, x_plus; namespace=ns); NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns)
            NeuroDSL.set!(g, S_FWD_SYM, x_minus; namespace=ns); NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns)
        end
        ts_jvp = Float64[]
        for _ in 1:N_TIMING
            t0 = time_ns()
            NeuroDSL.set!(g, S_FWD_SYM, x_plus; namespace=ns); NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns)
            NeuroDSL.set!(g, S_FWD_SYM, x_minus; namespace=ns); NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns)
            CUDA.synchronize()
            push!(ts_jvp, (time_ns()-t0)/1e6)
        end
        ms_jvp = median(ts_jvp)
        g.rules[ns][S_FWD_SYM] = orig_rule_s
        NeuroDSL.set!(g, S_FWD_SYM, x_s_orig; namespace=ns)

        cost_direct = (Bt+1) * ms_full
        cost_relay  = ms_full + Bt * ms_pruned + ms_jvp
        speedup = cost_direct / cost_relay
        ratio_pass = ms_pruned / ms_full
        ratio_B    = Bt / Bs

        emit(@sprintf("  Passe élaguée (is_param depuis couche %d)  : médiane sur %d = %.3f ms  (ratio vs pleine = %.4f, B(t)/B(s)=%.4f)",
                      T_LAYER, N_TIMING, ms_pruned, ratio_pass, ratio_B))
        emit(@sprintf("  Relais JVP (2 passes forward s->t)         : médiane sur %d = %.3f ms", N_TIMING, ms_jvp))
        emit(@sprintf("  Coût DIRECT (mesuré, isolé x compté)  = (B(t)+1)*ms_full                = (%d+1)*%.3f = %.1f ms",
                      Bt, ms_full, cost_direct))
        emit(@sprintf("  Coût RELAIS (mesuré, isolé x compté)  = ms_full + B(t)*ms_pruned + ms_jvp = %.3f + %d*%.3f + %.3f = %.1f ms",
                      ms_full, Bt, ms_pruned, ms_jvp, cost_relay))
        emit(@sprintf("  SPEEDUP prédit (passes isolées, chronométrées séparément, amorti sur CETTE seule ancre) = %.3fx",
                      speedup))
    end

    emit("\n" * "="^92)
    emit("NOTE : la passe g^(s) plète (ms_full, une seule mesure) est comptée UNE FOIS PAR")
    emit("ANCRE ci-dessus par prudence (comparaison conservative par paire) ; en pratique elle")
    emit("est calculée UNE SEULE FOIS pour toutes les ancres t d'une étude (déjà fait dans ce")
    emit("script pour g_s/x_s_orig/u_dev), donc le gain réel amorti sur plusieurs ancres est")
    emit("légèrement SUPÉRIEUR à celui rapporté par ancre isolée ci-dessus.")
end
println("\nÉcrit : ", OUT)
