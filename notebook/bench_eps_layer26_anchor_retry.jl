# ══════════════════════════════════════════════════════════════════════════════
# RETEST layer_26 (T_LAYER=26, B(t)=6, ancre la plus profonde/la plus élaguée) :
# ce même protocole (bench_eps_jvp_relay_cost.jl) avait bloqué le process
# pendant 40+ min AVANT le correctif GEMM (src/backward.jl, GRAD_RULES[:matmul]
# qui calculait inconditionnellement dB même sous prune_frozen) -- activité CPU
# réelle, GPU quasi inactif, aucun crash, tué manuellement depuis l'extérieur.
# Retesté ici APRÈS le correctif, avec :
#   - un plafond de temps RÉEL (`timeout` du process, voir invocation) au lieu
#     d'un kill manuel non supervisé ;
#   - une impression de progression (flush immédiat, `emit()`) à CHAQUE étape
#     et CHAQUE itération d'une boucle interne (Bt=6, donc peu coûteux) --
#     si ça bloque encore, le fichier de sortie contiendra la dernière étape
#     franchie, pas juste "toujours rien".
# NOTE : layer_16 et layer_22 sont volontairement RETIRÉS (déjà revalidés dans
# bench_eps_vocab_projection_profile.jl et bench_eps_layer22_anomaly_profile.jl)
# -- ce script cible SEULEMENT layer_26, pour isoler la cause si un blocage
# persiste encore.
#
# USAGE : timeout 480 julia --project=. notebook/bench_eps_layer26_anchor_retry.jl
# ÉCRIT  notebook/bench_eps_layer26_anchor_retry_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, Statistics, LinearAlgebra
using CUDA

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const PROMPTS_F = joinpath(@__DIR__, "qwen_sweep_prompts.json")
const OUT       = joinpath(@__DIR__, "bench_eps_layer26_anchor_retry_results.txt")

const EPS_RELAY = 1.0f-2
const T_LAYER   = 26
const N_TIMING  = 6
const N_WARMUP  = 2

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

    emit("RETEST layer_26 APRÈS le correctif GEMM -- avec plafond de temps et")
    emit("progression détaillée à chaque étape (voir invocation `timeout`).")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))

    dev = NeuroDSL.Backend.CUDADevice()
    ns, L = :qwen2, 28
    logits, loss = :lm_head_out, :ce_loss

    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    emit("\n[progress] avant load_graph!")
    NeuroDSL.load_graph!(g, ns, CKPT)
    emit("[progress] après load_graph!")
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
    all_param_syms = vcat(values(param_syms_by_layer)...)
    function set_trainable_from!(from_layer::Int)
        for s_ in all_param_syms
            li = layer_of(s_)
            g.nodes[ns][s_].is_param = (li >= from_layer)
        end
    end

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
    emit("[progress] eps-ops installés sur les 56 branches")

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

    emit("\n[progress] avant g^(s) plein réseau (backward complet is_param depuis couche 1)")
    set_trainable_from!(1)
    for b in branches; EPSV[b] = 1.0f0; end
    base_s = backward!()
    emit("[progress] après g^(s) plein réseau")
    g_s = Float64.(vec(base_s[S_SYM]))
    x_s_orig = copy(NeuroDSL.demand!(g, S_FWD_SYM; namespace=ns))
    u_cpu = Float32.(reshape(g_s ./ norm(g_s), size(x_s_orig)))
    u_dev = NeuroDSL.Backend.to_device(dev, u_cpu)
    emit("[progress] u_dev calculé")

    function time_pass!(from_layer::Int, probe_branch::Symbol; n=N_TIMING, warmup=N_WARMUP, label="")
        set_trainable_from!(from_layer)
        emit(@sprintf("  [progress] time_pass! %s : warmup (%d passes)...", label, warmup))
        for w in 1:warmup
            EPSV[probe_branch] = 0.0f0; backward!(); EPSV[probe_branch] = 1.0f0
            emit(@sprintf("    [progress] warmup %d/%d fait", w, warmup))
        end
        ts = Float64[]
        for k in 1:n
            EPSV[probe_branch] = 0.0f0
            t0 = time_ns(); backward!(); CUDA.synchronize(); push!(ts, (time_ns()-t0)/1e6)
            EPSV[probe_branch] = 1.0f0
            emit(@sprintf("    [progress] timing %d/%d fait (%.1f ms)", k, n, ts[end]))
        end
        return median(ts)
    end
    ms_full = time_pass!(1, branches[2]; label="FULL (is_param depuis couche 1)")
    emit(@sprintf("\nPasse backward COMPLÈTE (is_param depuis couche 1) : médiane sur %d = %.3f ms",
                  N_TIMING, ms_full))

    T_SYM = Symbol("layer_", T_LAYER, "_mha_output_out")
    T_SITE_POS = pos[T_SYM]
    cone_t = [b for b in branches if pos[b] >= T_SITE_POS]
    Bt = length(cone_t)
    T_FWD_SYM = BRANCH_TO_JOIN[T_SYM]

    emit("\n" * "="^92)
    emit(@sprintf("ANCRE t = %s (pos=%d, B(t)=%d)   vs   s = %s (B(s)=%d)",
                  T_SYM, T_SITE_POS, Bt, S_SYM, Bs))

    emit("[progress] avant PORTE (g_t_full, is_param depuis couche 1)")
    set_trainable_from!(1)
    for b in branches; EPSV[b] = 1.0f0; end
    g_t_full = Float64.(vec(backward!()[T_SYM]))
    emit("[progress] g_t_full calculé -- avant PORTE (g_t_pruned, is_param depuis couche $T_LAYER)")
    set_trainable_from!(T_LAYER)
    for b in branches; EPSV[b] = 1.0f0; end
    g_t_pruned = Float64.(vec(backward!()[T_SYM]))
    emit("[progress] g_t_pruned calculé")
    max_err = maximum(abs.(g_t_full .- g_t_pruned))
    emit(@sprintf("  PORTE  g^(t) élagué == g^(t) plein réseau : max|diff| = %.3e %s",
                  max_err, max_err < 1f-4 ? "OK" : "❌ ÉCHEC"))
    max_err < 1f-4 || error("❌ prune_frozen casse la valeur du gradient à $T_SYM -- arrêt")

    emit("[progress] avant boucle DIRECTE (base, is_param depuis couche 1)")
    set_trainable_from!(1)
    for b in branches; EPSV[b] = 1.0f0; end
    base = backward!()
    emit("[progress] base calculé -- entrée boucle DIRECTE ($Bt itérations)")
    g_s2 = Float64.(vec(base[S_SYM]))
    qs_direct = Vector{Float64}(undef, Bt)
    for (row, bj) in enumerate(cone_t)
        emit(@sprintf("  [progress] DIRECTE %d/%d : ablation %s...", row, Bt, bj))
        EPSV[bj] = 0.0f0
        abl_s = Float64.(vec(backward!()[S_SYM]))
        EPSV[bj] = 1.0f0
        qs_direct[row] = 1.0 - dot(abl_s, g_s2)/dot(g_s2, g_s2)
        emit(@sprintf("  [progress] DIRECTE %d/%d fait (q=%.4e)", row, Bt, qs_direct[row]))
        reclaim()
    end

    emit("[progress] avant boucle RELAIS (base_t, is_param depuis couche $T_LAYER)")
    set_trainable_from!(T_LAYER)
    for b in branches; EPSV[b] = 1.0f0; end
    base_t = backward!()
    emit("[progress] base_t calculé -- entrée boucle RELAIS ($Bt itérations)")
    g_t = Float64.(vec(base_t[T_SYM]))
    D = length(g_t)
    Bmat = Matrix{Float64}(undef, Bt, D)
    for (row, bj) in enumerate(cone_t)
        emit(@sprintf("  [progress] RELAIS %d/%d : ablation %s...", row, Bt, bj))
        EPSV[bj] = 0.0f0
        abl_t = Float64.(vec(backward!()[T_SYM]))
        EPSV[bj] = 1.0f0
        Bmat[row, :] = g_t .- abl_t
        emit(@sprintf("  [progress] RELAIS %d/%d fait", row, Bt))
        reclaim()
    end

    emit("[progress] avant relais JVP (delete! S_FWD_SYM)")
    orig_rule_s = g.rules[ns][S_FWD_SYM]
    delete!(g.rules[ns], S_FWD_SYM)
    emit("[progress] règle supprimée -- calcul x_plus/x_minus")
    x_plus  = x_s_orig .+ EPS_RELAY .* u_dev
    x_minus = x_s_orig .- EPS_RELAY .* u_dev
    emit("[progress] avant set!+demand! (x_plus)")
    NeuroDSL.set!(g, S_FWD_SYM, x_plus; namespace=ns)
    t_plus = Float64.(vec(Array(NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns))))
    emit("[progress] t_plus calculé -- avant set!+demand! (x_minus)")
    NeuroDSL.set!(g, S_FWD_SYM, x_minus; namespace=ns)
    t_minus = Float64.(vec(Array(NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns))))
    emit("[progress] t_minus calculé")
    hs_hat = norm(g_s) .* (t_plus .- t_minus) ./ (2.0*EPS_RELAY)
    denom = dot(g_t, hs_hat)
    qhat = (Bmat * hs_hat) ./ denom
    g.rules[ns][S_FWD_SYM] = orig_rule_s
    NeuroDSL.set!(g, S_FWD_SYM, x_s_orig; namespace=ns)
    emit("[progress] relais JVP terminé -- règle restaurée")

    errs = abs.(qhat .- qs_direct)
    emit(@sprintf("  Exactitude relais : erreur médiane=%.4e  max=%.4e  (q_j^(s) médian=%.4e, %d branches)",
                  median(errs), maximum(errs), median(abs.(qs_direct)), Bt))

    probe_t = length(cone_t) > 1 ? cone_t[2] : cone_t[1]
    ms_pruned = time_pass!(T_LAYER, probe_t; label="ÉLAGUÉ (is_param depuis couche $T_LAYER)")

    emit("[progress] avant timing relais JVP (delete! S_FWD_SYM à nouveau)")
    delete!(g.rules[ns], S_FWD_SYM)
    for w in 1:N_WARMUP
        NeuroDSL.set!(g, S_FWD_SYM, x_plus; namespace=ns); NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns)
        NeuroDSL.set!(g, S_FWD_SYM, x_minus; namespace=ns); NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns)
        emit(@sprintf("  [progress] warmup JVP %d/%d fait", w, N_WARMUP))
    end
    ts_jvp = Float64[]
    for k in 1:N_TIMING
        t0 = time_ns()
        NeuroDSL.set!(g, S_FWD_SYM, x_plus; namespace=ns); NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns)
        NeuroDSL.set!(g, S_FWD_SYM, x_minus; namespace=ns); NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns)
        CUDA.synchronize()
        push!(ts_jvp, (time_ns()-t0)/1e6)
        emit(@sprintf("  [progress] timing JVP %d/%d fait (%.1f ms)", k, N_TIMING, ts_jvp[end]))
    end
    ms_jvp = median(ts_jvp)
    g.rules[ns][S_FWD_SYM] = orig_rule_s
    NeuroDSL.set!(g, S_FWD_SYM, x_s_orig; namespace=ns)
    emit("[progress] timing relais JVP terminé -- règle restaurée")

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
    emit(@sprintf("  SPEEDUP prédit (passes isolées, chronométrées séparément) = %.3fx", speedup))

    emit("\n" * "="^92)
    emit("RETEST layer_26 TERMINÉ SANS BLOCAGE.")
end
println("\nÉcrit : ", OUT)
