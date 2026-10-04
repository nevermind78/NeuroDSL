# ══════════════════════════════════════════════════════════════════════════════
# WIND-T19 -- Gemma-2-2B : ESQUISSE du produit P_V = J_25···J_1 (x_1 -> x_26, dernier token) pour V = A, AN
# (pré-enregistrement : wind_T19_preregistration.md, amendement de méthode). K = 256 directions gaussiennes (graine
# 20261004, la même que l'étalonnage E1), dérivées multi-couches centrées ; A/AN : probabilités d'attention des couches
# 2..26 épinglées ; AN : les 5 normes par bloc + finale figées. Portes : G4b (forward figé = propre), E3 (ε vs 2ε).
# Sorties : wind_data_GEMMA/wind_sketch_p{p}.bin (P_A·V, P_AN·V, V = directions unitaires, D×K Float32 chacun)
#           + wind_sketch_meta_p{p}.json (X, γ final, rms_inv final, normes des g_i, portes, degré d'Euler).
# USAGE : WIND_PROMPTS=1,...,50 julia --project=. notebook/wind_T19_sketch.jl
# ══════════════════════════════════════════════════════════════════════════════
using NeuroDSL, Random, Printf, JSON, LinearAlgebra, Statistics, Dates
const MD = joinpath(@__DIR__, "gemma-2-2b"); const CKPT = joinpath(MD, "gemma2_neurodsl")
const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data_GEMMA"))
const RES = joinpath(@__DIR__, "wind_T19_sketch_results.txt")
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const K, EPS_REL = 256, 5e-3
mkpath(OUTDIR)
const FROZEN_RMS = Dict{Symbol,Any}()
NeuroDSL.register_op!(:rmsnorm_frozen, (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> begin
    out_buf .= inputs[1] .* FROZEN_RMS[out_sym] .* reshape(inputs[2], 1, :); out_buf
end)
NeuroDSL.CUSTOM_SHAPE_RULES[:rmsnorm_frozen] = (inputs, attrs) -> size(inputs[1])
include(joinpath(@__DIR__, "gemma_ops.jl"))

open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T19 esquisse Gemma -- " * Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ") * "   prompts $(PIDX)")
    dev = NeuroDSL.Backend.CUDADevice(); ns, L, D, NH = :gemma2, 26, 2304, 8
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev); NeuroDSL.load_graph!(g, ns, CKPT)
    res_sym(k) = k == 0 ? :tok_out : Symbol("layer_", k, "_out")
    pr_sym(l, h) = Symbol("layer_", l, "_mha_pr_h", h)
    norm_syms = vcat([Symbol("layer_", l, "_", j, "_out") for l in 1:L for j in ("norm1", "attnpost", "norm2", "mlppost")], [:final_norm_out])
    orig = Dict(s => g.rules[ns][s] for s in norm_syms)
    all(r.op == :rmsnorm for r in values(orig)) || error("règle de norme inattendue")
    swap_norms!(fr::Bool) = for s in norm_syms
        r = orig[s]
        NeuroDSL.addrule!(g, fr ? NeuroDSL.GraphRule(s, r.inputs, :rmsnorm_frozen; attrs=r.attrs, namespace=ns, atom_type=r.atom_type) : r)
    end
    prompts = JSON.parsefile(joinpath(MD, "qwen50_gemmatok.json"))
    Gm = randn(MersenneTwister(20261004), D, K)                        # mêmes g_i que l'étalonnage E1
    gnorm = [norm(Gm[:, i]) for i in 1:K]; Vd = Gm ./ reshape(gnorm, 1, :)
    for p in PIDX
        isfile(joinpath(OUTDIR, "wind_sketch_p$(p).bin")) && (emit("p$p déjà fait : sauté"); continue)
        t0 = time(); ids = Int.(prompts[p]["token_ids"]) .+ 1; n = length(ids)
        swap_norms!(false)
        NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns); NeuroDSL.demand!(g, :lm_head_out; namespace=ns)
        clean = NeuroDSL.capture_activations(g, ns)
        for s in norm_syms; FROZEN_RMS[s] = copy(NeuroDSL.node(g, s; namespace=ns).aux_data[:rms_inv]); end
        X = [Float64.(Array(clean[res_sym(k)])[n, :]) for k in 0:L]
        eps = EPS_REL * norm(X[2])
        function jvp(v, e, frozen_att::Bool)
            P = zeros(Float32, n, D); P[n, :] .= Float32.(e .* v); Pd = NeuroDSL.Backend.to_device(dev, P)
            r = Vector{Vector{Float64}}(undef, 2)
            for (i, sg) in enumerate((1f0, -1f0))
                NeuroDSL.patch_node!(g, res_sym(1), Dict(res_sym(1) => clean[res_sym(1)] .+ sg .* Pd); namespace=ns)
                frozen_att && for l in 2:L, h in 1:NH; NeuroDSL.patch_node!(g, pr_sym(l, h), clean; namespace=ns); end
                r[i] = Float64.(vec(Array(NeuroDSL.demand!(g, res_sym(L); namespace=ns)[n:n, :])))
            end
            NeuroDSL.patch_node!(g, res_sym(1), clean; namespace=ns)
            (r[1] .- r[2]) ./ (2e)
        end
        gates = Dict{String,Any}(); S = Dict{String,Matrix{Float32}}()
        for V in ("A", "AN")
            swap_norms!(V == "AN")
            NeuroDSL.patch_node!(g, :tok_out, clean; namespace=ns)
            lg = Float64.(Array(NeuroDSL.demand!(g, :lm_head_out; namespace=ns))[n, :]); l0 = Float64.(Array(clean[:lm_head_out])[n, :])
            gates["G4b_$V"] = norm(lg .- l0) / norm(l0)
            M = zeros(Float32, D, K)
            for i in 1:K; M[:, i] .= Float32.(jvp(Vd[:, i], eps, true)); end
            e3 = [(a = jvp(Vd[:, i], eps, true); b = jvp(Vd[:, i], 2eps, true); norm(a .- b) / norm(a)) for i in 1:4]
            gates["E3_$V"] = maximum(e3); S[V] = M
        end
        swap_norms!(false); NeuroDSL.patch_node!(g, :tok_out, clean; namespace=ns)
        ok = gates["G4b_AN"] < 1e-5 && gates["E3_A"] < 1e-2 && gates["E3_AN"] < 1e-2
        open(joinpath(OUTDIR, "wind_sketch_p$(p).bin"), "w") do f; write(f, S["A"]); write(f, S["AN"]); end
        open(joinpath(OUTDIR, "wind_sketch_meta_p$(p).json"), "w") do f
            JSON.print(f, Dict("prompt" => prompts[p]["prompt"], "n" => n, "X" => X, "K" => K, "gnorm" => gnorm,
                               "gamma_final" => Float64.(Array(NeuroDSL.node(g, :final_norm_gamma; namespace=ns).value)),
                               "rms_inv_final" => Float64(Array(FROZEN_RMS[:final_norm_out])[n]), "gates" => gates, "ok" => ok))
        end
        emit(@sprintf("  p%-3d G4b %.1e | E3 A %.1e AN %.1e | %s | %.0f s", p, gates["G4b_AN"], gates["E3_A"], gates["E3_AN"],
                      ok ? "retenu" : "EXCLU", time() - t0))
        GC.gc()
    end
end
