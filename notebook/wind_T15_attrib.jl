# ══════════════════════════════════════════════════════════════════════════════
# WIND-T15 -- attributions sous la règle du demi (pré-enregistré : notebook/wind_T15_preregistration.md).
# Gradients exacts de R avec : probabilités d'attention figées (stop-gradient), RMSNorm gelées, op :swiglu_half
# (forward affine au point propre ; backward ½·dy⊙s₀⊙(u₀, g₀)). Variantes AH (norme finale gelée) et AHf (vivante).
# Porte GV : ∂R/∂x_k (backward) = J^AHᵀ ∂R/∂x_{k+1} avec les Jacobiennes AH collectées (différences finies).
# USAGE : WIND_PROMPTS=1,...,50 julia --project=. notebook/wind_T15_attrib.jl
# ══════════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "wind3_common.jl"))
using Dates
const T8DIR = joinpath(@__DIR__, "wind_data_T8")
const PROMPTS = JSON.parsefile(joinpath(@__DIR__, "wind_T8_prompts.json"))
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const OUT = get(ENV, "WIND_T15_OUT", joinpath(@__DIR__, "wind_T15_attrib.json"))
const RES = get(ENV, "WIND_T15_RES", joinpath(@__DIR__, "wind_T15_attrib_results.txt"))
mlp_sym(l) = Symbol("layer_", l, "_mlp_out")
sw_sym(l) = Symbol("layer_", l, "_swiglu")

const FROZEN_SW = Dict{Symbol,Any}()
NeuroDSL.register_op!(:swiglu_half, (dev, out, inp, attrs, osym, onode, ctx) -> begin
    c = FROZEN_SW[osym]
    out .= c[:f0] .+ 0.5f0 .* c[:s0] .* (c[:u0] .* (inp[1] .- c[:g0]) .+ c[:g0] .* (inp[2] .- c[:u0]))
end)
NeuroDSL.CUSTOM_SHAPE_RULES[:swiglu_half] = (i, a) -> size(i[1])
NeuroDSL.CTX_REBUILD[:swiglu_half] = (dev, rule, nd, iv) -> Dict{Symbol,Any}(:c => FROZEN_SW[rule.output])
NeuroDSL.GRAD_RULES[:swiglu_half] = (dev, dy, ctx, inp) -> begin
    c = ctx[:c]
    (0.5f0 .* dy .* c[:s0] .* c[:u0], 0.5f0 .* dy .* c[:s0] .* c[:g0])
end

metas = Dict(p => JSON.parsefile(joinpath(T8DIR, "wind_meta_p$(p).json")) for p in 1:50)
const μMHA = [sum(Float64.(metas[p]["MHA"][l]) for p in 1:50) ./ 50 for l in 1:L]
const μMLP = [sum(Float64.(metas[p]["MLP"][l]) for p in 1:50) ./ 50 for l in 1:L]

function set_variant!(g, orig, origsw, frozen_norms, half::Bool)
    for s in NORM_SYMS
        r = orig[s]; want = s in frozen_norms ? :rmsnorm_frozen : :rmsnorm
        g.rules[NS][s].op == want && continue
        NeuroDSL.addrule!(g, want == :rmsnorm_frozen ? NeuroDSL.GraphRule(s, r.inputs, :rmsnorm_frozen; attrs=r.attrs,
                                                                            namespace=NS, atom_type=r.atom_type) : r)
    end
    for l in 1:L
        s = sw_sym(l); r = origsw[s]; want = half ? :swiglu_half : :swiglu
        g.rules[NS][s].op == want && continue
        NeuroDSL.addrule!(g, half ? NeuroDSL.GraphRule(s, r.inputs, :swiglu_half; attrs=r.attrs, namespace=NS,
                                                        atom_type=r.atom_type) : r)
    end
end

out = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T15 attributions règle du demi -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ") * "   prompts $(PIDX)")
    g, dev, orig, W_U = load_instrumented()
    origsw = Dict(sw_sym(l) => g.rules[NS][sw_sym(l)] for l in 1:L)
    all(r.op == :swiglu for r in values(origsw)) || error("règle swiglu inattendue")
    ALL = Set(NORM_SYMS); BLOCKS = setdiff(Set(NORM_SYMS), Set([:final_norm_out])); NONE = Set{Symbol}()
    for p in PIDX
        haskey(out, string(p)) && continue
        t0 = time(); pr = PROMPTS[p]; ids = Int.(pr["token_ids"]) .+ 1
        set_variant!(g, orig, origsw, NONE, false); SG_ON[] = false
        n = set_prompt!(g, ids)
        lg0 = Float64.(Array(NeuroDSL.demand!(g, :lm_head_out; namespace=NS))[n, :])
        a_id = pr["answer_id"] + 1; c_id = pr["cf_id"] + 1
        w64 = Float64.(vec(Array(W_U[a_id:a_id, :]))) .- Float64.(vec(Array(W_U[c_id:c_id, :])))
        set_readout!(dev, w64)
        cache = capture_rule_nodes(g)
        for s in NORM_SYMS; FROZEN_RMS[s] = copy(NeuroDSL.node(g, s; namespace=NS).aux_data[:rms_inv]); end
        for l in 1:L
            g0 = copy(cache[Symbol("layer_", l, "_gate")]); u0 = copy(cache[Symbol("layer_", l, "_up")])
            FROZEN_SW[sw_sym(l)] = Dict(:g0 => g0, :u0 => u0, :f0 => copy(cache[sw_sym(l)]), :s0 => 1f0 ./ (1f0 .+ exp.(-g0)))
        end
        # porte forward : variante AH au point propre = logits propres
        set_variant!(g, orig, origsw, ALL, true)
        lgh = Float64.(Array(NeuroDSL.demand!(g, :lm_head_out; namespace=NS))[n, :])
        gf = norm(lgh .- lg0) / norm(lg0)
        row(s) = Float64.(vec(Array(cache[s][n:n, :])))
        dM = [row(mha_sym(l)) for l in 1:L] .- μMHA; dP = [row(mlp_sym(l)) for l in 1:L] .- μMLP
        att = Dict{String,Any}(); Gl = nothing
        for (nm, fs) in (("AH", ALL), ("AHf", BLOCKS))
            SG_ON[] = true; set_variant!(g, orig, origsw, fs, true)
            NeuroDSL.demand!(g, :w3_R; namespace=NS)
            NeuroDSL.backward_graph!(g, :w3_R; namespace=NS, prune_frozen=true)
            Gx = [Float64.(Array(NeuroDSL.node(g, zx(k); namespace=NS).gradient)) for k in 0:L]
            Gr = [Float64.(Array(NeuroDSL.node(g, zr(l); namespace=NS).gradient)) for l in 1:L]
            SG_ON[] = false
            att[nm] = vcat([dot(Gr[l][n, :], dM[l]) for l in 1:L], [dot(Gx[l+1][n, :], dP[l]) for l in 1:L])
            nm == "AH" && (Gl = [Gx[k+1][n, :] for k in 0:L])
        end
        set_variant!(g, orig, origsw, NONE, false); NeuroDSL.demand!(g, :w3_R; namespace=NS)
        gv = NaN
        fJ = joinpath(T8DIR, "wind_J_p$(p)_AH.bin")
        if isfile(fJ)
            Js = Array{Float32}(undef, D, D, L); open(fJ, "r") do f; read!(f, Js); end
            gv = maximum(norm(Float64.(Js[:, :, k+1])' * Gl[k+2] .- Gl[k+1]) / norm(Gl[k+1]) for k in 1:L-1)
            Js = nothing; reclaim()
        end
        out[string(p)] = Dict("attr" => att, "gate_forward" => gf, "gate_GV" => gv)
        open(OUT, "w") do f; JSON.print(f, out); end
        emit(@sprintf("  p%-3d porte forward %.1e | GV %.1e | %.0f s", p, gf, gv, time() - t0))
    end
end
println("Écrit : ", OUT)
