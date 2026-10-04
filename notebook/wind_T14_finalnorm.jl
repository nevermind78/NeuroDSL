# ══════════════════════════════════════════════════════════════════════════════
# WIND-T14 (EXPLORATOIRE) -- le correctif avec la norme finale vivante (addendum de wind_T13_preregistration.md).
# Gradients exacts de R (probabilités d'attention figées) pour AAf (normes 1 figées), ANf (normes 1+2 figées),
# et AA (normes 1 + finale figées, porte vs T13). Attributions des 56 sous-couches, vérité terrain WIND-T9.
# USAGE : julia --project=. notebook/wind_T14_finalnorm.jl
# ══════════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "wind3_common.jl"))
using Dates
const T8DIR = joinpath(@__DIR__, "wind_data_T8")
const PROMPTS = JSON.parsefile(joinpath(@__DIR__, "wind_T8_prompts.json"))
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const OUT = joinpath(@__DIR__, "wind_T14_finalnorm.json")
const RES = joinpath(@__DIR__, "wind_T14_finalnorm_results.txt")
mlp_sym(l) = Symbol("layer_", l, "_mlp_out")
n1_sym(l) = Symbol("layer_", l, "_norm1_out"); n2_sym(l) = Symbol("layer_", l, "_norm2_out")
const FSETS = [("AAf", Set([n1_sym(l) for l in 1:L])),
               ("ANf", Set(vcat([n1_sym(l) for l in 1:L], [n2_sym(l) for l in 1:L]))),
               ("AA",  Set(vcat([n1_sym(l) for l in 1:L], [:final_norm_out])))]
metas = Dict(p => JSON.parsefile(joinpath(T8DIR, "wind_meta_p$(p).json")) for p in 1:50)
const μMHA = [sum(Float64.(metas[p]["MHA"][l]) for p in 1:50) ./ 50 for l in 1:L]
const μMLP = [sum(Float64.(metas[p]["MLP"][l]) for p in 1:50) ./ 50 for l in 1:L]
const T13 = JSON.parsefile(joinpath(@__DIR__, "wind_T13_factors.json"))
function set_frozen!(g, orig, fset)
    for s in NORM_SYMS
        r = orig[s]; want = s in fset ? :rmsnorm_frozen : :rmsnorm
        g.rules[NS][s].op == want && continue
        NeuroDSL.addrule!(g, want == :rmsnorm_frozen ? NeuroDSL.GraphRule(s, r.inputs, :rmsnorm_frozen; attrs=r.attrs,
                                                                            namespace=NS, atom_type=r.atom_type) : r)
    end
end
out = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T14 (EXPLORATOIRE) norme finale vivante -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    g, dev, orig, W_U = load_instrumented()
    for p in PIDX
        haskey(out, string(p)) && continue
        t0 = time(); pr = PROMPTS[p]; ids = Int.(pr["token_ids"]) .+ 1
        set_frozen!(g, orig, Set{Symbol}()); SG_ON[] = false
        n = set_prompt!(g, ids); NeuroDSL.demand!(g, :lm_head_out; namespace=NS)
        a_id = pr["answer_id"] + 1; c_id = pr["cf_id"] + 1
        w64 = Float64.(vec(Array(W_U[a_id:a_id, :]))) .- Float64.(vec(Array(W_U[c_id:c_id, :])))
        set_readout!(dev, w64)
        cache = capture_rule_nodes(g)
        for s in NORM_SYMS; FROZEN_RMS[s] = copy(NeuroDSL.node(g, s; namespace=NS).aux_data[:rms_inv]); end
        row(s) = Float64.(vec(Array(cache[s][n:n, :])))
        dM = [row(mha_sym(l)) for l in 1:L] .- μMHA; dP = [row(mlp_sym(l)) for l in 1:L] .- μMLP
        att = Dict{String,Any}()
        for (nm, fs) in FSETS
            SG_ON[] = true; set_frozen!(g, orig, fs)
            NeuroDSL.demand!(g, :w3_R; namespace=NS)
            NeuroDSL.backward_graph!(g, :w3_R; namespace=NS, prune_frozen=true)
            Gx = [Float64.(Array(NeuroDSL.node(g, zx(k); namespace=NS).gradient)) for k in 0:L]
            Gr = [Float64.(Array(NeuroDSL.node(g, zr(l); namespace=NS).gradient)) for l in 1:L]
            SG_ON[] = false
            att[nm] = vcat([dot(Gr[l][n, :], dM[l]) for l in 1:L], [dot(Gx[l+1][n, :], dP[l]) for l in 1:L])
        end
        set_frozen!(g, orig, Set{Symbol}()); NeuroDSL.demand!(g, :w3_R; namespace=NS)
        ref = Float64.(T13[string(p)]["attr"]["AA"])
        gate = maximum(abs.(att["AA"] .- ref)) / maximum(abs.(ref))
        out[string(p)] = Dict("attr" => att, "gate" => gate)
        open(OUT, "w") do f; JSON.print(f, out); end
        emit(@sprintf("  p%-3d porte AA vs T13 %.1e | %.0f s", p, gate, time() - t0))
    end
end
