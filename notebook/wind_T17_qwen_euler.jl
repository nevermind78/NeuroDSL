# WIND-T17 -- degré d'Euler effectif du MLP de Qwen à norme d'entrée figée (même mesure que wind_T17_collect.jl pour
# GPT-2) : d_l = ⟨∂_s m(s·h)|_{s=1}, m(h)⟩ / ‖m(h)‖², h = layer_l_res1 au dernier token, différence finie ±1e-3.
using NeuroDSL, JSON, LinearAlgebra, Statistics, Printf
const CKPT = joinpath(@__DIR__, "qwen2.5-1.5b-instruct", "qwen2_neurodsl")
const PROMPTS = JSON.parsefile(joinpath(@__DIR__, "wind_T8_prompts.json"))
const OUT = joinpath(@__DIR__, "wind_T17_qwen_euler.json")
const FROZEN_RMS = Dict{Symbol,Any}()
NeuroDSL.register_op!(:rmsnorm_frozen, (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> begin
    out_buf .= inputs[1] .* FROZEN_RMS[out_sym] .* reshape(inputs[2], 1, :); out_buf
end)
NeuroDSL.CUSTOM_SHAPE_RULES[:rmsnorm_frozen] = (inputs, attrs) -> size(inputs[1])
const ns, L = :qwen2, 28
g = NeuroDSL.NeuroGraph(namespace=ns, device=NeuroDSL.Backend.CUDADevice()); NeuroDSL.load_graph!(g, ns, CKPT)
n2(l) = Symbol("layer_", l, "_norm2_out")
const orig = Dict(n2(l) => g.rules[ns][n2(l)] for l in 1:L)
out = Dict{String,Any}()
for p in 1:50
    ids = Int.(PROMPTS[p]["token_ids"]) .+ 1; n = length(ids)
    for l in 1:L; NeuroDSL.addrule!(g, orig[n2(l)]); end
    NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns); NeuroDSL.demand!(g, :lm_head_out; namespace=ns)
    clean = NeuroDSL.capture_activations(g, ns)
    for l in 1:L
        FROZEN_RMS[n2(l)] = copy(NeuroDSL.node(g, n2(l); namespace=ns).aux_data[:rms_inv])
        r = orig[n2(l)]
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(n2(l), r.inputs, :rmsnorm_frozen; attrs=r.attrs, namespace=ns, atom_type=r.atom_type))
    end
    d = Float64[]
    for l in 1:L
        r1s = Symbol("layer_", l, "_res1"); ms = Symbol("layer_", l, "_mlp_out"); h = clean[r1s]; es = 1e-3
        o = Vector{Vector{Float64}}(undef, 2)
        for (i, sg) in enumerate((1.0, -1.0))
            P = copy(h); P[n:n, :] .= h[n:n, :] .* Float32(1 + sg * es)
            NeuroDSL.patch_node!(g, r1s, Dict(r1s => P); namespace=ns)
            o[i] = Float64.(vec(Array(NeuroDSL.demand!(g, ms; namespace=ns)[n:n, :])))
        end
        NeuroDSL.patch_node!(g, r1s, clean; namespace=ns)
        m0 = Float64.(vec(Array(clean[ms][n:n, :])))
        push!(d, dot((o[1] .- o[2]) ./ (2es), m0) / dot(m0, m0))
    end
    out[string(p)] = d
    @printf("p%-3d degre d'Euler median %.3f\n", p, median(d))
end
open(OUT, "w") do f; JSON.print(f, out); end
println("Écrit : ", OUT)
