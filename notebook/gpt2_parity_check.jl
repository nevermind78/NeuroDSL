# gpt2_parity_check.jl -- porte de validation du portage GPT-2 : NeuroDSL vs transformers (gpt2/reference.json).
# Compare, au dernier token : sortie de chaque bloc (couches 1..11), sortie après ln_f, logits ; et le top-1.
using NeuroDSL, JSON, LinearAlgebra, Printf
include(joinpath(@__DIR__, "gpt2_ops.jl"))
const MD = joinpath(@__DIR__, "gpt2"); ns = :gpt2
dev = NeuroDSL.Backend.CUDADevice()
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev); NeuroDSL.load_graph!(g, ns, joinpath(MD, "gpt2_neurodsl"))
ref = JSON.parsefile(joinpath(MD, "reference.json"))
worst = 0.0; allok = true
for r in ref
    ids = Int.(r["token_ids"]) .+ 1; n = length(ids)
    NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    lg = Float64.(Array(NeuroDSL.demand!(g, :lm_head_out; namespace=ns))[n, :])
    lr = Float64.(r["logits_last"])
    hs = [Float64.(Array(NeuroDSL.node(g, Symbol("layer_", l, "_out"); namespace=ns).value)[n, :]) for l in 1:11]
    push!(hs, Float64.(Array(NeuroDSL.node(g, :final_norm_out; namespace=ns).value)[n, :]))
    he = [norm(hs[l] .- Float64.(r["hidden_last"][l+1])) / norm(Float64.(r["hidden_last"][l+1])) for l in 1:12]
    e0 = norm(Float64.(Array(NeuroDSL.node(g, :tok_out; namespace=ns).value)[n, :]) .- Float64.(r["hidden_last"][1])) /
         norm(Float64.(r["hidden_last"][1]))
    el = maximum(abs.(lg .- lr)); top = argmax(lg) - 1 == r["top1"]
    global worst = max(worst, el); global allok &= top && el < 1e-2
    @printf("%-62s logits max|Δ| %.2e | top-1 %s | états : plongement %.1e, blocs max %.1e, ln_f %.1e\n",
            repr(r["prompt"]), el, top ? "identique" : "DIFFÉRENT", e0, maximum(he[1:11]), he[12])
end
println(allok ? "PORTE DE PARITÉ : OK" : "PORTE DE PARITÉ : ÉCHEC", @sprintf("  (pire max|Δlogit| = %.2e)", worst))
