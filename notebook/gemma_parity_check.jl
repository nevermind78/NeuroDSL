# gemma_parity_check.jl -- porte de validation du portage Gemma-2-2B : NeuroDSL vs transformers (gemma-2-2b/reference.json).
# Compare au dernier token : plongement mis à l'échelle, sortie de chaque bloc (1..25), sortie après la norme finale,
# logits plafonnés ; et le top-1. Convention transformers : hidden_states[0] = plongement, [l] = sortie du bloc l,
# le dernier = norme finale appliquée à la sortie du dernier bloc.
using NeuroDSL, JSON, LinearAlgebra, Printf
include(joinpath(@__DIR__, "gemma_ops.jl"))
const MD = joinpath(@__DIR__, "gemma-2-2b"); ns = :gemma2
const NL = JSON.parsefile(joinpath(MD, "config.json"))["num_hidden_layers"]
dev = NeuroDSL.Backend.CUDADevice()
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev); NeuroDSL.load_graph!(g, ns, joinpath(MD, "gemma2_neurodsl"))
ref = JSON.parsefile(joinpath(MD, "reference.json"))
rel(a, b) = norm(a .- b) / norm(b)
row(s, n) = Float64.(Array(NeuroDSL.node(g, s; namespace=ns).value)[n, :])
worst = 0.0; allok = true
for r in ref
    ids = Int.(r["token_ids"]) .+ 1; n = length(ids)
    NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    lg = Float64.(Array(NeuroDSL.demand!(g, :lm_head_out; namespace=ns))[n, :]); lr = Float64.(r["logits_last"])
    hr = [Float64.(h) for h in r["hidden_last"]]
    e0 = rel(row(:tok_out, n), hr[1])
    eb = maximum(rel(row(Symbol("layer_", l, "_out"), n), hr[l+1]) for l in 1:NL-1)
    ef = rel(row(:final_norm_out, n), hr[NL+1])
    el = maximum(abs.(lg .- lr)); top = argmax(lg) - 1 == r["top1"]
    global worst = max(worst, el); global allok &= top && el < 1e-2
    @printf("%-62s logits max|Δ| %.2e | top-1 %s | états : plongement %.1e, blocs max %.1e, norme finale %.1e\n",
            repr(r["prompt"]), el, top ? "identique" : "DIFFÉRENT", e0, eb, ef)
end
println(allok ? "PORTE DE PARITÉ : OK" : "PORTE DE PARITÉ : ÉCHEC", @sprintf("  (pire max|Δlogit| = %.2e)", worst))
