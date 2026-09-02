# Regenerates figures/patching_structural_signal.pdf (margin fix only).
#
# This figure's source cell lives in notebook/patching.ipynb (cell id "4d5e4c10"),
# not in a standalone generate_*.jl script -- unlike the article2 figures, its
# inputs (layer_cone_sizes, layer_recoveries, head_cone_sizes, head_recoveries)
# were never saved to JSON in that notebook. Reproducing the plot therefore
# requires recomputing those four small arrays. This recomputation is CPU-only
# (NeuroDSL.Backend.CPUDevice()), a tiny model (dim=128, n_heads=8, n_layers=8,
# seq_len=16), and fully deterministic (Random.seed!(42)) -- it is the exact
# same code path as notebook/patching.ipynb cells dfcd652b/0b2a5ef5/2ff46447/
# 2b5de7af, trimmed to just what feeds the plot (no timing loops, no GPU, no
# training). Runs in a couple of seconds. Confirmed against the notebook's
# printed table before trusting this script's numbers.
#
# Original bug: no margin kwargs at all -> "Downstream cone size" x-axis label
# clipped at the bottom on both panels. Fix: explicit bottom_margin (project
# convention, see notebook/generate_ceiling_figure.jl etc.).

using NeuroDSL, Statistics, Random, Printf, LinearAlgebra, Plots
gr()

dev = NeuroDSL.Backend.CPUDevice()
dim, n_heads, hidden_dim, n_layers, seq_len = 128, 8, 512, 8, 16
ns = :patch_bench

g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
Random.seed!(42)
NeuroDSL.set!(g, :input, randn(Float32, seq_len, dim); namespace=ns)
output_sym = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim)(g, :input; namespace=ns)

X_clean = randn(Float32, seq_len, dim)
NeuroDSL.set!(g, :input, X_clean; namespace=ns)
clean_output = copy(NeuroDSL.demand!(g, output_sym; namespace=ns))
clean_cache = NeuroDSL.capture_activations(g, ns)

X_corrupted = copy(X_clean)
X_corrupted[1, :] .= randn(Float32, dim)   # single corrupted token (position 1)
NeuroDSL.set!(g, :input, X_corrupted; namespace=ns)
corrupted_output = copy(NeuroDSL.demand!(g, output_sym; namespace=ns))
corrupted_cache = NeuroDSL.capture_activations(g, ns)

function position_patch_cache(base_cache, patch_sym, clean_cache, row::Int)
    hybrid = copy(base_cache[patch_sym])
    hybrid[row, :] .= clean_cache[patch_sym][row, :]
    return Dict(patch_sym => hybrid)
end

# ── Regime 1: across layers (depth varies), 8 points ────────────────────────
layer_cone_sizes = Int[]
layer_recoveries = Float64[]
for i in 1:n_layers
    patch_sym = Symbol(:layer_, i, :_out)
    push!(layer_cone_sizes, length(NeuroDSL._downstream_nodes(g, patch_sym, ns)))
    hybrid_cache = position_patch_cache(corrupted_cache, patch_sym, clean_cache, 1)
    r = NeuroDSL.patch_and_measure!(g, output_sym, patch_sym, hybrid_cache, corrupted_cache,
                                     clean_output, corrupted_output; namespace=ns)
    push!(layer_recoveries, r.recovery)
end

# ── Regime 2: across sibling heads (depth fixed), layer-1 heads ─────────────
head_syms = [Symbol(:layer_1_mha_ao_h, h) for h in 1:n_heads]
head_cone_sizes = Int[]
head_recoveries = Float64[]
for hs in head_syms
    push!(head_cone_sizes, length(NeuroDSL._downstream_nodes(g, hs, ns)))
    hybrid = position_patch_cache(corrupted_cache, hs, clean_cache, 1)
    r = NeuroDSL.patch_and_measure!(g, output_sym, hs, hybrid, corrupted_cache,
                                     clean_output, corrupted_output; namespace=ns)
    push!(head_recoveries, r.recovery)
end

println("Couche | |cône aval| | recovery  (sanity check vs. notebook printout)")
for i in 1:n_layers
    @printf "  %d    |    %4d     |  %.3f\n" i layer_cone_sizes[i] layer_recoveries[i]
end
println("Tête | |cône aval| | recovery")
for h in 1:n_heads
    @printf "  h%d  |    %4d     |  %+.4f\n" h head_cone_sizes[h] head_recoveries[h]
end

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

p1 = scatter(layer_cone_sizes, layer_recoveries,
    marker = :circle, markersize = 6, color = :steelblue, legend = false,
    xlabel = "Downstream cone size", ylabel = "Recovery",
    title = "Across layers (depth varies)",
    left_margin = 10Plots.mm, bottom_margin = 12Plots.mm)

p2 = scatter(head_cone_sizes, head_recoveries,
    marker = :diamond, markersize = 6, color = :firebrick, legend = false,
    xlabel = "Downstream cone size", ylabel = "Recovery",
    title = "Across sibling heads (depth fixed)",
    xlim = (minimum(head_cone_sizes) - 2, maximum(head_cone_sizes) + 2),
    left_margin = 10Plots.mm, bottom_margin = 12Plots.mm)

plot(p1, p2, layout = (1, 2), size = (900, 430))
savefig(joinpath(figdir, "patching_structural_signal.pdf"))
println("Saved -> figures/patching_structural_signal.pdf")
