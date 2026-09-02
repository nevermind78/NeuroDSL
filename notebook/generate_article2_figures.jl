# Generates the 4 new figures for article2.tex (English titles/axes), reusing
# already-saved results from this session -- no retraining, no new measurement.

using Plots, StatsPlots, JSON, Statistics
gr()

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

# ══ 1. P4 — three-way patch cost vs. depth ══════════════════════════════════
nd = JSON.parsefile(joinpath(@__DIR__, "p4_export", "neurodsl_bench_results.json"))
pt = JSON.parsefile(joinpath(@__DIR__, "p4_export", "pytorch_bench_results.json"))
nd_sites = Dict(r["site"] => r for r in nd["sites"])

layer_order = ["layer_1", "surgery_layer_1_out", "layer_2", "layer_3", "layer_4"]
layer_labels = ["Layer 1", "Graft", "Layer 2", "Layer 3", "Layer 4"]
xvals = Float64[]
nd_ms = Float64[]; naive_ms = Float64[]; partial_ms = Float64[]
for (i, r) in enumerate(pt["sites"])
    site = r["site"]
    push!(xvals, i)
    push!(nd_ms, nd_sites[site]["med_ms"])
    push!(naive_ms, r["full_forward_hook_med_ms"])
    push!(partial_ms, r["partial_forward_med_ms"])
end
p4 = plot(xvals, naive_ms, label="PyTorch (naive hook, full forward)", color=:firebrick, lw=2, marker=:circle, markersize=3,
          xlabel="Site (5 layers × 4 heads, grouped by layer)", ylabel="Median patch cost (ms)",
          title="Patch cost vs. depth: NeuroDSL vs. PyTorch (same weights, parity 1.23e-6)",
          legend=:topright, size=(900, 480), xticks=(2.5:4:18.5, layer_labels),
          bottom_margin=12Plots.mm)
plot!(p4, xvals, partial_ms, label="PyTorch (manual partial forward)", color=:seagreen, lw=2, marker=:diamond, markersize=3)
plot!(p4, xvals, nd_ms, label="NeuroDSL (sweep_patch_sites!)", color=:steelblue, lw=2.5, marker=:square, markersize=3)
for b in [4.5, 8.5, 12.5, 16.5]
    vline!(p4, [b], color=:gray, alpha=0.3, linestyle=:dot, label="")
end
savefig(p4, joinpath(figdir, "p4_cost_vs_depth_en.pdf"))
println("Saved -> figures/p4_cost_vs_depth_en.pdf")

# ══ 2. P1-bis — batched vs non-batched cone size (15 points) ════════════════
p1 = JSON.parsefile(joinpath(@__DIR__, "p1bis_results.json"))
rows = p1["rows"]
layer_names = ["layer_1", "surgery_layer_1_out", "layer_3"]
layer_names_labels = ["Layer 1", "Graft", "Layer 3"]
kinds = ["q_h", "k_h", "sc_h", "pr_h", "ao_h"]
groups = [(lp, k) for lp in layer_names for k in kinds]
batched_vals = Float64[]; nonbatched_vals = Float64[]; xt = String[]
for (lp, k) in groups
    rb = only(filter(r -> r["layer"] == lp && r["site_class"] == k && r["mode"] == "batched", rows))
    rn = only(filter(r -> r["layer"] == lp && r["site_class"] == k && r["mode"] == "non_batched", rows))
    push!(batched_vals, rb["cone_size"]); push!(nonbatched_vals, rn["cone_size"])
    push!(xt, k)
end
n_groups = length(groups)
x = 1:n_groups
p1bis_plot = groupedbar(hcat(nonbatched_vals, batched_vals), bar_position=:dodge, bar_width=0.7,
    label=["Non-batched" "Batched"], color=[:seagreen :steelblue],
    xlabel="Site type (grouped by layer: Layer 1 | Graft | Layer 3)", ylabel="Downstream cone size (nodes)",
    title="Patch cone size: batched vs. non-batched attention",
    xticks=(1:n_groups, xt), size=(1000, 480), legend=:topright, xrotation=0,
    left_margin=12Plots.mm, bottom_margin=12Plots.mm)
for b in [5.5, 10.5]
    vline!(p1bis_plot, [b], color=:gray, alpha=0.3, linestyle=:dot, label="")
end
savefig(p1bis_plot, joinpath(figdir, "p1bis_cone_size_en.pdf"))
println("Saved -> figures/p1bis_cone_size_en.pdf")

# ══ 3. Hot surgery — grafted vs. equal-budget control loss curve (English) ══
v2 = JSON.parsefile(joinpath(@__DIR__, "real_llm_surgery_v2_results.json"))
n_A = v2["n_steps_A"]; n_B = v2["n_steps_B"]; n_total = n_A + n_B
train_g = Float64.(v2["train_losses_grafted"]); train_c = Float64.(v2["train_losses_control"])
val_g = v2["val_history_grafted"]; val_c = v2["val_history_control"]

function downsample(arr, n=600)
    length(arr) <= n && return collect(1:length(arr)), arr
    step = length(arr) / n
    idx = [max(1, round(Int, i*step)) for i in 1:n]
    return idx, arr[idx]
end
idx_g, vg = downsample(train_g); idx_c, vc = downsample(train_c)
xg = idx_g ./ length(train_g) .* n_total
xc = idx_c ./ length(train_c) .* n_total

split_i = findlast(<=(n_A), xg)
hs = plot(xg[1:split_i], vg[1:split_i], label="Grafted -- before graft (steps 1-$(n_A))",
          color=:darkorange, lw=1.5, alpha=0.8,
          xlabel="Training step", ylabel="Loss (nats/char)",
          title="Hot surgery on a trained char-LM: grafted vs. equal-budget control",
          legend=:topright, size=(950, 500), bottom_margin=12Plots.mm)
plot!(hs, xg[split_i:end], vg[split_i:end], label="Grafted -- after graft (steps $(n_A+1)-$(n_total))", color=:steelblue, lw=1.5, alpha=0.8)
plot!(hs, xc, vc, label="Control -- fresh 4-layer, same total budget", color=:mediumpurple, lw=1.5, alpha=0.7)
vline!(hs, [n_A], label="graft insertion", linestyle=:dot, color=:red, lw=2)
scatter!(hs, [v[1] for v in val_g], [v[2] for v in val_g], label="Val loss (grafted)", color=:steelblue, markersize=3)
scatter!(hs, [v[1] for v in val_c], [v[2] for v in val_c], label="Val loss (control)", color=:mediumpurple, markersize=3, markershape=:diamond)
savefig(hs, joinpath(figdir, "hot_surgery_v2_loss_en.pdf"))
println("Saved -> figures/hot_surgery_v2_loss_en.pdf")

# ══ 4. Goulot vs témoin — directed-graft experiment summary (negative result) ══
# Hardcoded from the session's printed results (no JSON was saved for this run).
seeds = [1, 2, 3]
alpha_goulot = [0.0266, 0.0143, 0.0044]
alpha_temoin = [0.0085, 0.0043, 0.0086]
loss_goulot  = [2.1053, 2.2382, 2.1250]
loss_temoin  = [2.0590, 2.1779, 2.1244]

p_alpha = groupedbar(hcat(alpha_goulot, alpha_temoin), bar_position=:dodge, bar_width=0.6,
    label=["Bottleneck-placed graft" "Control-placed graft"], color=[:steelblue :seagreen],
    xlabel="Random seed", ylabel="|alpha| (final)", xticks=(1:3, string.(seeds)),
    title="Gate magnitude", legend=:topleft, size=(460, 400), bottom_margin=10Plots.mm)
p_loss = groupedbar(hcat(loss_goulot, loss_temoin), bar_position=:dodge, bar_width=0.6,
    label=["Bottleneck-placed graft" "Control-placed graft"], color=[:steelblue :seagreen],
    xlabel="Random seed", ylabel="Continued-training loss", xticks=(1:3, string.(seeds)),
    title="Final loss (lower is better)", legend=:topleft, size=(460, 400), bottom_margin=10Plots.mm)
gt = plot(p_alpha, p_loss, layout=(1, 2), size=(940, 480),
          plot_title="Directed graft placement: bottleneck vs. depth-matched control (3 seeds)",
          plot_titlevspan=0.14, top_margin=4Plots.mm)
savefig(gt, joinpath(figdir, "goulot_temoin_en.pdf"))
println("Saved -> figures/goulot_temoin_en.pdf")

println("\nAll 4 figures generated.")
