# Figure pour article3.tex -- ablation LR cosinus (§growthresults). Panneau (a)
# : la forme du schedule cosinus lui-même (lr_max -> lr_min sur les 3000 pas).
# Panneau (b) : les 3 courbes de validation (fixed-16, lr cosinus) comparées
# aux plages déjà mesurées à lr constant (baseline et meilleur schedule de
# croissance) -- montre que l'écart se referme partiellement sans s'inverser.

using Plots, JSON
gr()
default(titlefontsize=13, guidefontsize=12, tickfontsize=10, legendfontsize=9)

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

results = JSON.parsefile(joinpath(@__DIR__, "growth_lr_ablation_results.json"))
lr_max, lr_min = results["lr_max"], results["lr_min"]
budget = results["budget"]
n_layers = 16
total_steps = budget ÷ n_layers  # 3000

cosine_lr(t, T) = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*min(t,T)/T))

p1 = plot(1:total_steps, [cosine_lr(t, total_steps) for t in 1:total_steps],
          label="", color=:steelblue, lw=2.5,
          xlabel="Step", ylabel="Learning rate",
          title="(a) Cosine decay schedule (fixed-16 ablation)",
          left_margin=12Plots.mm, bottom_margin=10Plots.mm,
          top_margin=6Plots.mm, right_margin=4Plots.mm)
hline!(p1, [lr_max], color=:gray, linestyle=:dash, alpha=0.5, label="lr_max=1e-3")
hline!(p1, [lr_min], color=:gray, linestyle=:dot, alpha=0.5, label="lr_min=1e-4")

p2 = plot(xlabel="Step", ylabel="Validation loss (nats/char)",
          title="(b) fixed-16: cosine LR vs. constant LR vs. growth",
          legend=:topright, left_margin=12Plots.mm, bottom_margin=10Plots.mm,
          top_margin=6Plots.mm, right_margin=4Plots.mm)

colors = [:seagreen, :darkorange, :purple]
for (i, vh) in enumerate(results["val_histories"])
    steps = [Int(v[1]) for v in vh]
    losses = [Float64(v[2]) for v in vh]
    plot!(p2, steps, losses, color=colors[i], lw=2, marker=:circle, markersize=3,
          label="cosine LR, seed $(results["seeds"][i])")
end

hspan!(p2, [2.0562, 2.0723], color=:gray, alpha=0.15, label="fixed-16, constant LR (range)")
hspan!(p2, [1.6542, 1.7191], color=:steelblue, alpha=0.15, label="best growth schedule (range)")
hline!(p2, [results["mean"]], color=:black, linestyle=:dash, lw=1.5,
       label="cosine LR mean ($(round(results["mean"], digits=3)))")

combined = plot(p1, p2, layout=(1, 2), size=(1500, 620))
savefig(combined, joinpath(figdir, "growth_lr_ablation_en.pdf"))
println("Saved -> figures/growth_lr_ablation_en.pdf")
