# Figure pour jalon 1b (comparaison à NOMBRE DE PAS ÉGAL, 2026-07-13) --
# réutilise notebook/growth_jalon1b_results.json, aucune nouvelle mesure.
# Axe X = indice de pas brut (pas le proxy layer-steps de jalon 1 : ici les
# FLOPs varient délibérément entre bras, seul le nombre de pas est fixé à
# 4000 partout -- c'est l'axe qui a du sens pour CETTE comparaison).
#
# Deux panneaux : (a) perte de validation vs. pas, une courbe par densité ;
# (b) perte finale vs. densité -- montre le classement INVERSÉ par rapport à
# jalon 1 (FLOPs égal) : le bras fixe (1 étape) gagne maintenant.

using Plots, JSON, Statistics
gr()
default(titlefontsize=13, guidefontsize=12, tickfontsize=10, legendfontsize=9)

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

d = JSON.parsefile(joinpath(@__DIR__, "growth_jalon1b_results.json"))
const STAGE_COUNTS = [1, 2, 4, 8, 16]
const SEEDS1 = [1, 2]
const COLORS = Dict(1=>:gray, 2=>:seagreen, 4=>:steelblue, 8=>:darkorange, 16=>:purple)

# ── Panneau (a) : val loss vs. pas brut, une courbe par densité ────────────
p1 = plot(xlabel="Training step", ylabel="Validation loss (nats/char)",
          title="(a) Val loss vs. step (equal step count = 4000, mean of 2 seeds)",
          legend=:topright, left_margin=12Plots.mm, bottom_margin=10Plots.mm,
          top_margin=6Plots.mm, right_margin=4Plots.mm)

for n in STAGE_COUNTS
    runs = [d["stages$(n)_seed$(s)"] for s in SEEDS1]
    xs = [e[1] for e in runs[1]["val_history"]]
    ys_mat = hcat([[e[2] for e in r["val_history"]] for r in runs]...)
    ys_mean = vec(mean(ys_mat, dims=2))
    plot!(p1, xs, ys_mean, label="$n stage(s)", color=COLORS[n], lw=2.5)
end

# ── Panneau (b) : perte finale vs. nb étapes -- classement inversé vs jalon 1 ──
final_means = [mean([d["stages$(n)_seed$(s)"]["final_val"] for s in SEEDS1]) for n in STAGE_COUNTS]
final_all = [[d["stages$(n)_seed$(s)"]["final_val"] for s in SEEDS1] for n in STAGE_COUNTS]

p2 = plot(xlabel="Number of growth stages", ylabel="Final validation loss (nats/char)",
          title="(b) Equal steps: fixed-depth (1 stage) now WINS",
          legend=false, xticks=(1:5, string.(STAGE_COUNTS)),
          left_margin=12Plots.mm, bottom_margin=10Plots.mm, top_margin=6Plots.mm, right_margin=4Plots.mm)
bar!(p2, 1:5, final_means, color=[COLORS[n] for n in STAGE_COUNTS], alpha=0.75, bar_width=0.55)
for (i, vals) in enumerate(final_all)
    scatter!(p2, fill(i, length(vals)), vals, color=:black, markersize=5)
end

combined = plot(p1, p2, layout=(1, 2), size=(1500, 620))
savefig(combined, joinpath(figdir, "growth_jalon1b_equal_steps_en.pdf"))
println("Saved -> figures/growth_jalon1b_equal_steps_en.pdf")
