# Figure pour article3.tex §growthexact -- les deux expériences discriminantes
# sur l'universalité de c. Panneau (a) : a(L) vs ln(L) pour les 3 largeurs
# (dim=128/256/512) -- montre l'élévation quasi-uniforme de dim=512 (signature
# d'un sous-entraînement à budget fixe, pas d'un décalage de plancher).
# Panneau (b) : a(L) vs ln(L) pour les 2 configurations de têtes (heads=2/8)
# vs le contrôle dim=256 -- montre l'écart modeste, proche du bruit de graine.

using Plots, JSON
gr()
default(titlefontsize=13, guidefontsize=12, tickfontsize=10, legendfontsize=9)

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

fits_data = JSON.parsefile(joinpath(@__DIR__, "growth_discriminating_fits.json"))
means = fits_data["means_by_cfg"]
depths = Int.(fits_data["depths"])
lnL = log.(depths)

p1 = plot(xlabel="ln(Depth L)", ylabel="Loss at 4000 steps (nats/char)",
          title="(a) Width sweep: dim=128/256/512",
          legend=:topright, left_margin=12Plots.mm, bottom_margin=10Plots.mm,
          top_margin=6Plots.mm, right_margin=4Plots.mm)
scatter!(p1, lnL, means["dim128"], color=:seagreen, markersize=6, label="dim=128")
plot!(p1, lnL, means["dim128"], color=:seagreen, lw=1, alpha=0.5, label="")
scatter!(p1, lnL, means["dim256"], color=:steelblue, markersize=6, label="dim=256 (matched control)")
plot!(p1, lnL, means["dim256"], color=:steelblue, lw=1, alpha=0.5, label="")
scatter!(p1, lnL, means["dim512"], color=:firebrick, markersize=6, label="dim=512")
plot!(p1, lnL, means["dim512"], color=:firebrick, lw=1, alpha=0.5, label="")

p2 = plot(xlabel="ln(Depth L)", ylabel="Loss at 4000 steps (nats/char)",
          title="(b) Head-count sweep at dim=256 fixed",
          legend=:topright, left_margin=12Plots.mm, bottom_margin=10Plots.mm,
          top_margin=6Plots.mm, right_margin=4Plots.mm)
scatter!(p2, lnL, means["heads2"], color=:darkorange, markersize=6, label="n_heads=2")
plot!(p2, lnL, means["heads2"], color=:darkorange, lw=1, alpha=0.5, label="")
scatter!(p2, lnL, means["dim256"], color=:steelblue, markersize=6, label="n_heads=4 (matched control)")
plot!(p2, lnL, means["dim256"], color=:steelblue, lw=1, alpha=0.5, label="")
scatter!(p2, lnL, means["heads8"], color=:purple, markersize=6, label="n_heads=8")
plot!(p2, lnL, means["heads8"], color=:purple, lw=1, alpha=0.5, label="")

combined = plot(p1, p2, layout=(1, 2), size=(1500, 620))
savefig(combined, joinpath(figdir, "growth_discriminating_en.pdf"))
println("Saved -> figures/growth_discriminating_en.pdf")
