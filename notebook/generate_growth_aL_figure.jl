# Figure pour l'article 2 -- a(L) ajusté sur les 16 profondeurs (table du
# growth_theory_README.md §5.2), avec la courbe log retenue (§5.3) superposée,
# et le résidu par rapport à cette courbe en second panneau (montre l'outlier
# L=3, exclu, et le bruit résiduel des profondeurs à seed unique).

using Plots
gr()
default(titlefontsize=13, guidefontsize=12, tickfontsize=10, legendfontsize=9)

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

Ls  = collect(1:16)
avals = [1.5715,1.4485,1.9756,1.4390,1.5429,1.5327,1.4388,1.4422,1.4251,1.4145,1.3723,1.4007,1.3562,1.3372,1.3042,1.2803]
ases = [0.0024,0.0151,0.0777,0.0094,0.0687,0.0758,0.0695,0.0651,0.0581,0.0636,0.0672,0.0599,0.0596,0.0577,0.0572,0.0199]
is_trunc = [false,false,true,false,true,true,true,true,true,true,true,true,true,true,true,false]  # dummy delta a applies (4000-step runs)
is_outlier = [false,false,true,false,false,false,false,false,false,false,false,false,false,false,false,false]

a0, a1, da = 1.5710, -0.1042, 0.0738
fit_curve(L) = a0 + a1*log(L)

Lgrid = range(1, 16, length=200)
p1 = plot(Lgrid, fit_curve.(Lgrid), label="a0 + a1 ln(L)", color=:steelblue, lw=2.5,
          xlabel="Depth L", ylabel="Capacity floor a(L) (nats/char)",
          title="(a) Fitted capacity floor across 16 depths",
          legend=:topright, left_margin=12Plots.mm, bottom_margin=10Plots.mm,
          top_margin=6Plots.mm, right_margin=4Plots.mm)
plot!(p1, Lgrid, fit_curve.(Lgrid) .+ da, label="+ truncation bias δa (4000-step fits)",
      color=:steelblue, lw=1.5, linestyle=:dash, alpha=0.6)

good_idx = .!is_outlier
scatter!(p1, Ls[good_idx .& .!is_trunc], avals[good_idx .& .!is_trunc],
         yerror=ases[good_idx .& .!is_trunc], color=:seagreen, markersize=6,
         label="long runs (≥8000 steps or 3 seeds)")
scatter!(p1, Ls[good_idx .& is_trunc], avals[good_idx .& is_trunc],
         yerror=ases[good_idx .& is_trunc], color=:darkorange, markersize=6,
         label="4000-step, single-seed runs")
scatter!(p1, Ls[is_outlier], avals[is_outlier], yerror=ases[is_outlier],
         color=:firebrick, markersize=7, markershape=:x,
         label="L=3 (excluded outlier, late plateau escape)")

# Panneau (b) : résidus par rapport à la courbe retenue (avec dummy pour les points 4000-pas)
resids = Float64[]
for i in 1:16
    pred = fit_curve(Ls[i]) + (is_trunc[i] ? da : 0.0)
    push!(resids, avals[i] - pred)
end
p2 = plot(xlabel="Depth L", ylabel="Residual (nats/char)",
          title="(b) Residuals vs. retained fit (incl. truncation-bias term)",
          legend=:topright, left_margin=12Plots.mm, bottom_margin=10Plots.mm,
          top_margin=6Plots.mm, right_margin=4Plots.mm)
hline!(p2, [0.0], color=:gray, linestyle=:dash, label="")
scatter!(p2, Ls[good_idx .& .!is_trunc], resids[good_idx .& .!is_trunc],
         yerror=ases[good_idx .& .!is_trunc], color=:seagreen, markersize=6, label="long runs")
scatter!(p2, Ls[good_idx .& is_trunc], resids[good_idx .& is_trunc],
         yerror=ases[good_idx .& is_trunc], color=:darkorange, markersize=6, label="4000-step runs")
scatter!(p2, Ls[is_outlier], resids[is_outlier], yerror=ases[is_outlier],
         color=:firebrick, markersize=7, markershape=:x, label="L=3 (excluded, −5.7σ)")

combined = plot(p1, p2, layout=(1, 2), size=(1500, 620))
savefig(combined, joinpath(figdir, "growth_aL_fit_en.pdf"))
println("Saved -> figures/growth_aL_fit_en.pdf")
