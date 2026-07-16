# Figure pour article3.tex §growthexact -- ré-confirmation de L=3 comme
# outlier et découverte du problème de longueur de run à L=2, via 10 graines
# fraîches. Superpose les nouvelles mesures (10 graines, barres d'erreur SEM)
# sur le fit a(L) déjà retenu et les 16 points originaux.

using Plots, JSON, Statistics
gr()
default(titlefontsize=13, guidefontsize=12, tickfontsize=10, legendfontsize=9)

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

Ls  = collect(1:16)
avals = [1.5715,1.4485,1.9756,1.4390,1.5429,1.5327,1.4388,1.4422,1.4251,1.4145,1.3723,1.4007,1.3562,1.3372,1.3042,1.2803]
ases = [0.0024,0.0151,0.0777,0.0094,0.0687,0.0758,0.0695,0.0651,0.0581,0.0636,0.0672,0.0599,0.0596,0.0577,0.0572,0.0199]
is_trunc = [false,false,true,false,true,true,true,true,true,true,true,true,true,true,true,false]
is_outlier = [false,false,true,false,false,false,false,false,false,false,false,false,false,false,false,false]

a0, a1 = 1.5710, -0.1042
fit_curve(L) = a0 + a1*log(L)

# Nouvelles mesures : a(2), a(3), 10 graines, via fit courbe a+b/(n+n0)
new_a2, new_se2 = 1.3840, 0.0415
new_a3, new_se3 = 1.4096, 0.0513

Lgrid = range(1, 16, length=200)
p = plot(Lgrid, fit_curve.(Lgrid), label="retained fit: a0 + a1 ln(L)", color=:steelblue, lw=2.5,
         xlabel="Depth L", ylabel="Capacity floor a(L) (nats/char)",
         title="Re-confirming L=3 with 10 fresh seeds -- and finding a new limitation at L=2",
         legend=:topright, left_margin=12Plots.mm, bottom_margin=10Plots.mm,
         top_margin=6Plots.mm, right_margin=6Plots.mm, size=(950, 620))

good_idx = .!is_outlier
scatter!(p, Ls[good_idx .& .!is_trunc], avals[good_idx .& .!is_trunc],
         yerror=ases[good_idx .& .!is_trunc], color=:seagreen, markersize=6,
         label="original: long runs")
scatter!(p, Ls[good_idx .& is_trunc], avals[good_idx .& is_trunc],
         yerror=ases[good_idx .& is_trunc], color=:darkorange, markersize=6,
         label="original: 4000-step, single-seed")
scatter!(p, Ls[is_outlier], avals[is_outlier], yerror=ases[is_outlier],
         color=:firebrick, markersize=7, markershape=:x,
         label="original L=3: excluded outlier (5.7σ)")

scatter!(p, [2], [new_a2], yerror=[new_se2], color=:purple, markersize=8, markershape=:diamond,
         label="NEW: L=2, 10 seeds (2.8σ from trend)")
scatter!(p, [3], [new_a3], yerror=[new_se3], color=:blue, markersize=8, markershape=:diamond,
         label="NEW: L=3, 10 seeds (0.9σ from trend -- confirms outlier)")

savefig(p, joinpath(figdir, "growth_mu_reconfirmation_en.pdf"))
println("Saved -> figures/growth_mu_reconfirmation_en.pdf")
