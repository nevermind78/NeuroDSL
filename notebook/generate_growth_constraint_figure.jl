# Figure pour article3.tex §growthexact -- résultat de la campagne de test de
# la contrainte -1/3 : les 9 nouvelles mesures a(L), bien plus précises
# (4 graines, runs longs), comparées à l'ancien fit retenu et montrant le
# décrochage non-monotone inattendu autour de L=4-6.

using Plots, JSON
gr()
default(titlefontsize=13, guidefontsize=12, tickfontsize=10, legendfontsize=9)

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

fit = JSON.parsefile(joinpath(@__DIR__, "growth_constraint_fit_results.json"))
depths = Int.(fit["depths"])
avals = Float64.(fit["avals"])
ases = Float64.(fit["ases"])
coef_ext = Float64.(fit["coef_ext"])
coef_log = Float64.(fit["coef_log"])

old_a0, old_a1 = 1.5710, -0.1042
old_curve(L) = old_a0 + old_a1*log(L)
log_curve(L) = coef_log[1] + coef_log[2]*log(L)
ext_curve(L) = coef_ext[1] + coef_ext[2]*log(L) + coef_ext[3]/L + coef_ext[4]/L^2

Lgrid = range(1, 16, length=200)
p = plot(Lgrid, old_curve.(Lgrid), label="previously retained fit (16 depths, mostly 1 seed)",
         color=:gray, lw=2, linestyle=:dash,
         xlabel="Depth L", ylabel="Capacity floor a(L) (nats/char)",
         title="Constraint-test campaign: 9 depths, 4 seeds, long runs (20-40k steps)",
         legend=:topright, left_margin=12Plots.mm, bottom_margin=10Plots.mm,
         top_margin=6Plots.mm, right_margin=6Plots.mm, size=(1000, 650))
plot!(p, Lgrid, log_curve.(Lgrid), label="new log-only fit", color=:steelblue, lw=2.5)
plot!(p, Lgrid, ext_curve.(Lgrid), label="new extended fit (a0+a1 lnL+a2/L+a3/L²)", color=:firebrick, lw=2, linestyle=:dot)
scatter!(p, depths, avals, yerror=ases, color=:darkorange, markersize=7,
         label="new measurements (4 seeds, pooled)")

savefig(p, joinpath(figdir, "growth_constraint_test_en.pdf"))
println("Saved -> figures/growth_constraint_test_en.pdf")
