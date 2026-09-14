import Pkg
Pkg.activate(@__DIR__; io=devnull)
using Plots
using DelimitedFiles

patched = readdlm(joinpath(@__DIR__, "scaling_patched_results.tsv"), '\t'; skipstart=1)
stock = readdlm(joinpath(@__DIR__, "..", "mtk_translator_env", "scaling_stock_results.tsv"), '\t'; skipstart=1)

gr()
plt = plot(patched[:,1], patched[:,2]; label = "patched (compact dot derivative)", marker = :circle, lw = 2,
    xlabel = "n (position/velocity array size)", ylabel = "mtkcompile wall time (s)",
    title = "mtkcompile cost vs array size: patched vs stock", size = (700, 450), titlefontsize = 11,
    legend = :topleft)
plot!(plt, stock[:,1], stock[:,2]; label = "stock (scalarized dot derivative)", marker = :diamond, lw = 2, ls = :dash)
savefig(plt, joinpath(@__DIR__, "plots", "05_mtkcompile_scaling_patched_vs_stock.png"))
println("saved 05_mtkcompile_scaling_patched_vs_stock.png")
