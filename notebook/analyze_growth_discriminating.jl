# Analyse des 2 expériences discriminantes (§growthexact) : ajuste a(L)=A+slope*ln(L)
# par config (moyenne des 2 graines par profondeur, OLS simple sur les 5
# profondeurs {1,2,4,8,16}, même convention pour toutes -- 4000 pas, aucune
# correction de biais de troncature, pour rester strictement comparable).

using JSON, Statistics, LinearAlgebra

results = JSON.parsefile(joinpath(@__DIR__, "growth_discriminating_results.json"))

const DEPTHS = [1, 2, 4, 8, 16]
const CONFIGS = ["dim128", "dim256", "dim512", "heads2", "heads8"]

function mean_val(cfg, L)
    vals = [results["$(cfg)_L$(L)_seed$(s)"]["final_val"] for s in (1,2)]
    return mean(vals), std(vals)
end

function ols_fit(x, y)
    X = hcat(ones(length(x)), x)
    coef = X \ y
    resid = y .- X*coef
    return coef[1], coef[2], resid  # intercept, slope, residuals
end

println("="^70)
fits = Dict{String,Tuple{Float64,Float64}}()
means_by_cfg = Dict{String,Vector{Float64}}()
for cfg in CONFIGS
    means = Float64[]; stds = Float64[]
    for L in DEPTHS
        m, s = mean_val(cfg, L)
        push!(means, m); push!(stds, s)
    end
    means_by_cfg[cfg] = means
    A, slope, resid = ols_fit(log.(DEPTHS), means)
    fits[cfg] = (A, slope)
    println(rpad(cfg, 8), " : A=", round(A, digits=4), "  slope=", round(slope, digits=4),
            "   vals=", round.(means, digits=4), "  seed-std=", round.(stds, digits=4))
end
println("="^70)

println("\n--- Comparaison LARGEUR (prédiction : pente inchangée, décalage d'ordonnée ±c*ln4≈±0.144) ---")
A128, s128 = fits["dim128"]; A256, s256 = fits["dim256"]; A512, s512 = fits["dim512"]
println("dim128 - dim256 (ordonnée) : ", round(A128 - A256, digits=4), "  (prédit : +0.144)")
println("dim512 - dim256 (ordonnée) : ", round(A512 - A256, digits=4), "  (prédit : -0.144)")
println("pentes : dim128=", round(s128,digits=4), " dim256=", round(s256,digits=4), " dim512=", round(s512,digits=4),
        "  (prédit : ~identiques, ~-0.104)")

println("\n--- Comparaison TÊTES (prédiction : a(L) inchangé) ---")
Ah2, sh2 = fits["heads2"]; Ah8, sh8 = fits["heads8"]
println("heads2 - dim256 (ordonnée) : ", round(Ah2 - A256, digits=4), "  (prédit : 0)")
println("heads8 - dim256 (ordonnée) : ", round(Ah8 - A256, digits=4), "  (prédit : 0)")
println("pentes : heads2=", round(sh2,digits=4), " heads8=", round(sh8,digits=4), " dim256=", round(s256,digits=4))
println("\nÉcart max point-par-point (heads2 vs dim256, par profondeur) :")
for (i,L) in enumerate(DEPTHS)
    println("  L=$L : dim256=", round(means_by_cfg["dim256"][i],digits=4),
            "  heads2=", round(means_by_cfg["heads2"][i],digits=4),
            "  heads8=", round(means_by_cfg["heads8"][i],digits=4))
end

open(joinpath(@__DIR__, "growth_discriminating_fits.json"), "w") do io
    JSON.print(io, Dict("fits"=>Dict(k=>Dict("A"=>v[1],"slope"=>v[2]) for (k,v) in fits),
                         "means_by_cfg"=>means_by_cfg, "depths"=>DEPTHS))
end
println("\nÉcrit -> notebook/growth_discriminating_fits.json")
