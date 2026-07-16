# Ré-estime a(2) et a(3) à partir des 10 graines fraîches (§growthexact,
# tentative d'identification de μ), en réutilisant la même forme canonique
# ℓ(n)=a+b/(n+n0), n0=3000 (Hypothèse 1, §growthmodel), ajustée par régression
# linéaire simple sur x=1/(n+n0) -- même méthode que le fit original par
# profondeur, sur la queue n>=1000 (exclut le tout premier checkpoint, encore
# loin du régime asymptotique).

using JSON, Statistics, LinearAlgebra, Printf

results = JSON.parsefile(joinpath(@__DIR__, "growth_mu_identification_results.json"))
const N0 = 3000

function fit_a_b(depth::Int)
    xs = Float64[]; ys = Float64[]
    for seed in 1:10
        vh = results["L$(depth)_seed$(seed)"]["val_history"]
        for (n, loss, _) in vh
            n < 1000 && continue
            push!(xs, 1.0/(n+N0))
            push!(ys, loss)
        end
    end
    X = hcat(ones(length(xs)), xs)
    coef = X \ ys
    resid = ys .- X*coef
    # erreur standard de l'intercept (a), régression OLS classique
    n_obs = length(ys)
    sigma2 = sum(resid.^2) / (n_obs - 2)
    XtX_inv = inv(X'X)
    se_a = sqrt(sigma2 * XtX_inv[1,1])
    return coef[1], coef[2], se_a, n_obs
end

println("="^70)
a2, b2, se_a2, n2 = fit_a_b(2)
a3, b3, se_a3, n3 = fit_a_b(3)
@printf "a(2) = %.4f ± %.4f  (b=%.1f, n_points=%d, 10 graines)\n" a2 se_a2 b2 n2
@printf "a(3) = %.4f ± %.4f  (b=%.1f, n_points=%d, 10 graines)\n" a3 se_a3 b3 n3
println()
println("Valeurs déjà publiées (16 profondeurs, single-seed ou peu de graines) :")
println("  a(2) = 1.4485 ± 0.0151")
println("  a(3) = 1.9756 ± 0.0777  (exclu comme outlier à 5.7σ dans le fit retenu)")
println("="^70)

println("\nPente log retenue : a0 + a1*ln(L), a0=1.5710, a1=-0.1042")
pred2 = 1.5710 - 0.1042*log(2)
pred3 = 1.5710 - 0.1042*log(3)
@printf "Prédiction à L=2 : %.4f  (nouvelle mesure : %.4f, écart=%.2fσ)\n" pred2 a2 (a2-pred2)/se_a2
@printf "Prédiction à L=3 : %.4f  (nouvelle mesure : %.4f, écart=%.2fσ)\n" pred3 a3 (a3-pred3)/se_a3
println("\n(l'ancienne mesure à L=3, 1.9756, était à (1.9756-", round(pred3,digits=4), ")/0.0777 = ",
        round((1.9756-pred3)/0.0777, digits=2), "σ de la prédiction -- c'est l'outlier à 5.7σ documenté)")
