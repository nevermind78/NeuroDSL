# ══════════════════════════════════════════════════════════════════════════════
# RECALCUL DES CORRÉLATIONS DE SPEARMAN CITÉES DANS artilce/code4.tex, §7
# -- aucune nouvelle mesure sur les checkpoints : uniquement de la statistique
# sur les JSON déjà produits par verify_config_band_oos.jl et
# verify_robust_collapse.jl.
#
# POURQUOI CE SCRIPT EXISTE
# --------------------------
# Un audit indépendant a constaté que les rho de Spearman initialement cités
# dans l'article (-0.81 sur les 39 configurations, -0.79 sur les 25 hors
# échantillon, -0.80 sur les 14 hors-échantillon-et-non-nulles, 0.67 entre
# taux de certificat et taux de collapse réalisé) ne se reproduisent QU'avec
# une règle de rangs non standard : les ex æquo y sont départagés par position
# dans le tableau plutôt que par rang moyen (ex. `rankdata(..., method="ordinal")`
# plutôt que `method="average"`). Ce n'est pas la définition manuel du rho de
# Spearman -- avec la variable `interaction`, qui vaut exactement (ou quasi)
# zéro sur 12 à 16 des 39 configurations (les ablations à un seul carrier),
# le choix du départage des ex æquo change le résultat de façon non
# négligeable. Le rho de Spearman correctement calculé (rangs moyens, comme
# le fait `StatsBase.corspearman`, comme `scipy.stats.spearmanr`, comme R
# `cor(method="spearman")`) diffère de 0.02 à 0.06 des valeurs initialement
# publiées. La conclusion qualitative n'en est pas affaiblie -- au contraire,
# la corrélation correctement calculée est plus forte, pas plus faible.
#
# Ce script recalcule les quatre rho avec la définition standard et un test
# de permutation à deux côtés (200000 tirages), et remplace les valeurs
# citées dans le texte.
# ══════════════════════════════════════════════════════════════════════════════

using JSON, StatsBase, Random

function perm_pvalue(x::Vector{Float64}, y::Vector{Float64}; n::Int=200_000, seed::Int=1)
    rho_obs = corspearman(x, y)
    rng = MersenneTwister(seed)
    yp = copy(y)
    cnt = 0
    for _ in 1:n
        shuffle!(rng, yp)
        rho = corspearman(x, yp)
        (abs(rho) >= abs(rho_obs) - 1e-12) && (cnt += 1)
    end
    return rho_obs, (cnt + 1) / (n + 1)
end

const HERE = @__DIR__

d1 = JSON.parsefile(joinpath(HERE, "config_band_oos_results.json"))
cs = d1["configs"]
interaction = Float64[c["interaction"] for c in cs]
c1a         = Float64[c["c1a"] for c in cs]
in_sample   = Bool[c["in_sample"] for c in cs]
oos         = .!in_sample
nz          = interaction .> 1e-4
oos_nz      = oos .& nz

println("configurations : ", length(cs), " (", count(in_sample), " in-sample, ", count(oos), " hors éch.)")

rho_all, p_all = perm_pvalue(interaction, c1a)
println("\n[all 39]              rho = ", round(rho_all, digits=4), "   perm-p = ", p_all)

rho_oos, p_oos = perm_pvalue(interaction[oos], c1a[oos])
println("[25 held-out]         rho = ", round(rho_oos, digits=4), "   perm-p = ", p_oos)

rho_oosnz, p_oosnz = perm_pvalue(interaction[oos_nz], c1a[oos_nz])
println("[14 held-out+nonzero] rho = ", round(rho_oosnz, digits=4), "   perm-p = ", p_oosnz, "  (n=", count(oos_nz), ")")

d2 = JSON.parsefile(joinpath(HERE, "robust_collapse_results.json"))
cs2  = d2["configs"]
hyp  = Float64[c["hyp_rate"] for c in cs2]
conc = Float64[c["conc_rate"] for c in cs2]
rho_cert, p_cert = perm_pvalue(hyp, conc)
println("[cert-rate vs collapse-rate] rho = ", round(rho_cert, digits=4), "   perm-p = ", p_cert)

# Contrôle : documente le MÉCANISME de l'écart trouvé le 2026-07-27 (rangs
# ex æquo départagés par position plutôt que par rang moyen) -- ne reproduit
# plus exactement un chiffre publié précis, puisque les valeurs elles-mêmes
# ont depuis bougé une seconde fois (correctif de noyau CUDA du 2026-07-28,
# `_warp_reduce_add`/`_warp_reduce_max`, sans rapport avec les rangs) ; garde
# néanmoins un écart net et dans le même sens vis-à-vis de la méthode
# standard ci-dessus, ce qui suffit à illustrer le mécanisme.
rank_ordinal(v) = invperm(sortperm(v))
rho_ordinal(x, y) = corspearman(Float64.(rank_ordinal(x)), Float64.(rank_ordinal(y)))
println("\n[contrôle -- ex æquo départagés par position, NON standard]")
println("  all 39      : ", round(rho_ordinal(interaction, c1a), digits=4), "  (vs standard ci-dessus)")
println("  25 held-out : ", round(rho_ordinal(interaction[oos], c1a[oos]), digits=4), "  (vs standard ci-dessus)")
println("  cert vs coll: ", round(rho_ordinal(hyp, conc), digits=4), "  (vs standard ci-dessus)")
