# ══════════════════════════════════════════════════════════════════════════════
# Refit joint global : loss(n,L) = A - c*psi(mu*L+1) + b(L)/(n+n0), n0=3000,
# b(L)=b0+b1*ln(L), ajusté en UNE SEULE fois sur TOUS les points bruts
# "from-scratch" (jamais post-greffe) déjà collectés dans ce projet -- jalon0
# (bras fixe), jalon1/1b (bras stages1=fixe), missing_depths (toutes
# profondeurs, single-seed), mu_identification (L=2,3 x10 graines), et la
# nouvelle campagne constraint_test (L=1..6,8,12,16 x4 graines, runs longs).
#
# Pour chaque mu sur une grille, (A,c,b0,b1) se résolvent par moindres carrés
# linéaires exacts (le modèle est linéaire en ces 4 coefficients à mu fixé) --
# pas besoin d'optimiseur non-linéaire. On regarde ensuite si la courbe de
# RSS-profil en mu a un minimum net (identifiabilité) ou reste plate (comme
# avant). Bootstrap en grappes par PROFONDEUR (pas par point) pour l'IC final.
#
# digamma implémenté à la main (récursion + série asymptotique de Stirling),
# aucune dépendance nouvelle -- même convention que le reste de la session.
# ══════════════════════════════════════════════════════════════════════════════

using JSON, Statistics, LinearAlgebra, Random, Printf

function digamma_asympt(x::Float64)
    r = 1.0/x; r2 = r*r
    return log(x) - 0.5*r - r2*(1/12 - r2*(1/120 - r2*(1/252)))
end
function digamma(x::Float64)
    acc = 0.0
    while x < 6.0
        acc -= 1.0/x
        x += 1.0
    end
    return acc + digamma_asympt(x)
end

const N0 = 3000.0
dir = @__DIR__
pts_n = Float64[]; pts_loss = Float64[]; pts_L = Float64[]

function add_entry!(vh, L)
    for p in vh
        push!(pts_n, Float64(p[1])); push!(pts_loss, Float64(p[2])); push!(pts_L, Float64(L))
    end
end

# jalon0 : seul le bras "A_fixed_4" est from-scratch (B,C sont des schedules de croissance)
j0 = JSON.parsefile(joinpath(dir, "growth_jalon0_results.json"))
for (k,v) in j0
    startswith(k, "A_fixed_4_seed") && add_entry!(v["val_history"], v["schedule"][1])
end

# jalon1 / jalon1_fixed / jalon1b : seul stages1 (fixe) est from-scratch
for fname in ["growth_jalon1_results.json", "growth_jalon1_fixed_results.json", "growth_jalon1b_results.json"]
    fpath = joinpath(dir, fname)
    isfile(fpath) || continue
    d = JSON.parsefile(fpath)
    for (k,v) in d
        startswith(k, "stages1_seed") && add_entry!(v["val_history"], v["schedule"][1])
    end
end

# missing_depths : toutes les entrées sont from-scratch, profondeur unique
md = JSON.parsefile(joinpath(dir, "growth_theory_missing_depths_results.json"))
for (k,v) in md
    add_entry!(v["val_history"], v["depth"])
end

# mu_identification : L=2,3, 10 graines chacun, from-scratch
mu_id = JSON.parsefile(joinpath(dir, "growth_mu_identification_results.json"))
for (k,v) in mu_id
    add_entry!(v["val_history"], v["depth"])
end

# constraint_test : les 36 runs longs, from-scratch
ct = JSON.parsefile(joinpath(dir, "growth_constraint_test_results.json"))
for (k,v) in ct
    add_entry!(v["val_history"], v["depth"])
end

println("Total de points bruts poolés (from-scratch uniquement) : ", length(pts_n))
depths_present = sort(unique(Int.(pts_L)))
println("Profondeurs présentes : ", depths_present)
for L in depths_present
    n_at_L = count(==(Float64(L)), pts_L)
    println("  L=$L : $n_at_L points")
end

# ── Fit par grille sur mu, (A,c,b0,b1) linéaires à mu fixé ──────────────────
function fit_at_mu(mu, ns, losses, Ls)
    npts = length(ns)
    X = Matrix{Float64}(undef, npts, 4)
    for i in 1:npts
        invn = 1.0/(ns[i]+N0)
        X[i,1] = 1.0
        X[i,2] = -digamma(mu*Ls[i]+1.0)
        X[i,3] = invn
        X[i,4] = invn*log(Ls[i])
    end
    coef = X \ losses
    resid = losses .- X*coef
    rss = sum(resid.^2)
    return coef, rss
end

mu_grid = 10.0 .^ range(-2, 2, length=400)
rss_grid = Float64[]
for mu in mu_grid
    _, rss = fit_at_mu(mu, pts_n, pts_loss, pts_L)
    push!(rss_grid, rss)
end
best_idx = argmin(rss_grid)
mu_hat = mu_grid[best_idx]
coef_hat, rss_hat = fit_at_mu(mu_hat, pts_n, pts_loss, pts_L)
A_hat, c_hat, b0_hat, b1_hat = coef_hat

println("\n", "="^70)
println("Fit joint global -- optimum sur la grille")
println("="^70)
@printf "mu_hat = %.4f\n" mu_hat
@printf "A=%.4f  c=%.4f  b0=%.1f  b1=%.1f\n" A_hat c_hat b0_hat b1_hat
@printf "RSS au minimum = %.4f  (n=%d points, %d paramètres effectifs)\n" rss_hat length(pts_n) 5

# Courbe de profil : RSS relatif au minimum, à quelques mu de référence
println("\nProfil RSS (relatif au minimum) à quelques valeurs de mu de référence :")
for mu_ref in [0.1, 0.3, 0.7, 1.0, 2.0, 5.0, 10.0, 30.0]
    idx = argmin(abs.(mu_grid .- mu_ref))
    @printf "  mu=%.2f : RSS/RSS_min = %.6f\n" mu_grid[idx] (rss_grid[idx]/rss_hat)
end

# Seuil de vraisemblance-ratio approx (F-test à 1 param effectif) pour un IC
n_eff = length(pts_n)
s2 = rss_hat / (n_eff - 5)
delta_rss_95 = s2 * 3.84   # seuil chi2(1) à 95%
ci_idx = findall(rss_grid .<= rss_hat + delta_rss_95)
if !isempty(ci_idx)
    println("\nIC95% (profil de vraisemblance, seuil chi2(1)) sur mu : [",
            round(mu_grid[minimum(ci_idx)], digits=4), ", ",
            round(mu_grid[maximum(ci_idx)], digits=4), "]")
else
    println("\nIC95% : vide (ne devrait pas arriver)")
end

# ── Bootstrap en grappes par PROFONDEUR ──────────────────────────────────────
println("\n", "="^70)
println("Bootstrap en grappes par profondeur (500 tirages)")
println("="^70)
Random.seed!(123)
mu_boot = Float64[]
for _ in 1:500
    boot_depths = rand(depths_present, length(depths_present))
    bn = Float64[]; bl = Float64[]; bL = Float64[]
    for L in boot_depths
        idx = findall(==(Float64(L)), pts_L)
        append!(bn, pts_n[idx]); append!(bl, pts_loss[idx]); append!(bL, pts_L[idx])
    end
    local_rss = Float64[]
    for mu in mu_grid
        _, rss = fit_at_mu(mu, bn, bl, bL)
        push!(local_rss, rss)
    end
    push!(mu_boot, mu_grid[argmin(local_rss)])
end
lo, hi = quantile(mu_boot, [0.025, 0.975])
@printf "IC95%% bootstrap (grappes par profondeur) sur mu : [%.4f, %.4f]\n" lo hi
@printf "Médiane bootstrap : %.4f\n" median(mu_boot)

open(joinpath(dir, "growth_joint_fit_results.json"), "w") do io
    JSON.print(io, Dict(
        "n_points"=>length(pts_n), "depths_present"=>depths_present,
        "mu_hat"=>mu_hat, "A"=>A_hat, "c"=>c_hat, "b0"=>b0_hat, "b1"=>b1_hat,
        "rss_hat"=>rss_hat,
        "ci95_profile"=>isempty(ci_idx) ? nothing : [mu_grid[minimum(ci_idx)], mu_grid[maximum(ci_idx)]],
        "ci95_bootstrap"=>[lo, hi], "mu_boot_median"=>median(mu_boot),
    ))
end
println("\nÉcrit -> notebook/growth_joint_fit_results.json")
