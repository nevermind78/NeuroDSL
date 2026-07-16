# ══════════════════════════════════════════════════════════════════════════════
# Analyse de la campagne growth_theory_constraint_test.jl : rendre testable la
# contrainte a1*a3/a2^2 = -1/3. Suit le protocole conçu par Fable :
#   1. Par profondeur, fit a(L)+b(L)/(n+n0) par OLS (linéaire en x=1/(n+n0)),
#      poolé sur les 4 graines -- pas de nonlinear fit nécessaire.
#   2. Fit polynomial WLS a0 + a1 lnL + a2/L + a3/L^2 sur les 9 profondeurs.
#   3. Comparaison ΔAIC vs modèle log-seul (refit sur CES données, pas
#      l'ancien fit à 16 profondeurs -- comparaison apples-to-apples).
#   4. Leave-one-out (retrait d'une profondeur à la fois).
#   5. Bootstrap en grappes (graines ET profondeurs) sur R=a1*a3/a2^2.
#   6. Verdict appliqué STRICTEMENT selon les 3 critères pré-enregistrés.
# ══════════════════════════════════════════════════════════════════════════════

using JSON, Statistics, LinearAlgebra, Random, Printf

results = JSON.parsefile(joinpath(@__DIR__, "growth_constraint_test_results.json"))
const N0 = 3000.0
const DEPTHS = [1,2,3,4,5,6,8,12,16]
const SEEDS = 1:4

# ── Étape 1 : fit a(L), b(L) par profondeur (OLS poolé sur les 4 graines) ────
function fit_ab(L)
    xs = Float64[]; ys = Float64[]
    for s in SEEDS
        key = "L$(L)_n$(L in (8,12,16) ? 10000 : 20000)_seed$(s)"
        vh = results[key]["val_history"]
        for pt in vh
            n, loss = Float64(pt[1]), Float64(pt[2])
            push!(xs, 1.0/(n+N0)); push!(ys, loss)
        end
    end
    X = hcat(ones(length(xs)), xs)
    coef = X \ ys
    resid = ys .- X*coef
    n = length(ys)
    # erreur standard de l'intercept a via (X'X)^-1 * sigma^2
    sigma2 = sum(resid.^2) / (n - 2)
    XtX_inv = inv(X'X)
    se_a = sqrt(sigma2 * XtX_inv[1,1])
    return coef[1], coef[2], se_a, n
end

println("="^70)
println("Étape 1 : fit a(L), b(L) par profondeur (pooled 4 graines)")
println("="^70)
avals = Float64[]; bvals = Float64[]; ases = Float64[]
for L in DEPTHS
    a, b, se_a, n = fit_ab(L)
    push!(avals, a); push!(bvals, b); push!(ases, se_a)
    @printf "L=%-3d : a=%.4f ± %.4f   b=%.1f   (n_points=%d)\n" L a se_a b n
end

# Comparaison à l'ancien fit retenu a0=1.5710, a1=-0.1042 (16 profondeurs)
println("\n--- Comparaison au fit précédemment retenu (a0=1.5710, a1=-0.1042) ---")
old_a0, old_a1 = 1.5710, -0.1042
for (i,L) in enumerate(DEPTHS)
    pred_old = old_a0 + old_a1*log(L)
    @printf "L=%-3d : nouveau=%.4f  ancien_predit=%.4f  ecart=%.4f (%.1f σ)\n" L avals[i] pred_old (avals[i]-pred_old) ((avals[i]-pred_old)/ases[i])
end

# ── Étape 2 : fit polynomial WLS a0 + a1 lnL + a2/L + a3/L^2 ────────────────
function design_matrix(Ls)
    return hcat(ones(length(Ls)), log.(Ls), 1.0 ./ Ls, 1.0 ./ (Ls.^2))
end

function wls_fit(Ls, ys, weights)
    X = design_matrix(Ls)
    W = Diagonal(weights)
    XtWX = X' * W * X
    XtWy = X' * W * ys
    coef = XtWX \ XtWy
    resid = ys .- X*coef
    return coef, resid, XtWX
end

function log_only_fit(Ls, ys, weights)
    X = hcat(ones(length(Ls)), log.(Ls))
    W = Diagonal(weights)
    coef = (X'*W*X) \ (X'*W*ys)
    resid = ys .- X*coef
    return coef, resid
end

weights = 1.0 ./ (ases.^2)
coef_ext, resid_ext, XtWX = wls_fit(Float64.(DEPTHS), avals, weights)
coef_log, resid_log = log_only_fit(Float64.(DEPTHS), avals, weights)

a0f, a1f, a2f, a3f = coef_ext
R = a1f*a3f / a2f^2

println("\n", "="^70)
println("Étape 2 : fit polynomial étendu (WLS, pondéré par 1/SE²)")
println("="^70)
@printf "a0=%.4f  a1=%.4f  a2=%.4f  a3=%.4f\n" a0f a1f a2f a3f
@printf "R = a1*a3/a2^2 = %.4f   (cible : -1/3 = %.4f)\n" R (-1/3)

# AIC : k params, RSS pondéré -> logL approx via chi2 pondéré
function aic(resid, weights, k)
    n = length(resid)
    wrss = sum(weights .* resid.^2)
    # log-vraisemblance gaussienne pondérée (à une constante additive près,
    # constante identique entre modèles emboîtés -- valide pour ΔAIC)
    logL = -0.5*n*log(wrss/n)
    return 2*k - 2*logL
end
aic_ext = aic(resid_ext, weights, 4)
aic_log = aic(resid_log, weights, 2)
@printf "\nAIC (log-seul, 2 params) = %.3f\n" aic_log
@printf "AIC (étendu, 4 params)   = %.3f\n" aic_ext
@printf "ΔAIC (étendu - log)      = %.3f  (favorable si < -6)\n" (aic_ext-aic_log)

# ── Étape 3 : leave-one-out ──────────────────────────────────────────────────
println("\n", "="^70)
println("Étape 3 : leave-one-out (retrait d'une profondeur)")
println("="^70)
loo_pass = true
for i in eachindex(DEPTHS)
    idx = setdiff(1:length(DEPTHS), i)
    Ls_loo = Float64.(DEPTHS)[idx]; ys_loo = avals[idx]; w_loo = weights[idx]
    ce, re = wls_fit(Ls_loo, ys_loo, w_loo)
    cl, rl = log_only_fit(Ls_loo, ys_loo, w_loo)
    a_loo = aic(re, w_loo, 4); al_loo = aic(rl, w_loo, 2)
    dAIC = a_loo - al_loo
    Rloo = ce[2]*ce[4]/ce[3]^2
    ok = dAIC < -6
    global loo_pass &= ok
    @printf "sans L=%-3d : ΔAIC=%.3f  R=%.4f  %s\n" DEPTHS[i] dAIC Rloo (ok ? "✓" : "✗ ÉCHEC")
end
println("Critère LOO (ΔAIC<-6 pour TOUT retrait) : ", loo_pass ? "PASSÉ" : "ÉCHOUÉ")

# ── Étape 4 : bootstrap en grappes (graines ET profondeurs) ─────────────────
println("\n", "="^70)
println("Étape 4 : bootstrap en grappes (1000 tirages, resample graines + profondeurs)")
println("="^70)

Random.seed!(42)
NBOOT = 1000
Rboot = Float64[]
sign_ok_count = 0
ext_selected_count = 0
for _ in 1:NBOOT
    # Tirage avec remise des profondeurs (grappe = profondeur)
    boot_depth_idx = rand(1:length(DEPTHS), length(DEPTHS))
    Ls_b = Float64.(DEPTHS)[boot_depth_idx]
    # Pour chaque profondeur tirée, tirage avec remise des graines -> nouveau a(L),SE
    avals_b = Float64[]; ases_b = Float64[]
    for di in boot_depth_idx
        L = DEPTHS[di]
        boot_seeds = rand(SEEDS, 4)
        xs = Float64[]; ys = Float64[]
        for s in boot_seeds
            key = "L$(L)_n$(L in (8,12,16) ? 10000 : 20000)_seed$(s)"
            vh = results[key]["val_history"]
            for pt in vh
                n, loss = Float64(pt[1]), Float64(pt[2])
                push!(xs, 1.0/(n+N0)); push!(ys, loss)
            end
        end
        Xb = hcat(ones(length(xs)), xs)
        cb = Xb \ ys
        rb = ys .- Xb*cb
        sigma2b = sum(rb.^2)/(length(ys)-2)
        se_ab = sqrt(sigma2b * inv(Xb'Xb)[1,1])
        push!(avals_b, cb[1]); push!(ases_b, se_ab)
    end
    w_b = 1.0 ./ (ases_b.^2)
    try
        ceb, reb = wls_fit(Ls_b, avals_b, w_b)
        clb, rlb = log_only_fit(Ls_b, avals_b, w_b)
        dAICb = aic(reb, w_b, 4) - aic(rlb, w_b, 2)
        if dAICb < -6
            global ext_selected_count += 1
        end
        Rb = ceb[2]*ceb[4]/ceb[3]^2
        push!(Rboot, Rb)
        if ceb[2] < 0 && ceb[3] < 0 && ceb[4] > 0
            global sign_ok_count += 1
        end
    catch
        continue
    end
end

ci_lo, ci_hi = quantile(Rboot, [0.025, 0.975])
frac_ext_selected = ext_selected_count / NBOOT
frac_sign_ok = sign_ok_count / length(Rboot)

@printf "Fraction de tirages où le modèle étendu est sélectionné (ΔAIC<-6) : %.1f%%  (requis ≥90%%)\n" (100*frac_ext_selected)
@printf "Fraction de tirages avec signes stables (a1<0,a2<0,a3>0)          : %.1f%%  (requis ≥95%%)\n" (100*frac_sign_ok)
@printf "IC95%% bootstrap de R = a1*a3/a2^2                                : [%.4f, %.4f]\n" ci_lo ci_hi
println("Bande théorique attendue : [-0.333, -0.29]")

# ── Verdict pré-enregistré ───────────────────────────────────────────────────
println("\n", "="^70)
println("VERDICT PRÉ-ENREGISTRÉ")
println("="^70)
crit1 = (aic_ext - aic_log < -6) && loo_pass && (frac_ext_selected >= 0.90)
crit2 = frac_sign_ok >= 0.95
band_lo, band_hi = -0.333, -0.29
crit3 = ci_lo <= band_hi && ci_hi >= band_lo  # chevauchement avec la bande

println("Critère 1 (sélection robuste du modèle étendu) : ", crit1 ? "PASSÉ" : "ÉCHOUÉ")
println("Critère 2 (signes stables ≥95%)                : ", crit2 ? "PASSÉ" : "ÉCHOUÉ")
println("Critère 3 (IC95% de R recoupe la bande théorique) : ", crit3 ? "PASSÉ" : "ÉCHOUÉ")

if crit1 && !crit3
    println("\n>>> RÉFUTATION PROPRE : modèle robuste mais R hors bande théorique.")
elseif crit1 && crit2 && crit3
    println("\n>>> CONFIRMATION : les trois critères passent.")
else
    println("\n>>> INCONCLUSIF : le modèle étendu n'est pas assez robuste pour interpréter R.")
    println("    (R nominal = ", round(R,digits=4), " -- NE PAS CITER comme mesure fiable)")
end

open(joinpath(@__DIR__, "growth_constraint_fit_results.json"), "w") do io
    JSON.print(io, Dict(
        "depths"=>DEPTHS, "avals"=>avals, "bvals"=>bvals, "ases"=>ases,
        "coef_ext"=>coef_ext, "coef_log"=>coef_log, "R"=>R,
        "aic_ext"=>aic_ext, "aic_log"=>aic_log,
        "loo_pass"=>loo_pass,
        "boot_ci"=>[ci_lo, ci_hi], "frac_ext_selected"=>frac_ext_selected,
        "frac_sign_ok"=>frac_sign_ok,
        "crit1"=>crit1, "crit2"=>crit2, "crit3"=>crit3,
    ))
end
println("\nÉcrit -> notebook/growth_constraint_fit_results.json")
