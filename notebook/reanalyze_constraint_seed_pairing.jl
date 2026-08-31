# ══════════════════════════════════════════════════════════════════════════════
# Réanalyse : la campagne growth_theory_constraint_test.jl (via growth_jalon0.jl,
# train_growth_arm!) appelle Random.seed!(seed) AVANT build_char_lm_graph, donc
# pour un seed fixe, les blocs partagés (couches 1..min(L1,L2)) reçoivent les
# MÊMES poids initiaux à toute profondeur, ET rng=MersenneTwister(seed) pilote
# sample_window de façon identique entre profondeurs (même nombre de pas par
# palier de budget -> même séquence de fenêtres d'entraînement). "seed k à L1"
# et "seed k à L2" ne sont donc PAS des tirages indépendants.
#
# Le bootstrap Étape 4 de analyze_growth_constraint.jl tire pourtant les graines
# INDÉPENDAMMENT à chaque profondeur (`boot_seeds = rand(SEEDS,4)` dans la
# boucle sur les profondeurs). Ce script :
#   1. Calcule un fit a(L,seed) PAR graine (même méthode OLS que fit_ab, mais
#      sur une seule graine à la fois).
#   2. Teste empiriquement s'il y a une corrélation croisée-profondeurs pour
#      l'indice de graine apparié (résidu par rapport à la moyenne des graines
#      à profondeur fixe, corrélation poolée sur toutes les paires de
#      profondeurs, test de permutation pour la significativité).
#   3. Si corrélation mesurable : refait le bootstrap Étape 4 en tirant UN
#      SEUL jeu de 4 indices de graines par itération, appliqué identiquement
#      aux 9 profondeurs (bootstrap par blocs / apparié), et recalcule les 3
#      critères pré-enregistrés.
#   4. Sinon : rapporte l'absence de corrélation mesurable, sans fabriquer de
#      résultat.
# ══════════════════════════════════════════════════════════════════════════════

using JSON, Statistics, LinearAlgebra, Random, Printf

results = JSON.parsefile(joinpath(@__DIR__, "growth_constraint_test_results.json"))
const N0 = 3000.0
const DEPTHS = [1,2,3,4,5,6,8,12,16]
const SEEDS = 1:4
nsteps_for(L) = L in (8,12,16) ? 10000 : 20000

out = IOBuffer()
function say(io, fmt::String, args...)
    s = Printf.format(Printf.Format(fmt), args...)
    println(s); println(io, s)
end
sayln(io, s="") = (println(s); println(io, s))

sayln(out, "="^78)
sayln(out, "Réanalyse : structure partagée seed/profondeur dans le bootstrap Étape 4")
sayln(out, "="^78)

# ── Vérification du code (rapport de ce qui a été lu, pas recalculé) ────────
sayln(out, "\n--- Étape 0 : vérification directe du code (growth_jalon0.jl) ---")
sayln(out, "train_growth_arm!(schedule, budget; seed, ...) :")
sayln(out, "  ligne 128 : Random.seed!(seed)          <- RNG global, AVANT construction")
sayln(out, "  ligne 129 : build_char_lm_graph(...)     <- poids tirés via ce RNG global")
sayln(out, "  ligne 146 : rng = MersenneTwister(seed)  <- RNG séparé, dédié aux fenêtres")
sayln(out, "  campagne (growth_theory_constraint_test.jl) : schedule=[L] (1 seul palier),")
sayln(out, "  layer_step_budget = n_steps*L => n_steps_stage = budget÷L = n_steps,")
sayln(out, "  IDENTIQUE pour toutes les profondeurs d'un même groupe de budget")
sayln(out, "  (20000 pour L∈{1..6}, 10000 pour L∈{8,12,16}).")
sayln(out, "  => même seed : même graine d'init pour les blocs partagés (couches 1..min(L1,L2))")
sayln(out, "     ET même séquence exacte de fenêtres d'entraînement (même rng, même nb d'appels).")
sayln(out, "  => \"seed k à L1\" et \"seed k à L2\" NE SONT PAS des tirages indépendants.")

# ── Étape 1 : fit a(L,seed) par graine individuelle (même méthode que fit_ab
#    de analyze_growth_constraint.jl, mais sur une seule graine) ────────────
function fit_a_single(L, s)
    key = "L$(L)_n$(nsteps_for(L))_seed$(s)"
    vh = results[key]["val_history"]
    xs = Float64[]; ys = Float64[]
    for pt in vh
        n, loss = Float64(pt[1]), Float64(pt[2])
        push!(xs, 1.0/(n+N0)); push!(ys, loss)
    end
    X = hcat(ones(length(xs)), xs)
    coef = X \ ys
    return coef[1], coef[2]  # a, b
end

nD = length(DEPTHS); nS = length(SEEDS)
A = zeros(nD, nS)
for (i,L) in enumerate(DEPTHS), (j,s) in enumerate(SEEDS)
    a, b = fit_a_single(L, s)
    A[i,j] = a
end

sayln(out, "\n" * "="^78)
sayln(out, "Étape 1 : fit a(L,seed) PAR graine individuelle (matrice profondeur x graine)")
sayln(out, "="^78)
for (i,L) in enumerate(DEPTHS)
    row = join([@sprintf("%.4f", A[i,j]) for j in 1:nS], "   ")
    say(out, "L=%-3d : %s", L, row)
end

# ── Étape 2 : corrélation croisée-profondeurs pour l'indice de graine apparié
#    Résidu par rapport à la moyenne des graines à profondeur fixe, poolé sur
#    toutes les paires de profondeurs (C(9,2)=36 paires x 4 graines = 144 pts).
Rres = similar(A)
for i in 1:nD
    m = mean(A[i,:])
    Rres[i,:] = A[i,:] .- m
end

function pooled_corr(Rmat)
    px = Float64[]; py = Float64[]
    for i in 1:nD, k in (i+1):nD
        for j in 1:nS
            push!(px, Rmat[i,j]); push!(py, Rmat[k,j])
        end
    end
    return cor(px, py), length(px)
end

obs_corr, npairs = pooled_corr(Rres)

sayln(out, "\n" * "="^78)
sayln(out, "Étape 2 : corrélation croisée-profondeurs, indice de graine apparié")
sayln(out, "="^78)
say(out, "Résidu a(L,s) - moyenne_s(a(L,:)), poolé sur %d paires de profondeurs x 4 graines = %d points", binomial(nD,2), npairs)
say(out, "Corrélation de Pearson observée : r = %.4f", obs_corr)

# Test de permutation : permute l'ordre des graines INDÉPENDAMMENT à chaque
# profondeur (= casse l'appariement par indice de graine tout en conservant
# la structure marginale profondeur-par-profondeur), recalcule r, répète.
Random.seed!(2026)
NPERM = 20_000
perm_corrs = Float64[]
for _ in 1:NPERM
    Rp = similar(Rres)
    for i in 1:nD
        perm = shuffle(1:nS)
        Rp[i,:] = Rres[i, perm]
    end
    rp, _ = pooled_corr(Rp)
    push!(perm_corrs, rp)
end
p_value = mean(abs.(perm_corrs) .>= abs(obs_corr))
say(out, "Test de permutation (%d tirages, permutation des graines par profondeur) : p = %.4f", NPERM, p_value)
say(out, "  (moyenne |r| sous H0 = %.4f, écart-type = %.4f)", mean(abs.(perm_corrs)), std(perm_corrs))

MEASURABLE = p_value < 0.05
sayln(out, MEASURABLE ?
    "\n>>> Corrélation croisée-profondeurs MESURABLE (p<0.05) -- structure réelle, pas du bruit." :
    "\n>>> PAS de corrélation croisée-profondeurs mesurable (p>=0.05) -- l'hypothèse de structure partagée ne laisse pas de trace détectable dans a(L,seed).")

if !MEASURABLE
    sayln(out, "\n" * "="^78)
    sayln(out, "CONCLUSION")
    sayln(out, "="^78)
    sayln(out, "Le partage d'initialisation/ordre de fenêtres entre profondeurs (confirmé au")
    sayln(out, "niveau du code) n'imprime PAS de corrélation détectable sur a(L,seed) dans ces")
    sayln(out, "données -- le bruit d'entraînement (20000 pas, dynamique non-linéaire, insertion")
    sayln(out, "de nouvelles couches profondes après les couches partagées) efface apparemment")
    sayln(out, "la corrélation d'origine. Le bootstrap indépendant-par-profondeur de l'Étape 4")
    sayln(out, "de analyze_growth_constraint.jl n'est donc PAS invalidé par cet effet : pas de")
    sayln(out, "biais démontré, pas de raison de refaire le bootstrap par blocs. Le verdict")
    sayln(out, "inconclusif d'article3.tex (R=-0.1247, IC95% [-0.19,0.57]) est confirmé tel quel.")
    open(joinpath(@__DIR__, "reanalyze_constraint_seed_pairing_results.txt"), "w") do io
        write(io, String(take!(out)))
    end
    println("\nÉcrit -> notebook/reanalyze_constraint_seed_pairing_results.txt")
    exit(0)
end

# ── Étape 3 (uniquement si corrélation mesurable) : bootstrap par blocs ─────
# Réutilise EXACTEMENT la logique de fit_ab / wls_fit / log_only_fit / aic de
# analyze_growth_constraint.jl pour rester comparable au fit original.
function fit_ab_pooled(L, seeds_used)
    xs = Float64[]; ys = Float64[]
    for s in seeds_used
        key = "L$(L)_n$(nsteps_for(L))_seed$(s)"
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
    sigma2 = sum(resid.^2) / (n - 2)
    se_a = sqrt(sigma2 * inv(X'X)[1,1])
    return coef[1], se_a
end

function design_matrix(Ls)
    return hcat(ones(length(Ls)), log.(Ls), 1.0 ./ Ls, 1.0 ./ (Ls.^2))
end
function wls_fit(Ls, ys, weights)
    X = design_matrix(Ls)
    W = Diagonal(weights)
    coef = (X'*W*X) \ (X'*W*ys)
    resid = ys .- X*coef
    return coef, resid
end
function log_only_fit(Ls, ys, weights)
    X = hcat(ones(length(Ls)), log.(Ls))
    W = Diagonal(weights)
    coef = (X'*W*X) \ (X'*W*ys)
    resid = ys .- X*coef
    return coef, resid
end
function aic(resid, weights, k)
    n = length(resid)
    wrss = sum(weights .* resid.^2)
    logL = -0.5*n*log(wrss/n)
    return 2*k - 2*logL
end

# Fit complet poolé sur les 4 graines (référence, IDENTIQUE à analyze_growth_constraint.jl)
avals = Float64[]; ases = Float64[]
for L in DEPTHS
    a, se_a = fit_ab_pooled(L, SEEDS)
    push!(avals, a); push!(ases, se_a)
end
weights = 1.0 ./ (ases.^2)
coef_ext, resid_ext = wls_fit(Float64.(DEPTHS), avals, weights)
coef_log, resid_log = log_only_fit(Float64.(DEPTHS), avals, weights)
a0f, a1f, a2f, a3f = coef_ext
R_point = a1f*a3f / a2f^2
aic_ext = aic(resid_ext, weights, 4)
aic_log = aic(resid_log, weights, 2)

sayln(out, "\n" * "="^78)
sayln(out, "Étape 3 : fit complet de référence (identique à analyze_growth_constraint.jl)")
sayln(out, "="^78)
say(out, "R nominal = a1*a3/a2^2 = %.4f", R_point)
say(out, "ΔAIC (étendu-log) = %.3f", aic_ext-aic_log)

# LOO (identique à l'original, ne dépend pas du schéma de bootstrap)
loo_pass = true
for i in eachindex(DEPTHS)
    idx = setdiff(1:length(DEPTHS), i)
    ce, re = wls_fit(Float64.(DEPTHS)[idx], avals[idx], weights[idx])
    cl, rl = log_only_fit(Float64.(DEPTHS)[idx], avals[idx], weights[idx])
    dAIC = aic(re, weights[idx], 4) - aic(rl, weights[idx], 2)
    global loo_pass &= (dAIC < -6)
end
say(out, "LOO (ΔAIC<-6 pour tout retrait) : %s", loo_pass ? "PASSÉ" : "ÉCHOUÉ")

sayln(out, "\n" * "="^78)
sayln(out, "Étape 4bis : bootstrap PAR BLOCS (1 tirage de 4 graines appliqué aux 9 profondeurs)")
sayln(out, "="^78)

Random.seed!(42)
NBOOT = 1000
Rboot = Float64[]
sign_ok_count = 0
ext_selected_count = 0
for _ in 1:NBOOT
    boot_depth_idx = rand(1:nD, nD)
    Ls_b = Float64.(DEPTHS)[boot_depth_idx]
    # Un SEUL tirage de graines pour CETTE itération, appliqué à toutes les profondeurs.
    boot_seeds = rand(SEEDS, nS)
    avals_b = Float64[]; ases_b = Float64[]
    for di in boot_depth_idx
        L = DEPTHS[di]
        a_b, se_a_b = fit_ab_pooled(L, boot_seeds)
        push!(avals_b, a_b); push!(ases_b, se_a_b)
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

say(out, "Fraction de tirages où le modèle étendu est sélectionné (ΔAIC<-6) : %.1f%%  (requis ≥90%%)", 100*frac_ext_selected)
say(out, "Fraction de tirages avec signes stables (a1<0,a2<0,a3>0)          : %.1f%%  (requis ≥95%%)", 100*frac_sign_ok)
say(out, "IC95%% bootstrap par blocs de R = a1*a3/a2^2                      : [%.4f, %.4f]", ci_lo, ci_hi)
sayln(out, "Bande théorique attendue : [-0.333, -0.29]")

crit1 = (aic_ext - aic_log < -6) && loo_pass && (frac_ext_selected >= 0.90)
crit2 = frac_sign_ok >= 0.95
band_lo, band_hi = -0.333, -0.29
crit3 = ci_lo <= band_hi && ci_hi >= band_lo

sayln(out, "\n" * "="^78)
sayln(out, "VERDICT (bootstrap par blocs / apparié)")
sayln(out, "="^78)
sayln(out, "Critère 1 (sélection robuste du modèle étendu) : " * (crit1 ? "PASSÉ" : "ÉCHOUÉ"))
sayln(out, "Critère 2 (signes stables ≥95%)                : " * (crit2 ? "PASSÉ" : "ÉCHOUÉ"))
sayln(out, "Critère 3 (IC95% de R recoupe la bande théorique) : " * (crit3 ? "PASSÉ" : "ÉCHOUÉ"))

open(joinpath(@__DIR__, "reanalyze_constraint_seed_pairing_results.txt"), "w") do io
    write(io, String(take!(out)))
end
println("\nÉcrit -> notebook/reanalyze_constraint_seed_pairing_results.txt")
