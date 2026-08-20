# ══════════════════════════════════════════════════════════════════════════════
# Balayage taille-de-corpus a profondeur fixe -- mesure DIRECTE de l'exposant de
# Zipf epsilon du modele de quantification (Michaud et al. 2023, arXiv:2303.13506),
# sur l'axe des donnees plutot que via la courbure de a(L) (article3.tex,
# §growthexact, ligne 641-643 -- piste explicitement laissee de cote).
#
# Protocole fige dans notebook/corpus_size_zipf_preregistration.md AVANT ce script
# (derivation du modele, grille de corpus, criteres de verdict, calcul de
# puissance). Resume :
#   - profondeur fixe L=6 (ancree par growth_constraint_test_results.json,
#     4 graines, 20000 pas, en dehors de l'anomalie L=4-5 documentee dans
#     article3.tex ligne 646-658)
#   - grille de fractions du corpus TinyShakespeare : 1/32,1/16,1/8,1/4,1/2,1
#   - 4 graines x 20000 pas par point (meme convention que la campagne de
#     profondeur "porteuse de signal", PAS le budget 4000-pas biaise)
#   - sous-ensemble = portion contigue tiree aleatoirement (par graine) plutot
#     qu'un prefixe fixe, pour ne pas confondre "taille du corpus" et "quelle
#     piece de Shakespeare se trouve au debut du fichier"
#   - ensemble de validation COMMUN a tous les runs (10% final, jamais sous-
#     echantillonne)
#   - fit primaire : a(D) = A0 + A1*ln(D), robuste, ne necessite ni a_inf ni
#     K_max (miroir exact de a_1=-c sur l'axe profondeur)
#   - fit secondaire exploratoire : + A2*(ln D)^2, epsilon_hat = 2*A2/A1_hat ;
#     pre-enregistre comme probablement sous-puissant (meme argument que la
#     courbure de profondeur, deja infructueuse a 36 runs)
#   - incertitude : bootstrap en grappes de graines (PAS pooling naif -- le
#     pooling sous-estime l'incertitude reelle jusqu'a 4x, cf.
#     growth_constraint_fit_results.json vs growth_joint_fit_results.json)
# ══════════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "growth_jalon0.jl"))

using JSON, Statistics, LinearAlgebra, Printf, Random

# Surcharges optionnelles (smoke test uniquement -- non utilisees par la
# campagne pre-enregistree si les variables d'environnement ne sont pas
# definies, auquel cas les valeurs par defaut PRE-ENREGISTREES ci-dessous
# s'appliquent).
const FIXED_DEPTH = 6
const N_STEPS     = parse(Int, get(ENV, "CORPUSZIPF_NSTEPS", "20000"))
const VAL_EVERY    = parse(Int, get(ENV, "CORPUSZIPF_VALEVERY", "250"))
const N0           = 3000.0
const SEEDS        = 1:parse(Int, get(ENV, "CORPUSZIPF_NSEEDS", "4"))
const FRACTIONS    = haskey(ENV, "CORPUSZIPF_FRACTIONS") ?
    [eval(Meta.parse(s)) for s in split(ENV["CORPUSZIPF_FRACTIONS"], ",")] :
    [1//32, 1//16, 1//8, 1//4, 1//2, 1//1]

const N_TRAIN_FULL = length(train_ids)

# ── Sous-ensemble de corpus : portion contigue aleatoire, deterministe par
#    (graine, fraction) pour permettre la reprise apres interruption. ────────
function corpus_subset(seed::Int, target_len::Int)
    target_len >= N_TRAIN_FULL && return train_ids
    rng_off = MersenneTwister(hash((:corpus_offset, seed, target_len)))
    max_start = N_TRAIN_FULL - target_len + 1
    start = rand(rng_off, 1:max_start)
    return train_ids[start:start+target_len-1]
end

# ── Entrainement a profondeur fixe sur un sous-ensemble de corpus donne ──────
function train_fixed_depth!(n_layers::Int, train_subset::Vector{Int};
                             seed::Int, n_steps::Int, val_every::Int=VAL_EVERY,
                             lr=1f-3, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM)
    ns = Symbol(:corpuszipf_L, n_layers, :_s, seed, :_n, length(train_subset))
    Random.seed!(seed)
    g, logits_sym, cur_out = build_char_lm_graph(dev, ns; n_layers=n_layers, dim=dim,
                                                  n_heads=n_heads, hidden_dim=hidden_dim)
    ps = NeuroDSL.params(g; namespace=ns)
    m1 = Dict{Symbol,Any}(nd.name => NeuroDSL.Backend.zeros32(dev, size(nd.value)...) for nd in ps)
    m2 = Dict{Symbol,Any}(nd.name => NeuroDSL.Backend.zeros32(dev, size(nd.value)...) for nd in ps)
    rng = MersenneTwister(seed)
    val_history = Tuple{Int,Float64}[]
    t_start = time()
    for t in 1:n_steps
        tokens, labels = sample_window(rng, train_subset, BLOCK_SIZE)
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:BLOCK_SIZE); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        loss_val = NeuroDSL.demand!(g, :loss; namespace=ns)
        NeuroDSL.backward_graph!(g, :loss; namespace=ns)
        m1_flat = [m1[nd.name] for nd in ps]; m2_flat = [m2[nd.name] for nd in ps]
        NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps], [p.gradient for p in ps],
                                     m1_flat, m2_flat, Float32(lr), 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        if t % val_every == 0
            vl = val_loss(g, ns, BLOCK_SIZE)
            push!(val_history, (t, vl))
        end
    end
    elapsed = time() - t_start
    return (; n_layers, seed, n_train_chars=length(train_subset), n_steps, elapsed,
            val_history, final_val=val_history[end][2])
end

# ── Campagne : 6 fractions x 4 graines, reprise apres interruption ──────────
results_path = joinpath(@__DIR__, "bench_corpus_size_zipf_check_raw.json")
all_results = isfile(results_path) ? JSON.parsefile(results_path) : Dict{String,Any}()

t0 = time()
for frac in FRACTIONS
    target_len = round(Int, N_TRAIN_FULL * frac)
    for seed in SEEDS
        key = "f$(frac.num)_$(frac.den)_n$(target_len)_seed$(seed)"
        haskey(all_results, key) && continue
        println("\n", "="^60, "\n>>> ", key, " (L=$FIXED_DEPTH, corpus=$target_len chars)\n", "="^60)
        t_run = time()
        sub = corpus_subset(seed, target_len)
        res = train_fixed_depth!(FIXED_DEPTH, sub; seed=seed, n_steps=N_STEPS, val_every=VAL_EVERY)
        dt = time() - t_run
        @printf "%s : val finale=%.4f, temps=%.1fs\n" key res.final_val dt
        all_results[key] = Dict("fraction_num"=>frac.num, "fraction_den"=>frac.den,
                                 "n_train_chars"=>target_len, "seed"=>seed, "depth"=>FIXED_DEPTH,
                                 "n_steps"=>res.n_steps, "final_val"=>res.final_val,
                                 "elapsed_s"=>dt, "val_history"=>res.val_history)
        open(results_path, "w") do io
            JSON.print(io, all_results)
        end
    end
end
total_elapsed = time() - t0
@printf "\n\nTemps total (cette invocation) : %.1f min (%.2f h)\n" (total_elapsed/60) (total_elapsed/3600)
println("Runs bruts ecrits -> ", results_path)

# ══════════════════════════════════════════════════════════════════════════════
# Analyse : fit a(D), bootstrap en grappes de graines, verdict pre-enregistre.
# Ne s'execute que si tous les runs attendus sont presents dans le cache.
# ══════════════════════════════════════════════════════════════════════════════
expected_keys = String[]
for frac in FRACTIONS
    target_len = round(Int, N_TRAIN_FULL * frac)
    for seed in SEEDS
        push!(expected_keys, "f$(frac.num)_$(frac.den)_n$(target_len)_seed$(seed)")
    end
end
missing_keys = [k for k in expected_keys if !haskey(all_results, k)]
if !isempty(missing_keys)
    println("\n", length(missing_keys), " runs manquants -- analyse reportee. Relancer ce script pour reprendre.")
    exit(0)
end

println("\n", "="^70)
println("ANALYSE : campagne complete (", length(expected_keys), " runs). Fit + bootstrap.")
println("="^70)

function fit_ab_pooled(frac, target_len)
    xs = Float64[]; ys = Float64[]
    for s in SEEDS
        key = "f$(frac.num)_$(frac.den)_n$(target_len)_seed$(s)"
        vh = all_results[key]["val_history"]
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
    return coef[1], coef[2], se_a, n
end

# ── AMENDEMENT 2026-08-20 (voir corpus_size_zipf_preregistration.md, §5bis) ──
# f=1/32 (4/4 graines) surapprend severement : la perte de val touche un
# minimum vers le pas 2500-3500 (2.44-2.54 nats) puis DIVERGE jusqu'a 6.2-7.0
# nats au pas 20000 -- le fit a(D)+b(D)/(n+n0), concu pour une courbe
# monotone-decroissante vers un plancher, n'a plus de sens ici. Remplace par
# un estimateur "minimum de la courbe lissee" (moyenne mobile 5 points, puis
# argmin), robuste que la courbe soit monotone (grand D, aucun changement) ou
# en U (petit D, recupere le plancher pre-divergence au lieu d'extrapoler du
# bruit). Budget de pas, graines, fractions : INCHANGES (comparabilite avec
# le pre-enregistrement) -- seule la methode d'EXTRACTION change, decidee ici
# a partir d'un seul point de corpus, avant tout fit sur la grille complete.
function smoothed_min(vh; window::Int=5)
    ys = [Float64(pt[2]) for pt in vh]
    n = length(ys)
    half = window ÷ 2
    smoothed = similar(ys)
    for i in 1:n
        lo = max(1, i-half); hi = min(n, i+half)
        smoothed[i] = mean(ys[lo:hi])
    end
    return minimum(smoothed)
end

function smoothed_min_pooled(frac, target_len, seeds_subset)
    vals = Float64[]
    for s in seeds_subset
        key = "f$(frac.num)_$(frac.den)_n$(target_len)_seed$(s)"
        push!(vals, smoothed_min(all_results[key]["val_history"]))
    end
    return mean(vals)
end

println("\n--- Etape 1a : a(D) via extrapolation a+b/(n+n0) (methode ORIGINALE, sanity check grand D uniquement) ---")
lnDs = Float64[]; avals_extrap = Float64[]; bvals = Float64[]; ases = Float64[]
target_lens = Int[]
for frac in FRACTIONS
    target_len = round(Int, N_TRAIN_FULL * frac)
    push!(target_lens, target_len)
    a, b, se_a, n = fit_ab_pooled(frac, target_len)
    push!(lnDs, log(target_len)); push!(avals_extrap, a); push!(bvals, b); push!(ases, se_a)
    @printf "D=%-8d (f=%s) : a_extrap=%.4f ± %.4f (naive)  b=%.1f  n_points=%d\n" target_len string(frac) a se_a b n
end

println("\n--- Etape 1b : a(D) via minimum de courbe lissee (methode PRIMAIRE, amendement) ---")
avals = Float64[]
for (fi, frac) in enumerate(FRACTIONS)
    a = smoothed_min_pooled(frac, target_lens[fi], collect(SEEDS))
    push!(avals, a)
    agree = abs(a - avals_extrap[fi])
    @printf "D=%-8d (f=%s) : a_min_lisse=%.4f   (vs a_extrap=%.4f, ecart=%.4f)\n" target_lens[fi] string(frac) a avals_extrap[fi] agree
end

# ── Bootstrap en grappes de graines : SE honnete par point (methode primaire) ─
Random.seed!(20260820)
const NBOOT = 1000
boot_avals = zeros(NBOOT, length(FRACTIONS))
for bi in 1:NBOOT
    for (fi, frac) in enumerate(FRACTIONS)
        target_len = target_lens[fi]
        boot_seeds = rand(SEEDS, 4)
        boot_avals[bi, fi] = smoothed_min_pooled(frac, target_len, boot_seeds)
    end
end
boot_se = [std(boot_avals[:, fi]) for fi in eachindex(FRACTIONS)]
println("\n--- SE honnete par point (bootstrap en grappes de graines, 1000 tirages, methode primaire) ---")
for (fi, frac) in enumerate(FRACTIONS)
    @printf "D=%-8d : a_min_lisse=%.4f  SE_bootstrap=%.4f\n" target_lens[fi] avals[fi] boot_se[fi]
end

# ── Etape 2 : fit primaire (log-lineaire), WLS pondere par 1/SE_bootstrap^2 ──
function linfit(lnD, a, w)
    X = hcat(ones(length(lnD)), lnD)
    W = Diagonal(w)
    coef = (X'*W*X) \ (X'*W*a)
    resid = a .- X*coef
    return coef, resid
end
function quadfit(lnD, a, w)
    X = hcat(ones(length(lnD)), lnD, lnD.^2)
    W = Diagonal(w)
    coef = (X'*W*X) \ (X'*W*a)
    resid = a .- X*coef
    return coef, resid
end
function aic_w(resid, w, k)
    n = length(resid)
    wrss = sum(w .* resid.^2)
    logL = -0.5*n*log(wrss/n)
    return 2*k - 2*logL
end

w = 1.0 ./ (boot_se.^2)
coef_lin, resid_lin = linfit(lnDs, avals, w)
coef_quad, resid_quad = quadfit(lnDs, avals, w)
A0_lin, A1_lin = coef_lin
c_D_hat = -A1_lin
A0_q, A1_q, A2_q = coef_quad

println("\n", "="^70)
println("Etape 2 : fit primaire a(D) = A0 + A1*ln(D)")
println("="^70)
@printf "A0=%.4f  A1=%.4f   =>   c_D_hat = -A1 = %.4f\n" A0_lin A1_lin c_D_hat
println("Comparaison a c (axe profondeur) : 0.104 (fit 16 profondeurs), 0.133±0.012 (fit joint), 0.083 (campagne contrainte 9 profondeurs) -- bande [0.08,0.13]")

# Bootstrap de c_D_hat (grappes de graines, refit complet)
c_D_boot = Float64[]
for bi in 1:NBOOT
    avals_b = Float64[]
    for (fi, frac) in enumerate(FRACTIONS)
        boot_seeds = rand(SEEDS, 4)
        push!(avals_b, smoothed_min_pooled(frac, target_lens[fi], boot_seeds))
    end
    se_b = [std(boot_avals[:, fi]) for fi in eachindex(FRACTIONS)]  # reutilise SE deja estimee
    w_b = 1.0 ./ (se_b.^2)
    try
        cb, _ = linfit(lnDs, avals_b, w_b)
        push!(c_D_boot, -cb[2])
    catch
        continue
    end
end
cD_lo, cD_hi = quantile(c_D_boot, [0.025, 0.975])
@printf "IC95%% bootstrap de c_D_hat : [%.4f, %.4f]\n" cD_lo cD_hi

band_lo, band_hi = 0.08, 0.13
wide_lo, wide_hi = 0.04, 0.20
primary_consistent = (cD_lo <= band_hi && cD_hi >= band_lo)
primary_contradicts = !(cD_lo <= wide_hi && cD_hi >= wide_lo)
println("\nVerdict PRIMAIRE (c_D vs c, universalite inter-axes) :")
if primary_consistent
    println("  -> COHERENT : IC95% de c_D_hat recoupe la bande [0.08,0.13].")
elseif primary_contradicts
    println("  -> CONTREDIT : IC95% de c_D_hat exclut meme la bande elargie [0.04,0.20].")
else
    println("  -> INCONCLUSIF : ni dans la bande cible, ni clairement hors de la bande elargie.")
end

# ── Etape 3 : fit secondaire (quadratique), courbure -> epsilon ─────────────
println("\n", "="^70)
println("Etape 3 : fit secondaire a(D) = A0 + A1*ln(D) + A2*(ln D)^2  (exploratoire)")
println("="^70)
@printf "A0=%.4f  A1=%.4f  A2=%.4f\n" A0_q A1_q A2_q
eps_hat = c_D_hat != 0 ? 2*A2_q / (-A1_lin) : NaN
@printf "epsilon_hat = 2*A2/(-A1_lineaire) = %.4f\n" eps_hat

aic_lin = aic_w(resid_lin, w, 2)
aic_quad = aic_w(resid_quad, w, 3)
dAIC = aic_quad - aic_lin
@printf "AIC (lineaire, 2 params) = %.3f\n" aic_lin
@printf "AIC (quadratique, 3 params) = %.3f\n" aic_quad
@printf "ΔAIC (quad - lineaire) = %.3f  (favorable si < -6)\n" dAIC

println("\n--- Leave-one-out (retrait d'un point de corpus) ---")
loo_pass = true
for i in eachindex(FRACTIONS)
    idx = setdiff(1:length(FRACTIONS), i)
    lnD_loo, a_loo, w_loo = lnDs[idx], avals[idx], w[idx]
    cl, rl = linfit(lnD_loo, a_loo, w_loo)
    cq, rq = quadfit(lnD_loo, a_loo, w_loo)
    al = aic_w(rl, w_loo, 2); aq = aic_w(rq, w_loo, 3)
    ok = (aq - al) < -6
    global loo_pass &= ok
    @printf "sans D=%-8d : ΔAIC=%.3f  %s\n" target_lens[i] (aq-al) (ok ? "✓" : "✗ ECHEC")
end

println("\n--- Bootstrap en grappes (1000 tirages) : selection, signe(A2), epsilon_hat ---")
Random.seed!(31415)
quad_selected_count = 0
sign_ok_count = 0
eps_boot = Float64[]
for bi in 1:NBOOT
    avals_b = Float64[]
    for (fi, frac) in enumerate(FRACTIONS)
        boot_seeds = rand(SEEDS, 4)
        push!(avals_b, smoothed_min_pooled(frac, target_lens[fi], boot_seeds))
    end
    se_b = boot_se
    w_b = 1.0 ./ (se_b.^2)
    try
        cl_b, rl_b = linfit(lnDs, avals_b, w_b)
        cq_b, rq_b = quadfit(lnDs, avals_b, w_b)
        dAIC_b = aic_w(rq_b, w_b, 3) - aic_w(rl_b, w_b, 2)
        if dAIC_b < -6
            global quad_selected_count += 1
        end
        if cq_b[3] > 0
            global sign_ok_count += 1
        end
        push!(eps_boot, 2*cq_b[3] / (-cl_b[2]))
    catch
        continue
    end
end
frac_quad_selected = quad_selected_count / NBOOT
frac_sign_ok = sign_ok_count / length(eps_boot)
eps_lo, eps_hi = quantile(eps_boot, [0.025, 0.975])
@printf "Fraction de tirages selectionnant le modele quadratique (ΔAIC<-6) : %.1f%%  (requis >=90%%)\n" (100*frac_quad_selected)
@printf "Fraction de tirages avec signe(A2)>0                              : %.1f%%  (requis >=95%%)\n" (100*frac_sign_ok)
@printf "IC95%% bootstrap de epsilon_hat                                   : [%.4f, %.4f]\n" eps_lo eps_hi
println("Bande de coherence attendue : [0, 0.15] (autour du point faible ~0.076 de l'axe profondeur)")

crit_s1 = (dAIC < -6) && loo_pass && (frac_quad_selected >= 0.90)
crit_s2 = frac_sign_ok >= 0.95
crit_s3 = eps_lo <= 0.15 && eps_hi >= 0.0
println("\nVerdict SECONDAIRE (epsilon via courbure) :")
println("  Critere S1 (selection robuste du modele quadratique) : ", crit_s1 ? "PASSE" : "ECHOUE")
println("  Critere S2 (signe(A2)>0 stable >=95%)                 : ", crit_s2 ? "PASSE" : "ECHOUE")
println("  Critere S3 (IC95% epsilon recoupe [0,0.15])           : ", crit_s3 ? "PASSE" : "ECHOUE")
if crit_s1 && crit_s2 && crit_s3
    println("  -> CONFIRME : les trois criteres passent.")
elseif crit_s1 && !crit_s3
    println("  -> REFUTATION PROPRE : courbure robuste mais epsilon_hat hors bande attendue.")
else
    println("  -> INCONCLUSIF : modele quadratique pas assez robuste pour interpreter epsilon_hat.")
    println("     (epsilon_hat nominal = ", round(eps_hat, digits=4), " -- NE PAS CITER comme mesure fiable)")
end

# ── Sauvegarde ────────────────────────────────────────────────────────────
open(joinpath(@__DIR__, "bench_corpus_size_zipf_check_results.json"), "w") do io
    JSON.print(io, Dict(
        "depth"=>FIXED_DEPTH, "target_lens"=>target_lens, "lnDs"=>lnDs,
        "avals"=>avals, "avals_extrap_sanity_check"=>avals_extrap,
        "bvals"=>bvals, "ases_naive"=>ases, "boot_se"=>boot_se,
        "coef_lin"=>coef_lin, "coef_quad"=>coef_quad, "c_D_hat"=>c_D_hat,
        "c_D_boot_ci"=>[cD_lo, cD_hi], "eps_hat_nominal"=>eps_hat,
        "eps_boot_ci"=>[eps_lo, eps_hi],
        "aic_lin"=>aic_lin, "aic_quad"=>aic_quad, "dAIC"=>dAIC, "loo_pass"=>loo_pass,
        "frac_quad_selected"=>frac_quad_selected, "frac_sign_ok"=>frac_sign_ok,
        "primary_consistent"=>primary_consistent, "primary_contradicts"=>primary_contradicts,
        "crit_s1"=>crit_s1, "crit_s2"=>crit_s2, "crit_s3"=>crit_s3,
    ))
end
println("\nEcrit -> notebook/bench_corpus_size_zipf_check_results.json")
