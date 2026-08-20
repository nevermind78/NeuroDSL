# ══════════════════════════════════════════════════════════════════════════════
# Test PRÉ-ENREGISTRÉ (voir notebook/quantization_signature_preregistration.md,
# écrit AVANT ce script) de la piste laissée ouverte par article3.tex (ligne
# ~639) après l'échec structurel du test de la contrainte a1*a3/a2^2=-1/3
# (notebook/growth_constraint_test_results.json, notebook/analyze_growth_
# constraint.jl) : une distribution de perte PAR POSITION, à un seul
# checkpoint fixe (un seul forward pass, aucune extrapolation temporelle --
# donc aucune des sources d'erreur qui ont tué le test -1/3), peut-elle
# montrer une signature de "quanta" discrets (modèle de quantification de
# Michaud et al.) plutôt qu'une variation continue de difficulté ?
#
# src/kernels.jl N'EST PAS MODIFIÉ. cross_entropy_loss (ligne 760) renvoie la
# MOYENNE sur les positions ; la formule softmax/-log est reproduite ICI,
# SANS la réduction finale, pour obtenir une perte par position.
#
# Aucun checkpoint entraîné n'existe dans le dépôt (vérifié : growth_
# constraint_test_results.json ne stocke que des val_history scalaires, pas
# de poids) -- les 6 runs ci-dessous (3 profondeurs x 2 graines) sont
# entraînés depuis zéro, avec EXACTEMENT le protocole de growth_theory_
# constraint_test.jl / growth_jalon0.jl (mêmes corpus, archi, hyperparamètres,
# convention de pas) pour rester comparables aux campagnes existantes.
#
# Aucune dépendance ajoutée à Project.toml : le test de multimodalité (GMM
# 1 vs 2 composantes, comparaison BIC) est implémenté ici à la main (EM,
# vectorisé) avec seulement Statistics/LinearAlgebra/Random, déjà déclarés.
# ══════════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "growth_jalon0.jl"))
using Statistics, Random, LinearAlgebra, JSON, Printf

# ── Config pré-enregistrée ────────────────────────────────────────────────────
const CONFIGS = [(1, 20_000), (4, 20_000), (16, 10_000)]   # (L, n_steps)
const SEEDS_QSIG = 1:2
const RNG_ANALYSIS_SEED = 20260820

# Référence connue (growth_constraint_test_results.json, seed1, mêmes L/n_steps)
# -- pur contrôle de cohérence, n'entre dans aucun critère de verdict.
const REFERENCE_FINAL_VAL = Dict(1=>1.7346, 4=>1.6867, 16=>1.7194)

# ── Entraînement à profondeur fixe, retournant le graphe (pas de sauvegarde
#    checkpoint : l'évaluation par position se fait dans le MÊME processus,
#    sur le MÊME graphe, immédiatement après entraînement) ───────────────────
function train_fixed_depth!(L::Int, n_steps::Int; seed::Int, lr=1f-3, val_every=1000,
                             dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM)
    ns = Symbol(:qsig_L, L, :_seed, seed)
    Random.seed!(seed)
    g, logits_sym, cur_out = build_char_lm_graph(dev, ns; n_layers=L, dim=dim,
                                                  n_heads=n_heads, hidden_dim=hidden_dim)
    m1 = Dict{Symbol,Any}(); m2 = Dict{Symbol,Any}()
    for nd in NeuroDSL.params(g; namespace=ns)
        m1[nd.name] = NeuroDSL.Backend.zeros32(dev, size(nd.value)...)
        m2[nd.name] = NeuroDSL.Backend.zeros32(dev, size(nd.value)...)
    end
    rng = MersenneTwister(seed)
    ps = NeuroDSL.params(g; namespace=ns)
    val_history = Tuple{Int,Float64}[]
    t0 = time()
    for t in 1:n_steps
        tokens, labels = sample_window(rng, train_ids, BLOCK_SIZE)
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:BLOCK_SIZE); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.demand!(g, :loss; namespace=ns)
        NeuroDSL.backward_graph!(g, :loss; namespace=ns)
        m1_flat = [m1[nd.name] for nd in ps]; m2_flat = [m2[nd.name] for nd in ps]
        NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps], [p.gradient for p in ps],
                                     m1_flat, m2_flat, lr, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        if t % val_every == 0
            vl = val_loss(g, ns, BLOCK_SIZE)
            push!(val_history, (t, vl))
        end
    end
    elapsed = time() - t0
    return (; g, ns, logits_sym, val_history, elapsed, n_steps, L, seed)
end

# ── Fenêtres NON chevauchantes sur le split de validation (une seule mesure
#    par caractère tenu à l'écart, pas de re-mesure au même caractère à
#    différentes longueurs de contexte) ────────────────────────────────────
function val_windows_nonoverlapping(ids, block_size)
    n_windows = length(ids) ÷ block_size
    return [1 + (w - 1) * block_size for w in 1:n_windows]
end

# ── Perte PAR POSITION -- reproduit EXACTEMENT la formule de
#    src/kernels.jl:760 (softmax stable + clip 1f-10 + -log), SANS le -mean
#    final. src/kernels.jl n'est pas touché. ─────────────────────────────────
function per_position_losses(logits_cpu::AbstractMatrix{Float32}, labels::AbstractVector{Int})
    n = size(logits_cpu, 1)
    losses = Vector{Float32}(undef, n)
    @inbounds for i in 1:n
        row = @view logits_cpu[i, :]
        m = maximum(row)
        e = exp.(row .- m)
        p = e ./ sum(e)
        losses[i] = -log(max(p[labels[i]], 1f-10))
    end
    return losses
end

function collect_checkpoint_losses(g, ns, logits_sym, block_size, ids)
    starts = val_windows_nonoverlapping(ids, block_size)
    all_losses = Float32[]; all_targets = Int[]; all_posidx = Int[]
    for i in starts
        tokens, labels = ids[i:i+block_size-1], ids[i+1:i+block_size]
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        logits_val = NeuroDSL.demand_release!(g, logits_sym; namespace=ns)
        logits_cpu = Array{Float32}(Array(logits_val))
        append!(all_losses, per_position_losses(logits_cpu, labels))
        append!(all_targets, labels)
        append!(all_posidx, collect(1:block_size))
    end
    return all_losses, all_targets, all_posidx
end

# ── GMM 1D à 2 composantes par EM (vectorisé), à la main -- aucune dépendance
#    ajoutée. Restarts aléatoires + plancher de variance anti-dégénérescence. ─
function gmm_em_2comp(x::Vector{Float64}; n_restarts=12, max_iter=400, tol=1e-7,
                       rng=MersenneTwister(0))
    n = length(x)
    vtot = var(x)
    var_floor = max(1e-4 * vtot, 1e-10)
    best = nothing
    for _ in 1:n_restarts
        qs = sort(0.15 .+ 0.7 .* rand(rng, 2))
        mus = Float64[quantile(x, qs[1]), quantile(x, qs[2])]
        if abs(mus[1] - mus[2]) < 1e-6
            mus[2] += 1e-3
        end
        sigma2s = fill(vtot / 2, 2)
        weights = [0.5, 0.5]
        prev_ll = -Inf
        for _iter in 1:max_iter
            lp1 = @. -0.5 * log(2π * sigma2s[1]) - (x - mus[1])^2 / (2 * sigma2s[1])
            lp2 = @. -0.5 * log(2π * sigma2s[2]) - (x - mus[2])^2 / (2 * sigma2s[2])
            l1 = lp1 .+ log(weights[1] + 1e-300)
            l2 = lp2 .+ log(weights[2] + 1e-300)
            m = max.(l1, l2)
            s = exp.(l1 .- m) .+ exp.(l2 .- m)
            ll = sum(m .+ log.(s))
            r1 = exp.(l1 .- m) ./ s
            N1 = max(sum(r1), 1e-8); N2 = max(n - N1, 1e-8)
            r2 = 1.0 .- r1
            weights = [N1 / n, N2 / n]
            mus = [sum(r1 .* x) / N1, sum(r2 .* x) / N2]
            sigma2s = [max(sum(r1 .* (x .- mus[1]) .^ 2) / N1, var_floor),
                       max(sum(r2 .* (x .- mus[2]) .^ 2) / N2, var_floor)]
            if abs(ll - prev_ll) < tol * (abs(prev_ll) + 1e-10)
                prev_ll = ll
                break
            end
            prev_ll = ll
        end
        if best === nothing || prev_ll > best.ll
            best = (weights=weights, mus=mus, sigma2s=sigma2s, ll=prev_ll)
        end
    end
    return best
end

gmm_1comp_ll(x::Vector{Float64}) = begin
    mu = mean(x); s2 = max(var(x), 1e-10); n = length(x)
    sum(@. -0.5 * log(2π * s2) - (x - mu)^2 / (2 * s2))
end

# ── Test de multimodalité pré-enregistré (3 critères stricts, §4 de la
#    pré-enregistration) ──────────────────────────────────────────────────────
function multimodality_test(x::Vector{Float64}; rng)
    n = length(x)
    ll1 = gmm_1comp_ll(x)
    fit2 = gmm_em_2comp(x; rng=rng)
    bic1 = -2 * ll1 + 2 * log(n)
    bic2 = -2 * fit2.ll + 5 * log(n)
    dbic = bic2 - bic1
    mu1, mu2 = fit2.mus
    s1, s2 = sqrt.(fit2.sigma2s)
    D = abs(mu2 - mu1) / sqrt((s1^2 + s2^2) / 2)
    w1, w2 = fit2.weights
    nondeg = (0.05 <= w1 <= 0.95) && (0.05 <= w2 <= 0.95)
    fires = (dbic < -10) && (D > 2) && nondeg
    return (; n, ll1, ll2=fit2.ll, bic1, bic2, dbic, mu1, mu2, s1, s2, w1, w2, D, nondeg, fires, fit2)
end

# ── Régression de contrôle : fréquence du caractère cible (unigramme, calculé
#    sur train_ids UNIQUEMENT) + log(position dans la fenêtre) -- §5 de la
#    pré-enregistration ───────────────────────────────────────────────────────
function unigram_neglogfreq(train_ids, vocab_size)
    counts = zeros(Float64, vocab_size)
    for c in train_ids
        counts[c] += 1
    end
    total = sum(counts)
    freq = counts ./ total
    return Dict(c => -log(max(freq[c], 1e-12)) for c in 1:vocab_size)
end

function residualize(losses::Vector{Float64}, targets::Vector{Int}, posidx::Vector{Int},
                      neglogfreq::Dict{Int,Float64})
    n = length(losses)
    f = [neglogfreq[c] for c in targets]
    lp = log.(Float64.(posidx))
    X = hcat(ones(n), f, lp)
    coef = X \ losses
    resid = losses .- X * coef
    return resid, coef
end

# ── Jaccard entre ensembles de positions "haute perte" (composante GMM de
#    moyenne la plus haute, responsabilité > 0.5) -- §6 de la pré-registration ─
function highloss_set(x::Vector{Float64}, fit2)
    mus = fit2.mus; sigma2s = fit2.sigma2s; weights = fit2.weights
    hi = argmax(mus)
    lp_hi = @. -0.5 * log(2π * sigma2s[hi]) - (x - mus[hi])^2 / (2 * sigma2s[hi]) + log(weights[hi] + 1e-300)
    lo = hi == 1 ? 2 : 1
    lp_lo = @. -0.5 * log(2π * sigma2s[lo]) - (x - mus[lo])^2 / (2 * sigma2s[lo]) + log(weights[lo] + 1e-300)
    return Set(findall(lp_hi .> lp_lo)), weights[hi]
end

jaccard(a::Set, b::Set) = length(a) == 0 && length(b) == 0 ? 1.0 : length(intersect(a, b)) / length(union(a, b))

# ══════════════════════════════════════════════════════════════════════════════
# Exécution
# ══════════════════════════════════════════════════════════════════════════════

results = Dict{String,Any}()
checkpoint_losses = Dict{Tuple{Int,Int},Vector{Float64}}()   # (L,seed) -> losses
checkpoint_targets = Dict{Tuple{Int,Int},Vector{Int}}()
checkpoint_fits = Dict{Tuple{Int,Int},Any}()

neglogfreq = unigram_neglogfreq(train_ids, vocab_size)

t_total0 = time()
for (L, n_steps) in CONFIGS
    for seed in SEEDS_QSIG
        key = "L$(L)_seed$(seed)"
        println("\n", "="^70, "\n>>> Entraînement ", key, " (", n_steps, " pas)\n", "="^70)
        t0 = time()
        run = train_fixed_depth!(L, n_steps; seed=seed, val_every=max(1000, n_steps ÷ 10))
        dt_train = time() - t0
        sanity_val = val_loss(run.g, run.ns, BLOCK_SIZE)
        ref = get(REFERENCE_FINAL_VAL, L, NaN)
        @printf "  entraînement : %.1fs   val_loss(64 fenêtres) = %.4f   (référence campagne, seed1 = %.4f)\n" dt_train sanity_val ref

        t1 = time()
        losses_f32, targets, posidx = collect_checkpoint_losses(run.g, run.ns, run.logits_sym,
                                                                  BLOCK_SIZE, val_ids)
        dt_eval = time() - t1
        losses = Float64.(losses_f32)
        n_pos = length(losses)
        @printf "  extraction perte par position : %.1fs, %d positions (mean=%.4f, std=%.4f)\n" dt_eval n_pos mean(losses) std(losses)

        checkpoint_losses[(L, seed)] = losses
        checkpoint_targets[(L, seed)] = targets

        rng_a = MersenneTwister(RNG_ANALYSIS_SEED + L * 100 + seed)
        raw_test = multimodality_test(losses; rng=rng_a)
        resid, coef = residualize(losses, targets, posidx, neglogfreq)
        rng_b = MersenneTwister(RNG_ANALYSIS_SEED + L * 100 + seed + 50000)
        resid_test = multimodality_test(resid; rng=rng_b)

        checkpoint_fits[(L, seed)] = raw_test.fit2

        @printf "  [RAW]      ΔBIC=%.2f  D=%.3f  w=(%.3f,%.3f)  mu=(%.4f,%.4f)  fires=%s\n" raw_test.dbic raw_test.D raw_test.w1 raw_test.w2 raw_test.mu1 raw_test.mu2 raw_test.fires
        @printf "  [RESIDUAL] ΔBIC=%.2f  D=%.3f  w=(%.3f,%.3f)  mu=(%.4f,%.4f)  fires=%s   (coef: intercept=%.4f, freq=%.4f, log_pos=%.4f)\n" resid_test.dbic resid_test.D resid_test.w1 resid_test.w2 resid_test.mu1 resid_test.mu2 resid_test.fires coef[1] coef[2] coef[3]

        results[key] = Dict(
            "L"=>L, "seed"=>seed, "n_steps"=>n_steps, "n_positions"=>n_pos,
            "train_elapsed_s"=>dt_train, "eval_elapsed_s"=>dt_eval,
            "sanity_val_loss_64win"=>sanity_val, "reference_final_val_seed1"=>ref,
            "mean_loss"=>mean(losses), "std_loss"=>std(losses),
            "val_history"=>run.val_history,
            "raw_test"=>Dict("dbic"=>raw_test.dbic, "D"=>raw_test.D, "w1"=>raw_test.w1, "w2"=>raw_test.w2,
                              "mu1"=>raw_test.mu1, "mu2"=>raw_test.mu2, "s1"=>raw_test.s1, "s2"=>raw_test.s2,
                              "bic1"=>raw_test.bic1, "bic2"=>raw_test.bic2, "fires"=>raw_test.fires),
            "residual_test"=>Dict("dbic"=>resid_test.dbic, "D"=>resid_test.D, "w1"=>resid_test.w1, "w2"=>resid_test.w2,
                                   "mu1"=>resid_test.mu1, "mu2"=>resid_test.mu2, "fires"=>resid_test.fires,
                                   "coef"=>coef),
        )

        # Libérer la mémoire GPU du graphe avant la profondeur suivante.
        run = nothing
        GC.gc()
    end
end
total_elapsed = time() - t_total0
@printf "\nEntraînement + extraction total : %.1f min\n" (total_elapsed/60)

# ── §6 : comparaison inter-profondeurs (graduation) ──────────────────────────
println("\n", "="^70, "\nÉtape secondaire : comparaison inter-profondeurs (graduation, §6)\n", "="^70)
graduation = Dict{String,Any}()
for seed in SEEDS_QSIG
    sets = Dict{Int,Tuple{Set{Int},Float64}}()
    for (L, _) in CONFIGS
        losses = checkpoint_losses[(L, seed)]
        fit2 = checkpoint_fits[(L, seed)]
        hs, w_hi = highloss_set(losses, fit2)
        sets[L] = (hs, w_hi)
        @printf "  seed %d, L=%-2d : poids composante haute-perte = %.4f, |ensemble| = %d\n" seed L w_hi length(hs)
    end
    Ls = [c[1] for c in CONFIGS]
    for i in 1:length(Ls), j in (i+1):length(Ls)
        La, Lb = Ls[i], Ls[j]
        setA, wA = sets[La]; setB, wB = sets[Lb]
        jac = jaccard(setA, setB)
        jac_null = (wA * wB) / (wA + wB - wA * wB)
        @printf "  seed %d, L=%d vs L=%d : Jaccard=%.4f   Jaccard_null(indépendance)=%.4f   %s\n" seed La Lb jac jac_null (jac > jac_null ? "> baseline" : "<= baseline")
        graduation["seed$(seed)_L$(La)_L$(Lb)"] = Dict("jaccard"=>jac, "jaccard_null"=>jac_null,
                                                         "w_hi_a"=>wA, "w_hi_b"=>wB)
    end
    Ls_sorted = sort(Ls)
    weights_by_depth = [sets[L][2] for L in Ls_sorted]
    monotone = issorted(weights_by_depth; rev=true)
    @printf "  seed %d : poids haute-perte %s  (profondeurs %s, ordre croissant)  -> monotone décroissant=%s\n" seed string(round.(weights_by_depth, digits=4)) string(Ls_sorted) monotone
    graduation["seed$(seed)_monotone_weight_decrease"] = monotone
    graduation["seed$(seed)_shallow_deep_pair"] = "L$(Ls_sorted[1])_L$(Ls_sorted[end])"
end
const SHALLOW_L = minimum(c[1] for c in CONFIGS)
const DEEP_L = maximum(c[1] for c in CONFIGS)

# ── Verdict pré-enregistré (§7) ───────────────────────────────────────────────
println("\n", "="^70, "\nVERDICT PRÉ-ENREGISTRÉ (§7 de quantization_signature_preregistration.md)\n", "="^70)

# on exige : le test brut "tire" à AU MOINS 2 des profondeurs testées (au moins une graine)
fires_any_seed(L, testkey) = any(seed -> results["L$(L)_seed$(seed)"][testkey]["fires"], SEEDS_QSIG)
depths_firing = [L for (L,_) in CONFIGS if fires_any_seed(L, "raw_test")]
n_depths_firing = length(depths_firing)
n_depths_total = length(CONFIGS)

survives_control = all(L -> fires_any_seed(L, "residual_test"), depths_firing)

monotone_both_seeds = all(seed -> graduation["seed$(seed)_monotone_weight_decrease"], SEEDS_QSIG)
grad_overlap_both = all(seed -> begin
    k = "seed$(seed)_L$(SHALLOW_L)_L$(DEEP_L)"
    graduation[k]["jaccard"] > graduation[k]["jaccard_null"]
end, SEEDS_QSIG)

@printf "Profondeurs où le test brut (RAW) tire : %s (%d/%d)\n" string(depths_firing) n_depths_firing n_depths_total
@printf "Survit au contrôle fréquence+position (RESIDUAL tire aux mêmes profondeurs) : %s\n" survives_control
@printf "Poids haute-perte décroissant avec la profondeur, dans toutes les graines : %s\n" monotone_both_seeds
@printf "Recouvrement de graduation (L=%d vs L=%d) au-dessus du hasard, dans toutes les graines : %s\n" SHALLOW_L DEEP_L grad_overlap_both

verdict = if n_depths_firing == 0
    "REFUTATION : aucune structure multimodale détectée à AUCUNE des 3 profondeurs sur la perte brute."
elseif n_depths_firing >= 2 && survives_control && monotone_both_seeds && grad_overlap_both
    "SUPPORT : signature multimodale robuste, survit au contrôle fréquence+position, et la composante haute-perte décroît/se réduit de façon cohérente avec la profondeur (graduation)."
else
    "INCONCLUSIF : " * (n_depths_firing < 2 ? "signature multimodale détectée à moins de 2 profondeurs. " : "") *
    (n_depths_firing >= 2 && !survives_control ? "la signature brute ne survit PAS au contrôle fréquence+position (confondue avec la difficulté ordinaire). " : "") *
    (n_depths_firing >= 2 && survives_control && !(monotone_both_seeds && grad_overlap_both) ? "la signature survit au contrôle mais l'histoire d'acquisition (monotonie/graduation) n'est pas corroborée dans les 2 graines. " : "")
end
println("\n>>> ", verdict)

results["graduation"] = graduation
results["verdict"] = verdict
results["depths_firing_raw"] = depths_firing
results["survives_control"] = survives_control
results["monotone_both_seeds"] = monotone_both_seeds
results["grad_overlap_both"] = grad_overlap_both
results["total_elapsed_min"] = total_elapsed / 60

open(joinpath(@__DIR__, "bench_quantization_signature_check_results.json"), "w") do io
    JSON.print(io, results)
end
println("\nRésultats détaillés -> notebook/bench_quantization_signature_check_results.json")
println("(voir aussi notebook/bench_quantization_signature_check_results.txt pour le log complet)")
