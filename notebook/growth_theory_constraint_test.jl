# ══════════════════════════════════════════════════════════════════════════════
# Protocole conçu par Fable pour rendre testable la contrainte parameter-free
# a1*a3/a2^2 = -1/3 (§growthexact d'article3.tex). Décision centrale : PAS
# d'extension à L=32/64 (tout le signal sur a2/a3 vit aux PETITES profondeurs
# -- à L=16 leur contribution est déjà ~1e-5 ; aller plus profond ne raffine
# que a1, déjà robuste, pour un coût 2-3x plus élevé). À la place : runs plus
# LONGS (pas plus de graines à budget court -- leçon de l'ablation μ) et
# densément échantillonnés aux profondeurs 1-6, pour casser (a) la
# dégénérescence a(L)/b(L) sur fenêtre étroite et (b) la colinéarité de la
# base polynomiale {1, lnL, 1/L, 1/L^2} sur peu de profondeurs distinctes.
#
# 20 000 pas aux profondeurs porteuses de signal (1-6) : le biais de
# troncature mesuré à 4000 pas (+0.092 à L=2) est ~3x le signal a2/a3 recherché
# (~-0.033) et de signe opposé -- exactement le piège qui a probablement
# fabriqué le faux signal du premier test (ΔAIC=-10.1 non robuste). 20 000 pas
# dépasse les runs déjà jugés fiables dans le README (13 333 pas pour a(2),
# 20 000 pour a(4)).
#
# 10 000 pas aux profondeurs d'ancrage (8,12,16) : ancrent a1 (pente) et
# vérifient l'absence de saturation du log -- contribution a2/a3 quasi nulle
# à ces profondeurs, donc convergence exacte moins critique là.
#
# val_every=250 -- courbe complète par run (~80 points), pour un vrai fit
# a(L)+b(L)/(n+n0) par profondeur, pas juste la perte finale.
# ══════════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "growth_jalon0.jl"))

# (depth, n_steps_per_depth)
const DEPTH_STEPS = [
    (1, 20_000), (2, 20_000), (3, 20_000), (4, 20_000), (5, 20_000), (6, 20_000),
    (8, 10_000), (12, 10_000), (16, 10_000),
]
const SEEDS = 1:4

# Fichier séparé dédié à cette campagne (ne pas mélanger avec les runs
# existants qui utilisent d'autres conventions de budget/graines).
results_path = joinpath(@__DIR__, "growth_constraint_test_results.json")
all_results = isfile(results_path) ? JSON.parsefile(results_path) : Dict{String,Any}()

t0 = time()
for (L, n_steps) in DEPTH_STEPS
    for seed in SEEDS
        key = "L$(L)_n$(n_steps)_seed$(seed)"
        haskey(all_results, key) && continue   # reprise après interruption
        println("\n", "="^60, "\n>>> ", key, "\n", "="^60)
        t_run = time()
        res = train_growth_arm!([L], n_steps * L; seed=seed, val_every=250)
        dt = time() - t_run
        @printf "%s : %d pas, val finale=%.4f, temps=%.1fs\n" key res.n_steps_total res.final_val dt
        all_results[key] = Dict("depth"=>L, "n_steps_target"=>n_steps, "seed"=>seed,
                                 "n_steps_total"=>res.n_steps_total, "final_val"=>res.final_val,
                                 "elapsed_s"=>dt, "val_history"=>res.val_history)
        open(results_path, "w") do io
            JSON.print(io, all_results)
        end
    end
end

total_elapsed = time() - t0
@printf "\n\nTemps total (cette invocation) : %.1f min (%.2f h)\n" (total_elapsed/60) (total_elapsed/3600)
println("Résultats écrits -> notebook/growth_constraint_test_results.json")
