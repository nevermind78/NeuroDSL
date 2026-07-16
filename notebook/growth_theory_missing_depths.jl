# ══════════════════════════════════════════════════════════════════════════════
# Modélisation a(L)/b(L) -- comble les profondeurs manquantes pour ajuster une
# vraie forme fonctionnelle (pas une interpolation sur 3-4 points). Déjà connu
# depuis zéro : L=1 (nombreux runs), L=2 (peu, via B_2_3_4), L=4 (jalon0 A),
# L=16 (4000 pas seulement -- extrapolation la plus fragile signalée par
# l'agent d'ajustement). Ce script entraîne DEPUIS ZÉRO, à profondeur
# CONSTANTE (schedule=[L], un seul palier, aucune greffe), chaque profondeur
# manquante 2,3,5,6,...,15, plus un run L=16 plus long (8000 pas au lieu de
# 4000) pour réduire l'incertitude sur a(16).
# ══════════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "growth_jalon0.jl"))

const MISSING_DEPTHS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
const TARGET_STEPS = 4000
const L16_STEPS = 8000
const SEED = 1

all_results = Dict{String,Any}()
t0 = time()

for L in MISSING_DEPTHS
    key = "depth$(L)"
    println("\n", "="^60, "\n>>> ", key, " (", TARGET_STEPS, " pas depuis zéro)\n", "="^60)
    t_run = time()
    res = train_growth_arm!([L], TARGET_STEPS * L; seed=SEED, val_every=200)
    dt = time() - t_run
    @printf "%s : %d pas, val finale=%.4f, temps=%.1fs\n" key res.n_steps_total res.final_val dt
    all_results[key] = Dict("depth"=>L, "n_steps_total"=>res.n_steps_total,
                             "final_val"=>res.final_val, "elapsed_s"=>dt,
                             "val_history"=>res.val_history)
    open(joinpath(@__DIR__, "growth_theory_missing_depths_results.json"), "w") do io
        JSON.print(io, all_results)
    end
end

println("\n", "="^60, "\n>>> depth16_extended (", L16_STEPS, " pas depuis zéro)\n", "="^60)
t_run = time()
res16 = train_growth_arm!([16], L16_STEPS * 16; seed=SEED, val_every=250)
dt16 = time() - t_run
@printf "depth16_extended : %d pas, val finale=%.4f, temps=%.1fs\n" res16.n_steps_total res16.final_val dt16
all_results["depth16_extended"] = Dict("depth"=>16, "n_steps_total"=>res16.n_steps_total,
                                        "final_val"=>res16.final_val, "elapsed_s"=>dt16,
                                        "val_history"=>res16.val_history)
open(joinpath(@__DIR__, "growth_theory_missing_depths_results.json"), "w") do io
    JSON.print(io, all_results)
end

total_elapsed = time() - t0
@printf "\n\nTemps total : %.1f min\n" (total_elapsed/60)
println("Résultats écrits -> notebook/growth_theory_missing_depths_results.json")
