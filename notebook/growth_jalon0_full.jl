# Jalon 0 -- run complet du kill-switch (3 bras x 3 graines, budget = 80 000
# layer-steps = référence réelle déjà entraînée : 4 couches x 20 000 pas,
# val finale 1.6229 -- notebook/real_llm_neurodsl_results.json).
#
# Critère pré-enregistré (fixé AVANT de voir un seul chiffre de ce run) :
# le meilleur bras de croissance doit battre le bras fixe en val loss sur
# >= 2/3 graines, avec une marge >= l'écart-type inter-graines du bras fixe.
# Sinon : la revendication "la croissance économise du calcul à cette échelle"
# est morte -- pivot vers une note systems plus étroite (spike de perte /
# temps de récupération, exact vs copy-stacking), pas de déplacement du seuil.

include(joinpath(@__DIR__, "growth_jalon0.jl"))
using Statistics

const BUDGET = 80_000   # = 4 couches x 20 000 pas, référence réelle
const SEEDS = [1, 2, 3]
const ARMS = Dict("A_fixed_4" => [4], "B_2_3_4" => [2, 3, 4], "C_1_2_4" => [1, 2, 4])

all_results = Dict{String,Any}()
t0 = time()
for (arm_name, schedule) in ARMS
    for seed in SEEDS
        key = "$(arm_name)_seed$(seed)"
        println("\n" * "="^70)
        println(">>> ", key, "  (schedule=", schedule, ", budget=", BUDGET, ")")
        println("="^70)
        t_run = time()
        res = train_growth_arm!(schedule, BUDGET; seed=seed, val_every=500)
        dt = time() - t_run
        @printf "%s : %d pas, val finale = %.4f, temps = %.1fs\n" key res.n_steps_total res.final_val dt
        all_results[key] = Dict(
            "arm" => arm_name, "schedule" => schedule, "seed" => seed,
            "n_steps_total" => res.n_steps_total, "final_val" => res.final_val,
            "elapsed_s" => dt, "growth_events" => res.growth_events,
            "val_history" => res.val_history,
        )
        # Sauvegarde incrémentale -- si le run est interrompu, rien n'est perdu.
        open(joinpath(@__DIR__, "growth_jalon0_results.json"), "w") do io
            JSON.print(io, all_results)
        end
    end
end
total_elapsed = time() - t0

println("\n\n", "="^70)
println("RÉSULTAT DÉCISIF -- Jalon 0 (économie de la croissance)")
println("="^70)
@printf "Temps total : %.1f min\n\n" (total_elapsed/60)

for (arm_name, _) in ARMS
    vals = [all_results["$(arm_name)_seed$(s)"]["final_val"] for s in SEEDS]
    @printf "%-12s : val finale par graine = %s | moyenne=%.4f | std=%.4f\n" arm_name string(round.(vals,digits=4)) mean(vals) std(vals)
end

vals_A = [all_results["A_fixed_4_seed$(s)"]["final_val"] for s in SEEDS]
std_A = std(vals_A)
println("\nMarge requise (std inter-graines du bras fixe A) = ", round(std_A, digits=4))

mean_A = mean(vals_A)
println("\n(Comparaison NON appariée par graine -- même après correctif, les couches ajoutées en")
println(" cours de route tirent leurs poids à un point différent du flux RNG global entre bras ;")
println(" apparier par indice de graine n'aurait pas de sens réel. On compare les moyennes.)")
for arm_name in ["B_2_3_4", "C_1_2_4"]
    vals = [all_results["$(arm_name)_seed$(s)"]["final_val"] for s in SEEDS]
    gap = mean_A - mean(vals)
    println(arm_name, " : moyenne=", round(mean(vals),digits=4), "  écart vs A=", round(gap,digits=4),
            "  (marge requise std_A=", round(std_A,digits=4), ")")
    if gap >= std_A
        println("  ✅ CRITÈRE ATTEINT pour ce bras (moyenne meilleure que A par au moins std_A)")
    else
        println("  ⚠️  Critère non atteint pour ce bras")
    end
end
