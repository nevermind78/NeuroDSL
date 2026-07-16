# Jalon 1b -- run complet : comparaison à NOMBRE DE PAS ÉGAL (4000 pas pour
# chaque densité), complémentaire de jalon 1 (FLOPs égal). Isole si la forme
# du calendrier compte indépendamment du nombre de mises à jour de gradient.

include(joinpath(@__DIR__, "growth_jalon1b_equal_steps.jl"))

const TARGET_STEPS = 4000
const SEEDS1 = [1, 2]
const STAGE_COUNTS = [1, 2, 4, 8, 16]

all_results = Dict{String,Any}()
t0 = time()
for n_stages in STAGE_COUNTS
    sched = schedule_for_stages(n_stages)
    B = budget_for_equal_steps(sched, TARGET_STEPS)
    for seed in SEEDS1
        key = "stages$(n_stages)_seed$(seed)"
        println("\n" * "="^70)
        println(">>> ", key, "  (schedule=", sched, ", B=", B, ", cible=", TARGET_STEPS, " pas égaux/palier)")
        println("="^70)
        t_run = time()
        res = train_growth_arm!(sched, B; seed=seed, val_every=250, weight_fn=(L -> Float64(L)))
        dt = time() - t_run
        @printf "%s : %d pas, val finale = %.4f, temps = %.1fs\n" key res.n_steps_total res.final_val dt
        all_results[key] = Dict(
            "n_stages" => n_stages, "schedule" => sched, "seed" => seed, "budget" => B,
            "n_steps_total" => res.n_steps_total, "final_val" => res.final_val,
            "elapsed_s" => dt, "growth_events" => res.growth_events,
            "val_history" => res.val_history,
        )
        open(joinpath(@__DIR__, "growth_jalon1b_results.json"), "w") do io
            JSON.print(io, all_results)
        end
    end
end
total_elapsed = time() - t0

println("\n\n", "="^70)
println("RÉSULTAT -- Jalon 1b (nombre de pas égal = ", TARGET_STEPS, ")")
println("="^70)
@printf "Temps total : %.1f min\n\n" (total_elapsed/60)

for n_stages in STAGE_COUNTS
    vals = [all_results["stages$(n_stages)_seed$(s)"]["final_val"] for s in SEEDS1]
    @printf "n_stages=%-3d : val finale = %s | moyenne=%.4f\n" n_stages string(round.(vals,digits=4)) mean(vals)
end
