# Jalon 1 -- run complet : balayage de la granularité de croissance
# n_stages ∈ {1,2,4,8,16}, profondeur finale 16, budget layer-steps identique
# (48 000, réduit vs jalon 0 par discipline de coût -- jalon 1 est exploratoire,
# 2 graines seulement, pas un test d'hypothèse strict comme jalon 0).

include(joinpath(@__DIR__, "growth_jalon1.jl"))
using Statistics

const BUDGET1 = 48_000
const SEEDS1 = [1, 2]
const STAGE_COUNTS = [1, 2, 4, 8, 16]

all_results = Dict{String,Any}()
t0 = time()
for n_stages in STAGE_COUNTS
    sched = schedule_for_stages(n_stages)
    for seed in SEEDS1
        key = "stages$(n_stages)_seed$(seed)"
        println("\n" * "="^70)
        println(">>> ", key, "  (schedule=", sched, ", budget=", BUDGET1, ")")
        println("="^70)
        t_run = time()
        res = train_growth_arm!(sched, BUDGET1; seed=seed, val_every=500)
        dt = time() - t_run
        @printf "%s : %d pas, val finale = %.4f, temps = %.1fs\n" key res.n_steps_total res.final_val dt
        all_results[key] = Dict(
            "n_stages" => n_stages, "schedule" => sched, "seed" => seed,
            "n_steps_total" => res.n_steps_total, "final_val" => res.final_val,
            "elapsed_s" => dt, "growth_events" => res.growth_events,
            "val_history" => res.val_history,
        )
        open(joinpath(@__DIR__, "growth_jalon1_results.json"), "w") do io
            JSON.print(io, all_results)
        end
    end
end
total_elapsed = time() - t0

println("\n\n", "="^70)
println("RÉSULTAT -- Jalon 1 (carte de densité de croissance)")
println("="^70)
@printf "Temps total : %.1f min\n\n" (total_elapsed/60)

for n_stages in STAGE_COUNTS
    vals = [all_results["stages$(n_stages)_seed$(s)"]["final_val"] for s in SEEDS1]
    @printf "n_stages=%-3d : val finale par graine = %s | moyenne=%.4f\n" n_stages string(round.(vals,digits=4)) mean(vals)
end
