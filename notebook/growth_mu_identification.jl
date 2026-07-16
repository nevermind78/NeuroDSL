# ══════════════════════════════════════════════════════════════════════════════
# Mesure de μ (quanta appris par couche, modèle de quantification, §growthexact
# d'article3.tex) : l'analyse de non-identifiabilité montre que TOUTE
# l'information sur μ vient des profondeurs L=1,2,4 (le terme de correction
# a2/L+a3/L^2 s'écrase au bruit dès que L grandit), et que le bootstrap par
# graine seule (pas par profondeur) donne un IC à 95% sur [1e-3, 1e4] --
# 7 ordres de grandeur, inexploitable. Plutôt que plus de profondeurs ou des
# runs plus longs (qui n'apportent rien sur μ), on concentre l'effort : 10
# graines à L=2 et L=3 (là où μL~1 maximise le terme de correction, μ̂≈0.7
# du fit joint précédent), pour réduire le bruit de graine d'un facteur ~√10
# exactement là où le signal existe.
# ══════════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "growth_jalon0.jl"))

const DEPTHS = [2, 3]
const SEEDS = 1:10
const TARGET_STEPS = 4000

results_path = joinpath(@__DIR__, "growth_mu_identification_results.json")
all_results = isfile(results_path) ? JSON.parsefile(results_path) : Dict{String,Any}()

t0 = time()
for L in DEPTHS
    for seed in SEEDS
        key = "L$(L)_seed$(seed)"
        haskey(all_results, key) && continue   # reprise après interruption
        println("\n", "="^60, "\n>>> ", key, "\n", "="^60)
        t_run = time()
        res = train_growth_arm!([L], TARGET_STEPS * L; seed=seed, val_every=500)
        dt = time() - t_run
        @printf "%s : %d pas, val finale=%.4f, temps=%.1fs\n" key res.n_steps_total res.final_val dt
        all_results[key] = Dict("depth"=>L, "seed"=>seed, "n_steps_total"=>res.n_steps_total,
                                 "final_val"=>res.final_val, "elapsed_s"=>dt,
                                 "val_history"=>res.val_history)
        open(results_path, "w") do io
            JSON.print(io, all_results)
        end
    end
end

total_elapsed = time() - t0
@printf "\n\nTemps total (cette invocation) : %.1f min\n" (total_elapsed/60)

for L in DEPTHS
    vals = [all_results["L$(L)_seed$(s)"]["final_val"] for s in SEEDS]
    @printf "L=%d : moyenne=%.4f  std=%.4f  (n=%d graines)\n" L mean(vals) std(vals) length(vals)
end
println("Résultats écrits -> notebook/growth_mu_identification_results.json")
