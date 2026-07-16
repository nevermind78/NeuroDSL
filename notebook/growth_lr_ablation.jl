# ══════════════════════════════════════════════════════════════════════════════
# Ablation LR-schedule (objection Fable/G_stack-lineage) : le renversement à
# budget FLOPs égal (croissance bat fixed-depth-16, cf. table 18-schedules,
# campagne "coarsest wins") tient-il si le baseline fixed-16 utilise une VRAIE
# décroissance de learning rate (cosinus) au lieu du lr constant utilisé
# partout jusqu'ici ? Toutes les mesures growth_jalon1(_fixed) sont à lr=1e-3
# CONSTANT (confirmé par lecture directe de train_growth_arm!, aucun schedule
# nul part) -- c'est exactement l'objection qu'un relecteur de la lignée
# G_stack/LiGO poserait en premier.
#
# Protocole (le moins cher qui répond directement à la question) : reproduire
# UNIQUEMENT le baseline fixed-16 (schedule=[16]), même budget nominal que la
# campagne "coarsest wins" (48 000 layer-steps -> 3000 pas), 3 graines, avec
# une décroissance cosinus lr_max=1e-3 -> lr_min=1e-4 sur toute la durée du
# run -- pas de warm-up (les runs existants démarrent déjà depuis l'init
# aléatoire sans warm-up, donc ce n'est pas un facteur de confusion introduit
# ici). Comparé aux valeurs déjà mesurées, lr constant :
#   - fixed-16 (3k steps, lr constant) : 2.0562 (equal-share) / 2.0723 (1/L)
#   - meilleur schedule de croissance   : 1.7191 (equal-share) / 1.6542 (1/L)
#
# Critère pré-enregistré (avant tout calcul) :
#   - renversement SURVIT si fixed-16 (lr cosinus) reste > 1.75 nats/char
#     (marge confortable au-dessus du meilleur schedule de croissance).
#   - renversement EN RISQUE si la valeur tombe dans [1.70, 1.75].
#   - renversement INFIRMÉ si la valeur tombe <= 1.70 (rejoint/bat la
#     croissance) -- le résultat serait rapporté tel quel, sans forcer une
#     conclusion positive.
# ══════════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "growth_jalon0.jl"))

const BUDGET = 48_000   # layer-steps, identique à la campagne "coarsest wins"
const SEEDS = [1, 2, 3]
const LR_MAX, LR_MIN = 1f-3, 1f-4

cosine_lr(t, T) = LR_MIN + 0.5f0*(LR_MAX-LR_MIN)*(1f0+cos(Float32(pi)*min(t,T)/T))

println("="^60, "\n>>> Ablation LR cosinus -- fixed-depth-16, budget=", BUDGET, "\n", "="^60)

results = NamedTuple[]
t0 = time()
for seed in SEEDS
    t_run = time()
    res = train_growth_arm!([16], BUDGET; seed=seed, val_every=250, lr_schedule=cosine_lr)
    dt = time() - t_run
    @printf "seed=%d : %d pas, val finale=%.4f, temps=%.1fs\n" seed res.n_steps_total res.final_val dt
    push!(results, (; seed, final_val=res.final_val, val_history=res.val_history, elapsed=dt))
end
total_elapsed = time() - t0

vals = [r.final_val for r in results]
mean_val, std_val = mean(vals), std(vals)
@printf "\nMoyenne sur %d graines : %.4f ± %.4f nats/char\n" length(vals) mean_val std_val
@printf "Temps total : %.1f min\n" (total_elapsed/60)

verdict = mean_val > 1.75 ? "SURVIT" : (mean_val > 1.70 ? "EN RISQUE" : "INFIRMÉ")
println("\nComparaison :")
println("  fixed-16, lr constant (déjà mesuré)  : 2.0562 (equal-share) / 2.0723 (1/L-weighted)")
println("  fixed-16, lr cosinus (cette ablation) : ", round(mean_val, digits=4), " ± ", round(std_val, digits=4))
println("  meilleur schedule de croissance       : 1.7191 (equal-share) / 1.6542 (1/L-weighted)")
println("\nVerdict pré-enregistré : renversement ", verdict)

open(joinpath(@__DIR__, "growth_lr_ablation_results.json"), "w") do io
    JSON.print(io, Dict("seeds"=>SEEDS, "vals"=>vals, "mean"=>mean_val, "std"=>std_val,
                         "budget"=>BUDGET, "lr_max"=>LR_MAX, "lr_min"=>LR_MIN,
                         "verdict"=>verdict,
                         "val_histories"=>[r.val_history for r in results]))
end
println("Résultats écrits -> notebook/growth_lr_ablation_results.json")
