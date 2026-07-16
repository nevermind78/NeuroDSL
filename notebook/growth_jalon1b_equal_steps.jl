# ══════════════════════════════════════════════════════════════════════════════
# Axe complémentaire à jalon 1 (reporté à cette session, cf.
# project_neurodsl_growth_schedules_pitch_2026-07-13.md) : comparaison à
# NOMBRE DE PAS TOTAL ÉGAL (pas à budget FLOPs égal) entre les 5 densités de
# calendrier -- isole si la FORME du calendrier compte indépendamment du
# nombre de mises à jour de gradient, ce que jalon 1 (FLOPs égal) ne peut pas
# trancher puisque le nombre de pas y varie justement avec la densité.
#
# Astuce de réutilisation : train_growth_arm!(schedule, budget; weight_fn)
# donne des pas par palier = budget_palier ÷ L. Avec weight_fn = L -> L, le
# budget par palier ∝ L, donc budget_palier ÷ L = constante -- PAS ÉGAL PAR
# PALIER, quelle que soit sa profondeur. Il suffit de choisir le budget total
# par calendrier pour obtenir le nombre de pas total T souhaité :
#   B(schedule) = T * sum(schedule) / length(schedule)
# ══════════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "growth_jalon1.jl"))
using Statistics

"""Budget total (en layer-steps) à passer à train_growth_arm!(...; weight_fn=L->L)
pour obtenir exactement `total_steps` pas au total, répartis à parts égales
entre les paliers de `schedule`."""
budget_for_equal_steps(schedule, total_steps) =
    round(Int, total_steps * sum(schedule) / length(schedule))

for n in [1, 2, 4, 8, 16]
    sched = schedule_for_stages(n)
    B = budget_for_equal_steps(sched, 2000)
    println("n_stages=$n -> schedule=$sched, B=$B (cible 2000 pas)")
end
