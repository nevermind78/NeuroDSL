# ══════════════════════════════════════════════════════════════════════════════
# Jalon 1 (piste "économie de la croissance", suite de jalon 0 -- PASSÉ le
# 2026-07-13, mémoire project_neurodsl_growth_schedules_pitch_2026-07-13.md) :
# balayage de la GRANULARITÉ du calendrier de croissance, à budget layer-steps
# strictement égal entre toutes les configurations.
#
# Convention explicite (pour éviter toute ambiguïté) : on balaie le nombre
# D'ÉTAPES (longueur du schedule), pas le nombre d'événements de croissance
# au sens strict (= étapes - 1). n_stages=1 => schedule=[16] (aucune
# croissance, référence figée à la profondeur finale). n_stages=16 =>
# schedule=[1,2,...,16] (croissance couche par couche, la plus fine possible,
# 15 événements de croissance).
#
# Profondeur finale = 16 (pas 4 comme jalon 0) : nécessaire pour avoir assez
# de profondeurs entières distinctes pour tester jusqu'à 16 étapes.
# ══════════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "growth_jalon0.jl"))
using Statistics

const L_START = 1
const L_FINAL = 16

"""Schedule à `n_stages` paliers, profondeurs entières régulièrement espacées
entre L_START et L_FINAL inclus (n_stages=1 => juste [L_FINAL])."""
function schedule_for_stages(n_stages::Int)
    n_stages == 1 && return [L_FINAL]
    return unique(round.(Int, range(L_START, L_FINAL, length=n_stages)))
end

for n in [1, 2, 4, 8, 16]
    println("n_stages=$n -> schedule=", schedule_for_stages(n))
end
