# Pilote minuscule -- valide surtout le cas le plus risqué : n_stages=16
# (15 greffes d'affilée, une couche à la fois, jusqu'à 16 couches).

include(joinpath(@__DIR__, "growth_jalon1.jl"))

const B_PILOT = 16 * 50   # 50 pas-équivalents à profondeur finale -- minuscule

for n_stages in [1, 4, 16]
    sched = schedule_for_stages(n_stages)
    println("\n=== Pilote n_stages=$n_stages -> schedule=$sched ===")
    res = train_growth_arm!(sched, B_PILOT; seed=1, val_every=25)
    @printf "n_stages=%d : %d pas, val finale = %.4f, temps = %.1fs, %d événements de croissance\n" n_stages res.n_steps_total res.final_val res.elapsed length(res.growth_events)
end
println("\nSi tout ça tourne sans erreur, le mécanisme tient jusqu'à 15 greffes consécutives.")
