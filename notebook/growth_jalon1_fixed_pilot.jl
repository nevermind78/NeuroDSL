# Pilote du correctif (pondération 1/L au lieu de parts égales) -- vérifie
# que le budget par palier grossit bien vers les profondeurs bon marché, et
# que le mécanisme (surtout n_stages=16) tient toujours.

include(joinpath(@__DIR__, "growth_jalon1.jl"))

const B_PILOT = 16 * 50

for n_stages in [2, 16]
    sched = schedule_for_stages(n_stages)
    println("\n=== Pilote (1/L) n_stages=$n_stages -> schedule=$sched ===")
    res = train_growth_arm!(sched, B_PILOT; seed=1, val_every=25, weight_fn=(L -> 1.0/L))
    @printf "n_stages=%d : %d pas, val finale = %.4f, temps = %.1fs, %d événements\n" n_stages res.n_steps_total res.final_val res.elapsed length(res.growth_events)
    println("  Pas par palier (déduit des événements de croissance) :")
    prev_step = 0
    for ev in res.growth_events
        println("    -> jusqu'à ", ev[1] - prev_step, " pas avant la greffe vers ", ev[3], " couches")
        prev_step = ev[1]
    end
    println("    -> ", res.n_steps_total - prev_step, " pas au dernier palier")
end
