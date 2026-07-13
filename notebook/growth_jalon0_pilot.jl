# Pilote rapide (budget minuscule) -- vérifie le MÉCANISME avant tout chiffre :
# pas de crash, pas de saut de perte au moment de la croissance (identité
# exacte garantie par insert_block!, doit être visible dans train_losses), et
# les moments AdamW s'étendent correctement (aucune erreur de clé manquante).

include(joinpath(@__DIR__, "growth_jalon0.jl"))

const B_PILOT = 4 * 300   # équivalent à 300 pas à 4 couches -- minuscule, juste pour valider le mécanisme

println("\n=== Pilote : bras A (fixe, 4 couches) ===")
resA = train_growth_arm!([4], B_PILOT; seed=1, val_every=50)
@printf "Bras A : %d pas, val finale = %.4f, temps = %.1fs\n" resA.n_steps_total resA.final_val resA.elapsed

println("\n=== Pilote : bras B (croissance 2→3→4) ===")
resB = train_growth_arm!([2, 3, 4], B_PILOT; seed=1, val_every=50)
@printf "Bras B : %d pas, val finale = %.4f, temps = %.1fs\n" resB.n_steps_total resB.final_val resB.elapsed
println("Événements de croissance (pas global, layer-steps consommés, nouvelle profondeur) :")
for ev in resB.growth_events
    println("  ", ev)
end

println("\n--- Vérification de continuité au moment de la croissance (pas de saut de perte) ---")
for (step, ls, newL) in resB.growth_events
    idx_before = max(1, step - 2)
    idx_after = min(length(resB.train_losses), step + 3)
    window = resB.train_losses[idx_before:idx_after]
    @printf "  Croissance -> %d couches au pas %d : pertes autour de l'événement = %s\n" newL step string(round.(window, digits=3))
end

println("\n=== Pilote : bras C (croissance 1→2→4) ===")
resC = train_growth_arm!([1, 2, 4], B_PILOT; seed=1, val_every=50)
@printf "Bras C : %d pas, val finale = %.4f, temps = %.1fs\n" resC.n_steps_total resC.final_val resC.elapsed
println("Événements de croissance : ", resC.growth_events)

println("\n=== Résumé du pilote (budget layer-steps identique = $B_PILOT) ===")
@printf "A (fixe 4)      : %5d pas, val finale = %.4f\n" resA.n_steps_total resA.final_val
@printf "B (2→3→4)       : %5d pas, val finale = %.4f\n" resB.n_steps_total resB.final_val
@printf "C (1→2→4)       : %5d pas, val finale = %.4f\n" resC.n_steps_total resC.final_val
println("\n(Chiffres non conclusifs à ce budget minuscule -- ce pilote valide seulement le mécanisme.)")
