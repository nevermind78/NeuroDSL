# ══════════════════════════════════════════════════════════════════════════════
# Les deux expériences discriminantes sur l'universalité de c, annoncées mais
# jamais lancées dans article3.tex §growthexact ("Both are discriminating
# experiments we leave to future work, not yet run"). Prédictions du modèle de
# quantification (Michaud et al. 2023), à tester directement plutôt qu'à
# laisser en promesse :
#
#   (1) Balayage de LARGEUR (dim=128, dim=512 vs. dim=256 déjà mesuré) :
#       si c est universel (indépendant de la largeur), la pente -c du fit
#       log a(L) doit rester INCHANGÉE, et l'ordonnée doit se décaler de
#       exactement ∓c*ln(4) ≈ ∓0.144 nat (dim=128 : +0.144 ; dim=512 : -0.144)
#       -- puisque le nombre de paramètres par bloc est proportionnel à dim²
#       (4*dim²+3*dim*hidden+2*dim, hidden=2*dim ici), et ln(N(L)) ne diffère
#       de ln(L) que par une constante additive à largeur fixe.
#   (2) Balayage du NOMBRE DE TÊTES à dim=256 fixe (n_heads=2, n_heads=8) :
#       Q/K/V/O restent dim×dim quel que soit le nombre de têtes qui les
#       partage (src/layers.jl), donc le nombre de paramètres par bloc est
#       EXACTEMENT invariant au nombre de têtes -- si c dépend uniquement de
#       la capacité (comme le prédit le modèle de quantification), a(L) doit
#       rester ENTIÈREMENT inchangé, indépendamment de n_heads.
#
# Protocole : mêmes profondeurs {1,2,4,8,16} et même convention de mesure
# (single-run 4000 pas, comme la majorité du fit à 16 profondeurs de
# §growthfit) que le reste de cette ligne de travail, mais avec 2 graines par
# point plutôt qu'une seule -- cette expérience teste une prédiction nouvelle,
# pas une simple ré-confirmation, donc mérite un peu plus de robustesse que
# l'économie de graines déjà actée pour le fit original.
# ══════════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "growth_jalon0.jl"))

const DEPTHS = [1, 2, 4, 8, 16]
const SEEDS = [1, 2]
const TARGET_STEPS = 4000

# (nom, dim, n_heads, hidden_dim)
const CONFIGS = [
    ("dim128", 128, 4, 256),   # 4x moins cher/bloc que dim=256 -- prédit +0.144 nat d'ordonnée
    ("dim512", 512, 4, 1024),  # 4x plus cher/bloc -- prédit -0.144 nat d'ordonnée
    ("heads2", 256, 2, 512),   # paramètres/bloc inchangés -- prédit a(L) INCHANGÉ
    ("heads8", 256, 8, 512),   # paramètres/bloc inchangés -- prédit a(L) INCHANGÉ
    # Contrôle ajouté après coup : le fit dim=256 déjà publié (16 profondeurs,
    # L=16 à 8000 pas, biais de troncature corrigé) n'est pas sur la même
    # convention que les 4 configs ci-dessus (5 profondeurs, 4000 pas partout,
    # aucune correction) -- ce contrôle donne un dim=256 strictement comparable
    # (même profondeurs, même budget, même absence de correction).
    ("dim256", 256, 4, 512),
]

results_path = joinpath(@__DIR__, "growth_discriminating_results.json")
all_results = isfile(results_path) ? JSON.parsefile(results_path) : Dict{String,Any}()

t0 = time()
for (cfg_name, dim, n_heads, hidden_dim) in CONFIGS
    for L in DEPTHS
        for seed in SEEDS
            key = "$(cfg_name)_L$(L)_seed$(seed)"
            haskey(all_results, key) && continue   # reprise après interruption
            println("\n", "="^60, "\n>>> ", key, " (dim=$dim, n_heads=$n_heads, hidden=$hidden_dim)\n", "="^60)
            t_run = time()
            res = train_growth_arm!([L], TARGET_STEPS * L; seed=seed, val_every=500,
                                     dim=dim, n_heads=n_heads, hidden_dim=hidden_dim)
            dt = time() - t_run
            @printf "%s : %d pas, val finale=%.4f, temps=%.1fs\n" key res.n_steps_total res.final_val dt
            all_results[key] = Dict("config"=>cfg_name, "dim"=>dim, "n_heads"=>n_heads,
                                     "hidden_dim"=>hidden_dim, "depth"=>L, "seed"=>seed,
                                     "n_steps_total"=>res.n_steps_total, "final_val"=>res.final_val,
                                     "elapsed_s"=>dt)
            open(results_path, "w") do io
                JSON.print(io, all_results)
            end
        end
    end
end

total_elapsed = time() - t0
@printf "\n\nTemps total (cette invocation) : %.1f min\n" (total_elapsed/60)
println("Résultats écrits -> notebook/growth_discriminating_results.json")
