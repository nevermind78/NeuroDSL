# Figures pour le résultat jalon 0 (piste "économie de la croissance",
# 2026-07-13) -- réutilise notebook/growth_jalon0_results.json (déjà écrit par
# growth_jalon0_full.jl, run corrigé après le bug de seeding), aucune nouvelle
# mesure. Deux panneaux : (a) courbes de perte de validation vs. budget de
# calcul consommé (proxy layer-steps, l'axe qui rend les 3 bras comparables
# malgré des nombres de pas différents), moyenne sur 3 graines par bras,
# événements de croissance marqués ; (b) résumé final par graine (le
# "zéro chevauchement" du tableau, rendu visuellement).

using Plots, JSON, Statistics
gr()
default(titlefontsize=13, guidefontsize=12, tickfontsize=10, legendfontsize=9)

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

d = JSON.parsefile(joinpath(@__DIR__, "growth_jalon0_results.json"))

const ARMS = [("A_fixed_4", [4], :steelblue, "A -- fixed 4 layers"),
              ("B_2_3_4",   [2,3,4], :seagreen, "B -- growth 2→3→4"),
              ("C_1_2_4",   [1,2,4], :darkorange, "C -- growth 1→2→4")]
const SEEDS = [1, 2, 3]

"""Reconstruit le budget layer-steps cumulé à chaque point de val_history, à
partir des growth_events (step, layer_steps_au_moment_de_la_greffe, nouvelle_L)."""
function cumulative_layer_steps(val_history, growth_events, schedule)
    xs = Float64[]
    for entry in val_history
        step, _, _ = entry
        # trouve le dernier événement de croissance <= step
        base_step, base_ls, cur_L = 0, 0, schedule[1]
        for ev in growth_events
            ev_step, ev_ls, ev_L = ev
            if ev_step <= step
                base_step, base_ls, cur_L = ev_step, ev_ls, ev_L
            end
        end
        push!(xs, base_ls + (step - base_step) * cur_L)
    end
    return xs
end

# ── Panneau (a) : courbes val loss vs. budget consommé, moyenne sur 3 graines ──
p1 = plot(xlabel="Compute budget consumed (layer-steps)", ylabel="Validation loss (nats/char)",
          title="(a) Val loss vs. compute budget (mean of 3 seeds)",
          legend=:topright, left_margin=12Plots.mm, bottom_margin=10Plots.mm,
          top_margin=6Plots.mm, right_margin=4Plots.mm)

for (arm_key, schedule, color, label) in ARMS
    runs = [d["$(arm_key)_seed$(s)"] for s in SEEDS]
    xs_ref = cumulative_layer_steps(runs[1]["val_history"], runs[1]["growth_events"], schedule)
    ys_mat = hcat([[e[2] for e in r["val_history"]] for r in runs]...)
    ys_mean = vec(mean(ys_mat, dims=2))
    plot!(p1, xs_ref, ys_mean, label=label, color=color, lw=2.5)
    for ev in runs[1]["growth_events"]
        vline!(p1, [ev[2]], color=color, alpha=0.35, linestyle=:dot, label="")
    end
end
hline!(p1, [log(65)], label="Random init level (ln 65)", color=:gray, linestyle=:dash, lw=1)

# ── Panneau (b) : résumé final par graine (le "zéro chevauchement") ─────────
p2 = plot(xlabel="Arm", ylabel="Final validation loss (nats/char)",
          title="(b) Final val loss: 3 seeds per arm, equal FLOPs budget",
          legend=:topright, xlims=(0.5, 3.5), xticks=(1:3, ["A (fixed)", "B (2→3→4)", "C (1→2→4)"]),
          left_margin=12Plots.mm, bottom_margin=10Plots.mm, top_margin=6Plots.mm, right_margin=4Plots.mm)

for (i, (arm_key, schedule, color, label)) in enumerate(ARMS)
    vals = [d["$(arm_key)_seed$(s)"]["final_val"] for s in SEEDS]
    scatter!(p2, fill(i, length(vals)) .+ (rand(length(vals)) .- 0.5) .* 0.12, vals,
             color=color, markersize=7, label="", markerstrokewidth=0.5)
    m, s = mean(vals), std(vals)
    plot!(p2, [i-0.25, i+0.25], [m, m], color=color, lw=3, label="")
    plot!(p2, [i, i], [m-s, m+s], color=color, lw=1.5, label="")
end
scatter!(p2, [NaN], [NaN], color=:black, markersize=7, label="individual seed")
plot!(p2, [NaN], [NaN], color=:black, lw=3, label="mean ± std")

combined = plot(p1, p2, layout=(1, 2), size=(1500, 620))
savefig(combined, joinpath(figdir, "growth_jalon0_en.pdf"))
println("Saved -> figures/growth_jalon0_en.pdf")
