# Figure pour le résultat jalon 1 CORRIGÉ (pondération 1/L, 2026-07-13) --
# réutilise notebook/growth_jalon1_fixed_results.json, aucune nouvelle mesure.
# Deux panneaux : (a) perte de validation vs. budget de calcul consommé
# (layer-steps), une courbe par densité (moyenne sur 2 graines) ; (b) perte
# finale vs. nombre d'étapes, avec le nombre total de pas en axe secondaire --
# rend visuellement le point central : le calendrier le plus grossier (2
# étapes) gagne, corrélé au nombre de pas totaux, SAUF l'inversion 16 vs 8.

using Plots, JSON, Statistics
gr()
default(titlefontsize=13, guidefontsize=12, tickfontsize=10, legendfontsize=9)

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

d = JSON.parsefile(joinpath(@__DIR__, "growth_jalon1_fixed_results.json"))
const STAGE_COUNTS = [1, 2, 4, 8, 16]
const SEEDS1 = [1, 2]
const COLORS = Dict(1=>:gray, 2=>:seagreen, 4=>:steelblue, 8=>:darkorange, 16=>:purple)

function cumulative_layer_steps(val_history, growth_events)
    xs = Float64[]
    for entry in val_history
        step, _, cur_L = entry
        base_step, base_ls, L = 0, 0, val_history[1][3]
        for ev in growth_events
            ev_step, ev_ls, ev_L = ev
            if ev_step <= step
                base_step, base_ls, L = ev_step, ev_ls, ev_L
            end
        end
        push!(xs, base_ls + (step - base_step) * L)
    end
    return xs
end

# ── Panneau (a) : val loss vs. budget consommé, une courbe par densité ──────
p1 = plot(xlabel="Compute budget consumed (layer-steps)", ylabel="Validation loss (nats/char)",
          title="(a) Val loss vs. compute budget (1/L-weighted, mean of 2 seeds)",
          legend=:topright, left_margin=12Plots.mm, bottom_margin=10Plots.mm,
          top_margin=6Plots.mm, right_margin=4Plots.mm)

for n in STAGE_COUNTS
    runs = [d["stages$(n)_seed$(s)"] for s in SEEDS1]
    xs_ref = cumulative_layer_steps(runs[1]["val_history"], runs[1]["growth_events"])
    ys_mat = hcat([[e[2] for e in r["val_history"]] for r in runs]...)
    ys_mean = vec(mean(ys_mat, dims=2))
    plot!(p1, xs_ref, ys_mean, label="$n stage(s)", color=COLORS[n], lw=2.5)
end

# ── Panneau (b) : perte finale vs. nb étapes (barres) + pas totaux (ligne) ──
final_means = [mean([d["stages$(n)_seed$(s)"]["final_val"] for s in SEEDS1]) for n in STAGE_COUNTS]
total_steps = [d["stages$(n)_seed1"]["n_steps_total"] for n in STAGE_COUNTS]

p2 = plot(xlabel="Number of growth stages", ylabel="Final validation loss (nats/char)",
          title="(b) Final loss and total steps vs. schedule density",
          legend=:topright, xticks=(1:5, string.(STAGE_COUNTS)),
          left_margin=12Plots.mm, bottom_margin=10Plots.mm, top_margin=6Plots.mm, right_margin=16Plots.mm)
bar!(p2, 1:5, final_means, color=[COLORS[n] for n in STAGE_COUNTS], alpha=0.75, label="final val loss (left axis)", bar_width=0.55)

p2b = twinx(p2)
plot!(p2b, 1:5, total_steps, color=:black, lw=2.5, marker=:circle, markersize=6,
      ylabel="Total training steps", label="total steps (right axis)", legend=:topleft)

combined = plot(p1, p2, layout=(1, 2), size=(1500, 620))
savefig(combined, joinpath(figdir, "growth_jalon1_fixed_en.pdf"))
println("Saved -> figures/growth_jalon1_fixed_en.pdf")
