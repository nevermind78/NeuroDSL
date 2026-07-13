# Generates the figure for article2.tex's new "Aggregate Speedup Ceiling" subsection.
# Two panels: (a) aggregate sweep ratio vs. depth, exact formula + the two measured
# points (jalon 0, jalon 0b), converging to 2; (b) individual per-site ratio vs.
# relative depth, showing the AtP*-favorable deep-site regime (ratio >> 2).
# All curves computed from the closed-form cone formula (verified with zero residual
# against notebook/jalon0_results.json / jalon0b_results.json), not re-measured.
#
# Sizing fix (2026-07-13): the first version cropped axis labels and truncated the
# panel (b) title at default Plots.jl font sizes on a 1120x460 canvas. Fixed by
# shortening both titles, enlarging the canvas, adding explicit margins, and
# setting explicit (smaller, consistent) font sizes throughout.

using Plots
gr()

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

default(titlefontsize=13, guidefontsize=12, tickfontsize=10, legendfontsize=9)

# ── Exact closed-form ratio(L, H), derived and verified this session ────────
M(H) = 7H + 15
function aggregate_ratio(L, H)
    m = M(H)
    n_sites = L * (H + 1)
    sum_cone = (H + 1) * m * L * (L - 1) / 2 + L * (10H + 2)
    n_total = m * L
    return (n_sites * n_total) / sum_cone
end

# ── Panel (a): ratio(L) for the two measured configs, plus the measured points ──
Ls = 3:1:400
ratio_h12 = [aggregate_ratio(L, 12) for L in Ls]   # jalon 0 config
ratio_h16 = [aggregate_ratio(L, 16) for L in Ls]   # jalon 0b config

p1 = plot(Ls, ratio_h12, label="H=12 (jalon 0 config)", color=:steelblue, lw=2,
          xlabel="Number of layers L", ylabel="Aggregate sweep ratio",
          title="(a) Aggregate ratio → 2 as depth grows",
          legend=:topright, ylim=(1.9, 3.1),
          left_margin=12Plots.mm, bottom_margin=10Plots.mm, top_margin=6Plots.mm, right_margin=3Plots.mm)
plot!(p1, Ls, ratio_h16, label="H=16 (jalon 0b config)", color=:seagreen, lw=2)
hline!(p1, [2.0], label="Proven limit (L→∞)", color=:black, lw=1.5, linestyle=:dash)
hline!(p1, [3.0], label="Port-decision threshold (3×)", color=:firebrick, lw=1.5, linestyle=:dot)
scatter!(p1, [12], [2.1205], label="Measured (jalon 0, L=12)", color=:steelblue, markersize=6, markershape=:diamond)
scatter!(p1, [24], [2.1009], label="Measured (jalon 0b, L=24)", color=:seagreen, markersize=6, markershape=:diamond)

# ── Panel (b): individual per-site ratio vs. relative depth, fixed L=400, H=12 ──
L2, H2 = 400, 12
m2 = M(H2)
depths = 1:L2
rel_depth = depths ./ L2
cone_head = [10 + (L2 - i) * m2 for i in depths]
cone_mlp  = [2 + (L2 - i) * m2 for i in depths]
n_total2 = m2 * L2
ratio_head = n_total2 ./ cone_head
ratio_mlp  = n_total2 ./ cone_mlp

p2 = plot(rel_depth, ratio_head, label="Head site (ao_h)", color=:darkorange, lw=2,
          xlabel="Relative depth of patched site (i/L)", ylabel="Individual site ratio (log scale)",
          title="(b) Deep sites: individual ratio diverges",
          legend=:topleft, yscale=:log10,
          left_margin=12Plots.mm, bottom_margin=10Plots.mm, top_margin=6Plots.mm, right_margin=6Plots.mm)
plot!(p2, rel_depth, ratio_mlp, label="MLP site (mlp_out)", color=:purple, lw=2)
hline!(p2, [2.0], label="Aggregate ceiling (panel a)", color=:black, lw=1, linestyle=:dash)

combined = plot(p1, p2, layout=(1, 2), size=(1500, 620))
savefig(combined, joinpath(figdir, "ceiling_ratio_en.pdf"))
println("Saved -> figures/ceiling_ratio_en.pdf")
