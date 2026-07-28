# ══════════════════════════════════════════════════════════════════════════════
# Figures de math4.tex (tâche du marqueur). AUCUN réentraînement : toutes les
# données sont recopiées telles quelles depuis les logs vérifiés de la session :
#   - notebook/marker_seed_matrix.log   (matrice de graines 11/22/33/44,
#     tables de sweep "layer_l : [...]", configs [B]/[C]/[D1]/[DJ]/[DJA],
#     lignes RESUME_SEED)
#   - notebook/marker_task_p4bis.log    (instance 2 : sweep + configs)
#   - notebook/marker_task_adiag.jl run (instance 2 : diagnostic miroir,
#     inversé_A=0.86, pureté=1.0, acc_B=0.99)
# Sorties : artilce/figures/fig_marker_*.pdf/.png (même patron savefig que
# generate_e6_figure.jl).
# ══════════════════════════════════════════════════════════════════════════════
using Plots, JSON

const OUT = joinpath(@__DIR__, "..", "artilce", "figures")
mkpath(OUT)
gr()

site_labels = String[]
for l in 1:4
    for h in 1:4; push!(site_labels, "L$(l)h$(h)"); end
    push!(site_labels, "L$(l)m")
end

# Sweep individuel (r moyen, 8 paires) -- marker_seed_matrix.log, tables layer_l.
sweep = Dict(
    11 => [0.099, 0.226, 0.044, 0.663, 0.260,  0.141, 0.021, 0.044, 0.174, 0.050,
           0.147, 0.138, 0.161, 0.067, -0.012,  0.020, 0.043, -0.001, 0.008, 0.036],
    22 => [0.175, -0.001, 0.046, 0.841, 0.507,  0.394, 0.082, 0.330, 0.079, 0.054,
           0.011, 0.022, 0.037, 0.002, 0.016,   0.003, 0.003, 0.017, 0.005, 0.042],
    33 => [0.003, -0.001, 0.975, 0.002, 0.981,  0.056, 0.132, 0.536, 0.013, 0.062,
           0.065, 0.018, 0.021, 0.080, 0.046,   0.053, -0.000, 0.038, 0.026, -0.030],
    44 => [-0.006, 0.000, 0.989, -0.005, 0.621, 0.210, 0.028, 0.057, -0.013, 0.176,
           0.025, 0.001, 0.003, 0.114, 0.057,   0.027, 0.003, 0.006, 0.001, 0.472],
)

# ── Fig. 1 : sweep de carriers par graine ────────────────────────────────────
panels = []
for s in (11, 22, 33, 44)
    vals = sweep[s]
    cols = [i % 5 == 0 ? :darkorange : :steelblue for i in 1:20]  # mlp vs têtes
    p = bar(1:20, vals; color=cols, legend=false, title="seed $s",
            xticks=(1:20, site_labels), xrotation=90, tickfontsize=5,
            ylabel=(s in (11, 33) ? "mean patch recovery r" : ""), ylims=(-0.1, 1.05))
    hline!(p, [0.25]; color=:gray, ls=:dash)
    push!(panels, p)
end
fig1 = plot(panels...; layout=(2, 2), size=(950, 620),
            left_margin=5Plots.mm, bottom_margin=7Plots.mm)
savefig(fig1, joinpath(OUT, "fig_marker_sweep.pdf"))
savefig(fig1, joinpath(OUT, "fig_marker_sweep.png"))

# ── Fig. 2 : propreté du collapse vs concentration du drapeau ────────────────
# x = max_r (meilleur site individuel au patch), y = meilleur taux de collapse
# propre parmi les configs directionnelles = max(inversé_A, littéral_B).
#   seed 11 : max_r=0.663, max(0.41 [DJ inv], 0.4733 [DJ lit]) = 0.473
#   seed 22 : max_r=0.841, max(0.4933 [D1 inv], 0.2733)        = 0.493
#   seed 33 : max_r=0.981, max(0.8667 [D1 inv], 0.2033)        = 0.867
#   seed 44 : max_r=0.989, max(0.93 [D1 inv], 0.9767 [DJ lit]) = 0.977
#   instance 2 (p4bis+adiag) : max_r=0.71, inversé_A=0.86       = 0.86
# Point V=16 (nouveau, notebook/marker_v16_test8_dim128_10k.log, sec:v16 de
# math4.tex) -- carriers layer_2_mlp_out(0.508)+layer_1_mha_ao_h4(0.39),
# max_r=0.508 ; config DJA : inversé_A=0.227, littéral_B=0.37 -> max=0.37.
# Marqueur/couleur distincts : instance à V=16, pas V=8, comparaison honnête
# pas un 6e point de la même population.
xs   = [0.663, 0.841, 0.981, 0.989, 0.71]
ys   = [0.473, 0.493, 0.867, 0.977, 0.86]
labs = ["seed 11", "seed 22", "seed 33", "seed 44", "instance 2"]
fig2 = scatter(xs, ys; ms=8, color=:steelblue, label="V=8 instances",
               xlabel="flag concentration: best single-site patch recovery (max r)",
               ylabel="cleanest collapse rate\nmax(inverted_A, literal_B)",
               xlims=(0.45, 1.02), ylims=(0.2, 1.05), size=(700, 480),
               legend=:bottomright, left_margin=5Plots.mm, bottom_margin=5Plots.mm)
for (x, y, t) in zip(xs, ys, labs)
    annotate!(fig2, x, y + 0.045, text(t, 8, :center))
end
scatter!(fig2, [0.508], [0.37]; ms=9, color=:firebrick, marker=:diamond, label="V=16 (n=1)")
annotate!(fig2, 0.508, 0.37 - 0.05, text("V=16", 8, :center, :firebrick))
savefig(fig2, joinpath(OUT, "fig_marker_concentration.pdf"))
savefig(fig2, joinpath(OUT, "fig_marker_concentration.png"))

# ── Fig. 3 : double polarité de la graine 44 ─────────────────────────────────
# marker_seed_matrix.log, seed 44, configs (distributions filtrées) :
#   B   : A(correct=1.0,   inv=0.0)     B(correct=1.0,   lit=0.0)
#   D1  : A(correct=0.07,  inv=0.93)    B(correct=0.9633, lit=0.0367)
#   DJ  : A(correct=0.9733, inv=0.0267) B(correct=0.0233, lit=0.9767)
#   DJA : A(correct=0.1667, inv=0.8333) B(correct=0.82,   lit=0.18)
configs = ["baseline", "D1 (h3)", "DJ (h3+mlp)", "DJA (+L4mlp)"]
A_ok  = [1.0, 0.070, 0.973, 0.167]
A_inv = [0.0, 0.930, 0.027, 0.833]
B_ok  = [1.0, 0.963, 0.023, 0.820]
B_lit = [0.0, 0.037, 0.977, 0.180]
x = 1:4; w = 0.38
pA = bar(x .- w/2, A_ok; bar_width=w, color=:steelblue, label="correct v(k)",
         xticks=(x, configs), ylims=(0, 1.12), ylabel="rate (filtered A inputs)",
         title="Format A: [mA, k]", legend=:topright, tickfontsize=7)
bar!(pA, x .+ w/2, A_inv; bar_width=w, color=:firebrick, label="inverted v(inv-sigma(k))")
pB = bar(x .- w/2, B_ok; bar_width=w, color=:steelblue, label="correct v(k)",
         xticks=(x, configs), ylims=(0, 1.12), ylabel="rate (filtered B inputs)",
         title="Format B: [mB, sigma(k)]", legend=:topright, tickfontsize=7)
bar!(pB, x .+ w/2, B_lit; bar_width=w, color=:darkorange, label="literal v(sigma(k))")
fig3 = plot(pA, pB; layout=(1, 2), size=(1000, 420),
            left_margin=5Plots.mm, bottom_margin=6Plots.mm)
savefig(fig3, joinpath(OUT, "fig_marker_seed44_polarity.pdf"))
savefig(fig3, joinpath(OUT, "fig_marker_seed44_polarity.png"))


# ── Fig. 4 : magnitude d'interaction vs adéquation C1a, mono vs jointe ───────
# notebook/marker_conj1_verify.log -- 14 configs (5 instances x {D1,DJ,[DJA]},
# seed_11 n'a que 2 carriers donc pas de DJA). Colonnes : (instance, config,
# interaction_max, C1a, C1b_ok, mono/jointe). Recopié ligne à ligne du log.
conj_rows = [
    ("inst2","D1",1.0e-5,1.000,true,:mono),  ("inst2","DJ",12.34386,0.918,false,:joint), ("inst2","DJA",11.98751,0.908,false,:joint),
    ("s11","D1",0.0,0.993,true,:mono),       ("s11","DJ",4.00145,0.947,true,:joint),
    ("s22","D1",0.0,0.923,true,:mono),       ("s22","DJ",6.77653,0.918,true,:joint),  ("s22","DJA",9.94946,0.805,false,:joint),
    ("s33","D1",1.0e-5,0.998,true,:mono),    ("s33","DJ",16.70768,0.847,false,:joint),("s33","DJA",13.15389,0.795,false,:joint),
    ("s44","D1",1.0e-5,1.000,true,:mono),    ("s44","DJ",5.13752,0.980,true,:joint),  ("s44","DJA",5.19520,0.938,true,:joint),
]
xi   = [r[3] for r in conj_rows]
yi   = [r[4] for r in conj_rows]
colr = [r[6] == :mono ? :steelblue : (r[5] ? :seagreen : :firebrick) for r in conj_rows]
mk   = [r[6] == :mono ? :circle : (r[5] ? :utriangle : :xcross) for r in conj_rows]
fig4 = plot(; legend=:bottomleft, xlabel="interaction magnitude (weight-ablation vs frozen-activation gap, logit units)",
            ylabel="C1a: per-input agreement rate", ylims=(0.75, 1.02), xlims=(-1, 18),
            size=(750, 500), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
for grp in ((:mono,"single-carrier (D1)",:steelblue,:circle),
            (:joint_ok,"joint, both criteria pass",:seagreen,:utriangle),
            (:joint_fail,"joint, C1a and/or C1b fails",:firebrick,:xcross))
    key, lab, c, m = grp
    idx = if key == :mono
        findall(r -> r[6] == :mono, conj_rows)
    elseif key == :joint_ok
        findall(r -> r[6] == :joint && r[5], conj_rows)
    else
        findall(r -> r[6] == :joint && !r[5], conj_rows)
    end
    scatter!(fig4, xi[idx], yi[idx]; label=lab, color=c, marker=m, ms=8, markerstrokewidth=1)
end
hline!(fig4, [0.90]; color=:gray, ls=:dash, label=nothing)
vspan!(fig4, [6.78, 9.95]; color=:gray, alpha=0.15, label=nothing)
annotate!(fig4, 8.3, 0.78, text("clean separation\nband", 7, :center, :gray))
savefig(fig4, joinpath(OUT, "fig_marker_interaction.pdf"))
savefig(fig4, joinpath(OUT, "fig_marker_interaction.png"))

println("Écrit -> artilce/figures/fig_marker_{sweep,concentration,seed44_polarity,interaction}.pdf (+.png)")

# ══════════════════════════════════════════════════════════════════════════════
# Fig. 5-6 : vérification numérique de Theorem~\ref{thm:interaction} (§6.5/6.6).
# Données lues DIRECTEMENT depuis notebook/interaction_verify_results.json,
# produit par notebook/verify_marker_interaction_theorem.jl (45 sondes réelles,
# 3 checkpoints, aucun nombre recopié à la main -- même discipline que
# attribution_curve_results*.json / generate_e6_figure.jl).
# ══════════════════════════════════════════════════════════════════════════════
verify_json = joinpath(@__DIR__, "interaction_verify_results.json")
if isfile(verify_json)
    vd = JSON.parsefile(verify_json)

    # ── Fig. 5 : R par réarrangement vs R par quadrature (accord quasi parfait) ──
    inst_names = unique([p["instance"] for p in vd["per_probe"]])
    inst_colors = Dict(zip(inst_names, (:steelblue, :seagreen, :firebrick)))
    fig5 = plot(; legend=:topleft, xlabel="||R(x)|| -- direct rearrangement",
                ylabel="||R(x)|| -- independent Simpson quadrature (41 pts)",
                size=(600, 560), left_margin=5Plots.mm, bottom_margin=5Plots.mm, aspect_ratio=:equal)
    allR = [p["R_rearr"] for p in vd["per_probe"]]
    lim = maximum(allR) * 1.08
    plot!(fig5, [0, lim], [0, lim]; color=:gray, ls=:dash, label="y = x")
    for nm in inst_names
        idx = [p for p in vd["per_probe"] if p["instance"] == nm]
        scatter!(fig5, [p["R_rearr"] for p in idx], [p["R_quad"] for p in idx];
                 label=nm, color=inst_colors[nm], ms=6, markerstrokewidth=0.5)
    end
    savefig(fig5, joinpath(OUT, "fig_marker_theorem_identity.pdf"))
    savefig(fig5, joinpath(OUT, "fig_marker_theorem_identity.png"))

    # ── Fig. 6 : ordre quadratique -- ||R||/||v|| (décroît) vs ||R||/||v||^2 (plat) ──
    asc = sort(vd["alpha_scaling"]; by = a -> -a["alpha"])
    alphas = [a["alpha"] for a in asc]
    r1s    = [a["ratio1"] for a in asc]
    r2s    = [a["ratio2"] for a in asc]
    fig6 = plot(alphas, r1s; xaxis=:log10, yaxis=:log10, marker=:circle, ms=7, lw=2,
                color=:firebrick, label="||R||/||v||  (order 1, if it held)",
                xlabel="alpha (synthetic scale of a fixed direction v)",
                ylabel="ratio (log scale)", legend=:bottomright,
                size=(650, 480), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    plot!(fig6, alphas, r2s; xaxis=:log10, yaxis=:log10, marker=:utriangle, ms=7, lw=2,
          color=:steelblue, label="||R||/||v||^2  (order 2, flat if R=O(||v||^2))")
    savefig(fig6, joinpath(OUT, "fig_marker_theorem_order.pdf"))
    savefig(fig6, joinpath(OUT, "fig_marker_theorem_order.png"))
    println("Écrit -> artilce/figures/fig_marker_theorem_{identity,order}.pdf (+.png)")
else
    @warn "notebook/interaction_verify_results.json introuvable -- lancer verify_marker_interaction_theorem.jl d'abord"
end

# ══════════════════════════════════════════════════════════════════════════════
# Fig. 7 : trajectoire du pilotage conçu (C1c "at-will", §6.4 de math4.tex).
# Chiffres recopiés EXACTEMENT depuis notebook/marker_conj1_verify.log (lignes
# 79-87, cible=inverted_collapse, seed 44 DJ) -- aucune valeur inventée.
# Médianes s̃ des familles A et B après ajout de 0..4 directions ; la cible
# (basculer les DEUX médianes sous 0) n'est jamais atteinte -- illustre "moved
# substantially, did not achieve full control" par une vraie trajectoire plutôt
# que par les seuls points de départ/arrivée déjà cités en prose.
# ══════════════════════════════════════════════════════════════════════════════
n_dirs   = [0, 1, 2, 3, 4]
median_A = [2.53, 2.18, 1.95, 1.91, 1.71]
median_B = [2.04, 1.91, 1.76, 1.68, 1.35]
fig7 = plot(n_dirs, median_A; marker=:circle, ms=7, lw=2, color=:steelblue,
            label="median s̃ (format A, target: cross 0)",
            xlabel="directions added (greedy design, seed 44 DJ)",
            ylabel="median selector value s̃", legend=:topright,
            xticks=n_dirs, size=(650, 480), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
plot!(fig7, n_dirs, median_B; marker=:utriangle, ms=7, lw=2, color=:darkorange,
      label="median s̃ (format B, target: cross 0)")
hline!(fig7, [0.0]; color=:gray, ls=:dash, label="target (unreached)")
annotate!(fig7, 4, 1.55, text("budget exhausted\n(no candidate improves further)", 7, :right, :gray))
savefig(fig7, joinpath(OUT, "fig_marker_steering.pdf"))
savefig(fig7, joinpath(OUT, "fig_marker_steering.png"))
println("Écrit -> artilce/figures/fig_marker_steering.pdf (+.png)")
