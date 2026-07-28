# ══════════════════════════════════════════════════════════════════════════════
# Figure : magnitude d'interaction vs C1a sur les 39 configurations d'ablation
# (14 publiées "in-sample" + 25 hors échantillon), sondes tenues à l'écart.
#
# AUCUN nombre recopié à la main : tout est lu depuis
# notebook/config_band_oos_results.json, produit par verify_config_band_oos.jl.
# La bande GELÉE ]6.78, 9.95[ de la version publiée est tracée telle quelle,
# précisément pour montrer qu'elle N'EST PAS vide hors échantillon.
# ══════════════════════════════════════════════════════════════════════════════

using JSON, Plots
gr()

const HERE = @__DIR__
const OUT  = joinpath(HERE, "..", "artilce", "figures")
const BAND_LO, BAND_HI = 6.78, 9.95

d  = JSON.parsefile(joinpath(HERE, "config_band_oos_results.json"))
cs = d["configs"]
println("configurations lues : ", length(cs))

grp(ins, pass) = [c for c in cs if c["in_sample"] == ins && c["pass"] == pass]

fig = plot(; legend=:bottomleft,
           xlabel="interaction magnitude (frozen-activation vs weight-edit gap, logit units)",
           ylabel="C1a: per-input agreement rate",
           ylims=(0.74, 1.02), xlims=(-0.6, 13.2),
           size=(780, 520), left_margin=5Plots.mm, bottom_margin=5Plots.mm)

# bande gelée de la version publiée -- tracée pour montrer qu'elle est peuplée
vspan!(fig, [BAND_LO, BAND_HI]; color=:gray, alpha=0.15, label=nothing)
annotate!(fig, (BAND_LO + BAND_HI)/2, 0.762,
          text("band reported empty\non the original 14", 7, :center, :gray))
hline!(fig, [0.90]; color=:gray, ls=:dash, label=nothing)

for (ins, pass, lab, c, m) in (
        (true,  true,  "original 14, both criteria pass", :seagreen,  :utriangle),
        (true,  false, "original 14, a criterion fails",  :firebrick, :xcross),
        (false, true,  "held-out subset, both pass",      :steelblue, :circle),
        (false, false, "held-out subset, a criterion fails", :darkorange, :diamond))
    s = grp(ins, pass)
    isempty(s) && continue
    scatter!(fig, [c["interaction"] for c in s], [c["c1a"] for c in s];
             label="$lab (n=$(length(s)))", color=c, marker=m, ms=6.5, markerstrokewidth=1)
end

mkpath(OUT)
savefig(fig, joinpath(OUT, "fig_marker_interaction_oos.pdf"))
savefig(fig, joinpath(OUT, "fig_marker_interaction_oos.png"))
println("Écrit -> artilce/figures/fig_marker_interaction_oos.{pdf,png}")

inb = [c for c in cs if BAND_LO < c["interaction"] < BAND_HI]
println("points dans la bande gelée : ", length(inb), " / ", length(cs))
