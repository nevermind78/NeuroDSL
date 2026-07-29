# ══════════════════════════════════════════════════════════════════════════════
# generate_code4_step2_figures.jl — Deux figures pour la nouvelle sous-section
# de code4.tex (circuit IOI réel, Qwen2.5-1.5B-Instruct). AUCUNE donnée
# fabriquée : tout est lu directement depuis
# notebook/qwen2.5-1.5b-instruct/code4_step2_results.json, produit par
# notebook/code4_step2_ioi_patching.jl (5/5 instances IOI, terminé).
#
# Fig. 1 (fig_qwen_ioi_circuit_frequency) : fréquence de chaque site retenu
# (tête ou MLP) à travers les 5 circuits minimaux (`circuit_minimal` après
# backward_prune! -- aucun site n'a été élagué sur aucune instance ici,
# `pruned` est vide partout). Visualise le résultat central : head(layer=25,
# head=9) dans 5/5, head(layer=19,head=6) dans 4/5 -- un code redondant
# partagé à travers les instances, le reste largement idiosyncratique
# (beaucoup de sites à fréquence 1/5).
#
# Fig. 2 (fig_qwen_ioi_recovery_trajectory) : recovery cumulée par round de
# `greedy_patch_search!` (instrumenté), une courbe par instance -- visualise
# directement le `recovery_final` de chaque instance et le décrochage de
# Lucy/Sam (plafonne à 0.673 après 6 sites, contre >=0.96 pour les 4 autres).
#
# PAS de 3e figure (scatter collapse_ratio vs dissociation_gap) : seulement 5
# points, aucune relation pré-enregistrée à tester -- un nuage de 5 points
# suggérerait une corrélation qu'on n'a pas les moyens de soutenir. Le tableau
# texte de la sous-section suffit pour ces deux quantités.
# ══════════════════════════════════════════════════════════════════════════════
using Plots, JSON

const OUT = joinpath(@__DIR__, "..", "artilce", "figures")
mkpath(OUT)
gr()

const RESULTS_PATH = joinpath(@__DIR__, "qwen2.5-1.5b-instruct", "code4_step2_results.json")
data = JSON.parsefile(RESULTS_PATH)
instances = data["instances"]
@assert length(instances) == 5 "attendu 5 instances IOI, trouvé $(length(instances))"

# ── Fig. 1 : fréquence des sites à travers les 5 circuits minimaux ──────────
freq = Dict{String,Int}()
for inst in instances
    for site in inst["circuit_minimal"]
        freq[site] = get(freq, site, 0) + 1
    end
end
sorted_sites = sort(collect(keys(freq)); by = s -> (-freq[s], s))
counts = [freq[s] for s in sorted_sites]
is_mlp = [startswith(s, "mlp") for s in sorted_sites]

# Étiquettes courtes lisibles : "h25/9" pour une tête (layer=25,head=9), "m22" pour un MLP (layer=22).
function short_label(s::String)
    if startswith(s, "head(layer=")
        m = match(r"layer=(\d+),head=(\d+)", s)
        return "h$(m.captures[1])/$(m.captures[2])"
    else
        m = match(r"layer=(\d+)", s)
        return "m$(m.captures[1])"
    end
end
short_labels = short_label.(sorted_sites)

cols = [m ? :darkorange : :steelblue for m in is_mlp]
fig1 = bar(1:length(sorted_sites), counts; color=cols, legend=false,
           xticks=(1:length(sorted_sites), short_labels), xrotation=90, tickfontsize=7,
           ylabel="instances sur 5 où ce site est retenu (circuit minimal)",
           xlabel="site (h=tête layer/head, m=MLP layer)",
           ylims=(0, 5.5), yticks=0:1:5, size=(900, 420),
           title="Fréquence des sites à travers les 5 instances IOI (Qwen2.5-1.5B)",
           titlefontsize=10, left_margin=5Plots.mm, bottom_margin=9Plots.mm)
hline!(fig1, [2.5]; color=:gray, ls=:dash, label="")
annotate!(fig1, 1, 5.3, text("head(25,9) : 5/5", 8, :left, :steelblue))
annotate!(fig1, 2, 4.3, text("head(19,6) : 4/5", 8, :left, :steelblue))
savefig(fig1, joinpath(OUT, "fig_qwen_ioi_circuit_frequency.pdf"))
savefig(fig1, joinpath(OUT, "fig_qwen_ioi_circuit_frequency.png"))
println("Fig. 1 écrite (", length(sorted_sites), " sites uniques sur ", 5*6, " site-instances) -> ", joinpath(OUT, "fig_qwen_ioi_circuit_frequency.pdf"))

# ── Fig. 2 : trajectoire de recovery cumulée par round, une courbe/instance ──
labels5 = [inst["label"] for inst in instances]
colors5 = [:steelblue, :darkorange, :seagreen, :firebrick, :purple]
markers5 = [:circle, :square, :diamond, :utriangle, :star5]

fig2 = plot(; xlabel="sites retenus (ordre de la recherche gloutonne)", ylabel="recovery cumulée",
            xlims=(0.7, 6.3), ylims=(0, 1.05), xticks=1:6, size=(760, 480),
            legend=:bottomright, left_margin=5Plots.mm, bottom_margin=5Plots.mm,
            title="Recovery cumulée (greedy_patch_search!, max 6 sites), 5 instances IOI",
            titlefontsize=10)
for (i, inst) in enumerate(instances)
    traj = inst["greedy_trajectory"]
    ys = [t["cumulative_recovery"] for t in traj]
    plot!(fig2, 1:length(ys), ys; color=colors5[i], marker=markers5[i], ms=5, lw=2,
          label=labels5[i])
end
hline!(fig2, [1.0]; color=:gray, ls=:dash, label="")
savefig(fig2, joinpath(OUT, "fig_qwen_ioi_recovery_trajectory.pdf"))
savefig(fig2, joinpath(OUT, "fig_qwen_ioi_recovery_trajectory.png"))
println("Fig. 2 écrite -> ", joinpath(OUT, "fig_qwen_ioi_recovery_trajectory.pdf"))
