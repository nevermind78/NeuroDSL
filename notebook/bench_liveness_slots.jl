#=
TEST DÉCISIF DU PLANIFICATEUR MÉMOIRE (coloration d'intervalles, src/liveness.jl).

CE QUI A ÉTÉ DIAGNOSTIQUÉ (2026-08-08) : `compute_liveness` étendait
`last_use = n` pour TOUT nœud backpropable, INCONDITIONNELLEMENT -- il ne
demandait jamais si un backward allait suivre. Conséquence : aucun intervalle ne
se fermait, la coloration ne pouvait partager aucun slot, et `n_slots` restait
égal au nombre de nœuds. Le planificateur était donc rigoureusement sans effet,
y compris sur un graphe forward seul (validation, génération, inférence) où
aucune activation n'a besoin de survivre.

Constat qui a mis la puce à l'oreille : un notebook antérieur
(`PMLJULIA/notebook.ipynb`, cellule 83) rapportait 15.75 MiB de "baseline" à
depth=20 -- soit 63 slots pour un graphe de 63 nœuds. Zéro réduction, présentée
comme une référence.

CE QUE CE BANC MESURE, sans chronomètre et sans horloge : `n_slots` rendu par
`plan_memory!` avec `for_backward=true` (comportement historique) contre
`for_backward=false` (le drapeau ajouté), rapporté au nombre de nœuds. C'est une
quantité STRUCTURELLE et déterministe -- aucune allocation, aucun temps mesuré.

CRITÈRES PRÉ-ENREGISTRÉS (fixés avant le premier lancement) :
  ABANDON  si n_slots >= 0.9 * n_noeuds même avec `for_backward=false`
           -> les intervalles ne se ferment pas, il n'y a rien à allouer et le
              planificateur est sans objet, y compris en inférence.
  FEU VERT si n_slots <= 0.6 * n_noeuds
           -> réduction réelle ; justifie ensuite UN bras chronométré contre les
              51.44 / 51.80 MB mesurés pour val_window / gen_token.
  AMBIGU   entre 0.6 et 0.9 -> à trancher au cas par cas, pas d'engagement.

CONTRÔLE NÉGATIF (bras `skip`) : une chaîne identique PLUS une longue connexion
résiduelle de l'entrée vers la sortie. Cette valeur doit rester vivante d'un
bout à l'autre, donc `n_slots` doit AUGMENTER par rapport à la chaîne nue. Un
planificateur qui rendrait le même compte dans les deux cas ne mesurerait rien.

USAGE : julia --project=. notebook/bench_liveness_slots.jl
=#
using NeuroDSL, Printf

dev = NeuroDSL.Backend.CPUDevice()
const ROWS, DIM = 32, 64

"Chaîne de `depth` motifs matmul->relu. `skip=true` ajoute une résiduelle entrée->sortie."
function build_chain(ns::Symbol, depth::Int; skip::Bool=false)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.set!(g, :h0, NeuroDSL.Backend.randn32(dev, ROWS, DIM); namespace=ns)
    prev = :h0
    for i in 1:depth
        W = Symbol(:W, i)
        NeuroDSL.set!(g, W, randn(Float32, DIM, DIM).*0.05f0; is_param=true, namespace=ns)
        z = Symbol(:z, i); h = Symbol(:h, i)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(z, [prev, W], :matmul; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(h, [z], :relu; namespace=ns))
        prev = h
    end
    if skip
        # Longue durée de vie forcée : :h0 doit survivre jusqu'à la toute fin.
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:skipout, [prev, :h0], :add; namespace=ns))
        prev = :skipout
    end
    return g, prev
end

function slots(g, ns; for_backward::Bool)
    plan, _ = NeuroDSL.plan_memory!(g; namespace=ns, for_backward=for_backward)
    return plan.n_slots, length(plan.order)
end

println("Planificateur mémoire : slots contre nombre de nœuds (structurel, sans horloge)")
println("="^78)
@printf("%-26s %8s %10s %10s %10s %9s\n", "graphe", "nœuds", "slots(bw)", "slots(fwd)", "part fwd", "verdict")
println("-"^78)

results = Tuple{String,Int,Int,Int}[]

for depth in (8, 20, 40)
    ns = Symbol(:lv_chain_, depth)
    g, _ = build_chain(ns, depth)
    s_bw, n = slots(g, ns; for_backward=true)
    s_fw, _ = slots(g, ns; for_backward=false)
    frac = s_fw / n
    verdict = frac <= 0.6 ? "FEU VERT" : (frac >= 0.9 ? "ABANDON" : "ambigu")
    @printf("%-26s %8d %10d %10d %9.1f%% %9s\n", "chaîne depth=$depth", n, s_bw, s_fw, 100*frac, verdict)
    push!(results, ("chain_$depth", n, s_bw, s_fw))
end

# ── contrôle négatif : la résiduelle longue doit AUGMENTER le compte ──────────
ns_a = :lv_ctrl_plain; ga, _ = build_chain(ns_a, 20)
ns_b = :lv_ctrl_skip;  gb, _ = build_chain(ns_b, 20; skip=true)
sa, na = slots(ga, ns_a; for_backward=false)
sb, nb = slots(gb, ns_b; for_backward=false)
println("-"^78)
@printf("%-26s %8d %10s %10d %9s %9s\n", "contrôle : sans skip", na, "-", sa, "-", "-")
@printf("%-26s %8d %10s %10d %9s %9s\n", "contrôle : AVEC skip", nb, "-", sb, "-",
        sb > sa ? "REAGIT" : "INERTE")

# ── un vrai LlamaModel : hétérogénéité et résiduelles réelles ─────────────────
ns_l = :lv_llama
gl = NeuroDSL.NeuroGraph(namespace=ns_l, device=dev)
NeuroDSL.set!(gl, :input, randn(Float32, 8, 32); namespace=ns_l)
NeuroDSL.LlamaModel(4, 32, 4, 64)(gl, :input; namespace=ns_l)
sl_bw, nl = slots(gl, ns_l; for_backward=true)
sl_fw, _  = slots(gl, ns_l; for_backward=false)
println("-"^78)
@printf("%-26s %8d %10d %10d %9.1f%% %9s\n", "LlamaModel 4 couches", nl, sl_bw, sl_fw,
        100*sl_fw/nl, sl_fw/nl <= 0.6 ? "FEU VERT" : (sl_fw/nl >= 0.9 ? "ABANDON" : "ambigu"))

println("="^78)
open(joinpath(@__DIR__, "bench_liveness_slots_results.txt"), "w") do io
    println(io, "# Planificateur memoire : n_slots contre nombre de noeuds.")
    println(io, "# Structurel, deterministe, sans chronometre. for_backward=false coupe")
    println(io, "# l'extension last_use=n des activations backpropables (src/liveness.jl).")
    println(io, "# Criteres pre-enregistres : ABANDON si slots_fwd >= 0.9n, FEU VERT si <= 0.6n.")
    for (name, n, sbw, sfw) in results
        @printf(io, "RESULT graph=%s n_nodes=%d slots_backward=%d slots_forward=%d frac_forward=%.4f\n",
                name, n, sbw, sfw, sfw/n)
    end
    @printf(io, "RESULT graph=llama4 n_nodes=%d slots_backward=%d slots_forward=%d frac_forward=%.4f\n",
            nl, sl_bw, sl_fw, sl_fw/nl)
    @printf(io, "RESULT control=skip n_plain=%d slots_plain=%d n_skip=%d slots_skip=%d responds=%s\n",
            na, sa, nb, sb, sb > sa)
end
println("Résultats archivés dans notebook/bench_liveness_slots_results.txt")
