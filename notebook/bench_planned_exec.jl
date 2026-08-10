#=
`demand_planned!` : CORRECTION D'ABORD, puis réutilisation du pool.

Suite de `bench_liveness_slots.jl`, qui a montré que le plan mémoire peut
descendre à 19.9% du nombre de nœuds sur un LlamaModel dès qu'on cesse
d'étendre la durée de vie des activations backpropables (`for_backward=false`).
Restait à savoir si l'EXÉCUTEUR sait s'en servir. Il ne savait pas : il devinait
la forme de sortie comme celle de la première entrée, `execute_rule!` libérait
alors le tampon acquis pour en allouer un neuf, et le pool n'enregistrait aucune
réutilisation. Corrigé en interrogeant `_infer_output_shape`.

ORDRE DES PORTES, comme partout dans ce dépôt : la correction avant la mémoire.
  (1) `demand_planned!` doit rendre EXACTEMENT le même résultat que `demand!`
      -- égalité bit-à-bit, pas `isapprox`. Un plan qui économise de la mémoire
      en changeant le résultat n'économise rien, il casse.
  (2) Seulement ensuite : le pool enregistre-t-il des réutilisations, et
      combien de tampons distincts vivent au pic.

CONTRÔLE NÉGATIF : le même graphe planifié avec `for_backward=true` (durées de
vie étendues, aucun intervalle ne se ferme). Le taux de réutilisation doit y être
NUL ou quasi nul -- sinon la métrique ne mesure pas ce qu'on croit.

Aucun chronomètre : on compte des allocations et des réutilisations, quantités
déterministes rapportées par `pool_stats`.

USAGE : julia --project=. notebook/bench_planned_exec.jl
=#
using NeuroDSL, Printf

dev = NeuroDSL.Backend.CPUDevice()

function build_llama(ns::Symbol; n_layers=4, dim=32, heads=4, hidden=64, seq=8)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.set!(g, :input, randn(Float32, seq, dim); namespace=ns)
    out = NeuroDSL.LlamaModel(n_layers, dim, heads, hidden)(g, :input; namespace=ns)
    return g, out
end

# ── référence : demand! classique ────────────────────────────────────────────
ns_ref = :pe_ref
g_ref, out_ref = build_llama(ns_ref)
ref = copy(Array(NeuroDSL.demand!(g_ref, out_ref; namespace=ns_ref)))
n_nodes = length(NeuroDSL.topo_order!(g_ref; namespace=ns_ref))

println("Référence `demand!` : sortie $(size(ref)), graphe de $n_nodes nœuds")
println("="^72)

results = Tuple{String,Bool,Float64,Int,Int,Int}[]

for (label, fb) in (("for_backward=false (plan serré)", false),
                    ("for_backward=true  (contrôle)",  true))
    ns = Symbol(:pe_, fb)
    g, out = build_llama(ns)
    # mêmes poids que la référence, pour que l'égalité de sortie ait un sens
    for (s, nd) in g_ref.nodes[ns_ref]
        nd.is_param || continue
        NeuroDSL.set!(g, s, Array(nd.value); is_param=true, namespace=ns)
    end
    NeuroDSL.set!(g, :input, Array(g_ref.nodes[ns_ref][:input].value); namespace=ns)

    plan, pool = NeuroDSL.plan_memory!(g; namespace=ns, for_backward=fb)
    got = NeuroDSL.demand_planned!(g, out, plan, pool; namespace=ns)
    exact = (got !== nothing) && size(Array(got)) == size(ref) && Array(got) == ref
    st = NeuroDSL.pool_stats(pool)

    @printf("%-34s slots=%3d  exact=%-5s  reutilisations=%3d  allocs=%3d  taux=%.1f%%\n",
            label, plan.n_slots, exact, st.hits, st.allocs, st.hit_rate_pct)
    push!(results, (fb ? "for_backward_true" : "for_backward_false",
                    exact, st.hit_rate_pct, st.hits, st.allocs, plan.n_slots))
end

println("="^72)
open(joinpath(@__DIR__, "bench_planned_exec_results.txt"), "w") do io
    println(io, "# demand_planned! : correction (egalite bit-a-bit avec demand!) puis")
    println(io, "# reutilisation du pool. Deterministe, sans chronometre.")
    println(io, "# Correctif du 2026-08-08 : la forme de sortie vient desormais de")
    println(io, "# _infer_output_shape et non d'une supposition sur la premiere entree.")
    @printf(io, "RESULT reference=demand! n_nodes=%d out_shape=%s\n", n_nodes, size(ref))
    for (name, exact, rate, hits, allocs, slots) in results
        @printf(io, "RESULT arm=%s bitexact=%s pool_hits=%d pool_allocs=%d hit_rate_pct=%.1f n_slots=%d\n",
                name, exact, hits, allocs, rate, slots)
    end
end
println("Résultats archivés dans notebook/bench_planned_exec_results.txt")
