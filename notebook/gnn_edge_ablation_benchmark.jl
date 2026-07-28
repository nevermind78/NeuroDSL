# ══════════════════════════════════════════════════════════════════════════════
# Prototype minimal : sweep d'ablation d'arêtes (façon GNNExplainer/PGExplainer)
# sur un petit GNN synthétique, où CHAQUE ARÊTE est un nœud de message
# individuellement adressable (PAS une matrice dense partagée comme dans
# notebook/self_healing_graph_benchmark.jl). Objectif exact (demandé par le
# coordinateur) : vérifier si l'échec du prototype précédent (1,03-1,18x, pas
# de gain) était dû SPÉCIFIQUEMENT au paramètre W partagé par toutes les
# étapes de Bellman-Ford, ou si l'invalidation en cône ne marche jamais sur ce
# type de mutation. Réutilise patch_node!/_downstream_nodes/restore_from_cache!
# tels quels (déjà prouvés sur la tâche du marqueur) -- aucun nouveau
# mécanisme moteur.
# ══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra, NeuroDSL, Random, Statistics

# ── Génération d'un petit graphe orienté aléatoire (V noeuds, degré sortant ~k) ──
function random_digraph(V::Int, avg_out_degree::Int; seed=0)
    rng = MersenneTwister(seed)
    edges = Tuple{Int,Int}[]
    for u in 1:V
        targets = Set{Int}()
        n_out = max(1, avg_out_degree + rand(rng, -1:1))
        attempts = 0
        while length(targets) < n_out && attempts < 20
            v = rand(rng, 1:V)
            if v != u
                push!(targets, v)
            end
            attempts += 1
        end
        for v in targets
            push!(edges, (u, v))
        end
    end
    return unique(edges)
end

# ── Construction du graphe NeuroDSL : message passing à 1 couche, arêtes
#    individuellement adressables (chaque arête = son propre noeud :matmul) ──
function build_gnn_graph(dev, V::Int, d::Int, edges::Vector{Tuple{Int,Int}};
                          seed=0, skip_edge=nothing, nsx=:gnn)
    g = NeuroDSL.NeuroGraph(namespace=nsx, device=dev)
    rng = MersenneTwister(seed)

    # Features d'entrée h0_v -- fixes (comme un GNN déjà entraîné qu'on explique,
    # à la GNNExplainer : les poids ET les features sont figés, seule la
    # structure du graphe est testée).
    for v in 1:V
        NeuroDSL.set!(g, Symbol(:h0_, v), (rand(rng, Float32, 1, d) .- 0.5f0);
                       is_param=false, namespace=nsx)
    end
    W = (rand(rng, Float32, d, d) .- 0.5f0) .* (1f0 / sqrt(Float32(d)))
    NeuroDSL.set!(g, :W, W; is_param=false, namespace=nsx)

    # Un noeud de message PAR ARÊTE, individuellement adressable.
    incoming = [Symbol[] for _ in 1:V]
    for (u, v) in edges
        if skip_edge !== nothing && (u, v) == skip_edge
            continue   # arête absente de CE graphe (voie "reconstruction complète")
        end
        msg_sym = Symbol(:msg_, u, :_, v)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(msg_sym, [Symbol(:h0_, u), :W], :matmul; namespace=nsx))
        push!(incoming[v], msg_sym)
    end

    # Agrégation par noeud cible : somme des messages entrants (+ auto-boucle).
    h1_syms = Symbol[]
    for v in 1:V
        self_sym = Symbol(:h0_, v)
        terms = vcat([self_sym], incoming[v])
        acc = terms[1]
        for (i, t) in enumerate(terms[2:end])
            acc_sym = Symbol(:agg_, v, :_, i)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(acc_sym, [acc, t], :add; namespace=nsx))
            acc = acc_sym
        end
        h1_sym = Symbol(:h1_, v)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(h1_sym, [acc], :relu; namespace=nsx))
        push!(h1_syms, h1_sym)
    end

    # Lecture globale : somme de tous les h1_v, puis somme scalaire.
    acc = h1_syms[1]
    for (i, t) in enumerate(h1_syms[2:end])
        acc_sym = Symbol(:readout_, i)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(acc_sym, [acc, t], :add; namespace=nsx))
        acc = acc_sym
    end
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:output, [acc], :sum_matrix; namespace=nsx))

    return g
end

n_nodes(g, ns::Symbol) = length(g.nodes[ns])

# ══════════════════════════════════════════════════════════════════════════════
# Protocole
# ══════════════════════════════════════════════════════════════════════════════
function run_gnn_ablation_benchmark(V::Int, avg_out_degree::Int; d::Int=4, seed=0)
    println("="^70)
    println("SWEEP D'ABLATION D'ARÊTES -- V=$V noeuds, d=$d, degré sortant moyen≈$avg_out_degree")
    println("="^70)

    dev = NeuroDSL.Backend.CPUDevice()
    edges = random_digraph(V, avg_out_degree; seed=seed)
    println("Arêtes : $(length(edges))")

    ns = :gnn
    g = build_gnn_graph(dev, V, d, edges; seed=seed, nsx=ns)
    output_full = NeuroDSL.demand!(g, :output; namespace=ns)
    total_nodes = n_nodes(g, ns)
    println("Noeuds totaux dans le graphe : $total_nodes")
    println("Sortie (graphe complet) : $(round(output_full[1], digits=5))")

    full_cache = NeuroDSL.capture_activations(g, ns)

    cone_sizes = Int[]
    t_cone = Float64[]
    t_rebuild = Float64[]
    deltas = Float64[]
    max_output_mismatch = 0.0

    for (u, v) in edges
        msg_sym = Symbol(:msg_, u, :_, v)

        # --- Voie CÔNE : patch_node! + demand!, restauration via restore_from_cache! ---
        affected = NeuroDSL._downstream_nodes(g, msg_sym, ns)
        ablated_cache = Dict{Symbol,Any}(msg_sym => zeros(Float32, 1, d))
        t0 = time_ns()
        NeuroDSL.patch_node!(g, msg_sym, ablated_cache; namespace=ns)
        out_ablated = NeuroDSL.demand!(g, :output; namespace=ns)
        dt_cone = (time_ns() - t0) / 1e6
        Δ = abs(out_ablated[1] - output_full[1])
        NeuroDSL.restore_from_cache!(g, ns, full_cache, affected)
        NeuroDSL.demand!(g, :output; namespace=ns)  # re-valide :output (hors du cône restauré si non inclus)

        # --- Voie RECONSTRUCTION COMPLÈTE : nouveau graphe SANS cette arête ---
        t1 = time_ns()
        g2 = build_gnn_graph(dev, V, d, edges; seed=seed, skip_edge=(u, v), nsx=Symbol(:gnn_rb_, u, :_, v))
        out_rebuilt = NeuroDSL.demand!(g2, :output; namespace=Symbol(:gnn_rb_, u, :_, v))
        dt_rebuild = (time_ns() - t1) / 1e6

        push!(cone_sizes, length(affected))
        push!(t_cone, dt_cone)
        push!(t_rebuild, dt_rebuild)
        push!(deltas, Δ)
        max_output_mismatch = max(max_output_mismatch, abs(out_ablated[1] - out_rebuilt[1]))
    end

    println()
    println("── Correction (voie cône vs reconstruction complète -- doivent coïncider) ──")
    println("Écart max entre les deux voies sur la sortie ablatée : $max_output_mismatch")
    println()
    println("── Taille du cône d'invalidation ──")
    println("Cône moyen  : $(round(mean(cone_sizes), digits=2)) noeuds  (sur $total_nodes noeuds totaux)")
    println("Cône max    : $(maximum(cone_sizes))   Cône min : $(minimum(cone_sizes))")
    println("Fraction moyenne du graphe touchée : $(round(100*mean(cone_sizes)/total_nodes, digits=2))%")
    println()
    println("── Latence (moyenne sur $(length(edges)) arêtes) ──")
    println("Voie CÔNE (patch_node!+demand!)        : $(round(mean(t_cone), digits=4)) ms")
    println("Voie RECONSTRUCTION COMPLÈTE            : $(round(mean(t_rebuild), digits=4)) ms")
    println("Ratio rebuild/cône                      : $(round(mean(t_rebuild)/mean(t_cone), digits=2))x")
    println()
    println("── Scores d'attribution (façon GNNExplainer : |Δoutput| par arête ablatée) ──")
    order = sortperm(deltas, rev=true)
    for i in order[1:min(5, length(order))]
        u, v = edges[i]
        println("  arête ($u -> $v) : Δoutput = $(round(deltas[i], digits=5))  (cône = $(cone_sizes[i]) noeuds)")
    end

    return (; V, total_nodes, cone_sizes, t_cone, t_rebuild, deltas, max_output_mismatch)
end

# ── Chauffe JIT explicite : le premier appel à demand!/patch_node!/
#    restore_from_cache!/build_gnn_graph absorbe la compilation Julia (déjà vu
#    et diagnostiqué sur self_healing_graph_benchmark.jl -- V=16 y semblait
#    plus lent que V=100 pour la même raison). On chauffe sur un tout petit
#    graphe jetable AVANT de mesurer quoi que ce soit pour de vrai. ──
println("(chauffe JIT sur un graphe jetable, résultat ignoré)")
_ = run_gnn_ablation_benchmark(6, 2; seed=99)
println("\n" * "="^70)
println("MESURES RÉELLES (après chauffe JIT)")
println("="^70 * "\n")

res_small = run_gnn_ablation_benchmark(10, 2; seed=0)

println("\n" * "#"^70)
println("# Second test, plus grand (V=100) -- même protocole")
println("#"^70)
res_big = run_gnn_ablation_benchmark(100, 3; seed=0)
