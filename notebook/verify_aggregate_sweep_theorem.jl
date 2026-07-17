# ══════════════════════════════════════════════════════════════════════════════
# B1 : vérification numérique du Théorème "Aggregate sweep ratio" ajouté à
# math.tex (§sec:aggregate). Construit un DAG séquentiel RÉEL dans NeuroDSL
# (pas une simulation combinatoire abstraite) où chaque "couche" j est
# gonflée à w_j = round(M * profil(j)) nœuds via une chaîne de :relu
# unaires, mesure les VRAIS cônes aval via _downstream_nodes (la même
# fonction utilisée par patch_node!/sweep_patch_sites! partout ailleurs dans
# le projet), et vérifie que rho(L) converge vers la limite prédite par le
# théorème : (q+2)/(q+1) pour un profil "output-heavy" (poids croissant vers
# la sortie), q+2 pour un profil "input-heavy" (poids croissant vers
# l'entrée), et 2 dans les deux cas pour q=0 (profil uniforme, cas déjà
# établi dans l'article2 companion).
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL

const TARGET_N = 4000   # total de nœuds visé, indépendant de q -- garde la construction
                        # rapide même pour les profils les plus déséquilibrés (q élevé)

function build_layered_graph(L::Int, q::Float64, profile::Symbol)
    ns = Symbol(:agg_test, :_, L, :_, replace(string(q), "."=>"p"), :_, profile)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=NeuroDSL.Backend.CPUDevice())
    NeuroDSL.set!(g, :input, zeros(Float32, 4); atom_type=NeuroDSL.Datom, namespace=ns)
    cur = :input
    layer_outputs = Symbol[]
    weights = Int[]
    raw = profile == :output_heavy ? [Float64(j)^q for j in 1:L] : [Float64(L-j+1)^q for j in 1:L]
    scale = TARGET_N / sum(raw)
    for j in 1:L
        wj = max(1, round(Int, raw[j] * scale))
        push!(weights, wj)
        for k in 1:wj
            nxt = Symbol(:L, j, :_, k)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(nxt, [cur], :relu; namespace=ns))
            cur = nxt
        end
        push!(layer_outputs, cur)
    end
    NeuroDSL.demand!(g, cur; namespace=ns)  # évalue tout le graphe une fois -> tous les
                                             # nœuds deviennent `valid`, condition requise
                                             # pour que _downstream_nodes traverse le cône
                                             # complet (elle ne suit que les consommateurs
                                             # déjà valides, cf. patching.jl:163)
    return g, ns, layer_outputs, weights
end

function aggregate_ratio(L::Int, q::Float64, profile::Symbol)
    g, ns, layer_outputs, weights = build_layered_graph(L, q, profile)
    N = sum(weights) + 1  # +1 pour :input lui-même, compté dans chaque cône
    cone_sizes = Int[]
    for out_sym in layer_outputs
        cone = NeuroDSL._downstream_nodes(g, out_sym, ns)
        push!(cone_sizes, length(cone))
    end
    # rho(L) = (S=1 site/couche) * L * N(L) / sum(tailles de cone)
    rho = (L * N) / sum(cone_sizes)
    return rho
end

println("="^70)
println("Profil OUTPUT-HEAVY (w_j ~ j^q, poids croissant vers la sortie)")
println("Prédiction du théorème : rho -> (q+2)/(q+1)")
println("="^70)
for q in [0.0, 0.5, 1.0, 2.0]
    pred = (q + 2) / (q + 1)
    print("q=$q  (prédit $(round(pred, digits=4)))  :  ")
    for L in [20, 40, 80, 160, 320]
        rho = aggregate_ratio(L, q, :output_heavy)
        print("L=$L:$(round(rho, digits=4))  ")
    end
    println()
end

println()
println("="^70)
println("Profil INPUT-HEAVY (w_j ~ (L-j+1)^q, poids croissant vers l'entrée)")
println("Prédiction du théorème : rho -> q+2")
println("="^70)
for q in [0.0, 0.5, 1.0, 2.0]
    pred = q + 2
    print("q=$q  (prédit $(round(pred, digits=4)))  :  ")
    for L in [20, 40, 80, 160, 320]
        rho = aggregate_ratio(L, q, :input_heavy)
        print("L=$L:$(round(rho, digits=4))  ")
    end
    println()
end
