# ══════════════════════════════════════════════════════════════════════════════
# Vérification numérique du Corollaire "Training-mode locality collapse" ajouté
# à math.tex (§sec:backward-locality). Contrairement à un premier essai (chemin
# séquentiel unique, où le corollaire tient comme identité EXACTE puisque
# c_k=w_i dégénère), cette version construit une vraie largeur intra-couche :
# chaque couche j a w_j branches PARALLÈLES (ne dépendant que de la couche
# précédente, pas les unes des autres), combinées par un repli de :add avant
# de passer à la couche suivante. Le site testé est UNE branche (pas
# l'agrégat), donc ses "frères" de la même couche restent hors de son cône
# amont ET aval -- exactement l'hypothèse générale du corollaire (untouched
# = w_i - c_k, pas 0). Teste la convergence ASYMPTOTIQUE rho_train(L) -> 1,
# pas une égalité exacte à L fini.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Statistics

const TARGET_N = 4000

function build_parallel_layered_graph(L::Int, q::Float64, profile::Symbol)
    ns = Symbol(:bwd_test2, :_, L, :_, replace(string(q), "."=>"p"), :_, profile)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=NeuroDSL.Backend.CPUDevice())
    NeuroDSL.set!(g, :input, zeros(Float32, 4); atom_type=NeuroDSL.Datom, namespace=ns)
    cur = :input
    site_per_layer = Symbol[]   # une branche représentative par couche (le "site" testé)
    weights = Int[]
    raw = profile == :output_heavy ? [Float64(j)^q for j in 1:L] : [Float64(L-j+1)^q for j in 1:L]
    scale = TARGET_N / sum(raw)
    for j in 1:L
        wj = max(2, round(Int, raw[j] * scale))  # >=2 pour avoir une vraie largeur
        push!(weights, wj)
        branches = Symbol[]
        for k in 1:wj
            b = Symbol(:L, j, :br, k)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(b, [cur], :relu; namespace=ns))
            push!(branches, b)
        end
        push!(site_per_layer, branches[1])   # site candidat = 1ère branche
        # repli séquentiel des branches en un seul agrégat de couche
        acc = branches[1]
        for k in 2:wj
            nxt = Symbol(:L, j, :agg, k)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(nxt, [acc, branches[k]], :add; namespace=ns))
            acc = nxt
        end
        cur = acc
    end
    NeuroDSL.demand!(g, cur; namespace=ns)
    return g, ns, site_per_layer, weights
end

function ancestors(g, ns, sym)
    visited = Set{Symbol}()
    stack = [sym]
    while !isempty(stack)
        cur = pop!(stack)
        rule = get(g.rules[ns], cur, nothing)
        rule === nothing && continue
        for inp in rule.inputs
            if inp ∉ visited
                push!(visited, inp)
                push!(stack, inp)
            end
        end
    end
    return visited
end

function training_ratio(L::Int, q::Float64, profile::Symbol)
    g, ns, sites, weights = build_parallel_layered_graph(L, q, profile)
    N = sum(w -> 2*w - 1, weights) + 1  # w_j branches + (w_j-1) agrégats par couche, +1 :input
    touched_sizes = Int[]
    for s in sites
        up = ancestors(g, ns, s)
        down = NeuroDSL._downstream_nodes(g, s, ns)
        push!(touched_sizes, length(union(up, down)))
    end
    rho_train = (length(sites) * N) / sum(touched_sizes)
    return rho_train
end

println("="^70)
println("rho_train(L) avec largeur intra-couche réelle -- prédiction : -> 1")
println("="^70)
for profile in [:output_heavy, :input_heavy]
    println("\n--- profil $profile ---")
    for q in [0.0, 1.0, 2.0]
        print("q=$q  :  ")
        for L in [10, 20, 40, 80, 160]
            rho_t = training_ratio(L, q, profile)
            print("L=$L:$(round(rho_t, digits=4))  ")
        end
        println()
    end
end
