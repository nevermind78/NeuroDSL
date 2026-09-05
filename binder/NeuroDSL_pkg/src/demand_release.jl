
"""
    demand_release!(g, sym; keep=Symbol[], namespace=g.active_ns, log=nothing)

Comme `demand!`, mais libère (`Backend.free!`) la valeur forward de chaque
nœud intermédiaire DÈS que son dernier consommateur (dans le cône
topologique de `sym`) a été calculé -- au lieu de la garder résidente pour
toujours (le comportement par défaut du graphe mutable, nécessaire pour la
réactivité incrémentale ET pour `CTX_REBUILD`, backward.jl).

**N'utiliser QUE pour un `demand!` qui ne sera JAMAIS suivi d'un
`backward_graph!` sur ce même graphe/namespace** (ex. validation,
génération autorégressive) : les activations intermédiaires libérées ne
sont PAS recalculées automatiquement pour un backward ultérieur --
`backward_graph!`/`backward_graph_sparse!` lèvent une erreur explicite
(pas un crash cryptique) si elles rencontrent une valeur manquante sur le
chemin du gradient. Pour de l'entraînement, utiliser `demand!` classique.

`keep` : symboles à ne jamais libérer (en plus de `sym` lui-même et de tout
paramètre, jamais libérés). Mêmes conditions d'éligibilité que
`release_intermediates!` ci-dessous.
"""
function demand_release!(g::NeuroGraph, name::Symbol; keep::Vector{Symbol}=Symbol[],
                          namespace=g.active_ns, log::Union{Nothing,ExecutionLog}=nothing)
    ns = namespace
    haskey(g.nodes, ns) && haskey(g.nodes[ns], name) ||
        error("❌ :$name introuvable dans :$ns")

    nd = g.nodes[ns][name]
    nd.valid && nd.value !== nothing && return _unwrap_value(nd.value)

    order = topo_order!(g; namespace=ns)
    pos = Dict{Symbol,Int}(sym => i for (i, sym) in enumerate(order))
    tpos = pos[name]

    # Dernière position (<= tpos) où chaque nœud est lu comme entrée d'une
    # règle du cône topologique de `name` -- calculé une fois, à l'avance,
    # depuis la structure du graphe (rule.inputs), pas en observant
    # l'exécution. Un consommateur situé APRÈS `name` dans l'ordre
    # topologique ne prolonge pas la vie : il n'est pas dans le cône qu'on
    # calcule ici, et un `demand!`/`demand_release!` ultérieur sur une
    # cible plus profonde recalculera proprement grâce à `valid=false`.
    last_use = Dict{Symbol,Int}()
    for i in 1:tpos
        sym = order[i]
        haskey(g.rules[ns], sym) || continue
        for inp in g.rules[ns][sym].inputs
            last_use[inp] = max(get(last_use, inp, 0), i)
        end
    end
    free_at = Dict{Int,Vector{Symbol}}()
    for (sym, lu) in last_use
        push!(get!(free_at, lu, Symbol[]), sym)
    end

    keep_set = Set(keep)

    # Boucle miroir de demand! (src/dispatch.jl) -- même séquence
    # d'exécution, donc bit-identique -- avec la libération interleavée
    # juste après chaque position, une fois qu'on sait que plus personne
    # dans le cône ne lira ce nœud.
    for i in 1:tpos
        sym = order[i]
        nd_i = g.nodes[ns][sym]
        if !(nd_i.valid && nd_i.value !== nothing) && haskey(g.rules[ns], sym)
            execute_rule!(g, g.rules[ns][sym]; ctx_store=nothing, namespace=ns, log=log)
        end
        for fsym in get(free_at, i, _EMPTY_SYMBOL_VEC)
            (fsym == name || fsym in keep_set) && continue
            fnd = g.nodes[ns][fsym]
            (!haskey(g.rules[ns], fsym) || fnd.is_param || fnd.value === nothing) && continue
            Backend.free!(g.device, fnd.value)
            fnd.value = nothing
            fnd.valid = false
        end
    end

    return _unwrap_value(g.nodes[ns][name].value)
end

"""
    release_intermediates!(g; keep=Symbol[], namespace=g.active_ns) -> Int

Balayage autonome : libère toute activation intermédiaire encore résidente
(nœud avec règle, non-paramètre, valeur non nulle, absent de `keep`).
Utile en frontière d'épisode (ex. juste après le dernier pas d'entraînement,
avant une salve de validation/génération) pour ne pas payer, dans le pic de
la salve suivante, la résidence laissée par la précédente. Retourne le
nombre de nœuds libérés. Mêmes garde-fous que `demand_release!` côté
backward -- ne jamais appeler avant un `backward_graph!` sur ce namespace
sans un `demand!`/`demand_release!` complet entre les deux.
"""
function release_intermediates!(g::NeuroGraph; keep::Vector{Symbol}=Symbol[], namespace=g.active_ns)
    ns = namespace
    keep_set = Set(keep)
    n_freed = 0
    for (sym, nd) in g.nodes[ns]
        haskey(g.rules[ns], sym) || continue
        nd.is_param && continue
        sym in keep_set && continue
        nd.value === nothing && continue
        Backend.free!(g.device, nd.value)
        nd.value = nothing
        nd.valid = false
        n_freed += 1
    end
    return n_freed
end
