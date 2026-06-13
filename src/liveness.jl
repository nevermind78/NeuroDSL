# ════════════════════════════════════════════════════════════════════════════════
# NeuroDSL — liveness.jl
# Analyse de durée de vie des tenseurs + coloration greedy d'intervalles
# Implémente le plan_memory! décrit dans le document de design HTML.
# Code existant inchangé : tout est additif.
# ════════════════════════════════════════════════════════════════════════════════

# ── Intervalle de durée de vie ────────────────────────────────────────────────
"""
    LivenessInterval(first_use, last_use)
Durée de vie d'un nœud dans l'ordre topologique : vivant de first_use à last_use inclus.
"""
struct LivenessInterval
    first_use :: Int
    last_use  :: Int
end

is_alive_at(iv::LivenessInterval, t::Int) = iv.first_use <= t <= iv.last_use

# ── BufferPool ───────────────────────────────────────────────────────────────
"""
    BufferPool(device)
Pool de buffers pré-alloués réutilisables, indexés par forme de tenseur.
API : acquire!(pool, shape) / release!(pool, buf)

Remplace _BUFFER_POOL de dispatch.jl pour les pipelines d'exécution planifiés.
Le pool existant de dispatch.jl reste intact pour la compatibilité.
"""
mutable struct BufferPool
    buckets :: Dict{Tuple, Vector{AbstractArray}}
    device  :: Union{Backend.CPUDevice, Backend.CUDADevice}
    n_alloc :: Int      # allocations fraîches (statistique)
    n_hits  :: Int      # réutilisations depuis le pool (statistique)
end

BufferPool(dev::Union{Backend.CPUDevice, Backend.CUDADevice}) =
    BufferPool(Dict{Tuple, Vector{AbstractArray}}(), dev, 0, 0)

"""
    acquire!(pool, shape) → AbstractArray{Float32}
Emprunte un buffer de forme `shape`. Alloue si le pool est vide pour cette forme.
"""
function acquire!(pool::BufferPool, shape::Tuple)
    if haskey(pool.buckets, shape) && !isempty(pool.buckets[shape])
        pool.n_hits += 1
        return pop!(pool.buckets[shape])::AbstractArray
    end
    pool.n_alloc += 1
    return Backend.zeros32(pool.device, shape...)
end

"""
    release!(pool, buf)
Rend le buffer au pool pour réutilisation future.
"""
function release!(pool::BufferPool, buf::AbstractArray)
    key = Tuple(size(buf))
    haskey(pool.buckets, key) || (pool.buckets[key] = AbstractArray[])
    push!(pool.buckets[key], buf)
    return nothing
end

"""Statistiques du pool : taux de réutilisation."""
function pool_stats(pool::BufferPool)
    total = pool.n_alloc + pool.n_hits
    rate  = total > 0 ? round(100 * pool.n_hits / total; digits=1) : 0.0
    return (allocs=pool.n_alloc, hits=pool.n_hits, hit_rate_pct=rate)
end

# ── Calcul de durée de vie ────────────────────────────────────────────────────
"""
    compute_liveness(g; namespace) → Dict{Symbol, LivenessInterval}

Pour chaque nœud du graphe, calcule :
- first_use : indice topologique où le nœud est calculé
- last_use  : dernier indice où sa valeur est consommée par un successeur

Les paramètres (is_param=true) ont last_use = n (vivent toute la durée d'une itération).
Les nœuds Quantom backpropables ont leur durée de vie étendue jusqu'au backward.

Complexité : O(n²) sur le nombre de nœuds — acceptable pour n < 10 000.
"""
function compute_liveness(g::NeuroGraph; namespace::Symbol = g.active_ns)
    order  = topo_order!(g; namespace = namespace)
    n      = length(order)
    idx_of = Dict{Symbol, Int}(sym => i for (i, sym) in enumerate(order))
    liveness = Dict{Symbol, LivenessInterval}()

    for (i, sym) in enumerate(order)
        first_use = i
        last_use  = i

        # Nœuds qui lisent sym comme entrée → étendent la durée de vie
        for (other_sym, rule) in g.rules[namespace]
            sym ∈ rule.inputs || continue
            j = get(idx_of, other_sym, 0)
            j > 0 && (last_use = max(last_use, j))
        end

        nd = get(g.nodes[namespace], sym, nothing)
        if nd !== nothing
            if nd.is_param
                # Les paramètres vivent toute l'itération (nécessaires au backward)
                last_use = n
            elseif is_backpropable(nd)
                # Les activations backpropables vivent jusqu'à la fin du backward
                last_use = max(last_use, n)
            end
        end

        liveness[sym] = LivenessInterval(first_use, last_use)
    end

    return liveness
end

# ── Coloration greedy d'intervalles ───────────────────────────────────────────
"""
    greedy_interval_coloring(liveness, order) → Dict{Symbol, Int}

Register allocation par sweep line :
- Trie les nœuds par first_use croissant
- Libère les slots dont le last_use est dépassé
- Assigne le plus petit slot libre, ou crée un nouveau slot

Le nombre de slots distincts = borne supérieure du pic de consommation mémoire.
C'est l'invariant topologique décrit dans le document théorique (Théorème 1.1),
ici nommé correctement : largeur maximale de l'antichain du DAG.
"""
function greedy_interval_coloring(liveness::Dict{Symbol, LivenessInterval},
                                   order::Vector{Symbol})
    sorted_nodes = sort(order, by = sym -> liveness[sym].first_use)
    slot_of      = Dict{Symbol, Int}()
    free_slots   = Int[]
    active       = Dict{Symbol, Int}()   # sym → last_use (nœuds courants vivants)
    next_slot    = 1

    for sym in sorted_nodes
        iv = liveness[sym]

        # Libérer les slots expirés (last_use < current first_use)
        expired = [s for (s, lu) in active if lu < iv.first_use]
        for s in expired
            push!(free_slots, slot_of[s])
            delete!(active, s)
        end
        sort!(free_slots)   # stabilité : on réutilise les plus petits slots en premier

        # Attribuer un slot
        slot = isempty(free_slots) ? next_slot : popfirst!(free_slots)
        next_slot = max(next_slot, slot + 1)

        slot_of[sym]   = slot
        active[sym]    = iv.last_use
    end

    return slot_of
end

# ── MemoryPlan ──────────────────────────────────────────────────────────────
"""
    MemoryPlan
Résultat de plan_memory! : association nœud → slot de buffer et métriques.
"""
struct MemoryPlan
    slot_of  :: Dict{Symbol, Int}
    liveness :: Dict{Symbol, LivenessInterval}
    n_slots  :: Int            # nombre de slots = pics de tenseurs simultanés
    order    :: Vector{Symbol}
end

function Base.show(io::IO, plan::MemoryPlan)
    n      = length(plan.order)
    peak   = 0
    for t in 1:n
        live = count(sym -> is_alive_at(plan.liveness[sym], t), plan.order)
        peak = max(peak, live)
    end
    println(io, "MemoryPlan:")
    @printf(io, "  Nœuds     : %d\n", n)
    @printf(io, "  Slots     : %d  (buffers physiques distincts)\n", plan.n_slots)
    @printf(io, "  Pic live  : %d  nœuds simultanément\n", peak)
    @printf(io, "  Réduction : −%.0f%% vs naïf (1 buffer / nœud)\n",
            100 * (1 - plan.n_slots / max(1, n)))
end

# ── Point d'entrée principal ──────────────────────────────────────────────────
"""
    plan_memory!(g; namespace) → (MemoryPlan, BufferPool)
Analyse topologique complète, calcule les durées de vie, attribue les slots
par coloration greedy, et instancie un BufferPool vide prêt à l'usage.

Usage :
    plan, pool = plan_memory!(g)
    println(plan)
    # → affiche les métriques de réduction mémoire
"""
function plan_memory!(g::NeuroGraph; namespace::Symbol = g.active_ns)
    order    = topo_order!(g; namespace = namespace)
    liveness = compute_liveness(g; namespace = namespace)
    slot_of  = greedy_interval_coloring(liveness, order)
    n_slots  = isempty(slot_of) ? 0 : maximum(values(slot_of))
    plan     = MemoryPlan(slot_of, liveness, n_slots, order)
    pool     = BufferPool(g.device)
    return plan, pool
end

# ── Exécution guidée par le plan mémoire ──────────────────────────────────────
"""
    demand_planned!(g, sym, plan, pool; namespace, ctx_store)

Version de `demand!` qui utilise le BufferPool planifié.
Les buffers sont alloués/libérés selon le MemoryPlan pour minimiser le pic VRAM.

Compatible avec le backward_graph! existant (les valeurs restent dans g.nodes).
"""
function demand_planned!(g::NeuroGraph, sym::Symbol,
                          plan::MemoryPlan, pool::BufferPool;
                          ctx_store::Union{CtxStore, Nothing} = nothing,
                          namespace::Symbol = g.active_ns)
    ns = namespace
    haskey(g.nodes, ns) && haskey(g.nodes[ns], sym) ||
        error("❌ :$sym introuvable dans :$ns")

    nd = g.nodes[ns][sym]
    nd.valid && nd.value !== nothing && return nd.value

    idx_of = Dict{Symbol, Int}(s => i for (i, s) in enumerate(plan.order))
    target_idx = get(idx_of, sym, 0)

    for (step, node_sym) in enumerate(plan.order)
        nd_i = g.nodes[ns][node_sym]
        nd_i.valid && nd_i.value !== nothing && step < target_idx && continue
        haskey(g.rules[ns], node_sym) || continue

        rule = g.rules[ns][node_sym]

        # Acquérir un buffer depuis le pool si le nœud n'en a pas encore
        if nd_i.value === nothing
            iv = plan.liveness[node_sym]
            inputs_avail = [g.nodes[ns][s].value for s in rule.inputs
                            if g.nodes[ns][s].value !== nothing]
            if !isempty(inputs_avail)
                estimated_shape = Tuple(size(inputs_avail[1]))
                nd_i.value = acquire!(pool, estimated_shape)
            end
        end

        execute_rule!(g, rule; ctx_store = ctx_store)

        # Libérer les buffers dont la durée de vie est expirée
        for prev_sym in plan.order[1:step]
            prev_nd = g.nodes[ns][prev_sym]
            prev_nd.value === nothing && continue
            iv = plan.liveness[prev_sym]
            if iv.last_use <= step && !prev_nd.is_param
                release!(pool, prev_nd.value)
                prev_nd.value = nothing
            end
        end
    end

    return g.nodes[ns][sym].value
end
