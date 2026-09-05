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
    # Désarme définitivement CTX_REBUILD (src/backward.jl) pour tout le
    # processus : c'est le seul point d'entrée du BufferPool (vérifié par
    # grep -- execute_rule_pooled!/demand_planned! sont les deux seuls
    # appelants), donc le seul mécanisme qui peut faire d'un `.value` de
    # nœud un buffer partagé/aliasé plutôt qu'un tableau alloué en propre.
    _POOLED_EXECUTION_SEEN[] = true
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
function compute_liveness(g::NeuroGraph; namespace::Symbol = g.active_ns,
                          for_backward::Bool = true)
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
            elseif is_backpropable(nd) && for_backward
                # Les activations backpropables vivent jusqu'à la fin du backward.
                #
                # `for_backward=false` (ajouté le 2026-08-08) coupe cette extension
                # pour les graphes où AUCUN backward ne suivra -- validation,
                # génération autorégressive, inférence. Sans ce drapeau, la
                # branche s'appliquait INCONDITIONNELLEMENT : elle ne demandait
                # jamais si un backward allait tourner, si bien que TOUT nœud
                # backpropable voyait `last_use = n` et qu'aucun intervalle ne se
                # fermait. La coloration ne pouvait donc partager aucun slot, et
                # `n_slots` restait égal au nombre de nœuds -- planificateur
                # rigoureusement sans effet, y compris en forward seul.
                # Diagnostiqué en constatant que `plan_memory!` rendait 63 slots
                # sur un graphe de 63 nœuds (depth=20), soit zéro réduction.
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
function plan_memory!(g::NeuroGraph; namespace::Symbol = g.active_ns,
                      for_backward::Bool = true)
    order    = topo_order!(g; namespace = namespace)
    liveness = compute_liveness(g; namespace = namespace, for_backward = for_backward)
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

        # Acquérir un buffer depuis le pool si le nœud n'en a pas encore.
        #
        # CORRECTIF 2026-08-08 -- la version précédente devinait la forme de
        # sortie comme celle de la PREMIÈRE entrée disponible
        # (`Tuple(size(inputs_avail[1]))`). C'est faux pour la plupart des ops,
        # y compris `:matmul` -- le plus courant : A(m,k)·B(k,n) sort en (m,n),
        # pas en (m,k). Et la conséquence était pire que l'erreur elle-même :
        # `execute_rule!` teste `size(out_node.value) != out_shape` et, la forme
        # devinée ne correspondant pas, il LIBÉRAIT le tampon fraîchement
        # acquis pour en allouer un neuf. Chaque nœud payait donc une
        # acquisition perdue, le pool n'enregistrait jamais aucune
        # réutilisation, et `demand_planned!` allouait STRICTEMENT PLUS que
        # `demand!` -- le plan mémoire n'avait aucun effet sur le pic.
        #
        # On interroge désormais l'inférence de forme du moteur lui-même
        # (`_infer_output_shape`, src/dispatch.jl), celle qu'`execute_rule!`
        # utilisera juste après : les deux concordent donc par construction, le
        # test de taille passe, et le tampon du pool est réellement réutilisé.
        if nd_i.value === nothing
            inputs_vals = [g.nodes[ns][s].value for s in rule.inputs]
            if all(v -> v !== nothing, inputs_vals)
                out_shape = _infer_output_shape(rule.op, inputs_vals, rule.attrs)
                nd_i.value = acquire!(pool, Tuple(out_shape))
            end
        end

        execute_rule!(g, rule; ctx_store = ctx_store)

        # Libérer les buffers dont la durée de vie est expirée, sans libérer
        # la cible demandée avant de la retourner.
        for prev_sym in plan.order[1:step]
            prev_sym == sym && continue
            prev_nd = g.nodes[ns][prev_sym]
            prev_nd.value === nothing && continue
            iv = plan.liveness[prev_sym]
            # Ne libérer QUE ce qu'une règle saura recalculer.
            #
            # CORRECTIF 2026-08-08 -- la condition était `!prev_nd.is_param`
            # seule, ce qui libérait aussi les FEUILLES D'ENTRÉE brutes
            # (`token_ids`, `pos_ids`, un batch injecté par `set!`...) : des
            # nœuds sans règle, donc que la boucle d'exécution ci-dessus saute
            # (`haskey(g.rules[ns], node_sym) || continue`) et que RIEN ne peut
            # reconstruire. `demand_planned!` détruisait ainsi ses propres
            # entrées et n'était utilisable qu'UNE SEULE FOIS : au deuxième
            # appel, `execute_rule!` recevait `nothing` et levait
            # `TypeError: expected AbstractArray{Float32}, got Nothing`.
            # Invisible pour tout test qui ne l'appelle qu'une fois -- c'est
            # pourquoi le défaut a survécu.
            recomputable = haskey(g.rules[ns], prev_sym)
            if iv.last_use <= step && !prev_nd.is_param && recomputable
                release!(pool, prev_nd.value)
                prev_nd.value = nothing
            end
        end

        node_sym == sym && return g.nodes[ns][sym].value
    end

    return g.nodes[ns][sym].value
end
