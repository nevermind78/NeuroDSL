# ══════════════════════════════════════════════════════════════════════════════
# patching.jl — Patching d'activation incrémental
#
# `set!` ne convient PAS pour patcher un nœud qui a déjà un GraphRule : il
# marque le nœud LUI-MÊME invalide (`_invalidate_downstream!` invalide `target`
# en plus de ses successeurs), donc le `demand!` suivant le recalculerait à
# partir de sa propre règle et écraserait la valeur qu'on vient d'imposer --
# vérifié empiriquement (`y.valid == false` après `set!(g,:y,...)` sur un nœud
# `:y` défini par une règle). `set!` est fait pour des feuilles (entrées,
# paramètres), pas pour piloter la valeur d'un nœud calculé.
#
# `patch_node!` mute donc directement `.value`/`.valid` (le nœud patché reste
# `valid=true`, `demand!` ne le recalculera pas) et invalide seulement ses
# VRAIS consommateurs via `_invalidate_downstream!` réutilisé tel quel sur
# chacun d'eux -- aucun nouveau mécanisme de graphe, juste une orchestration
# différente de l'invalidation existante.
# ══════════════════════════════════════════════════════════════════════════════

"""
    capture_activations(g, ns=g.active_ns) -> Dict{Symbol,Any}

Copie la valeur de tous les nœuds calculés du namespace `ns`. Utilisé pour
mettre en cache un run "propre" ou "corrompu" avant de patcher. La copie est
indispensable : `demand!`/`node(g,...).value` renvoient un alias vers le
buffer interne, qui peut être écrasé en place lors d'un appel ultérieur.
"""
capture_activations(g::NeuroGraph, ns::Symbol=g.active_ns) =
    Dict{Symbol,Any}(sym => copy(nd.value) for (sym, nd) in g.nodes[ns] if nd.value !== nothing)

"""
    patch_node!(g, sym, cache; namespace=g.active_ns)

Impose la valeur `cache[sym]` sur le nœud `sym` -- le nœud reste `valid=true`
(un `demand!` ultérieur ne le recalculera pas à partir de sa propre règle),
et seuls ses vrais consommateurs (successeurs dans le graphe de règles) sont
invalidés, en aval, via `_invalidate_downstream!` déjà existant.
"""
function patch_node!(g::NeuroGraph, sym::Symbol, cache; namespace::Symbol=g.active_ns)
    nd = node(g, sym; namespace=namespace)
    nd.value = Backend.to_device(g.device, cache[sym])
    nd.valid = true
    nd.backwarded = false
    for (out_sym, rule) in g.rules[namespace]
        sym ∈ rule.inputs || continue
        _invalidate_downstream!(g, out_sym, namespace)
    end
    return g
end

"""
    recovery_metric(patched, clean, corrupted) -> Float64

Métrique standard de causal tracing : 1.0 = récupération complète du
comportement propre, 0.0 = aucune récupération (sortie toujours corrompue).
"""
function recovery_metric(patched, clean, corrupted)
    patched_a, clean_a, corrupted_a = Array(patched), Array(clean), Array(corrupted)
    denom = norm(corrupted_a .- clean_a)
    denom == 0 && return 1.0
    return 1.0 - norm(patched_a .- clean_a) / denom
end

"""
    patch_and_measure!(g, output_sym, patch_sym, clean_cache, corrupted_cache,
                        clean_output, corrupted_output; namespace=g.active_ns)

Patch `patch_sym` avec sa valeur propre en cache, mesure le temps de
recalcul de `output_sym` et l'effet (`recovery_metric`) sur la sortie, puis
restaure `patch_sym` à sa valeur corrompue en cache et reconverge -- l'appel
suivant peut ainsi tester un autre site en repartant d'un état corrompu
cohérent, sans reconstruire le graphe.
"""
function patch_and_measure!(g::NeuroGraph, output_sym::Symbol, patch_sym::Symbol,
                             clean_cache, corrupted_cache,
                             clean_output, corrupted_output;
                             namespace::Symbol=g.active_ns)
    patch_node!(g, patch_sym, clean_cache; namespace=namespace)
    t0 = time_ns()
    patched_output = demand!(g, output_sym; namespace=namespace)
    dt_ms = (time_ns() - t0) / 1e6
    recovery = recovery_metric(patched_output, clean_output, corrupted_output)

    # Restauration : repart d'un état corrompu cohérent pour le site suivant.
    patch_node!(g, patch_sym, corrupted_cache; namespace=namespace)
    demand!(g, output_sym; namespace=namespace)

    return (; recovery, time_ms=dt_ms)
end

# ══════════════════════════════════════════════════════════════════════════════
# Balayage multi-sites amorti
#
# `patch_and_measure!` restaure en rappelant patch_node!+demand! sur le nœud
# corrompu -- ce qui RECALCULE le même cône qu'à l'étape de mesure. Sur un
# balayage complet (un site par couche), le coût cumulé de ces restaurations
# recalculées est du même ordre que la somme des patches eux-mêmes : la
# restauration n'apporte aucune information nouvelle (on connaît déjà la
# valeur corrompue, capturée une fois pour toutes par capture_activations),
# donc il n'y a aucune raison de la recalculer. `restore_from_cache!` la
# remplace par une copie directe des valeurs déjà connues suivie d'un
# marquage valid=true -- un coût mémoire, pas un coût de calcul (matmul,
# attention, normalisation). C'est une capacité propre à un graphe persistant
# où chaque nœud est individuellement adressable et son état de validité
# explicite ; un framework à bande n'a pas de notion de "nœud" à restaurer
# directement, seulement "refaire tourner le graphe".
# ══════════════════════════════════════════════════════════════════════════════

"""
    _downstream_nodes(g, sym, ns) -> Set{Symbol}

Calcule, en lecture seule, l'ensemble des nœuds que `patch_node!(g, sym, ...)`
toucherait -- `sym` lui-même plus tout nœud atteignable depuis ses
consommateurs directs par la même traversée (valid-gated) que
`_invalidate_downstream!`, sans muter `valid`. Appelé AVANT de patcher (donc
sur un graphe encore valide) pour capturer le cône affecté exactement comme
`patch_node!` le calculerait, avant qu'il ne soit invalidé puis recalculé.
"""
function _downstream_nodes(g::NeuroGraph, sym::Symbol, ns::Symbol)
    visited = Set{Symbol}([sym])
    queue = Symbol[]
    for (out_sym, rule) in g.rules[ns]
        sym ∈ rule.inputs || continue
        push!(queue, out_sym)
    end
    while !isempty(queue)
        cur = pop!(queue)
        cur ∈ visited && continue
        push!(visited, cur)
        nd = get(g.nodes[ns], cur, nothing)
        if nd !== nothing
            for w in nd.watchers
                w ∈ visited || push!(queue, w)
            end
        end
        for (out_sym, rule) in g.rules[ns]
            cur ∈ rule.inputs || continue
            out_nd = get(g.nodes[ns], out_sym, nothing)
            if out_nd !== nothing && out_nd.valid
                push!(queue, out_sym)
            end
        end
    end
    return visited
end

"""
    restore_from_cache!(g, ns, cache, nodes)

Remplace directement `.value` par `copy(cache[sym])` et force `.valid = true`
pour chaque `sym` de `nodes` -- aucune règle exécutée, aucun appel à
`demand!`. Correct si et seulement si `cache` a été capturé sur ce même état
(déterminisme du calcul garanti par construction : mêmes poids, mêmes
entrées, mêmes opérations).
"""
function restore_from_cache!(g::NeuroGraph, ns::Symbol, cache, nodes)
    for sym in nodes
        haskey(cache, sym) || continue
        nd = g.nodes[ns][sym]
        nd.value = copy(cache[sym])
        nd.valid = true
        nd.backwarded = false
    end
    return g
end

"""
    sweep_patch_sites!(g, output_sym, sites, clean_cache, corrupted_cache,
                        clean_output, corrupted_output; namespace=g.active_ns)

Balaie `sites` (patch + mesure comme `patch_and_measure!`), mais restaure
via `restore_from_cache!` au lieu de patch_node!+demand! -- élimine le
recalcul côté restauration pour un balayage complet. Retourne un vecteur de
`(; site, recovery, patch_ms, restore_ms)`.
"""
function sweep_patch_sites!(g::NeuroGraph, output_sym::Symbol, sites::AbstractVector{Symbol},
                             clean_cache, corrupted_cache,
                             clean_output, corrupted_output;
                             namespace::Symbol=g.active_ns)
    results = NamedTuple[]
    for site in sites
        affected = _downstream_nodes(g, site, namespace)

        patch_node!(g, site, clean_cache; namespace=namespace)
        t0 = time_ns()
        patched_output = demand!(g, output_sym; namespace=namespace)
        patch_ms = (time_ns() - t0) / 1e6
        recovery = recovery_metric(patched_output, clean_output, corrupted_output)

        t1 = time_ns()
        restore_from_cache!(g, namespace, corrupted_cache, affected)
        restore_ms = (time_ns() - t1) / 1e6

        push!(results, (; site, recovery, patch_ms, restore_ms))
    end
    return results
end

# ══════════════════════════════════════════════════════════════════════════════
# Patching composable multi-nœuds
#
# L'invalidation n'a jamais supposé qu'un seul nœud change à la fois :
# `_invalidate_downstream!` marque `valid=false` sur tout nœud atteint, peu
# importe combien de sources l'ont invalidé, et `demand!` visite chaque nœud
# de l'ordre topologique une seule fois, ne recalculant que ceux marqués
# invalides. Patcher plusieurs nœuds puis appeler `demand!` une seule fois
# calcule donc déjà l'union de leurs cônes en aval correctement, chaque nœud
# partagé n'étant recalculé qu'une fois -- sans code nouveau pour la fusion
# des cônes elle-même. Ce qui suit expose cette propriété via une API
# multi-sites, en réutilisant patch_node!/_downstream_nodes/restore_from_cache!
# tels quels.
# ══════════════════════════════════════════════════════════════════════════════

"""
    patch_nodes!(g, syms, cache; namespace=g.active_ns)

Applique `patch_node!` à chaque symbole de `syms`, sans appeler `demand!`
entre les deux -- l'appelant choisit quand demander la sortie, une seule
fois pour l'ensemble des patches. L'union de leurs cônes en aval est gérée
par le mécanisme d'invalidation existant, pas par cette fonction.
"""
function patch_nodes!(g::NeuroGraph, syms, cache; namespace::Symbol=g.active_ns)
    for sym in syms
        patch_node!(g, sym, cache; namespace=namespace)
    end
    return g
end

"""
    restore_nodes_from_cache!(g, ns, cache, syms)

Restaure l'union des cônes en aval de chaque symbole de `syms` (calculée
avant tout patch, sur le graphe encore valide) en un seul appel à
`restore_from_cache!` -- réutilise le mécanisme de restauration amortie déjà
construit, pas de nouvelle logique.
"""
function restore_nodes_from_cache!(g::NeuroGraph, ns::Symbol, cache, syms)
    affected = Set{Symbol}()
    for sym in syms
        union!(affected, _downstream_nodes(g, sym, ns))
    end
    restore_from_cache!(g, ns, cache, affected)
    return g
end
