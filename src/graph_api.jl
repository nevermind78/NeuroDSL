
function _ensure_namespace!(g::NeuroGraph, ns::Symbol)
    # Correction du type pour correspondre à NeuroGraph (GraphNode{Float32})
    haskey(g.nodes, ns) || (g.nodes[ns] = Dict{Symbol, GraphNode{Float32}}())
    haskey(g.rules, ns)       || (g.rules[ns]       = Dict{Symbol, GraphRule}())
    haskey(g._topo_cache, ns) || (g._topo_cache[ns] = nothing)
    haskey(g._consumers_cache, ns) || (g._consumers_cache[ns] = nothing)
    haskey(g._ancestors_cache, ns) || (g._ancestors_cache[ns] = Dict{Symbol,Vector{Symbol}}())
end

const _EMPTY_SYMBOL_VEC = Symbol[]

"""
    _consumers_index!(g, ns) -> Dict{Symbol,Vector{Symbol}}

Table (symbole d'entrée) -> (liste des symboles de sortie dont il est un input),
construite une seule fois puis mise en cache (même patron que `topo_order!`),
invalidée uniquement par `addrule!` (toute mutation structurelle du graphe passe
par lui, y compris les suppressions de règles dans `_fuse!`/`_apply_fusion!`, qui
réinitialisent aussi explicitement le cache par prudence).

Remplace le scan de TOUTES les règles du namespace, répété à chaque nœud visité
par `_invalidate_downstream!`/`_invalidate_upstream!`/`patch_node!`/
`_downstream_nodes` (coût O(cône × règles × arité)), par un unique parcours
O(règles × arité) suivi de lookups O(1) par nœud. Vérifié empiriquement avant
d'implémenter ce correctif : sur un graphe à ~9200 nœuds / ~6900 règles, un seul
`patch_node!` sur un cône de 28 nœuds coûtait déjà ~2.4ms rien que pour
l'invalidation (avant tout recalcul) -- le scan répété domine largement le
parcours de `demand!` lui-même à cette échelle.
"""
function _consumers_index!(g::NeuroGraph, ns::Symbol)
    cached = g._consumers_cache[ns]
    cached !== nothing && return cached
    idx = Dict{Symbol, Vector{Symbol}}()
    for (out_sym, rule) in g.rules[ns]
        for inp in rule.inputs
            push!(get!(idx, inp, Symbol[]), out_sym)
        end
    end
    g._consumers_cache[ns] = idx
    return idx
end

"""
    _ancestors_of!(g, ns, target) -> Vector{Symbol}

Ancêtres de `target` (lui inclus, en dernière position), EN ORDRE
TOPOLOGIQUE -- calculé par un DFS post-ordre sur `rule.inputs` (visite les
entrées d'abord, puis ajoute le nœud courant), qui produit un ordre valide
par construction sans jamais consulter `topo_order!(g)` (le préfixe complet
du graphe). Mis en cache par cible (`g._ancestors_cache[ns]`), invalidé aux
mêmes points que `_topo_cache`/`_consumers_cache`.

Corrige un défaut trouvé dans `demand!` (`src/dispatch.jl`) : sans cette
fonction, chaque appel parcourait tout le préfixe topologique jusqu'à la
cible -- O(position de la cible dans l'ordre du graphe entier), PAS O(cône
des ancêtres réels). Le théorème publié (`O(|V_θ|)`) ne couvre que la phase
d'invalidation (`_invalidate_downstream!`/`_invalidate_upstream!`, déjà
O(cône) via `_consumers_index!`) -- la phase de récupération avait un coût
séparé, plus grand, non couvert par ce théorème. C'est aussi la cause
directe d'un vrai bug déjà rencontré (2026-07-29, `copy_params_to_namespace!`) :
un `demand!` sur un nœud SANS RAPPORT avec un cache KV actif dans le même
namespace pouvait quand même le ré-exécuter, puisque le parcours ne
s'arrêtait pas aux seuls ancêtres réels de la cible.
"""
function _ancestors_of!(g::NeuroGraph, ns::Symbol, target::Symbol)
    cache = g._ancestors_cache[ns]
    cached = get(cache, target, nothing)
    cached !== nothing && return cached

    order = Symbol[]
    visited = Set{Symbol}()
    function visit(sym::Symbol)
        sym ∈ visited && return
        push!(visited, sym)
        rule = get(g.rules[ns], sym, nothing)
        if rule !== nothing
            for inp in rule.inputs
                visit(inp)
            end
        end
        push!(order, sym)
    end
    visit(target)

    cache[target] = order
    return order
end

activate!(g::NeuroGraph, ns::Symbol) = (_ensure_namespace!(g, ns); g.active_ns = ns; g)
namespaces(g::NeuroGraph) = collect(keys(g.nodes))

function set!(g::NeuroGraph, name::Symbol, value;
              is_param=false, atom_type=Quantom, namespace=g.active_ns)
    _ensure_namespace!(g, namespace)
    value_on_device = Backend.to_device(g.device, value)
    old_nd = get(g.nodes[namespace], name, nothing)
    watchers = old_nd !== nothing ? old_nd.watchers : Symbol[]
    on_change = old_nd !== nothing ? old_nd.on_change : nothing

    g.nodes[namespace][name] = GraphNode(name, value_on_device;
        atom_type=atom_type, is_param=is_param, namespace=namespace)
    g.nodes[namespace][name].watchers = watchers
    g.nodes[namespace][name].on_change = on_change

    # 1. Forward sweep: Invalidate values and gradients from here to the loss
    _invalidate_downstream!(g, name, namespace)

    # 2. Backward sweep: Invalidate gradients from here back to the inputs
    # Only necessary for parameters, as they are the "roots" of the gradient chain
    if is_param
        _invalidate_upstream!(g, name, namespace)
    end
    
    return g
end

function _invalidate_downstream!(g::NeuroGraph, target::Symbol, ns::Symbol,
                                 visited::Set{Symbol}=Set{Symbol}())
    queue = Symbol[target]
    while !isempty(queue)
        cur = pop!(queue)
        cur ∈ visited && continue
        push!(visited, cur)

        nd = get(g.nodes[ns], cur, nothing)
        if nd !== nothing
            # 🚀 THE FIX: If a node's value is invalid, its gradient is also invalid!
            nd.valid = false
            nd.backwarded = false

            # Trigger callback
            if nd.on_change !== nothing
                nd.on_change(g, cur, ns)
            end
            
            # Propagate to observers
            for w in nd.watchers
                w ∈ visited || push!(queue, w)
            end
        end

        # Propagate to successors -- via l'index de consommateurs (O(1) lookup)
        # plutôt qu'un scan de toutes les règles du namespace à chaque nœud visité.
        for out_sym in get(_consumers_index!(g, ns), cur, _EMPTY_SYMBOL_VEC)
            out_nd = get(g.nodes[ns], out_sym, nothing)
            if out_nd !== nothing && out_nd.valid
                push!(queue, out_sym)
            end
        end
    end
end

"""
    _watch!(g::NeuroGraph, observer::Symbol, observed::Symbol; ns=g.active_ns)

Enregistre `observer` comme observateur de `observed`.  
Quand `observed` est invalidé, `observer` le sera aussi (et si `observer` a un callback `on_change`, il sera déclenché).
"""
function _watch!(g::NeuroGraph, observer::Symbol, observed::Symbol; ns=g.active_ns)
    push!(g.nodes[ns][observed].watchers, observer)
end

const FUSION_TABLE = Dict{Tuple{Vararg{Symbol}}, Symbol}(
    (:matmul, :relu) => :fused_matmul_relu,
    (:relu, :matmul) => :fused_relu_matmul,
)

function _fuse!(g::NeuroGraph, chain::Vector{Symbol}; ns=g.active_ns)
    length(chain) < 2 && return false

    # 1. Verify linearity
    for i in 1:length(chain)-1
        sym = chain[i]
        users = get(_consumers_index!(g, ns), sym, _EMPTY_SYMBOL_VEC)
        if length(users) != 1 || users[1] != chain[i+1]
            return false
        end
    end

    # 2. Check if the sequence of operations is in our Fusion Table
    rules = [g.rules[ns][sym] for sym in chain]
    ops = tuple([r.op for r in rules]...)
    fused_op = get(FUSION_TABLE, ops, nothing)
    fused_op === nothing && return false

    # 🚀 FIX: Collect attributes from all rules in the chain
    # We merge the attribute dictionaries so the fused op knows about :trans_b, etc.
    fused_attrs = Dict{Symbol, Any}()
    for r in rules
        merge!(fused_attrs, r.attrs)
    end

    # 3. Construct external inputs
    fused_inputs = Symbol[]
    intermediate_outputs = [r.output for r in rules[1:end-1]]
    for r in rules
        for inp in r.inputs
            if !(inp in intermediate_outputs) && !(inp in fused_inputs)
                push!(fused_inputs, inp)
            end
        end
    end

    fused_output = rules[end].output

    # 4. Cleanup
    for sym in chain[1:end-1]
        delete!(g.rules[ns], sym)
        delete!(g.nodes[ns], sym)
    end
    delete!(g.rules[ns], chain[end])
    # Réinitialisation explicite du cache de consommateurs -- ne pas dépendre
    # implicitement du fait que l'addrule! ci-dessous le réinitialise aussi.
    g._consumers_cache[ns] = nothing
    empty!(g._ancestors_cache[ns])

    # 🚀 FIX: Pass the collected fused_attrs here!
    addrule!(g, GraphRule(fused_output, fused_inputs, fused_op; 
                                                        attrs=fused_attrs, 
                                                        namespace=ns))
    
    _invalidate_downstream!(g, fused_output, ns)
    return true
end





function node(g::NeuroGraph, name::Symbol; namespace=g.active_ns)
    haskey(g.nodes, namespace) && haskey(g.nodes[namespace], name) ||
        error("❌ Nœud :$name introuvable dans :$namespace")
    return g.nodes[namespace][name]
end

"""
    addrule!(g, rule) -> NeuroGraph

Installe `rule` sous `rule.output`. Si `rule.output` n'existe pas encore, crée un `GraphNode`
frais (`valid=false`). Si `rule.output` existe DÉJÀ (redéfinition d'une règle -- ex. fusion,
chirurgie de graphe), réinvalide explicitement le nœud lui-même et son cône aval via
`_invalidate_downstream!` : le nœud a désormais une règle différente (`inputs`/`op` différents),
sa valeur en cache ne correspond plus à ce qu'il est censé calculer. Découvert comme un vrai
trou du filet de sécurité pendant la conception de `src/graph_surgery.jl` (l'appelant devait
auparavant réinvalider lui-même, sans quoi une valeur périmée était servie silencieusement) --
corrigé ici à la source plutôt que de compter sur chaque appelant pour s'en souvenir.
"""
function addrule!(g::NeuroGraph, rule::GraphRule)
    ns = rule.namespace; _ensure_namespace!(g, ns)
    is_redefinition = haskey(g.nodes[ns], rule.output)
    g.rules[ns][rule.output] = rule
    g._topo_cache[ns] = nothing
    g._consumers_cache[ns] = nothing
    empty!(g._ancestors_cache[ns])
    if is_redefinition
        _invalidate_downstream!(g, rule.output, ns)
    else
        g.nodes[ns][rule.output] = GraphNode(rule.output, nothing;
            atom_type=rule.atom_type, namespace=ns, valid=false)
    end
    return g
end

# ── Problème 1+2+4 : topo_order! avec cache + buffers persistants ──────────
function topo_order!(g::NeuroGraph; namespace::Symbol)
    # Problème 1 : lire le cache avant de recalculer
    cached = g._topo_cache[namespace]
    cached !== nothing && return cached

    n_nodes = length(g.nodes[namespace])

    # Problème 2+4 : pré-allouer tous les buffers une seule fois
    order = sizehint!(Symbol[], n_nodes)
    perm  = sizehint!(Set{Symbol}(), n_nodes)
    temp  = sizehint!(Set{Symbol}(), n_nodes)
    work  = sizehint!(Tuple{Symbol,Int}[], n_nodes)

    function visit(start::Symbol)
        start ∈ perm && return
        push!(work, (start, 1))
        push!(temp, start)

        while !isempty(work)
            n, idx = work[end]
            deps = haskey(g.rules[namespace], n) ?
                       g.rules[namespace][n].inputs : Symbol[]

            if idx > length(deps)
                pop!(work)
                delete!(temp, n)
                push!(perm, n)
                push!(order, n)
            else
                child = deps[idx]
                work[end] = (n, idx + 1)
                child ∈ temp && error("Cycle détecté : $child est un ancêtre de $n")
                child ∈ perm && continue
                push!(work, (child, 1))
                push!(temp, child)
            end
        end
    end

    for root in keys(g.nodes[namespace])
        visit(root)
    end

    # Problème 1 : écrire dans le cache
    g._topo_cache[namespace] = order
    return order
end


function zero_grads!(g::NeuroGraph; namespace=g.active_ns)
    for (_, nd) in g.nodes[namespace]
        # ONLY clear the gradient if the node was invalidated
        # If nd.backwarded is true, the gradient from the previous pass is still valid!
        if !nd.backwarded
            nd.gradient = nothing
        end
    end
end

invalidate_all!(g::NeuroGraph; namespace=g.active_ns) =
    (for (_, nd) in g.nodes[namespace]; nd.is_param || (nd.valid = false); end)

"""
    copy_params_to_namespace!(g::NeuroGraph, src_ns::Symbol, dst_ns::Symbol)

Copie (par VALEUR -- `copy(nd.value)`, jamais un alias) chaque nœud
`is_param=true` de `src_ns` vers `dst_ns`, sous le MÊME symbole. Motivé par
le décodage incrémental avec cache KV (`build_cached_decode_graph!`,
`src/layers.jl`, 2026-07-28) : ce chemin doit lire EXACTEMENT les mêmes
poids qu'un modèle "recalcul complet" déjà construit, pour qu'un test de
parité entre les deux ait un sens -- mais NE PEUT PAS partager le même
`namespace` que ce modèle, pour une raison de correction, pas de style :

`demand!` (dispatch.jl:740) parcourt `topo_order!(g; namespace=ns)` DEPUIS
LE DÉBUT et exécute tout nœud invalide qu'il rencontre AVANT de s'arrêter
sur la cible demandée (`sym == name && break`) -- ce n'est PAS un parcours
restreint aux seuls ANCÊTRES de la cible. Combiné à `invalidate_all!`
(ci-dessus) qui invalide TOUT le namespace sans distinction, tout nœud
invalide du namespace qui se trouve être ordonné avant la cible dans le tri
topologique se fait recalculer comme effet de bord -- inoffensif pour des
ops pures (recalcul redondant, juste du gaspillage), mais RÉELLEMENT
INCORRECT pour un nœud à état comme `:kv_cache_append` (`src/kv_cache.jl`) :
un appel à `demand!` sur un tout autre nœud du même namespace peut ré-exécuter
un nœud de cache KV à l'insu de l'appelant, avec des valeurs de
`cur_step`/`pos` périmées, corrompant silencieusement `aux_data[:history]`.
Découvert exactement ainsi le 2026-07-29 (`kv_cache_qwen_gate.jl`, erreur
`historique absent ou de taille incohérente`) après qu'un premier essai avec
un namespace UNIQUE partagé entre les deux chemins ait semblé fonctionner
sur le jouet -- par PUR HASARD d'ordre topologique (le chemin caché, ajouté
APRÈS le chemin complet, se trouvait ordonné après lui ; l'inverse aurait pu
se produire tout aussi bien, rien dans l'API ne le garantit). Deux
namespaces séparés + cette fonction pour partager les poids EST le correctif
robuste, indépendant de tout ordre d'insertion non documenté.
"""
function copy_params_to_namespace!(g::NeuroGraph, src_ns::Symbol, dst_ns::Symbol)
    _ensure_namespace!(g, dst_ns)
    n = 0
    for (sym, nd) in g.nodes[src_ns]
        nd.is_param || continue
        set!(g, sym, copy(nd.value); is_param=true, namespace=dst_ns)
        n += 1
    end
    return n
end

"""
    alias_tied_param!(g, ns, keep_sym, alias_sym; atol=1f-6) -> Bool

Corrige une duplication de poids LIÉS (`tie_word_embeddings=true` côté HF --
`embed_tokens.weight`/`lm_head.weight` partagent mathématiquement le MÊME
tenseur) redevenue réelle après un aller-retour disque via
`save_graph!`/`load_graph!` (`src/serialization.jl`).

Pourquoi la duplication réapparaît malgré `load_qwen2.jl` qui fait déjà
`set!(g, :lm_head_W, tok_W; ...)` avec le MÊME objet Julia que `:tok_E` :
`save_graph!` itère sur `g.nodes[ns]` PAR SYMBOLE et écrit `Array(nd.value)`
pour chaque nœud feuille indépendamment -- aucune détection d'aliasing entre
deux noms différents pointant vers le même tableau. Le `.bin` contient donc
DEUX blobs de ~890 Mio identiques (vérifié : `qwen2_neurodsl.bin`,
`lm_head_W`/`tok_E`, mêmes 151936×1536 octets, offsets différents), et
`load_graph!` recrée deux `CuArray` séparés à partir d'eux -- l'alias
d'origine est perdu au chargement, pas à la construction.

Corrige ÇA, pas `save_graph!`/`load_graph!` eux-mêmes (format générique,
utilisé aussi par des modèles SANS poids liés -- y toucher pour ce seul cas
Qwen serait un changement bien plus large que nécessaire, voir la discipline
de ce dépôt sur les correctifs ciblés). À appeler UNE FOIS juste après
`load_graph!`, avant toute construction de graphe qui lirait `alias_sym`.

Vérifie d'abord que les deux tableaux sont NUMÉRIQUEMENT identiques
(`isapprox`, comme `load_qwen2.jl` le fait déjà à la conversion) -- ne fait
RIEN et retourne `false` si `alias_sym` est absent (modèle non lié / déjà
appelé) ou si les valeurs diffèrent (refuse de deviner un aliasing qui ne
serait pas réellement vrai). Si conforme : libère (`Backend.free!`) l'ancien
tableau GPU de `alias_sym` puis réassigne son `.value` au MÊME objet Julia
que `keep_sym` -- alias réel, zéro octet supplémentaire, exactement l'état
que `load_qwen2.jl` avait avant la sérialisation.
"""
function alias_tied_param!(g::NeuroGraph, ns::Symbol, keep_sym::Symbol, alias_sym::Symbol; atol=1f-6)
    haskey(g.nodes[ns], keep_sym) ||
        error("❌ alias_tied_param! : nœud :$keep_sym introuvable dans :$ns")
    haskey(g.nodes[ns], alias_sym) || return false

    keep_nd  = g.nodes[ns][keep_sym]
    alias_nd = g.nodes[ns][alias_sym]
    keep_nd.value === alias_nd.value && return false  # déjà aliasé (idempotent)

    size(keep_nd.value) == size(alias_nd.value) || return false
    isapprox(Array(keep_nd.value), Array(alias_nd.value); atol=atol) || return false

    old_value = alias_nd.value
    alias_nd.value = keep_nd.value
    Backend.free!(g.device, old_value)
    return true
end

params(g::NeuroGraph; namespace=g.active_ns) =
    [nd for (_, nd) in g.nodes[namespace] if nd.is_param && is_backpropable(nd)]

function graph_summary(g::NeuroGraph)
    println("╔══════════════════════════════════╗")
    println("║    NeuroGraph — NeuroDSL v4    ║")
    println("╚══════════════════════════════════╝")
    println("  Device    : ", g.device isa Backend.CUDADevice ? "CUDA" : "CPU")
    for ns in namespaces(g)
        n_p = count(nd -> nd.is_param, values(g.nodes[ns]))
        println("  [:$ns]  nodes=$(length(g.nodes[ns]))  rules=$(length(g.rules[ns]))  params=$n_p")
    end
end

"""
    _invalidate_upstream!(g::NeuroGraph, target::Symbol, ns::Symbol;
                               visited::Set{Symbol}=Set{Symbol}())

Propagates invalidation BACKWARDS from a changed parameter up toward the loss.
All nodes that depended on the changed value must have their gradients wiped
and their `backwarded` status reset to false.
"""
function _invalidate_upstream!(g::NeuroGraph, target::Symbol, ns::Symbol;
                               visited::Set{Symbol}=Set{Symbol}())
    queue = Symbol[target]
    
    while !isempty(queue)
        cur = pop!(queue)
        cur ∈ visited && continue
        push!(visited, cur)
        
        nd = get(g.nodes[ns], cur, nothing)
        if nd !== nothing
            nd.gradient = nothing      
            nd.backwarded = false      
        end
        
        # --- STEP 1: THE JUMP (For Parameters/Inputs) ---
        # If 'cur' is used as an input in any rule, that rule's output
        # must be invalidated so we can travel backward from it.
        # Via l'index de consommateurs (O(1)) plutôt qu'un scan de toutes les
        # règles du namespace.
        for out_sym in get(_consumers_index!(g, ns), cur, _EMPTY_SYMBOL_VEC)
            push!(queue, out_sym)
        end

        # --- STEP 2: THE CLIMB (For Rule Outputs) ---
        # If 'cur' is the output of a rule, invalidate all its inputs.
        # `g.rules[ns]` est déjà indexé PAR symbole de sortie -- un lookup direct
        # remplace le scan linéaire de toutes les règles pour trouver celle dont
        # la sortie est `cur`.
        rule = get(g.rules[ns], cur, nothing)
        if rule !== nothing
            for inp in rule.inputs
                push!(queue, inp)
            end
        end
    end
    return g
end
