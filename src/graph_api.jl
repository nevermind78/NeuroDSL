
function _ensure_namespace!(g::NeuroGraph, ns::Symbol)
    # Correction du type pour correspondre à NeuroGraph (GraphNode{Float32})
    haskey(g.nodes, ns) || (g.nodes[ns] = Dict{Symbol, GraphNode{Float32}}())
    haskey(g.rules, ns)       || (g.rules[ns]       = Dict{Symbol, GraphRule}())
    haskey(g._topo_cache, ns) || (g._topo_cache[ns] = nothing)
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

        # Propagate to successors
        for (out_sym, rule) in g.rules[ns]
            cur ∈ rule.inputs || continue
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
        users = [out for (out, rule) in g.rules[ns] if sym in rule.inputs]
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

function addrule!(g::NeuroGraph, rule::GraphRule)
    ns = rule.namespace; _ensure_namespace!(g, ns)
    g.rules[ns][rule.output] = rule
    g._topo_cache[ns] = nothing
    haskey(g.nodes[ns], rule.output) ||
        (g.nodes[ns][rule.output] = GraphNode(rule.output, nothing;
            atom_type=rule.atom_type, namespace=ns, valid=false))
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
        for (out_sym, rule) in g.rules[ns]
            if cur ∈ rule.inputs
                push!(queue, out_sym)
            end
        end

        # --- STEP 2: THE CLIMB (For Rule Outputs) ---
        # If 'cur' is the output of a rule, invalidate all its inputs.
        for (out_sym, rule) in g.rules[ns]
            if out_sym == cur
                for inp in rule.inputs
                    push!(queue, inp)
                end
            end
        end
    end
    return g
end
