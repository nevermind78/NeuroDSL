# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    NeuroDSL — compiler.jl                                  ║
# ║  Moteur de compilation par equality saturation + extraction par coût       ║
# ║  + Ré‑saturation incrémentale réactive via le NeuroGraph vivant           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


using Metatheory
using Metatheory.EGraphs
using Metatheory.TermInterface
using Metatheory.Rules

# ─────────────────────────────────────────────────────────────────────────────
# 1. Conversion NeuroGraph → Metatheory E-Graph
# ─────────────────────────────────────────────────────────────────────────────

function _neuro_term(g::NeuroGraph, ns::Symbol, sym::Symbol)
    rule = g.rules[ns][sym]
    return Term(rule.op, [Term(inp) for inp in rule.inputs])
end

function _graph_to_egraph(g::NeuroGraph, ns::Symbol, output_syms::Vector{Symbol})
    visited = Set{Symbol}()
    stack = copy(output_syms)
    terms = Dict{Symbol, Any}()
    egraph = EGraph()

    while !isempty(stack)
        sym = pop!(stack)
        sym in visited && continue
        push!(visited, sym)

        rule = g.rules[ns][sym]
        inputs = rule.inputs

        for inp in inputs
            if inp ∉ terms && inp ∉ visited
                push!(stack, inp)
            end
        end

        if isempty(inputs)
            id = addexpr!(egraph, QuoteNode(sym))
        else
            for inp in inputs
                if inp ∉ terms
                    terms[inp] = addexpr!(egraph, QuoteNode(inp))
                end
            end
            call_expr = Expr(:call, rule.op, inputs...)
            id = addexpr!(egraph, call_expr)
        end
        terms[sym] = id
    end
    return egraph, terms
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Saturation
# ─────────────────────────────────────────────────────────────────────────────

function _rule_to_metatheory(rule::RewriteRule)
    pat = rule.pattern
    length(pat) < 2 && error("RewriteRule pattern must have at least 2 operators")
    x = PatVar(:x)
    lhs = Term(pat[1], [Term(pat[2], [x])])
    if length(pat) > 2
        inner = x
        for i = length(pat):-1:2
            inner = Term(pat[i], [inner])
        end
        lhs = inner
    end
    rhs = Term(rule.result, [x])
    return DynamicRule(lhs, rhs)
end

function _saturate!(egraph::EGraph, rules::Vector{RewriteRule}, budget::Int)
    metarules = DynamicRule[]
    for r in rules
        try
            push!(metarules, _rule_to_metatheory(r))
        catch e
            @warn "Impossible de convertir la règle $(r.name) : $e"
        end
    end
    Metatheory.saturate!(egraph, metarules; timeout=budget)
    return egraph
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Extraction du plan optimal
# ─────────────────────────────────────────────────────────────────────────────

function _extract_plan(egraph::EGraph, terms::Dict{Symbol,Any}, g::NeuroGraph,
                       ns::Symbol, config::CompilerConfig)
    # Récupérer les formes des nœuds originaux
    shapes = Dict{Symbol, Tuple}()
    for (sym, nd) in g.nodes[ns]
        if nd.value !== nothing && nd.valid
            shapes[sym] = size(nd.value)
        end
    end
    for (sym, cid) in terms
        if haskey(shapes, sym)
            setdata!(egraph[cid], :shape, shapes[sym])
        end
    end

    true_outputs = [sym for sym in keys(g.nodes[ns]) if isempty(consumers(g, sym; ns=ns))]
    if isempty(true_outputs)
        true_outputs = collect(keys(g.nodes[ns]))
    end

    fused_ops = Dict{Symbol, Symbol}()
    exec_order = Symbol[]
    visited_classes = Dict{Int, Symbol}()
    rev_terms = Dict{Int, Symbol}()
    for (sym, cid) in terms
        rev_terms[cid] = sym
    end

    function extract_choice(eclass_id)
        if haskey(visited_classes, eclass_id)
            return visited_classes[eclass_id]
        end
        ecls = egraph[eclass_id]
        best_node = nothing
        best_cost = Inf
        for node in ecls
            op = head(node)
            child_shapes = []
            for child in arguments(node)
                cid_child = child
                child_ecl = egraph[cid_child]
                shape = getdata(child_ecl, :shape, nothing)
                push!(child_shapes, shape !== nothing ? shape : ())
            end
            c = symbolic_cost(op, child_shapes)
            if c < best_cost
                best_cost = c
                best_node = node
            end
        end
        if best_node === nothing
            best_node = first(ecls)
        end
        best_op = head(best_node)

        if haskey(rev_terms, eclass_id)
            sym = rev_terms[eclass_id]
            fused_ops[sym] = best_op
            push!(exec_order, sym)
            visited_classes[eclass_id] = sym
            return sym
        else
            new_sym = gensym("fused_")
            fused_ops[new_sym] = best_op
            push!(exec_order, new_sym)
            visited_classes[eclass_id] = new_sym
            return new_sym
        end
    end

    for sym in true_outputs
        if haskey(terms, sym)
            extract_choice(terms[sym])
        end
    end

    unique!(exec_order)

    deps = Dict{Symbol, Vector{Symbol}}()
    for sym in exec_order
        if haskey(g.rules[ns], sym)
            inputs = g.rules[ns][sym].inputs
            deps[sym] = [inp for inp in inputs if inp in exec_order]
        else
            deps[sym] = Symbol[]
        end
    end
    exec_order = topological_sort(exec_order, deps)

    liveness = compute_liveness(g; namespace=ns)
    slot_of  = greedy_interval_coloring(liveness, exec_order)
    n_slots  = isempty(slot_of) ? 0 : maximum(values(slot_of))
    memory_plan = MemoryPlan(slot_of, liveness, n_slots, exec_order)
    pool = BufferPool(g.device)

    plan = CompiledPlan(ns, exec_order, fused_ops, memory_plan, pool, config)
    return plan
end

function topological_sort(nodes::Vector{Symbol}, deps::Dict{Symbol, Vector{Symbol}})
    indegree = Dict{Symbol, Int}(s => 0 for s in nodes)
    for s in nodes
        for d in deps[s]
            indegree[d] += 1
        end
    end
    queue = [s for s in nodes if indegree[s] == 0]
    sorted = Symbol[]
    while !isempty(queue)
        s = popfirst!(queue)
        push!(sorted, s)
        for d in deps[s]
            indegree[d] -= 1
            indegree[d] == 0 && push!(queue, d)
        end
    end
    length(sorted) == length(nodes) || error("Cycle détecté dans le graphe compilé")
    return sorted
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Enregistrement des hooks de réactivité
# ─────────────────────────────────────────────────────────────────────────────

function _register_reactive_hook!(g::NeuroGraph, plan::CompiledPlan, ns::Symbol)
    for sym in plan.exec_order
        if haskey(g.nodes[ns], sym)
            nd = g.nodes[ns][sym]
            old_cb = nd.on_change
            nd.on_change = (g2, s, n) -> begin
                push!(plan.dirty_nodes, s)
                old_cb !== nothing && old_cb(g2, s, n)
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Exécution du plan compilé
# ─────────────────────────────────────────────────────────────────────────────

function (plan::CompiledPlan)(g::NeuroGraph, output_sym::Symbol;
                              ctx_store=CtxStore())
    if is_dirty(plan)
        _resaturate_region!(g, plan)
    end

    ns = plan.namespace
    for sym in plan.exec_order
        nd = g.nodes[ns][sym]
        nd.valid && nd.value !== nothing && continue

        fused_op = get(plan.fused_ops, sym, nothing)
        original_rule = g.rules[ns][sym]
        if fused_op === nothing || fused_op == original_rule.op
            rule = original_rule
        else
            rule = GraphRule(sym, original_rule.inputs, fused_op;
                             attrs=copy(original_rule.attrs))
        end
        execute_rule!(g, rule; ctx_store=ctx_store)
    end
    return g.nodes[ns][output_sym].value
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Re-saturation incrémentale
# ─────────────────────────────────────────────────────────────────────────────

function _resaturate_region!(g::NeuroGraph, plan::CompiledPlan)
    @info "Recompilation incrémentale ($(length(plan.dirty_nodes)) nœuds sales)"
    new_plan = compile(g, plan.config; namespace=plan.namespace)
    plan.exec_order   = new_plan.exec_order
    plan.fused_ops    = new_plan.fused_ops
    plan.memory_plan  = new_plan.memory_plan
    plan.pool         = new_plan.pool
    plan.egraph_cache = new_plan.egraph_cache
    plan.compiled_at  = time()
    plan.n_recompiles += 1
    empty!(plan.dirty_nodes)
    return plan
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Point d'entrée public
# ─────────────────────────────────────────────────────────────────────────────

function compile(g::NeuroGraph, config::CompilerConfig = CompilerConfig();
                 namespace::Symbol = g.active_ns)
    true_outputs = [sym for sym in keys(g.nodes[namespace])
                    if isempty(consumers(g, sym; ns=namespace))]
    if isempty(true_outputs)
        error("Aucun nœud de sortie trouvé dans le namespace $namespace")
    end

    egraph, terms = _graph_to_egraph(g, namespace, true_outputs)
    _saturate!(egraph, config.rules, config.budget)
    plan = _extract_plan(egraph, terms, g, namespace, config)

    if config.incremental || config.inspect
        plan.egraph_cache = egraph
    end

    _register_reactive_hook!(g, plan, namespace)
    return plan
end
