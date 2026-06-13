# ════════════════════════════════════════════════════════════════════════════════
# NeuroDSL — checkpoint.jl (version stable, avec BufferPool)
# ════════════════════════════════════════════════════════════════════════════════

struct CheckpointSchedule
    checkpoints  :: Set{Symbol}
    recomputable :: Set{Symbol}
    every        :: Int
    order        :: Vector{Symbol}
end

function CheckpointSchedule(g::NeuroGraph, cd::CheckpointData; namespace::Symbol = g.active_ns)
    order = topo_order!(g; namespace=namespace)
    every = cd.every

    # 1. Tous les paramètres sont des checkpoints
    checkpoints = Set{Symbol}()
    for (sym, nd) in g.nodes[namespace]
        nd.is_param && push!(checkpoints, sym)
    end

    # 2. Tous les nœuds source (sans règle de calcul) doivent être préservés
    #    Ils n'ont pas de règle ⇒ leur valeur ne peut pas être recomputée.
    #    On les ajoute systématiquement aux checkpoints.
    source_nodes = Set{Symbol}()
    for (sym, nd) in g.nodes[namespace]
        if !haskey(g.rules[namespace], sym) && !nd.is_param
            push!(source_nodes, sym)
        end
    end
    union!(checkpoints, source_nodes)

    # 3. Les nœuds intermédiaires backpropables tous les `every` steps
    quant_nodes = [sym for sym in order
                   if haskey(g.nodes[namespace], sym)
                   && is_backpropable(g.nodes[namespace][sym])
                   && !g.nodes[namespace][sym].is_param]
    for (k, sym) in enumerate(quant_nodes)
        k % every == 0 && push!(checkpoints, sym)
    end
    # 4. Le dernier nœud (sortie) est toujours un checkpoint
    isempty(order) || push!(checkpoints, last(order))

    recomputable = Set(sym for sym in quant_nodes if sym ∉ checkpoints)

    @info "CheckpointSchedule: $(length(checkpoints)) checkpoints, $(length(recomputable)) recomputables"
    return CheckpointSchedule(checkpoints, recomputable, every, order)
end

function forward_with_checkpointing!(g::NeuroGraph, output_sym::Symbol,
                                     ctx_store::CtxStore, schedule::CheckpointSchedule;
                                     namespace::Symbol = g.active_ns)
    ns = namespace
    for sym in schedule.order
        nd = g.nodes[ns][sym]
        nd.valid && nd.value !== nothing && continue
        haskey(g.rules[ns], sym) || continue
        rule = g.rules[ns][sym]

        for inp in rule.inputs
            inp_nd = g.nodes[ns][inp]
            if !(inp_nd.valid && inp_nd.value !== nothing)
                demand!(g, inp; ctx_store=ctx_store, namespace=ns)
            end
        end

        demand!(g, sym; ctx_store=ctx_store, namespace=ns)
    end

    # On libère les activations non‑checkpointées pour économiser la mémoire
    for sym in schedule.recomputable
        nd = g.nodes[ns][sym]
        if nd.value !== nothing && !nd.is_param
            nd.value = nothing
            nd.valid = false
        end
        delete!(ctx_store, sym)
    end
    return nothing
end

function _recompute_segment!(g::NeuroGraph, target_sym::Symbol,
                             schedule::CheckpointSchedule,
                             ctx_store::Union{CtxStore,Nothing}=nothing;   # ← nouveau paramètre
                             namespace::Symbol = g.active_ns)
    order = schedule.order
    ns = namespace
    target_idx = findfirst(==(target_sym), order)
    target_idx === nothing && return nothing

    # Cherche le dernier checkpoint ou nœud encore valide avant la cible
    start_idx = 1
    for i in (target_idx-1):-1:1
        prev_sym = order[i]
        nd = get(g.nodes[ns], prev_sym, nothing)
        if nd !== nothing && nd.value !== nothing && nd.valid
            start_idx = i + 1
            break
        end
    end

    # Recalcule de start_idx jusqu'à target_idx
    for i in start_idx:target_idx
        sym = order[i]
        nd = get(g.nodes[ns], sym, nothing)
        nd === nothing && continue
        nd.valid && continue
        # Passe le ctx_store pour que les règles puissent y stocker leurs données
        demand!(g, sym; ctx_store=ctx_store, namespace=ns)
    end
    return g.nodes[ns][target_sym].value
end

function backward_with_checkpointing!(g::NeuroGraph, loss_sym::Symbol;
                                      ctx_store::CtxStore = CtxStore(),
                                      schedule::CheckpointSchedule,
                                      namespace::Symbol = g.active_ns)
    zero_grads!(g; namespace=namespace)
    ln = node(g, loss_sym; namespace=namespace)
    @assert length(ln.value)==1 "loss doit être scalaire"
    ln.gradient = Backend.ones32(g.device, size(ln.value)...)

    for out_sym in reverse(schedule.order)
        !haskey(g.rules[namespace], out_sym) && continue
        rule = g.rules[namespace][out_sym]
        nd_out = g.nodes[namespace][out_sym]
        nd_out.gradient === nothing && continue

        !haskey(GRAD_RULES, rule.op) &&
            error("❌ Pas de règle backward pour :$(rule.op)")

        # Vérifie que les entrées sont disponibles, sinon recompute
        for in_sym in rule.inputs
            in_nd = get(g.nodes[namespace], in_sym, nothing)
            if in_nd !== nothing && (in_nd.value === nothing || !in_nd.valid)
                # Recompute avec le ctx_store pour que les buffers de contexte soient remplis
                _recompute_segment!(g, in_sym, schedule, ctx_store; namespace=namespace)
            end
        end

        # Récupération du contexte
        ctx = get(ctx_store, out_sym, nothing)
        if ctx === nothing
            # Si pas de contexte, on le construit en exécutant la règle à blanc
            ctx_tmp = CtxStore()
            execute_rule!(g, rule; ctx_store=ctx_tmp)
            ctx = get(ctx_tmp, out_sym, Dict{Symbol, Any}())
        end

        inputs_vals = [g.nodes[namespace][s].value for s in rule.inputs]
        grads = GRAD_RULES[rule.op](g.device, nd_out.gradient, ctx, inputs_vals)

        for (i, in_sym) in enumerate(rule.inputs)
            accum_grad!(g.nodes[namespace][in_sym], grads[i])
        end

        delete!(ctx_store, out_sym)
        nd_out.gradient = nothing
    end
    return g
end