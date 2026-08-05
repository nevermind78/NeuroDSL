using NeuroDSL, CUDA, Random, Printf
using NeuroDSL: GRAD_RULES, accum_grad!, execute_rule!, CtxStore, node, Backend, topo_order!
dev = NeuroDSL.Backend.CUDADevice()

function build_deep_network(depth::Int, dim::Int, batch::Int; ns::Symbol)
    g = NeuroDSL.NeuroGraph(device=dev, namespace=ns)
    Random.seed!(1234)
    NeuroDSL.set!(g, :x0, randn(Float32, batch, dim) .* 0.01f0; namespace=ns)
    h = :x0
    for i in 1:depth
        Wsym = Symbol(:W, i)
        NeuroDSL.set!(g, Wsym, randn(Float32, dim, dim) .* 0.01f0; is_param=true, namespace=ns)
        mm = Symbol(:h, i, :_mm)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(mm, [h, Wsym], :matmul; namespace=ns))
        rl = Symbol(:h, i, :_relu)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(rl, [mm], :relu; namespace=ns))
        h = rl
    end
    NeuroDSL.set!(g, :target, randn(Float32, batch, dim) .* 0.01f0; namespace=ns)
    loss_sym = :loss
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss_sym, [h, :target], :mse_loss; namespace=ns))
    return g, loss_sym
end

# Copie tracee de demand! (src/dispatch.jl:740) -- IDENTIQUE, avec un print
# a chaque node qu'elle (re)calcule, pour voir si elle rescanne depuis le
# debut du graphe (position 1) a CHAQUE appel.
function traced_demand!(g, name; ctx_store, namespace, tag)
    ns = namespace
    nd = g.nodes[ns][name]
    nd.valid && nd.value !== nothing && return nd.value
    order = topo_order!(g; namespace=ns)
    for sym in order
        nd_i = g.nodes[ns][sym]
        nd_i.valid && nd_i.value !== nothing && continue
        haskey(g.rules[ns], sym) || continue
        @printf "      [demand! target=%-10s tag=%-8s] RECOMPUTE %s (was invalid)\n" String(name) tag String(sym)
        NeuroDSL.execute_rule!(g, g.rules[ns][sym]; ctx_store=ctx_store, namespace=ns)
        sym == name && break
    end
    return g.nodes[ns][name].value
end

# Copie tracee de _recompute_segment! (src/checkpoint.jl:122)
function traced_recompute_segment!(g, target_sym, schedule, ctx_store; namespace, tag)
    order = schedule.order
    ns = namespace
    target_idx = findfirst(==(target_sym), order)
    target_idx === nothing && return nothing
    start_idx = 1
    for i in (target_idx-1):-1:1
        prev_sym = order[i]
        nd = get(g.nodes[ns], prev_sym, nothing)
        if nd !== nothing && nd.value !== nothing && nd.valid
            start_idx = i + 1
            break
        end
    end
    @printf "    [_recompute_segment! target=%s tag=%s] start_idx=%d target_idx=%d (segment length=%d)\n" String(target_sym) tag start_idx target_idx (target_idx-start_idx+1)
    for i in start_idx:target_idx
        sym = order[i]
        nd = get(g.nodes[ns], sym, nothing)
        nd === nothing && continue
        nd.valid && continue
        traced_demand!(g, sym; ctx_store=ctx_store, namespace=ns, tag=tag)
    end
    return g.nodes[ns][target_sym].value
end

function backward_traced2!(g, loss_sym; ctx_store, schedule, namespace)
    ns = namespace
    NeuroDSL.zero_grads!(g; namespace=ns)
    ln = node(g, loss_sym; namespace=ns)
    ln.gradient = Backend.ones32(g.device, size(ln.value)...)

    pos = Dict(sym => i for (i,sym) in enumerate(schedule.order))

    for out_sym in reverse(schedule.order)
        if !haskey(g.rules[ns], out_sym)
            continue
        end
        rule = g.rules[ns][out_sym]
        nd_out = g.nodes[ns][out_sym]
        if nd_out.gradient === nothing
            continue
        end
        is_recomp = out_sym in schedule.recomputable
        @printf "OWN TURN pos=%-3d out_sym=%-10s is_recomp=%-5s value_before=%s\n" pos[out_sym] String(out_sym) is_recomp (nd_out.value===nothing ? "nothing" : "ALIVE")

        for in_sym in rule.inputs
            in_nd = get(g.nodes[ns], in_sym, nothing)
            if in_nd !== nothing && (in_nd.value === nothing || !in_nd.valid)
                traced_recompute_segment!(g, in_sym, schedule, ctx_store; namespace=ns, tag=String(out_sym))
            end
        end

        ctx = get(ctx_store, out_sym, nothing)
        if ctx === nothing
            ctx_tmp = CtxStore()
            execute_rule!(g, rule; ctx_store=ctx_tmp)
            ctx = get(ctx_tmp, out_sym, Dict{Symbol,Any}())
        end

        inputs_vals = [g.nodes[ns][s].value for s in rule.inputs]
        grads = GRAD_RULES[rule.op](g.device, nd_out.gradient, ctx, inputs_vals)
        for (i, in_sym) in enumerate(rule.inputs)
            accum_grad!(g.nodes[ns][in_sym], grads[i])
        end

        delete!(ctx_store, out_sym)
        nd_out.gradient = nothing

        if is_recomp
            if nd_out.value !== nothing
                Backend.free!(g.device, nd_out.value)
                nd_out.value = nothing
                nd_out.valid = false
                @printf "  -> FREED %s at its own turn\n" String(out_sym)
            end
        end
    end
    return g
end

depth, dim, batch, every = 16, 512, 4096, 4
ns = :dt
g, loss_sym = build_deep_network(depth, dim, batch; ns=ns)
cd = NeuroDSL.CheckpointData(every=every)
sch = NeuroDSL.CheckpointSchedule(g, cd; namespace=ns)
ctx = NeuroDSL.CtxStore()
NeuroDSL.forward_with_checkpointing!(g, loss_sym, ctx, sch; namespace=ns)
CUDA.synchronize()

backward_traced2!(g, loss_sym; ctx_store=ctx, schedule=sch, namespace=ns)

println("\n\n=== FINAL STATE of recomputable nodes ===")
pos = Dict(sym => i for (i,sym) in enumerate(sch.order))
for sym in sort(collect(sch.recomputable); by = s -> pos[s])
    nd = g.nodes[ns][sym]
    @printf "  pos=%3d %-10s alive=%s\n" pos[sym] String(sym) (nd.value !== nothing)
end
