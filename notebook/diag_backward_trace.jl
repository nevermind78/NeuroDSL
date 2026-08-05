using NeuroDSL, CUDA, Random, Printf
using NeuroDSL: GRAD_RULES, accum_grad!, execute_rule!, CtxStore, node, Backend
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

# Copie TRACÉE de backward_with_checkpointing! (src/checkpoint.jl), avec des
# compteurs à chaque étape clé -- pour voir EXACTEMENT pourquoi les nœuds
# recomputables ne sont pas libérés malgré le correctif déjà en place dans
# src/checkpoint.jl (identique ligne pour ligne, juste instrumenté).
function backward_traced!(g, loss_sym; ctx_store, schedule, namespace)
    ns = namespace
    NeuroDSL.zero_grads!(g; namespace=ns)
    ln = node(g, loss_sym; namespace=ns)
    ln.gradient = Backend.ones32(g.device, size(ln.value)...)

    n_no_rule = 0; n_skip_nograd = 0; n_processed = 0
    n_recomp_processed = 0; n_recomp_freed = 0; n_recomp_alreadynothing = 0

    for out_sym in reverse(schedule.order)
        if !haskey(g.rules[ns], out_sym)
            n_no_rule += 1
            continue
        end
        rule = g.rules[ns][out_sym]
        nd_out = g.nodes[ns][out_sym]
        if nd_out.gradient === nothing
            n_skip_nograd += 1
            continue
        end
        n_processed += 1
        is_recomp = out_sym in schedule.recomputable
        is_recomp && (n_recomp_processed += 1)

        for in_sym in rule.inputs
            in_nd = get(g.nodes[ns], in_sym, nothing)
            if in_nd !== nothing && (in_nd.value === nothing || !in_nd.valid)
                NeuroDSL._recompute_segment!(g, in_sym, schedule, ctx_store; namespace=ns)
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
            if nd_out.value === nothing
                n_recomp_alreadynothing += 1
            else
                Backend.free!(g.device, nd_out.value)
                nd_out.value = nothing
                nd_out.valid = false
                n_recomp_freed += 1
            end
        end
    end
    @printf "\n  TRACE backward :\n"
    @printf "    nœuds sans règle (leaves, skip)         : %d\n" n_no_rule
    @printf "    nœuds SKIP (gradient===nothing)         : %d\n" n_skip_nograd
    @printf "    nœuds traités (gradient présent)        : %d\n" n_processed
    @printf "    -- dont recomputables traités           : %d / %d attendus\n" n_recomp_processed length(schedule.recomputable)
    @printf "    -- dont recomputables réellement libérés: %d\n" n_recomp_freed
    @printf "    -- dont recomputables déjà nothing avant libération: %d\n" n_recomp_alreadynothing
    return g
end

depth, dim, batch, every = 16, 512, 4096, 4
ns = :tr
g, loss_sym = build_deep_network(depth, dim, batch; ns=ns)
cd = NeuroDSL.CheckpointData(every=every)
sch = NeuroDSL.CheckpointSchedule(g, cd; namespace=ns)
ctx = NeuroDSL.CtxStore()
NeuroDSL.forward_with_checkpointing!(g, loss_sym, ctx, sch; namespace=ns)
CUDA.synchronize()

backward_traced!(g, loss_sym; ctx_store=ctx, schedule=sch, namespace=ns)
