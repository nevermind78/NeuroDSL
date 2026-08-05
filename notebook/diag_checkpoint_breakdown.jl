using NeuroDSL, CUDA, Random, Printf
dev = NeuroDSL.Backend.CUDADevice()
_nb(x) = x isa CUDA.CuArray ? sizeof(x) : 0

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

function breakdown(g, ns, ctx_store, sch, label)
    bytes_param = 0; bytes_grad = 0; bytes_ckpt = 0; bytes_recomp = 0; bytes_ctx = 0
    n_ckpt_alive = 0; n_recomp_alive = 0
    for (sym, nd) in g.nodes[ns]
        if nd.value isa CUDA.CuArray
            n = _nb(nd.value)
            if nd.is_param
                bytes_param += n
            elseif sym in sch.recomputable
                bytes_recomp += n; n_recomp_alive += 1
            else
                bytes_ckpt += n; n_ckpt_alive += 1
            end
        end
        if nd.gradient isa CUDA.CuArray
            bytes_grad += _nb(nd.gradient)
        end
    end
    for (_, ctx) in ctx_store, (_, v) in ctx
        v isa CUDA.CuArray && (bytes_ctx += _nb(v))
    end
    mb(x) = x/1024^2
    @printf "  -- %s --\n" label
    @printf "     params              %9.2f MB\n" mb(bytes_param)
    @printf "     gradients           %9.2f MB\n" mb(bytes_grad)
    @printf "     checkpoints vivants %9.2f MB  (%d nœuds)\n" mb(bytes_ckpt) n_ckpt_alive
    @printf "     recomputables vivants %7.2f MB  (%d nœuds)  <- devrait être ~0 en fin de passe\n" mb(bytes_recomp) n_recomp_alive
    @printf "     ctx_store           %9.2f MB\n" mb(bytes_ctx)
    @printf "     TOTAL               %9.2f MB\n" mb(bytes_param+bytes_grad+bytes_ckpt+bytes_recomp+bytes_ctx)
end

depth, dim, batch, every = 16, 512, 4096, 4
ns = :bd
g, loss_sym = build_deep_network(depth, dim, batch; ns=ns)
cd = NeuroDSL.CheckpointData(every=every)
sch = NeuroDSL.CheckpointSchedule(g, cd; namespace=ns)
ctx = NeuroDSL.CtxStore()

NeuroDSL.forward_with_checkpointing!(g, loss_sym, ctx, sch; namespace=ns)
CUDA.synchronize()
breakdown(g, ns, ctx, sch, "juste après forward_with_checkpointing! (corrigé)")

NeuroDSL.backward_with_checkpointing!(g, loss_sym; ctx_store=ctx, schedule=sch, namespace=ns)
CUDA.synchronize()
breakdown(g, ns, ctx, sch, "juste après backward_with_checkpointing! (corrigé)")
