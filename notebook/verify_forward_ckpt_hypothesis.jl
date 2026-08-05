using NeuroDSL, CUDA, Random, Printf
dev = NeuroDSL.Backend.CUDADevice()
const _POOL = CUDA.CUDACore.pool_create(CUDA.CUDACore.active_state().device)
pool_current_mb() = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_CURRENT)) / 1024^2
pool_high_mb()     = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH)) / 1024^2
reset_high!()      = CUDA.attribute!(_POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
quiesce() = (CUDA.synchronize(); GC.gc(true); GC.gc(true); CUDA.synchronize())
function peak_mem(f)
    quiesce(); baseline = pool_current_mb(); reset_high!()
    f(); quiesce(); peak = pool_high_mb()
    return (baseline=baseline, peak=peak, delta=peak - baseline)
end

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

depth, dim, batch, every = 16, 512, 4096, 4
ns = :hyp
g, loss_sym = build_deep_network(depth, dim, batch; ns=ns)
ctx = NeuroDSL.CtxStore()
# warmup
for _ in 1:2
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, loss_sym; ctx_store=ctx, namespace=ns)
end
CUDA.synchronize()

cd = NeuroDSL.CheckpointData(every=every)
sch = NeuroDSL.CheckpointSchedule(g, cd; namespace=ns)

# (1) forward SEUL, plain demand! (pas de checkpointing du tout)
p1 = peak_mem() do
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, loss_sym; ctx_store=ctx, namespace=ns)
    CUDA.synchronize()
end

# (2) forward SEUL, forward_with_checkpointing! (stock, aucune modif)
ctx2 = NeuroDSL.CtxStore()
p2 = peak_mem() do
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.forward_with_checkpointing!(g, loss_sym, ctx2, sch; namespace=ns)
    CUDA.synchronize()
end

@printf "\n  forward SEUL (pas de backward) :\n"
@printf "    (1) demand! plein (pas de checkpointing)      : %9.2f MB\n" p1.delta
@printf "    (2) forward_with_checkpointing! (stock)       : %9.2f MB  (ratio %.3fx vs (1))\n" p2.delta (p2.delta/p1.delta)
