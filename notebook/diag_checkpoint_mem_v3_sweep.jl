#=
v3 -- balaie plusieurs densités de checkpoint (`every`), du plus dense (4,
comme les tests précédents) au plus clairsemé (32), pour voir si le
checkpointing finit par battre le backward standard quand `every` augmente
(hypothèse : every=4 est trop dense, chaque segment recomputé re-paie le
coût de ctx_store -- copy(A) pour :matmul -- trop souvent).

Même méthodologie que diag_checkpoint_mem_v2.jl (graphe frais et indépendant
par branche, watermark CUDA CU_MEMPOOL_ATTR_USED_MEM_HIGH) -- seule
différence : plusieurs valeurs de `every` testées sur le MÊME depth/dim/batch.
=#
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

function warmup_fresh!(depth, dim, batch, ns_base)
    ns = Symbol(ns_base, :_warm)
    g, loss_sym = build_deep_network(depth, dim, batch; ns=ns)
    ctx = NeuroDSL.CtxStore()
    for _ in 1:2
        NeuroDSL.demand!(g, loss_sym; ctx_store=ctx, namespace=ns)
        NeuroDSL.backward_graph!(g, loss_sym; ctx_store=ctx, namespace=ns, full=true)
        NeuroDSL.invalidate_all!(g; namespace=ns)
    end
    CUDA.synchronize()
    g = nothing; GC.gc(true); CUDA.reclaim()
end

function measure_standard(depth, dim, batch, tag)
    ga, loss_a = build_deep_network(depth, dim, batch; ns=Symbol(:a_, tag))
    ctx_a = NeuroDSL.CtxStore()
    a = peak_mem() do
        NeuroDSL.demand!(ga, loss_a; ctx_store=ctx_a, namespace=Symbol(:a_, tag))
        NeuroDSL.backward_graph!(ga, loss_a; ctx_store=ctx_a, namespace=Symbol(:a_, tag), full=true)
        CUDA.synchronize()
    end
    ga = nothing; GC.gc(true); CUDA.reclaim()
    return a.delta
end

function measure_checkpoint(depth, dim, batch, every, tag)
    gb, loss_b = build_deep_network(depth, dim, batch; ns=Symbol(:b_, tag))
    cd  = NeuroDSL.CheckpointData(every=every)
    sch = NeuroDSL.CheckpointSchedule(gb, cd; namespace=Symbol(:b_, tag))
    ctx_b = NeuroDSL.CtxStore()
    b = peak_mem() do
        NeuroDSL.forward_with_checkpointing!(gb, loss_b, ctx_b, sch; namespace=Symbol(:b_, tag))
        NeuroDSL.backward_with_checkpointing!(gb, loss_b;
            ctx_store=ctx_b, schedule=sch, namespace=Symbol(:b_, tag))
        CUDA.synchronize()
    end
    gb = nothing; GC.gc(true); CUDA.reclaim()
    return b.delta
end

function sweep(depth, dim, batch, everys)
    @printf "\n  depth=%d dim=%d batch=%d\n" depth dim batch
    warmup_fresh!(depth, dim, batch, Symbol(:w_, depth))
    a = measure_standard(depth, dim, batch, Symbol(depth, :_a))
    @printf "    (a) backward standard : %9.2f MB\n" a
    for every in everys
        b = measure_checkpoint(depth, dim, batch, every, Symbol(depth, :_, every))
        ratio = b / a
        verdict = ratio < 1.0 ? "GAIN" : "pire"
        @printf "    (b) every=%-3d          : %9.2f MB  (ratio %.3fx, %s)\n" every b ratio verdict
    end
end

println("="^78)
println("   v3 : balayage de densité de checkpoint (every)")
println("="^78)
sweep(32, 512, 8192, [4, 16])
