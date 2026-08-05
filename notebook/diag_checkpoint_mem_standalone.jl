#=
Script autonome (ne dépend pas de bench_mem_neurodsl.jl / fix_checkpointing.jl,
introuvables dans le dépôt ou sur le disque -- probablement seulement chargés
dans une session Julia locale de l'utilisateur, jamais sauvegardés sur disque).

Reconstruit une architecture équivalente à celle décrite implicitement par les
comptes de checkpoints observés (depth=16 -> "27 checkpoints, 24 recomputables",
depth=32 -> "51 checkpoints, 48 recomputables") : chaîne de `depth` couches
matmul+relu (2 nœuds recomputables par couche, chaque couche a son propre poids
dim×dim), plus une perte MSE finale. Vérifié ci-dessous par les mêmes comptes de
CheckpointSchedule avant de faire confiance aux chiffres.

Watermark CUDA (CU_MEMPOOL_ATTR_USED_MEM_HIGH), même méthodologie que
notebook/article_benchmark_vram_probe.jl / real_llm_vram_probe.jl.
=#
using NeuroDSL, CUDA, Random, Printf

dev = NeuroDSL.Backend.CUDADevice()

const _POOL = CUDA.CUDACore.pool_create(CUDA.CUDACore.active_state().device)
pool_current_mb() = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_CURRENT)) / 1024^2
pool_high_mb()     = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH)) / 1024^2
reset_high!()      = CUDA.attribute!(_POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
function quiesce()
    CUDA.synchronize(); GC.gc(true); GC.gc(true); CUDA.synchronize()
end

function peak_mem(f)
    quiesce()
    baseline = pool_current_mb()
    reset_high!()
    f()
    quiesce()
    peak = pool_high_mb()
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

function warmup!(g, loss_sym, ctx, ns)
    for _ in 1:2
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.zero_grads!(g; namespace=ns)
        NeuroDSL.demand!(g, loss_sym; ctx_store=ctx, namespace=ns)
        NeuroDSL.backward_graph!(g, loss_sym; ctx_store=ctx, namespace=ns, full=true)
    end
    CUDA.synchronize()
end

function bench(depth, dim, every, batch)
    w_mb = dim * dim * 4 / 1024^2
    a_mb = batch * dim * 4 / 1024^2
    @printf "\n  depth=%d dim=%d batch=%d every=%d  |  poids %.1f MB, activation %.1f MB (×%.1f)\n" depth dim batch every w_mb a_mb (a_mb/w_mb)

    ns = Symbol(:std_, depth, :_, dim, :_, batch)
    g, loss_sym = build_deep_network(depth, dim, batch; ns=ns)
    ctx = NeuroDSL.CtxStore()
    warmup!(g, loss_sym, ctx, ns)

    cd  = NeuroDSL.CheckpointData(every=every)
    sch = NeuroDSL.CheckpointSchedule(g, cd; namespace=ns)

    # (a) référence : backward standard, sans checkpointing
    a = peak_mem() do
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.zero_grads!(g; namespace=ns)
        NeuroDSL.demand!(g, loss_sym; ctx_store=ctx, namespace=ns)
        NeuroDSL.backward_graph!(g, loss_sym; ctx_store=ctx, namespace=ns, full=true)
        CUDA.synchronize()
    end

    # (b) checkpointing avec le correctif de src/checkpoint.jl (libération des
    # segments recomputés pendant le backward, au lieu de les garder résidents
    # jusqu'à la fin de la passe)
    ctx_b = NeuroDSL.CtxStore()
    b = peak_mem() do
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.zero_grads!(g; namespace=ns)
        NeuroDSL.forward_with_checkpointing!(g, loss_sym, ctx_b, sch; namespace=ns)
        NeuroDSL.backward_with_checkpointing!(g, loss_sym;
            ctx_store=ctx_b, schedule=sch, namespace=ns)
        CUDA.synchronize()
    end

    ratio = b.delta / a.delta
    @printf "    (a) backward standard              : activations %9.2f MB\n" a.delta
    @printf "    (b) checkpointing (correctif appliqué): activations %9.2f MB  (ratio %.3fx, %+.1f%%)\n" b.delta ratio (b.delta-a.delta)/a.delta*100

    g = nothing; GC.gc(true); CUDA.reclaim()
    return (a=a.delta, b=b.delta, ratio=ratio)
end

println("="^78)
println("   Vérification architecture (comptes attendus : 27/24 puis 51/48)")
println("="^78)
g16, _ = build_deep_network(16, 512, 4096; ns=:chk16)
NeuroDSL.CheckpointSchedule(g16, NeuroDSL.CheckpointData(every=4); namespace=:chk16)
g32, _ = build_deep_network(32, 512, 8192; ns=:chk32)
NeuroDSL.CheckpointSchedule(g32, NeuroDSL.CheckpointData(every=4); namespace=:chk32)
g16 = nothing; g32 = nothing; GC.gc(true); CUDA.reclaim()

println()
println("="^78)
println("   Comparaison backward standard vs checkpointing corrigé")
println("="^78)
results = [bench(d, dim, e, b) for (d, dim, e, b) in [(16, 512, 4, 4096), (32, 512, 4, 8192)]]

println()
println("="^78)
println("   Résumé")
println("="^78)
for (i, (d, _, _, _)) in enumerate([(16, 512, 4, 4096), (32, 512, 4, 8192)])
    r = results[i]
    verdict = r.ratio < 1.0 ? "GAIN mémoire (checkpointing bat le backward standard)" :
              r.ratio ≈ 1.0 ? "parité" : "toujours PIRE que le backward standard"
    @printf "  depth=%-3d : ratio checkpointing/standard = %.3fx  -> %s\n" d r.ratio verdict
end
