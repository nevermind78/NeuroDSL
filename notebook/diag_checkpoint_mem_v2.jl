#=
v2 -- corrige un biais méthodologique trouvé dans les tests précédents
(diag_checkpoint_mem_standalone.jl, verify_forward_ckpt_hypothesis.jl) :
`invalidate_all!` (src/graph_api.jl:276-277) ne fait QUE `nd.valid = false`,
il ne libère jamais `.value`. En réutilisant le MÊME graphe `g` pour
plusieurs mesures à la suite (warmup -> test A -> test B), les buffers
d'activation restent résidents depuis le tout premier run et sont réutilisés
EN PLACE (même forme -> pas de réallocation) -- invisibles pour toute mesure
de delta ultérieure. Ça a faussé la vérification précédente ("forward_with_
checkpointing! == demand! plein, 136.00 MB pile identique") : ce qui était
mesuré, c'était le bruit des copies de ctx_store, pas le vrai coût des
activations.

Ici : un graphe FRAIS et INDÉPENDANT par branche mesurée, comme le fait déjà
notebook/article_benchmark_vram_probe.jl (méthodologie déjà validée cette
session pour la comparaison NeuroDSL/PyTorch de l'article).
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
    # warmup JIT sur un graphe JETABLE, distinct de celui mesuré -- garantit
    # que le graphe mesuré démarre bien vierge (seulement les poids/leaves).
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

function bench_fresh(depth, dim, every, batch)
    w_mb = dim * dim * 4 / 1024^2
    a_mb = batch * dim * 4 / 1024^2
    @printf "\n  depth=%d dim=%d batch=%d every=%d  |  poids %.1f MB, activation %.1f MB (×%.1f)\n" depth dim batch every w_mb a_mb (a_mb/w_mb)

    warmup_fresh!(depth, dim, batch, Symbol(:w_, depth))

    # (a) graphe FRAIS, backward standard
    ga, loss_a = build_deep_network(depth, dim, batch; ns=Symbol(:a_, depth))
    ctx_a = NeuroDSL.CtxStore()
    a = peak_mem() do
        NeuroDSL.demand!(ga, loss_a; ctx_store=ctx_a, namespace=Symbol(:a_, depth))
        NeuroDSL.backward_graph!(ga, loss_a; ctx_store=ctx_a, namespace=Symbol(:a_, depth), full=true)
        CUDA.synchronize()
    end
    ga = nothing; GC.gc(true); CUDA.reclaim()

    # (b) graphe FRAIS SÉPARÉ, checkpointing (correctifs forward+backward appliqués)
    gb, loss_b = build_deep_network(depth, dim, batch; ns=Symbol(:b_, depth))
    cd  = NeuroDSL.CheckpointData(every=every)
    sch = NeuroDSL.CheckpointSchedule(gb, cd; namespace=Symbol(:b_, depth))
    ctx_b = NeuroDSL.CtxStore()
    b = peak_mem() do
        NeuroDSL.forward_with_checkpointing!(gb, loss_b, ctx_b, sch; namespace=Symbol(:b_, depth))
        NeuroDSL.backward_with_checkpointing!(gb, loss_b;
            ctx_store=ctx_b, schedule=sch, namespace=Symbol(:b_, depth))
        CUDA.synchronize()
    end
    gb = nothing; GC.gc(true); CUDA.reclaim()

    ratio = b.delta / a.delta
    @printf "    (a) backward standard (graphe frais)      : activations %9.2f MB\n" a.delta
    @printf "    (b) checkpointing (graphe frais séparé)    : activations %9.2f MB  (ratio %.3fx, %+.1f%%)\n" b.delta ratio (b.delta-a.delta)/a.delta*100
    return (a=a.delta, b=b.delta, ratio=ratio)
end

println("="^78)
println("   v2 : graphe frais par branche (corrige le biais de réutilisation)")
println("="^78)
results = [bench_fresh(d, dim, e, b) for (d, dim, e, b) in [(16, 512, 4, 4096), (32, 512, 4, 8192)]]

println()
println("="^78)
println("   Résumé")
println("="^78)
for (i, (d, _, _, _)) in enumerate([(16, 512, 4, 4096), (32, 512, 4, 8192)])
    r = results[i]
    verdict = r.ratio < 1.0 ? "GAIN mémoire" : r.ratio ≈ 1.0 ? "parité" : "PIRE que le backward standard"
    @printf "  depth=%-3d : ratio checkpointing/standard = %.3fx  -> %s\n" d r.ratio verdict
end
