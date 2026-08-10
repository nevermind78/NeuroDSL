#=
Étape 2 du plan (validation backward_sparse à l'échelle) -- critère de
succès FIXÉ AVANT LE LANCEMENT (voir conversation) : prune_frozen=true doit
réduire le temps d'horloge ET le pic mémoire d'AU MOINS 10% chacun par
rapport à prune_frozen=false, avec ≥50% des couches gelées, reproductible
sur 3 lancements indépendants à <5% de variance. Résultat réel rapporté
tel quel, positif/nul/négatif.

Fixture : LlamaModel à échelle réelle (pas un réseau jouet D=4), comparable
en esprit au fixture GPT-2-small (85M params, 3000 pas, déjà validé stable
dans une session précédente) -- dim=512, n_heads=8, hidden_dim=1376,
n_layers=16, seq_len=128, ~50M paramètres. Moitié basse des couches gelée
(is_param=false), moitié haute entraînable.
=#
using NeuroDSL, CUDA, Random, Printf, Statistics

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

# n_frozen_layers : combien des n_layers couches (à partir du DÉBUT, comme un
# préfixe de fine-tuning partiel classique) sont gelées -- créées avec
# is_param=false au lieu de is_param=true.
function build_llama_partial_freeze(; n_layers, dim, n_heads, hidden_dim, seq_len,
                                      n_frozen_layers, ns::Symbol)
    g = NeuroDSL.NeuroGraph(device=dev, namespace=ns)
    Random.seed!(42)
    NeuroDSL.set!(g, :x0, randn(Float32, seq_len, dim) .* 0.02f0; namespace=ns)
    h = :x0
    d_head = dim ÷ n_heads
    for l in 1:n_layers
        frozen = l <= n_frozen_layers
        prefix = Symbol(:layer_, l)
        # Reprend le câblage de LlamaBlock à la main pour contrôler is_param
        # par couche (LlamaModel(...) marque tout is_param=true par défaut).
        wq = Symbol(prefix, :_mha_q_W); wk = Symbol(prefix, :_mha_k_W)
        wv = Symbol(prefix, :_mha_v_W); wo = Symbol(prefix, :_mha_output_W)
        g1 = Symbol(prefix, :_norm1_gamma); g2 = Symbol(prefix, :_norm2_gamma)
        w1 = Symbol(prefix, :_mlp_w1); w2 = Symbol(prefix, :_mlp_w2); w3 = Symbol(prefix, :_mlp_w3)
        isp = !frozen
        NeuroDSL.set!(g, wq, randn(Float32, dim, dim).*0.02f0; is_param=isp, namespace=ns)
        NeuroDSL.set!(g, wk, randn(Float32, dim, dim).*0.02f0; is_param=isp, namespace=ns)
        NeuroDSL.set!(g, wv, randn(Float32, dim, dim).*0.02f0; is_param=isp, namespace=ns)
        NeuroDSL.set!(g, wo, randn(Float32, dim, dim).*0.02f0; is_param=isp, namespace=ns)
        NeuroDSL.set!(g, g1, ones(Float32, dim); is_param=isp, namespace=ns)
        NeuroDSL.set!(g, g2, ones(Float32, dim); is_param=isp, namespace=ns)
        NeuroDSL.set!(g, w1, randn(Float32, hidden_dim, dim).*0.02f0; is_param=isp, namespace=ns)
        NeuroDSL.set!(g, w2, randn(Float32, dim, hidden_dim).*0.02f0; is_param=isp, namespace=ns)
        NeuroDSL.set!(g, w3, randn(Float32, hidden_dim, dim).*0.02f0; is_param=isp, namespace=ns)

        n1 = Symbol(prefix, :_norm1)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(n1, [h, g1], :rmsnorm; namespace=ns))
        q = Symbol(prefix, :_q); k = Symbol(prefix, :_k); v = Symbol(prefix, :_v)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(q, [n1, wq], :matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(k, [n1, wk], :matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(v, [n1, wv], :matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=ns))
        sc = Symbol(prefix, :_sc)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(sc, [q, k], :matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=ns))
        scm = Symbol(prefix, :_scm)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(scm, [sc], :scale_mask; attrs=Dict{Symbol,Any}(:d_head=>d_head), namespace=ns))
        pr = Symbol(prefix, :_pr)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(pr, [scm], :softmax; namespace=ns))
        ao = Symbol(prefix, :_ao)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(ao, [pr, v], :matmul; namespace=ns))
        attn_out = Symbol(prefix, :_attn_out)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(attn_out, [ao, wo], :matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=ns))
        res1 = Symbol(prefix, :_res1)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(res1, [h, attn_out], :add; namespace=ns))

        n2 = Symbol(prefix, :_norm2)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(n2, [res1, g2], :rmsnorm; namespace=ns))
        gate = Symbol(prefix, :_gate); up = Symbol(prefix, :_up)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(gate, [n2, w1], :matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(up, [n2, w3], :matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=ns))
        swi = Symbol(prefix, :_swiglu)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(swi, [gate, up], :swiglu; namespace=ns))
        down = Symbol(prefix, :_down)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(down, [swi, w2], :matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=ns))
        res2 = Symbol(prefix, :_res2)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(res2, [res1, down], :add; namespace=ns))
        h = res2
    end
    NeuroDSL.set!(g, :target, randn(Float32, seq_len, dim) .* 0.02f0; namespace=ns)
    loss_sym = :loss
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss_sym, [h, :target], :mse_loss; namespace=ns))
    return g, loss_sym
end

CFG = (n_layers=16, dim=512, n_heads=8, hidden_dim=1376, seq_len=128, n_frozen_layers=8)
n_params = CFG.n_layers * (4*CFG.dim^2 + 3*CFG.dim*CFG.hidden_dim + 2*CFG.dim)
@printf "Config : n_layers=%d dim=%d n_heads=%d hidden_dim=%d seq_len=%d  (~%.1fM params, %d/%d couches gelées)\n" CFG.n_layers CFG.dim CFG.n_heads CFG.hidden_dim CFG.seq_len (n_params/1e6) CFG.n_frozen_layers CFG.n_layers

function measure(tag, sparse::Bool, prune_frozen::Bool)
    ns = Symbol(:bs_, tag)
    g, loss_sym = build_llama_partial_freeze(; CFG..., ns=ns)
    ctx = NeuroDSL.CtxStore()
    NeuroDSL.demand!(g, loss_sym; ctx_store=ctx, namespace=ns)  # warm-up JIT
    NeuroDSL.backward_graph!(g, loss_sym; ctx_store=ctx, namespace=ns, sparse=sparse, prune_frozen=prune_frozen)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, loss_sym; ctx_store=ctx, namespace=ns)
    NeuroDSL.backward_graph!(g, loss_sym; ctx_store=ctx, namespace=ns, sparse=sparse, prune_frozen=prune_frozen)
    CUDA.synchronize()

    times = Float64[]
    mems = Float64[]
    for _ in 1:10
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.demand!(g, loss_sym; ctx_store=ctx, namespace=ns)
        m = peak_mem() do
            NeuroDSL.backward_graph!(g, loss_sym; ctx_store=ctx, namespace=ns, sparse=sparse, prune_frozen=prune_frozen)
            CUDA.synchronize()
        end
        push!(mems, m.delta)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.demand!(g, loss_sym; ctx_store=ctx, namespace=ns)
        t = @elapsed begin
            NeuroDSL.backward_graph!(g, loss_sym; ctx_store=ctx, namespace=ns, sparse=sparse, prune_frozen=prune_frozen)
            CUDA.synchronize()
        end
        push!(times, t)
    end
    g = nothing; GC.gc(true); CUDA.reclaim()
    return times, mems
end

println()
for (tag, sparse, prune) in [("dense", false, false), ("dense_prune", false, true),
                              ("sparse_no_prune", true, false), ("sparse_prune", true, true)]
    times, mems = measure(tag, sparse, prune)
    tvar = (maximum(times)-minimum(times))/median(times)*100
    mvar = maximum(mems)==0 ? 0.0 : (maximum(mems)-minimum(mems))/max(median(mems),1e-6)*100
    @printf "  [%-16s] temps médian=%.4f s (min=%.4f max=%.4f, variance %.1f%%)   pic mémoire médian=%.2f MB (variance %.1f%%)\n" tag median(times) minimum(times) maximum(times) tvar median(mems) mvar
end
