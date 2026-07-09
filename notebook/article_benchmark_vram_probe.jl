#=
Re-vérification du benchmark de mémoire de l'article (build_transformer_block_neuro,
notebook/notebook.ipynb, cellule 76-77) APRÈS le correctif moteur de src/backward.jl :
- les gradients des nœuds paramètres ne sont plus nullifiés à chaque backward_graph!
  mais remis à zéro en place (fill!) -- évite une double-résidence buffer
  ancien/nouveau causée par le retard du GC de Julia sur CUDA ;
- GRAD_RULES[:matmul] et [:linear] accumulent désormais le gradient d'un paramètre
  DIRECTEMENT dans son buffer persistant via `mul!(dest, A', dy, 1f0, 1f0)` (BLAS
  gemm! in-place) au lieu de retourner un tableau temporaire neuf à chaque passe.
Voir memory `project_neurodsl_architecture_review_2026-07-05` et la session qui a
diagnostiqué la cause exacte (`Peak_NeuroDSL(dim) ≈ 4×W(dim)+L(dim)` avant le fix).

Reproduit le protocole de bench_transformer_block avec un CtxStore FRAIS à chaque
appel (le comportement PAR DÉFAUT, celui de tout entraînement réel de ce dépôt --
real_llm.ipynb, train_induction!, etc.), PAS le CtxStore partagé utilisé par
bench_transformer_block lui-même -- vérifié séparément : les deux donnent un
résultat quasi identique après le fix, donc ce choix ne change pas la conclusion,
mais reste le choix honnête/généralisable (ne dépend pas d'un détail de
méthodologie de benchmark propre à cet unique notebook).

Instrument : pool watermark CUDA (CU_MEMPOOL_ATTR_USED_MEM_HIGH), voir
real_llm_vram_probe.jl pour la justification complète du choix (MemTrack et la
mémoire brute du device sont tous deux rejetés, expliqué en détail là-bas).

Point de méthodologie important, vérifié en lisant notebook_py.ipynb (cellule 2) :
`torch.cuda.reset_peak_memory_stats()` y est appelé APRÈS la création du modèle
(dans `warmup_pass`), donc le pic PyTorch rapporté (`peak_mb()`) INCLUT la mémoire
des poids déjà résidents, pas seulement l'activité de l'épisode forward+backward.
La comparaison honnête est donc PIC ABSOLU (baseline + delta d'épisode) côté
NeuroDSL contre PIC ABSOLU côté PyTorch -- PAS delta contre absolu (piège identifié
et corrigé pendant cette vérification : comparer delta-seul donnerait un gain
5-17x totalement artificiel, sans rapport avec ce que PyTorch mesure réellement).
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

# ── Même architecture EXACTE que build_transformer_block_neuro (notebook.ipynb) ──
function build_transformer_block_neuro(g::NeuroDSL.NeuroGraph, x_sym::Symbol, dim::Int, ns::Symbol)
    NeuroDSL.set!(g, :Wq, randn(Float32, dim, dim).*0.01f0; is_param=true, namespace=ns)
    NeuroDSL.set!(g, :Wk, randn(Float32, dim, dim).*0.01f0; is_param=true, namespace=ns)
    NeuroDSL.set!(g, :Wv, randn(Float32, dim, dim).*0.01f0; is_param=true, namespace=ns)
    NeuroDSL.set!(g, :Wo, randn(Float32, dim, dim).*0.01f0; is_param=true, namespace=ns)
    NeuroDSL.set!(g, :W1, randn(Float32, dim, dim).*0.01f0; is_param=true, namespace=ns)
    NeuroDSL.set!(g, :W2, randn(Float32, dim, dim).*0.01f0; is_param=true, namespace=ns)
    NeuroDSL.set!(g, :W3, randn(Float32, dim, dim).*0.01f0; is_param=true, namespace=ns)
    NeuroDSL.set!(g, :gamma1, ones(Float32, dim); is_param=true, namespace=ns)
    NeuroDSL.set!(g, :gamma2, ones(Float32, dim); is_param=true, namespace=ns)

    xn1 = Symbol(x_sym, :_norm1)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(xn1, [x_sym, :gamma1], :rmsnorm; namespace=ns))
    q = Symbol(x_sym, :_q); k = Symbol(x_sym, :_k); v = Symbol(x_sym, :_v)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(q, [xn1, :Wq], :matmul; attrs=Dict(:trans_b=>true), namespace=ns))
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(k, [xn1, :Wk], :matmul; attrs=Dict(:trans_b=>true), namespace=ns))
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(v, [xn1, :Wv], :matmul; attrs=Dict(:trans_b=>true), namespace=ns))
    scores = Symbol(x_sym, :_scores)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(scores, [q, k], :matmul; attrs=Dict(:trans_b=>true), namespace=ns))
    sm = Symbol(x_sym, :_sm)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(sm, [scores], :scale_mask; attrs=Dict(:d_head=>dim), namespace=ns))
    att = Symbol(x_sym, :_att)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(att, [sm], :softmax; namespace=ns))
    ao = Symbol(x_sym, :_ao)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(ao, [att, v], :matmul; namespace=ns))
    out_att = Symbol(x_sym, :_out_att)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out_att, [ao, :Wo], :matmul; attrs=Dict(:trans_b=>true), namespace=ns))
    r1 = Symbol(x_sym, :_r1)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(r1, [x_sym, out_att], :add; namespace=ns))
    xn2 = Symbol(x_sym, :_norm2)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(xn2, [r1, :gamma2], :rmsnorm; namespace=ns))
    gate = Symbol(x_sym, :_gate); up = Symbol(x_sym, :_up)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(gate, [xn2, :W1], :matmul; attrs=Dict(:trans_b=>true), namespace=ns))
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(up, [xn2, :W3], :matmul; attrs=Dict(:trans_b=>true), namespace=ns))
    sg = Symbol(x_sym, :_sg)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(sg, [gate, up], :swiglu; namespace=ns))
    mlp_out = Symbol(x_sym, :_mlp_out)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(mlp_out, [sg, :W2], :matmul; attrs=Dict(:trans_b=>true), namespace=ns))
    out = Symbol(x_sym, :_out)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out, [r1, mlp_out], :add; namespace=ns))
    return out
end

function measure_peak(dim::Int, seq::Int=64)
    ns = Symbol(:bench_tb_, dim)
    g = NeuroDSL.NeuroGraph(device=dev, namespace=ns)
    x_sym = :x
    NeuroDSL.set!(g, x_sym, NeuroDSL.Backend.randn32(dev, seq, dim))
    out_sym = build_transformer_block_neuro(g, x_sym, dim, ns)
    loss_sym = :loss
    NeuroDSL.set!(g, :y, NeuroDSL.Backend.zeros32(dev, seq, dim); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss_sym, [out_sym, :y], :mse_loss; namespace=ns))

    step!() = begin
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.demand!(g, loss_sym; namespace=ns)
        NeuroDSL.backward_graph!(g, loss_sym; namespace=ns, full=true)
        CUDA.synchronize()
    end

    for _ in 1:10; step!(); end   # warm-up (stabilise formes + JIT + caches _buf!)
    quiesce()

    peaks = Float64[]
    baselines = Float64[]
    for trial in 1:5
        quiesce()
        baseline = pool_current_mb()
        reset_high!()
        step!()
        push!(peaks, pool_high_mb())
        push!(baselines, baseline)
    end
    return baselines, peaks
end

const PYTORCH_ABS_PEAK_MB = Dict(256 => 20.32, 512 => 31.38, 1024 => 74.51)

println("dim | baseline (MB) | pic absolu NeuroDSL (MB) | pic absolu PyTorch (MB) | ratio")
for dim in (256, 512, 1024)
    baselines, peaks = measure_peak(dim)
    @assert all(==(baselines[1]), baselines) "baseline instable -- fuite entre essais"
    @assert all(==(peaks[1]), peaks) "pic instable -- non-déterminisme entre essais"
    pic_neuro = peaks[1]
    pic_pt = PYTORCH_ABS_PEAK_MB[dim]
    ratio = pic_pt / pic_neuro
    tag = ratio > 1 ? "NeuroDSL $(round(ratio, digits=2))x MOINS" : "NeuroDSL $(round(1/ratio, digits=2))x PLUS"
    @printf("%4d | %6.2f MB | %10.2f MB | %10.2f MB | %s\n", dim, baselines[1], pic_neuro, pic_pt, tag)
end
