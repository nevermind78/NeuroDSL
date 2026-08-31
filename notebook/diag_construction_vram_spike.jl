# ══════════════════════════════════════════════════════════════════════════════
# diag_construction_vram_spike.jl — localise l'origine du pic transitoire
# pool_high_watermark=9076MB vs pool_current=6779MB observé dans
# diag_tied_embedding_alias.jl (avant même load_graph! -- déjà présent après
# la seule construction du graphe à poids ALÉATOIRES). Checkpoints fins pour
# savoir si le pic vient du JIT CUDA (premiers lancements de kernel), de la
# génération des grandes matrices aléatoires (tok_E/lm_head_W, 151936x1536),
# ou de l'empilement des 28 couches.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, CUDA

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=2))s] ", msg); flush(stdout))

const _POOL = CUDA.CUDACore.pool_create(CUDA.CUDACore.active_state().device)
pool_current_mb() = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_CURRENT)) / 1024^2
pool_high_mb()     = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH)) / 1024^2
quiesce()          = (CUDA.synchronize(); GC.gc(true); GC.gc(true); CUDA.synchronize())

function nvidia_smi_used_mb()
    out = try
        read(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`, String)
    catch
        return NaN
    end
    parse(Float64, strip(split(strip(out), '\n')[1]))
end

function checkpoint(label::String)
    quiesce()
    smi = nvidia_smi_used_mb()
    cur = pool_current_mb()
    hi  = pool_high_mb()
    _log("  [checkpoint] $label : nvidia-smi=$(round(smi,digits=1))MB  pool_current=$(round(cur,digits=1))MB  pool_high_watermark=$(round(hi,digits=1))MB")
    return (label=label, smi_mb=smi, pool_current_mb=cur, pool_high_mb=hi)
end

const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6

results = Any[]
push!(results, checkpoint("00_avant_tout"))

dev = NeuroDSL.Backend.CUDADevice()
# Force la création du contexte CUDA + JIT d'un premier kernel trivial, SANS
# rien construire côté NeuroDSL -- isole le coût "warmup CUDA pur".
warmup = CUDA.rand(Float32, 8, 8) * CUDA.rand(Float32, 8, 8)
push!(results, checkpoint("01_apres_contexte_cuda_et_1_kernel_trivial"))

ns = :spike
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
push!(results, checkpoint("02_apres_token_ids"))

tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
push!(results, checkpoint("03_apres_embedding_tok_E_151936x1536"))

out_sym = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                               batched_attn=true, n_kv_heads=N_KV_HEADS,
                               qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
push!(results, checkpoint("04_apres_28_couches_llamamodel"))

final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out_sym, :final_norm; namespace=ns)
push!(results, checkpoint("05_apres_layernorm_finale"))

logits_sym = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)
push!(results, checkpoint("06_apres_linear_lm_head_151936x1536"))

# Un premier forward pass RÉEL avec ces poids aléatoires -- teste si le pic
# vient plutôt de la COMPILATION des kernels d'attention/matmul au premier
# usage (JIT différé jusqu'au premier appel), pas de la construction en soi.
NeuroDSL.invalidate_all!(g; namespace=ns)
out = Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))
push!(results, checkpoint("07_apres_premier_forward_pass_poids_aleatoires"))

println("\n", "═"^78)
println("RÉSUMÉ")
println("═"^78)
prev_hi = 0.0
for r in results
    delta_hi = r.pool_high_mb - prev_hi
    println("  $(rpad(r.label,50))  smi=$(round(r.smi_mb,digits=0))MB  pool_cur=$(round(r.pool_current_mb,digits=0))MB  pool_high=$(round(r.pool_high_mb,digits=0))MB  (Δhigh=+$(round(delta_hi,digits=0))MB)")
    global prev_hi = r.pool_high_mb
end
flush(stdout)
