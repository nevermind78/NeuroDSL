# ══════════════════════════════════════════════════════════════════════════════
# diag_tied_embedding_alias.jl — vérifie et corrige la double matérialisation
# de tok_E/lm_head_W (poids liés, tie_word_embeddings=true côté HF Qwen2.5).
#
# Confirmé AVANT ce script par inspection du manifeste sur disque
# (qwen2_neurodsl.json) : tok_E et lm_head_W sont deux blobs SÉPARÉS de
# 933 494 784 octets (151936×1536×4, ~890 MiB chacun) à des offsets
# différents du .bin -- save_graph! (src/serialization.jl) itère par
# SYMBOLE et n'a aucune notion d'aliasing entre deux noms différents
# pointant vers le même tableau Julia au moment de la sauvegarde
# (load_qwen2.jl faisait bien `set!(g,:lm_head_W,tok_W;...)` avec le MÊME
# objet, mais ça se perd au roundtrip disque). load_graph! recrée donc
# deux CuArray indépendants en VRAM -- ~0.9 Go gaspillés.
#
# Correctif : `NeuroDSL.alias_tied_param!(g, ns, :tok_E, :lm_head_W)`
# (src/graph_api.jl) -- appelé UNE FOIS juste après load_graph!, vérifie
# l'égalité numérique puis réassigne :lm_head_W.value au MÊME objet que
# :tok_E, libère l'ancien buffer GPU.
#
# CE SCRIPT couvre les DEUX process séparés (avant/après) comme la
# méthodologie établie plus tôt cette session l'exige (mesurer les deux
# phases dans le MÊME process fausse la comparaison VRAM) -- passer
# `ARGS[1] == "before"` ou `"after"`.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, CUDA, JSON

mode = length(ARGS) >= 1 ? ARGS[1] : "after"
mode in ("before", "after") || error("usage: julia diag_tied_embedding_alias.jl [before|after]")

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

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6

results = Any[]
_log("Mode = $mode")
push!(results, checkpoint("0_avant_tout"))

dev = NeuroDSL.Backend.CUDADevice()
ns = :qwen2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
out_sym = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                               batched_attn=true, n_kv_heads=N_KV_HEADS,
                               qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out_sym, :final_norm; namespace=ns)
logits_sym = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)
push!(results, checkpoint("1_graphe_construit_poids_aleatoires"))

NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
push!(results, checkpoint("2_apres_load_graph"))

same_object_before = g.nodes[ns][:tok_E].value === g.nodes[ns][:lm_head_W].value
_log("  tok_E === lm_head_W (même objet Julia) AVANT correctif : $same_object_before")

if mode == "after"
    aliased = NeuroDSL.alias_tied_param!(g, ns, :tok_E, :lm_head_W)
    _log("  alias_tied_param!(:tok_E, :lm_head_W) -> $aliased")
    push!(results, checkpoint("3_apres_alias_tied_param"))
    same_object_after = g.nodes[ns][:tok_E].value === g.nodes[ns][:lm_head_W].value
    _log("  tok_E === lm_head_W (même objet Julia) APRÈS correctif : $same_object_after")
else
    push!(results, checkpoint("3_pas_de_correctif_applique"))
end

# Passage avant sur un prompt court, pour vérifier que les logits sont
# corrects après l'aliasing (pas seulement que la VRAM a baissé).
NeuroDSL.set!(g, :token_ids, [1,5,10,20,3,7], atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
out = Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))
push!(results, checkpoint("4_apres_forward_pass_test"))

peak = maximum(r.smi_mb for r in results)
println("\n", "═"^78)
println("RÉSUMÉ ($mode)")
println("═"^78)
for r in results
    println("  $(rpad(r.label,45))  smi=$(round(r.smi_mb,digits=0))MB  pool_high=$(round(r.pool_high_mb,digits=0))MB")
end
println("PIC VRAM ($mode) : $(round(peak,digits=0)) MB")
println("Logits forme : $(size(out)), somme=$(round(sum(out),digits=3)), premières valeurs (dernière ligne) : $(round.(out[end,1:5],digits=4))")

open(joinpath(@__DIR__, "diag_tied_embedding_alias_$(mode)_results.json"), "w") do io
    JSON.print(io, Dict(
        "mode"=>mode,
        "results"=>[Dict("label"=>r.label, "smi_mb"=>r.smi_mb, "pool_current_mb"=>r.pool_current_mb, "pool_high_mb"=>r.pool_high_mb) for r in results],
        "peak_vram_mb"=>peak,
        "same_object_before"=>same_object_before,
        "logits_last_row"=>Float64.(out[end, :]),
    ), 2)
end
_log("Écrit -> diag_tied_embedding_alias_$(mode)_results.json")
flush(stdout)
