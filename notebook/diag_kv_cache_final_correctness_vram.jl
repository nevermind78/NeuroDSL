# ══════════════════════════════════════════════════════════════════════════════
# diag_kv_cache_final_correctness_vram.jl — chemin de production FINAL après
# la passe complète de cette session (revue algorithmique du cache KV) :
#   1. Namespace UNIQUE (correctif déjà en place, 2026-08-31 plus tôt).
#   2. `alias_tied_param!(g, ns, :tok_E, :lm_head_W)` -- élimine la double
#      matérialisation du poids lié embedding/lm_head (~0.9 Go).
#   3. `load_graph!` libère désormais explicitement les anciens buffers avant
#      d'en recréer (`src/serialization.jl`) -- hygiène mémoire déterministe.
#   4. `LlamaModel` fait un nettoyage périodique (GC.gc()+reclaim! toutes les
#      4 couches) pendant la construction du graphe à poids ALÉATOIRES
#      (`src/layers.jl`) -- élimine ~3.2 Go de pic transitoire mesuré
#      (notebook/diag_construction_vram_spike.jl / _after_run.log).
#
# MÊME prompt / MÊME méthodologie que
# notebook/diag_kv_cache_double_alloc_phaseB.jl (référence "avant cette
# passe") -- process isolé, mêmes checkpoints VRAM, même trace de logits par
# pas -- pour une comparaison directe, chiffre pour chiffre.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, CUDA

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
_log("Chemin de production FINAL (namespace unique + alias_tied_param! + load_graph!/LlamaModel corrigés)")
push!(results, checkpoint("F0_avant_tout"))

dev = NeuroDSL.Backend.CUDADevice()
ns = :qwen2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
out_sym = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                               batched_attn=true, n_kv_heads=N_KV_HEADS,
                               qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out_sym, :final_norm; namespace=ns)
logits_load = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)
push!(results, checkpoint("F1_graphe_construit_poids_aleatoires"))

NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
push!(results, checkpoint("F2_apres_load_graph"))

aliased = NeuroDSL.alias_tied_param!(g, ns, :tok_E, :lm_head_W)
_log("  alias_tied_param!(:tok_E, :lm_head_W) -> $aliased")
push!(results, checkpoint("F3_apres_alias_tied_param"))

dec_logits = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns)
push!(results, checkpoint("F4_apres_construction_graphe_cache"))

const PYTHON_ENV = raw"C:\Users\Nevermind\anaconda3\envs\neurodsl_llm_check\python.exe"
const TOKENIZER_HELPER = joinpath(@__DIR__, "qwen2_tokenizer_helper.py")
function _call_helper(req::Dict)
    input_json = JSON.json(req)
    out = IOBuffer()
    run(pipeline(`$PYTHON_ENV $TOKENIZER_HELPER`, stdin=IOBuffer(input_json), stdout=out))
    return JSON.parse(String(take!(out)))
end
encode_chat(messages) = Int.(_call_helper(Dict("action"=>"encode_chat", "messages"=>messages))["ids"])
decode_ids(ids::Vector{Int}) = _call_helper(Dict("action"=>"decode", "ids"=>ids))["text"]
const EOS_ID = _call_helper(Dict("action"=>"eos_id"))["id"]
_log("EOS_ID = $EOS_ID.")

function batched_prefix_pass!(prefix1idx::Vector{Int})
    NeuroDSL.set!(g, :token_ids, prefix1idx; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    out = Array(NeuroDSL.demand!(g, logits_load; namespace=ns))
    NeuroDSL.prime_kv_cache_from_prefix!(g; src_ns=ns, dst_ns=ns,
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, use_rope=true)
    return Float32.(out[end, :])
end
function cached_step!(tok0::Int, cur_step::Int)
    NeuroDSL.set!(g, :dec_token_id, [tok0 + 1]; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :dec_cur_step, Float32[cur_step]; namespace=ns)
    NeuroDSL.set!(g, :dec_pos, Float32[cur_step-1]; namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, dec_logits; namespace=ns))[1, :]
end

const PROMPT = "Explain in one short sentence why the sky is blue."
const N_GEN = 40
history = Dict{String,Any}[Dict("role"=>"user", "content"=>PROMPT)]
prefix = encode_chat(history) .+ 1

t_prefix0 = time()
logits_row = batched_prefix_pass!(prefix)
t_prefix = time() - t_prefix0
push!(results, checkpoint("F5_apres_passage_prefixe"))
cur_step = length(prefix)
gen = Int[]
logit_trace = Vector{Float32}[]
gen_times = Float64[]
for step in 1:N_GEN
    nxt0 = argmax(logits_row) - 1
    nxt0 == EOS_ID && break
    push!(gen, nxt0)
    push!(logit_trace, copy(logits_row))
    global cur_step += 1
    ts = time()
    global logits_row = cached_step!(nxt0, cur_step)
    push!(gen_times, time() - ts)
    if step in (1, 10, 40)
        push!(results, checkpoint("F6_apres_$(step)_tokens_generes"))
    end
end
reply = decode_ids(gen)
_log("Réponse générée ($(length(gen)) tokens) : $(repr(reply))")
push!(results, checkpoint("F7_fin"))

steady = length(gen_times) > 3 ? gen_times[4:end] : gen_times
avg_steady = isempty(steady) ? NaN : sum(steady)/length(steady)
peak = maximum(r.smi_mb for r in results)
println("\n", "═"^78)
println("RÉSUMÉ FINAL")
println("═"^78)
for r in results
    println("  $(rpad(r.label,45))  smi=$(round(r.smi_mb,digits=0))MB  pool_high=$(round(r.pool_high_mb,digits=0))MB")
end
println("Préfixe ($(length(prefix)) tok) : $(round(t_prefix,digits=3))s")
println("Génération : 1er token=$(round(gen_times[1],digits=4))s  moy(régime établi)=$(round(avg_steady,digits=4))s/tok")
println("PIC VRAM FINAL : $(round(peak,digits=0)) MB")

open(joinpath(@__DIR__, "diag_kv_cache_final_correctness_vram_results.json"), "w") do io
    JSON.print(io, Dict(
        "results"=>[Dict("label"=>r.label, "smi_mb"=>r.smi_mb, "pool_current_mb"=>r.pool_current_mb, "pool_high_mb"=>r.pool_high_mb) for r in results],
        "gen"=>gen, "reply"=>reply,
        "logit_trace"=>[Float64.(v) for v in logit_trace],
        "t_prefix"=>t_prefix, "gen_times"=>gen_times, "avg_steady"=>avg_steady,
        "peak_vram_mb"=>peak,
    ), 2)
end
_log("Écrit -> diag_kv_cache_final_correctness_vram_results.json")
flush(stdout)
