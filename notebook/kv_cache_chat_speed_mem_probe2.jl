# Mesure vitesse (passage batché du préfixe + génération token par token) ET
# pic mémoire GPU (watermark CUDA, même méthodologie que le reste de la
# session) pour le chat Qwen -- adapté de kv_cache_chat_timing_probe_warm.jl
# pour comparer avant/après le correctif demand!/_ancestors_of! (ancêtres
# réels au lieu du préfixe topologique complet).

using NeuroDSL, JSON, CUDA

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=2))s] ", msg); flush(stdout))

const _POOL = CUDA.CUDACore.pool_create(CUDA.CUDACore.active_state().device)
pool_current_mb() = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_CURRENT)) / 1024^2
pool_high_mb()     = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH)) / 1024^2
reset_high!()      = CUDA.attribute!(_POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
quiesce()          = (CUDA.synchronize(); GC.gc(true); GC.gc(true); CUDA.synchronize())

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6

_log("Construction + chargement...")
dev = NeuroDSL.Backend.CUDADevice()
ns_load = :qwen2_load
ns_chat = :qwen2_chat
g = NeuroDSL.NeuroGraph(namespace=ns_load, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns_load)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns_load)
out_sym = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                               batched_attn=true, n_kv_heads=N_KV_HEADS,
                               qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns_load)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out_sym, :final_norm; namespace=ns_load)
logits_load = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns_load)
NeuroDSL.load_graph!(g, ns_load, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
n_copied = NeuroDSL.copy_params_to_namespace!(g, ns_load, ns_chat)
dec_logits = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns_chat)
_log("Prêt ($n_copied paramètres).")

function cached_step!(g, tok0::Int, cur_step::Int)
    NeuroDSL.set!(g, :dec_token_id, [tok0 + 1]; atom_type=NeuroDSL.Datom, namespace=ns_chat)
    NeuroDSL.set!(g, :dec_cur_step, Float32[cur_step]; namespace=ns_chat)
    NeuroDSL.set!(g, :dec_pos, Float32[cur_step-1]; namespace=ns_chat)
    NeuroDSL.invalidate_all!(g; namespace=ns_chat)
    return Array(NeuroDSL.demand!(g, dec_logits; namespace=ns_chat))[1, :]
end

function batched_prefix_pass!(g, prefix1idx::Vector{Int})
    t0 = time()
    NeuroDSL.set!(g, :token_ids, prefix1idx; atom_type=NeuroDSL.Datom, namespace=ns_load)
    NeuroDSL.invalidate_all!(g; namespace=ns_load)
    out = Array(NeuroDSL.demand!(g, logits_load; namespace=ns_load))
    t_fwd = time() - t0
    t1 = time()
    NeuroDSL.prime_kv_cache_from_prefix!(g; src_ns=ns_load, dst_ns=ns_chat,
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, use_rope=true)
    t_prime = time() - t1
    return Float32.(out[end, :]), t_fwd, t_prime
end

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

function run_turn!(label::String, prompt::String; max_new_tokens=40, measure_mem=false)
    history = Dict{String,Any}[Dict("role"=>"user", "content"=>prompt)]
    ids0 = encode_chat(history)
    prefix = ids0 .+ 1

    if measure_mem
        quiesce(); reset_high!()
    end
    logits_row, t_fwd, t_prime = batched_prefix_pass!(g, prefix)
    cur_step = length(prefix)
    gen0 = Int[]
    gen_times = Float64[]
    for step in 1:max_new_tokens
        nxt0 = argmax(logits_row) - 1
        nxt0 == EOS_ID && break
        push!(gen0, nxt0)
        cur_step += 1
        ts = time()
        logits_row = cached_step!(g, nxt0, cur_step)
        push!(gen_times, time() - ts)
    end
    mem_peak = measure_mem ? (quiesce(); pool_high_mb()) : NaN
    reply = decode_ids(gen0)
    steady = length(gen_times) > 2 ? gen_times[3:end] : gen_times
    avg_steady = isempty(steady) ? NaN : sum(steady)/length(steady)
    _log("  [$label] préfixe=$(length(prefix))tok fwd=$(round(t_fwd,digits=3))s amorce=$(round(t_prime,digits=3))s | " *
         "gen=$(length(gen0))tok 1er=$(round(gen_times[1],digits=4))s moy(régime établi)=$(round(avg_steady,digits=4))s/tok" *
         (measure_mem ? " | pic VRAM=$(round(mem_peak,digits=1))MB" : ""))
    return (t_fwd=t_fwd, t_prime=t_prime, n_prefix=length(prefix), n_gen=length(gen0),
            gen_times=gen_times, avg_steady=avg_steady, mem_peak=mem_peak, reply=reply)
end

# Warm-up : 2 tours pour amortir tout JIT CUDA restant
run_turn!("warmup1", "What is the capital of Egypt?"; max_new_tokens=20)
run_turn!("warmup2", "What is the capital of Japan?"; max_new_tokens=20)

# Mesures : 3 tours chauds, dont un avec suivi mémoire
r1 = run_turn!("mesure1", "Write a haiku about the ocean."; max_new_tokens=40)
r2 = run_turn!("mesure2 (+VRAM)", "What is the minimum of 2, 3, 15, 0?"; max_new_tokens=40, measure_mem=true)
r3 = run_turn!("mesure3", "Tell me a short fact about Mars.", max_new_tokens=40)

println("\n", "═"^74)
println("RÉSUMÉ")
println("═"^74)
for (label, r) in [("mesure1", r1), ("mesure2", r2), ("mesure3", r3)]
    println("  $label : préfixe fwd=$(round(r.t_fwd,digits=4))s | régime établi=$(round(r.avg_steady,digits=4))s/tok" *
            (isnan(r.mem_peak) ? "" : " | pic VRAM=$(round(r.mem_peak,digits=1))MB"))
end
flush(stdout)
