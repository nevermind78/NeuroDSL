# ══════════════════════════════════════════════════════════════════════════════
# diag_gpu_clock_during_decode.jl — mesure DIRECTE de l'état d'horloge GPU
# pendant un décodage réel, pour trancher : le "coût embedding=180-250ms"
# trouvé par diag_decode_op_breakdown.jl (constant, reproductible même après
# GC.gc()+reclaim() forcé, alors que l'op :embedding elle-même est triviale
# -- un seul gather d'une ligne) est-il en fait le coût de RAMPAGE D'HORLOGE
# GPU après le trou d'inactivité CPU-seul entre deux pas de décodage
# (argmax + plusieurs set! + invalidate_all!, AUCUN travail GPU) ?
#
# Écrit des horodatages MONOTONES (time_ns()) pour chaque pas de décodage
# dans decode_timestamps.log ; en parallèle, `nvidia-smi --loop-ms=25`
# (lancé par le script appelant, PAS ici -- Julia ne peut pas lancer un
# process détaché en arrière-plan facilement sur Windows) écrit
# clock_trace.csv. Les deux fichiers sont ensuite corrélés par horodatage.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, CUDA

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=2))s] ", msg); flush(stdout))

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6

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
_log("Graphe construit (poids aléatoires).")

NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
NeuroDSL.alias_tied_param!(g, ns, :tok_E, :lm_head_W)
dec_logits = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns)
_log("Poids réels chargés, cache KV construit.")

const PYTHON_ENV = raw"C:\Users\Nevermind\anaconda3\envs\neurodsl_llm_check\python.exe"
const TOKENIZER_HELPER = joinpath(@__DIR__, "qwen2_tokenizer_helper.py")
function _call_helper(req::Dict)
    input_json = JSON.json(req)
    out = IOBuffer()
    run(pipeline(`$PYTHON_ENV $TOKENIZER_HELPER`, stdin=IOBuffer(input_json), stdout=out))
    return JSON.parse(String(take!(out)))
end
encode_chat(messages) = Int.(_call_helper(Dict("action"=>"encode_chat", "messages"=>messages))["ids"])

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
history = Dict{String,Any}[Dict("role"=>"user", "content"=>PROMPT)]
prefix = encode_chat(history) .+ 1
logits_row = batched_prefix_pass!(prefix)
cur_step = length(prefix)
_log("Préfixe traité ($(length(prefix)) tok). Marqueur GPU_TRACE_START.")
println("GPU_TRACE_START wall=$(time())")
flush(stdout)

# ── Génère N_GEN tokens, en loggant pour CHAQUE pas : timestamp epoch juste
#    AVANT le premier appel GPU du pas (fin du travail CPU-seul), timestamp
#    juste APRÈS (fin du pas, incluant le Array() qui force la sync), et la
#    durée totale. Permet de corréler avec clock_trace.csv (nvidia-smi,
#    lancé en parallèle par le script appelant). ─────────────────────────
const N_GEN = 40
open(joinpath(@__DIR__, "diag_gpu_clock_during_decode_steps.csv"), "w") do io
    println(io, "step,cur_step,t_before_wall,t_after_wall,dur_s")
    for step in 1:N_GEN
        nxt0 = argmax(logits_row) - 1
        global cur_step += 1
        t_before = time()
        global logits_row = cached_step!(nxt0, cur_step)
        t_after = time()
        println(io, "$step,$cur_step,$t_before,$t_after,$(t_after-t_before)")
        flush(io)
    end
end
println("GPU_TRACE_END wall=$(time())")
_log("Terminé -> diag_gpu_clock_during_decode_steps.csv")
