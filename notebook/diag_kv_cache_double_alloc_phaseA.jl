# ══════════════════════════════════════════════════════════════════════════════
# diag_kv_cache_double_alloc_phaseA.jl — PHASE A seule, PROCESS ISOLÉ.
#
# Scindé de `diag_kv_cache_double_alloc.jl` après avoir constaté que mesurer A
# et B dans LE MÊME process Julia fausse la comparaison : le graphe de la
# phase A (gA, ~13.8 Go) n'est jamais libéré avant que la phase B ne
# commence à construire son propre graphe -- la VRAM de B s'empile sur celle
# de A au lieu de partir d'une base propre (mesuré : B affichait à tort un
# pic PLUS HAUT que A, alors que B alloue structurellement MOINS de poids).
# Deux process séparés, lancés l'un après l'autre, chacun parti de VRAM
# quasi nulle (contexte CUDA seul), donnent la vraie comparaison.
#
# Reproduit le chemin de PRODUCTION ACTUEL (`notebook/qwen2.ipynb`,
# `kv_cache_chat_speed_mem_probe2.jl`) : DEUX namespaces (`ns_load` pour le
# recalcul complet + passage avant batché du préfixe, `ns_chat` pour le
# décodage incrémental caché), poids copiés PAR VALEUR de l'un vers l'autre
# via `copy_params_to_namespace!` (`src/graph_api.jl:360`).
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
_log("PHASE A (process isolé) -- chemin de production actuel (2 namespaces, poids dupliqués)")
push!(results, checkpoint("A0_avant_tout"))

devA = NeuroDSL.Backend.CUDADevice()
ns_load = :A_load
ns_chat = :A_chat
gA = NeuroDSL.NeuroGraph(namespace=ns_load, device=devA)
NeuroDSL.set!(gA, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns_load)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(gA, :token_ids, :tok; namespace=ns_load)
out_sym = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                               batched_attn=true, n_kv_heads=N_KV_HEADS,
                               qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(gA, tok_emb; namespace=ns_load)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(gA, out_sym, :final_norm; namespace=ns_load)
logits_load = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(gA, final_norm, :lm_head; namespace=ns_load)
push!(results, checkpoint("A1_graphe_construit_poids_aleatoires"))

NeuroDSL.load_graph!(gA, ns_load, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
push!(results, checkpoint("A2_apres_chargement_poids_reels_1_copie"))

n_copied = NeuroDSL.copy_params_to_namespace!(gA, ns_load, ns_chat)
_log("  $n_copied paramètres copiés vers :$ns_chat (2e copie complète).")
push!(results, checkpoint("A3_apres_copy_params_to_namespace_2e_copie"))

dec_logits_A = NeuroDSL.build_cached_decode_graph!(gA;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns_chat)
push!(results, checkpoint("A4_apres_construction_graphe_cache"))

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

function batched_prefix_pass_A!(prefix1idx::Vector{Int})
    NeuroDSL.set!(gA, :token_ids, prefix1idx; atom_type=NeuroDSL.Datom, namespace=ns_load)
    NeuroDSL.invalidate_all!(gA; namespace=ns_load)
    out = Array(NeuroDSL.demand!(gA, logits_load; namespace=ns_load))
    NeuroDSL.prime_kv_cache_from_prefix!(gA; src_ns=ns_load, dst_ns=ns_chat,
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, use_rope=true)
    return Float32.(out[end, :])
end
function cached_step_A!(tok0::Int, cur_step::Int)
    NeuroDSL.set!(gA, :dec_token_id, [tok0 + 1]; atom_type=NeuroDSL.Datom, namespace=ns_chat)
    NeuroDSL.set!(gA, :dec_cur_step, Float32[cur_step]; namespace=ns_chat)
    NeuroDSL.set!(gA, :dec_pos, Float32[cur_step-1]; namespace=ns_chat)
    NeuroDSL.invalidate_all!(gA; namespace=ns_chat)
    return Array(NeuroDSL.demand!(gA, dec_logits_A; namespace=ns_chat))[1, :]
end

const PROMPT = "Explain in one short sentence why the sky is blue."
const N_GEN = 40
history = Dict{String,Any}[Dict("role"=>"user", "content"=>PROMPT)]
prefix = encode_chat(history) .+ 1

t_prefix0 = time()
logits_row = batched_prefix_pass_A!(prefix)
t_prefix = time() - t_prefix0
push!(results, checkpoint("A5_apres_passage_prefixe"))
cur_step = length(prefix)
gen_A = Int[]
logit_trace_A = Vector{Float32}[]
gen_times = Float64[]
for step in 1:N_GEN
    nxt0 = argmax(logits_row) - 1
    nxt0 == EOS_ID && break
    push!(gen_A, nxt0)
    push!(logit_trace_A, copy(logits_row))
    global cur_step += 1
    ts = time()
    global logits_row = cached_step_A!(nxt0, cur_step)
    push!(gen_times, time() - ts)
    if step in (1, 10, 40)
        push!(results, checkpoint("A6_apres_$(step)_tokens_generes"))
    end
end
reply_A = decode_ids(gen_A)
_log("Phase A -- réponse générée ($(length(gen_A)) tokens) : $(repr(reply_A))")
push!(results, checkpoint("A7_fin_phase_A"))

steady = length(gen_times) > 3 ? gen_times[4:end] : gen_times
avg_steady = isempty(steady) ? NaN : sum(steady)/length(steady)
peak_A = maximum(r.smi_mb for r in results)
println("\n", "═"^78)
println("RÉSUMÉ PHASE A")
println("═"^78)
for r in results
    println("  $(rpad(r.label,45))  smi=$(round(r.smi_mb,digits=0))MB  pool_high=$(round(r.pool_high_mb,digits=0))MB")
end
println("Préfixe ($(length(prefix)) tok) : $(round(t_prefix,digits=3))s")
println("Génération : 1er token=$(round(gen_times[1],digits=4))s  moy(régime établi)=$(round(avg_steady,digits=4))s/tok")
println("PIC VRAM Phase A : $(round(peak_A,digits=0)) MB")

open(joinpath(@__DIR__, "diag_kv_cache_double_alloc_phaseA_results.json"), "w") do io
    JSON.print(io, Dict(
        "results"=>[Dict("label"=>r.label, "smi_mb"=>r.smi_mb, "pool_current_mb"=>r.pool_current_mb, "pool_high_mb"=>r.pool_high_mb) for r in results],
        "gen"=>gen_A, "reply"=>reply_A,
        "logit_trace"=>[Float64.(v) for v in logit_trace_A],
        "t_prefix"=>t_prefix, "gen_times"=>gen_times, "avg_steady"=>avg_steady,
        "peak_vram_mb"=>peak_A,
    ), 2)
end
_log("Écrit -> diag_kv_cache_double_alloc_phaseA_results.json")
flush(stdout)
