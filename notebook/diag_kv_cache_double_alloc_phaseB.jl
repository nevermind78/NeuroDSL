# ══════════════════════════════════════════════════════════════════════════════
# diag_kv_cache_double_alloc_phaseB.jl — PHASE B seule, PROCESS ISOLÉ.
# Voir diag_kv_cache_double_alloc_phaseA.jl pour le contexte complet et le
# pourquoi de la scission en deux process.
#
# CORRECTIF PROPOSÉ : un SEUL namespace, AUCUNE copie de poids.
# `build_cached_decode_graph!`/`CachedLlamaModel` (`src/layers.jl:273-492`)
# retrouvent par CONVENTION DE NOMMAGE les nœuds de poids déjà créés par
# `LlamaModel` dans le MÊME namespace -- pas besoin de `copy_params_to_namespace!`.
# Ceci n'est SÛR que parce que `demand!` restreint désormais son parcours au
# cône des ancêtres RÉELS de la cible (`_ancestors_of!`, `src/graph_api.jl:44-88`,
# `src/dispatch.jl:773-805`) au lieu du préfixe topologique complet du
# namespace -- l'ancien comportement de `demand!` (avant ce correctif) est ce
# qui avait motivé la duplication complète des poids en premier lieu
# (2026-07-29, voir l'en-tête de `copy_params_to_namespace!`/`kv_cache.jl`).
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
_log("PHASE B (process isolé) -- correctif proposé (1 namespace, poids partagés, ZÉRO copie)")
push!(results, checkpoint("B0_avant_tout"))

devB = NeuroDSL.Backend.CUDADevice()
ns_single = :B_single
gB = NeuroDSL.NeuroGraph(namespace=ns_single, device=devB)
NeuroDSL.set!(gB, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns_single)
tok_embB = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(gB, :token_ids, :tok; namespace=ns_single)
out_symB = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                                batched_attn=true, n_kv_heads=N_KV_HEADS,
                                qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(gB, tok_embB; namespace=ns_single)
final_normB = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(gB, out_symB, :final_norm; namespace=ns_single)
logits_loadB = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(gB, final_normB, :lm_head; namespace=ns_single)
push!(results, checkpoint("B1_graphe_construit_poids_aleatoires"))

NeuroDSL.load_graph!(gB, ns_single, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
push!(results, checkpoint("B2_apres_chargement_poids_reels_1_seule_copie"))

# PAS de copy_params_to_namespace! ici -- build_cached_decode_graph! dans LE
# MÊME namespace retrouve les nœuds de poids déjà créés par LlamaModel ci-dessus.
dec_logits_B = NeuroDSL.build_cached_decode_graph!(gB;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns_single)
push!(results, checkpoint("B3_apres_construction_graphe_cache_MEME_namespace"))

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

function batched_prefix_pass_B!(prefix1idx::Vector{Int})
    NeuroDSL.set!(gB, :token_ids, prefix1idx; atom_type=NeuroDSL.Datom, namespace=ns_single)
    NeuroDSL.invalidate_all!(gB; namespace=ns_single)
    out = Array(NeuroDSL.demand!(gB, logits_loadB; namespace=ns_single))
    NeuroDSL.prime_kv_cache_from_prefix!(gB; src_ns=ns_single, dst_ns=ns_single,
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, use_rope=true)
    return Float32.(out[end, :])
end
function cached_step_B!(tok0::Int, cur_step::Int)
    NeuroDSL.set!(gB, :dec_token_id, [tok0 + 1]; atom_type=NeuroDSL.Datom, namespace=ns_single)
    NeuroDSL.set!(gB, :dec_cur_step, Float32[cur_step]; namespace=ns_single)
    NeuroDSL.set!(gB, :dec_pos, Float32[cur_step-1]; namespace=ns_single)
    NeuroDSL.invalidate_all!(gB; namespace=ns_single)
    return Array(NeuroDSL.demand!(gB, dec_logits_B; namespace=ns_single))[1, :]
end

const PROMPT = "Explain in one short sentence why the sky is blue."
const N_GEN = 40
history = Dict{String,Any}[Dict("role"=>"user", "content"=>PROMPT)]
prefix = encode_chat(history) .+ 1

t_prefix0 = time()
logits_row = batched_prefix_pass_B!(prefix)
t_prefix = time() - t_prefix0
push!(results, checkpoint("B4_apres_passage_prefixe"))
cur_step = length(prefix)
gen_B = Int[]
logit_trace_B = Vector{Float32}[]
gen_times = Float64[]
for step in 1:N_GEN
    nxt0 = argmax(logits_row) - 1
    nxt0 == EOS_ID && break
    push!(gen_B, nxt0)
    push!(logit_trace_B, copy(logits_row))
    global cur_step += 1
    ts = time()
    global logits_row = cached_step_B!(nxt0, cur_step)
    push!(gen_times, time() - ts)
    if step in (1, 10, 40)
        push!(results, checkpoint("B5_apres_$(step)_tokens_generes"))
    end
end
reply_B = decode_ids(gen_B)
_log("Phase B -- réponse générée ($(length(gen_B)) tokens) : $(repr(reply_B))")
push!(results, checkpoint("B6_fin_phase_B"))

steady = length(gen_times) > 3 ? gen_times[4:end] : gen_times
avg_steady = isempty(steady) ? NaN : sum(steady)/length(steady)
peak_B = maximum(r.smi_mb for r in results)
println("\n", "═"^78)
println("RÉSUMÉ PHASE B")
println("═"^78)
for r in results
    println("  $(rpad(r.label,45))  smi=$(round(r.smi_mb,digits=0))MB  pool_high=$(round(r.pool_high_mb,digits=0))MB")
end
println("Préfixe ($(length(prefix)) tok) : $(round(t_prefix,digits=3))s")
println("Génération : 1er token=$(round(gen_times[1],digits=4))s  moy(régime établi)=$(round(avg_steady,digits=4))s/tok")
println("PIC VRAM Phase B : $(round(peak_B,digits=0)) MB")

open(joinpath(@__DIR__, "diag_kv_cache_double_alloc_phaseB_results.json"), "w") do io
    JSON.print(io, Dict(
        "results"=>[Dict("label"=>r.label, "smi_mb"=>r.smi_mb, "pool_current_mb"=>r.pool_current_mb, "pool_high_mb"=>r.pool_high_mb) for r in results],
        "gen"=>gen_B, "reply"=>reply_B,
        "logit_trace"=>[Float64.(v) for v in logit_trace_B],
        "t_prefix"=>t_prefix, "gen_times"=>gen_times, "avg_steady"=>avg_steady,
        "peak_vram_mb"=>peak_B,
    ), 2)
end
_log("Écrit -> diag_kv_cache_double_alloc_phaseB_results.json")
flush(stdout)
