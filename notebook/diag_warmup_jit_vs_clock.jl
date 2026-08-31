# ══════════════════════════════════════════════════════════════════════════════
# diag_warmup_jit_vs_clock.jl — Piste 2 (horloge GPU) : diag_gpu_clock_correlate.jl
# a déjà montré que la durée d'un pas de décodage NE CORRÈLE PAS avec
# l'horloge SM (pas à 210MHz et pas à 1500MHz prennent le MÊME temps,
# ~420-480ms) -- donc le ralentissement massif isolé du tout premier pas de
# décodage (2163ms, mesuré dans diag_gpu_clock_during_decode.jl, à 210MHz
# comme les pas suivants qui eux sont normaux) n'est PAS un effet d'horloge.
# Hypothèse alternative testée ici : compilation JIT Julia/CUDA de première
# invocation (CUSTOM_OPS `:kv_cache_append`/`:scale_no_mask`/`:rope_at_pos`,
# jamais appelés avant le tout premier pas de décodage caché, contrairement
# au chemin `:matmul`/`:rmsnorm`/etc. déjà exercé par le passage batché sur
# le préfixe). Si c'est bien la cause, UN pas de décodage "jetable" (token
# bidon, cur_step=1) lancé juste après `build_cached_decode_graph!` --
# AVANT le vrai prompt -- devrait payer ce coût pendant le "chargement", pas
# pendant la première réponse visible par l'utilisateur.
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
_log("Graphe construit.")

NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
NeuroDSL.alias_tied_param!(g, ns, :tok_E, :lm_head_W)
dec_logits = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns)
_log("Poids réels chargés, cache KV construit (AVANT tout warm-up).")

const PYTHON_ENV = raw"C:\Users\Nevermind\anaconda3\envs\neurodsl_llm_check\python.exe"
const TOKENIZER_HELPER = joinpath(@__DIR__, "qwen2_tokenizer_helper.py")
function _call_helper(req::Dict)
    input_json = JSON.json(req)
    out = IOBuffer()
    run(pipeline(`$PYTHON_ENV $TOKENIZER_HELPER`, stdin=IOBuffer(input_json), stdout=out))
    return JSON.parse(String(take!(out)))
end
encode_chat(messages) = Int.(_call_helper(Dict("action"=>"encode_chat", "messages"=>messages))["ids"])

function cached_step!(tok0::Int, cur_step::Int)
    NeuroDSL.set!(g, :dec_token_id, [tok0 + 1]; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :dec_cur_step, Float32[cur_step]; namespace=ns)
    NeuroDSL.set!(g, :dec_pos, Float32[cur_step-1]; namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, dec_logits; namespace=ns))[1, :]
end

# ── LE WARM-UP : un pas de décodage JETABLE, cur_step=1, token bidon ------
# (aucune conséquence sur le cache réel -- `prime_kv_cache_from_prefix!`,
# appelé juste après pour le VRAI préfixe, écrase entièrement
# `aux_data[:history]` de chaque nœud de cache, voir sa docstring dans
# src/layers.jl -- vérifié : ce warm-up ne fuit aucun état vers la suite.)
CUDA.synchronize()
t_warmup0 = time()
cached_step!(0, 1)
CUDA.synchronize()
t_warmup = time() - t_warmup0
_log("Warm-up (1 pas jetable, cur_step=1) : $(round(t_warmup,digits=4))s")

function batched_prefix_pass!(prefix1idx::Vector{Int})
    NeuroDSL.set!(g, :token_ids, prefix1idx; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    out = Array(NeuroDSL.demand!(g, logits_load; namespace=ns))
    NeuroDSL.prime_kv_cache_from_prefix!(g; src_ns=ns, dst_ns=ns,
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, use_rope=true)
    return Float32.(out[end, :])
end

const PROMPT = "Explain in one short sentence why the sky is blue."
history = Dict{String,Any}[Dict("role"=>"user", "content"=>PROMPT)]
prefix = encode_chat(history) .+ 1
t_prefix0 = time()
logits_row = batched_prefix_pass!(prefix)
t_prefix = time() - t_prefix0
cur_step = length(prefix)
_log("Préfixe RÉEL traité ($(length(prefix)) tok) après priming (cache écrasé par le vrai préfixe) : $(round(t_prefix,digits=3))s")

const N_GEN = 10
gen_times = Float64[]
gen_tokens = Int[]
for step in 1:N_GEN
    nxt0 = argmax(logits_row) - 1
    push!(gen_tokens, nxt0)
    global cur_step += 1
    t0 = time()
    global logits_row = cached_step!(nxt0, cur_step)
    push!(gen_times, time() - t0)
end

println("\n", "="^78)
println("AVEC warm-up jetable AVANT le prompt réel")
println("="^78)
println("Coût du warm-up (payé PENDANT le chargement, invisible pour l'utilisateur) : $(round(1000*t_warmup,digits=1))ms")
println("Temps des $N_GEN premiers tokens RÉELS générés après le vrai prompt :")
for (i,t) in enumerate(gen_times)
    println("  token $i : $(round(1000*t,digits=1))ms")
end
println("Premier token réel : $(round(1000*gen_times[1],digits=1))ms -- ",
        gen_times[1] < 1.0 ? "PAS de pic de première invocation (le warm-up a bien absorbé le coût JIT)." :
        "pic encore présent -- le warm-up n'a PAS absorbé tout le coût.")

open(joinpath(@__DIR__, "diag_warmup_jit_vs_clock_results.json"), "w") do io
    JSON.print(io, Dict(
        "t_warmup_s"=>t_warmup, "t_prefix_s"=>t_prefix,
        "gen_times_s"=>gen_times, "gen_tokens"=>gen_tokens,
    ), 2)
end
_log("Écrit -> diag_warmup_jit_vs_clock_results.json")
