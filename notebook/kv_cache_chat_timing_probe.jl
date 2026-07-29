# ══════════════════════════════════════════════════════════════════════════════
# kv_cache_chat_timing_probe.jl — instrumentation temporelle de chat() (qwen2.ipynb)
#
# RÉVISION 2026-07-29 #2 : la 1ère version de ce script a permis de trouver un
# vrai bug (préfixe rempli TOKEN PAR TOKEN dans le cache, ~0.5s/token de
# préfixe -- corrigé par `NeuroDSL.prime_kv_cache_from_prefix!`, voir
# src/layers.jl et kv_cache_prefix_prime_qwen_gate.jl pour la porte de
# correction). Ce script est maintenant mis à jour pour mesurer la MÊME chose
# mais via le mécanisme CORRIGÉ (préfixe = 1 passage batché), afin de vérifier
# avec des chiffres réels, pas une extrapolation, où le temps part maintenant :
#   (a) temps des 2 appels au sous-processus tokenizer (encode_chat, decode_ids)
#   (b) temps du passage avant batché du préfixe (UN SEUL appel, plus une boucle)
#   (c) temps de la boucle de génération, PAS PAR PAS (pas juste le total)
#   (d) le premier pas de génération séparément des suivants (JIT CUDA à froid)
# Exécuté en dehors du notebook pour itérer plus vite -- même graphe, mêmes
# fonctions que qwen2.ipynb (dupliquées ici, pas de dépendance croisée fragile
# sur l'état du notebook).
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=2))s] ", msg); flush(stdout))

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

# -- Sous-processus tokenizer, mesuré isolément --
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
_log("EOS_ID = $EOS_ID (ce premier appel inclut le tout premier lancement d'un process Python -- mesuré séparément ci-dessous, hors boucle de chat).")

# ══════════════════ Mesure instrumentée d'UN échange réaliste ══════════════════
HISTORY = Dict{String,Any}[]
push!(HISTORY, Dict("role"=>"user", "content"=>"What is the capital of Egypt?"))

_log("── (a) encode_chat, appel #1 (à part, pour ne pas mélanger avec le JIT CUDA) ──")
t_tok1_0 = time()
ids0 = encode_chat(HISTORY)
t_tok1 = time() - t_tok1_0
_log("  encode_chat: $(round(t_tok1,digits=3))s -- $(length(ids0)) tokens de préfixe")

_log("── (b) préfixe : UN SEUL passage avant batché (ns_load) + amorçage du cache ──")
prefix = ids0 .+ 1   # 1-indexé
t_prefix0 = time()
NeuroDSL.set!(g, :token_ids, prefix; atom_type=NeuroDSL.Datom, namespace=ns_load)
NeuroDSL.invalidate_all!(g; namespace=ns_load)
prefix_out = Array(NeuroDSL.demand!(g, logits_load; namespace=ns_load))
t_prefix_fwd = time() - t_prefix0
logits_row = Float32.(prefix_out[end, :])
t_prime0 = time()
n_primed = NeuroDSL.prime_kv_cache_from_prefix!(g; src_ns=ns_load, dst_ns=ns_chat,
    n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, use_rope=true)
t_prime = time() - t_prime0
cur_step = length(prefix)
_log("  passage avant batché ($(length(prefix)) tokens) : $(round(t_prefix_fwd,digits=3))s   " *
     "amorçage du cache ($n_primed tableaux) : $(round(t_prime,digits=3))s")

_log("── (c) génération, PAS PAR PAS (jusqu'à 40 tokens ou EOS) ──")
gen0 = Int[]
gen_step_times = Float64[]
for step in 1:40
    nxt0 = argmax(logits_row) - 1
    if nxt0 == EOS_ID
        _log("  EOS atteint après $(length(gen0)) tokens générés.")
        break
    end
    push!(gen0, nxt0)
    global cur_step += 1
    ts0 = time()
    global logits_row = cached_step!(g, nxt0, cur_step)
    push!(gen_step_times, time() - ts0)
    _log("    pas génération $step ($(round(gen_step_times[end],digits=3))s)")
end

_log("── (a) decode_ids, appel final ──")
t_tok2_0 = time()
reply = decode_ids(gen0)
t_tok2 = time() - t_tok2_0
_log("  decode_ids: $(round(t_tok2,digits=3))s")
_log("  Réponse : $reply")

println("\n", "═"^74)
println("RÉSUMÉ -- où part le temps, RÉELLEMENT")
println("═"^74)
println("  encode_chat (sous-processus tokenizer)         : $(round(t_tok1,digits=2))s")
println("  decode_ids  (sous-processus tokenizer)          : $(round(t_tok2,digits=2))s")
println("  préfixe -- passage avant batché ($(length(prefix)) tokens) : $(round(t_prefix_fwd,digits=3))s")
println("  préfixe -- amorçage du cache (copie aux_data)   : $(round(t_prime,digits=3))s")
println("  génération -- 1er pas (JIT CUDA à froid)        : $(round(gen_step_times[1],digits=3))s")
if length(gen_step_times) > 1
    println("  génération -- pas suivants (moy., hors 1er)     : $(round(sum(gen_step_times[2:end])/length(gen_step_times[2:end]),digits=3))s/pas ($(length(gen_step_times)-1) pas)")
end
total_tok_overhead = t_tok1 + t_tok2
println("  Total tokenizer (2 appels)                      : $(round(total_tok_overhead,digits=2))s")
println("  Total calcul (préfixe batché + génération cache): $(round(t_prefix_fwd+t_prime+sum(gen_step_times),digits=2))s")
println("  Total chat() (tokenizer + calcul)               : $(round(total_tok_overhead+t_prefix_fwd+t_prime+sum(gen_step_times),digits=2))s")
flush(stdout)
