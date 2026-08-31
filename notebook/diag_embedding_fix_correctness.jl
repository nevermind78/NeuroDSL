# ══════════════════════════════════════════════════════════════════════════════
# diag_embedding_fix_correctness.jl — vérifie que le correctif de l'op
# `:embedding` (src/dispatch.jl, remplace l'aller-retour CPU de toute la
# table par `view(E, idx_cpu, :)`) ne change PAS le texte généré. Lancé en
# DEUX PROCESSUS SÉPARÉS ET ISOLÉS (voir le script appelant) -- un avec le
# code AVANT le correctif (`git stash` temporaire de src/dispatch.jl), un
# avec le code APRÈS -- PAS une comparaison dans le même process (une
# mesure même-process a déjà été trouvée contaminée plus tôt dans cette
# session). Génère 40 tokens en glouton (argmax, déterministe) à partir du
# même prompt, sur le chemin de production RÉEL (`build_cached_decode_graph!`
# + `prime_kv_cache_from_prefix!`, identique à `chat()` dans
# notebook/qwen2.ipynb), et écrit les IDs de tokens + les logits du DERNIER
# pas dans un fichier JSON étiqueté par argument CLI (before/after).
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, CUDA

const TAG = length(ARGS) >= 1 ? ARGS[1] : "unlabeled"

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=2))s] [$TAG] ", msg); flush(stdout))

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
_log("Préfixe traité ($(length(prefix)) tok). Génération gloutonne de 40 tokens...")

const N_GEN = 40
gen_tokens = Int[]
for step in 1:N_GEN
    nxt0 = argmax(logits_row) - 1
    push!(gen_tokens, nxt0)
    global cur_step += 1
    global logits_row = cached_step!(nxt0, cur_step)
end

decoded = _call_helper(Dict("action"=>"decode", "ids"=>gen_tokens))["text"]
_log("Tokens générés : $gen_tokens")
_log("Texte décodé : $decoded")

open(joinpath(@__DIR__, "diag_embedding_fix_correctness_$(TAG)_results.json"), "w") do io
    JSON.print(io, Dict(
        "tag"=>TAG, "prompt"=>PROMPT, "prefix_len"=>length(prefix),
        "gen_tokens"=>gen_tokens, "decoded_text"=>decoded,
        "final_logits_sample"=>Float64.(logits_row[1:20]),  # premières 20 valeurs, pour une comparaison numérique fine
    ), 2)
end
_log("Écrit -> diag_embedding_fix_correctness_$(TAG)_results.json")
