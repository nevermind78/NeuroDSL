# ══════════════════════════════════════════════════════════════════════════════
# diag_decode_e2e_verify.jl — mesure bout-en-bout (SANS checkpoints, chemin
# chaud réel) + vérification de correction (tokens/logits identiques) pour
# le correctif RoPE (src/kv_cache.jl, `rope_at_pos!` : theta mis en cache +
# plus de lecture device->host de `pos`) et le kernel `hcat_heads` fusionné
# (src/kernels.jl). Lancé UNE FOIS par état du code (avant/après, via git
# stash), résultats comparés hors-processus.
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
_log("Poids chargés, cache KV construit.")

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
_log("Préfixe traité ($(length(prefix)) tok). Warmup 100 pas...")

for step in 1:100
    nxt0 = argmax(logits_row) - 1
    global cur_step += 1
    global logits_row = cached_step!(nxt0, cur_step)
end
_log("Régime établi (cur_step=$cur_step). Génération de 40 tokens mesurée + capture logits...")

# ── Génération mesurée : 40 pas, chemin chaud RÉEL (aucun checkpoint,
#    aucun CUDA.synchronize() intermédiaire hors de la boucle de mesure). ──
gen_tokens = Int[]
gen_logits_first5 = Vector{Float32}[]  # capture les logits complets des 5 premiers pas pour comparaison bit-exacte
CUDA.synchronize()
t_start = time()
for step in 1:40
    nxt0 = argmax(logits_row) - 1
    global cur_step += 1
    global logits_row = cached_step!(nxt0, cur_step)
    push!(gen_tokens, nxt0)
    if step <= 5
        push!(gen_logits_first5, copy(logits_row))
    end
end
CUDA.synchronize()
t_total = time() - t_start
push!(gen_tokens, argmax(logits_row) - 1)  # dernier token généré

ms_per_tok = 1000 * t_total / 40
_log("40 tokens en $(round(t_total,digits=3))s -> $(round(ms_per_tok,digits=2))ms/tok, $(round(40/t_total,digits=2)) tok/s")

open(joinpath(@__DIR__, "diag_decode_e2e_verify_results.json"), "w") do io
    JSON.print(io, Dict(
        "ms_per_tok"=>ms_per_tok, "tok_per_s"=>40/t_total, "t_total"=>t_total,
        "gen_tokens"=>gen_tokens,
        "gen_logits_first5"=>[Float64.(l) for l in gen_logits_first5],
    ), 2)
end
_log("Écrit -> diag_decode_e2e_verify_results.json")
