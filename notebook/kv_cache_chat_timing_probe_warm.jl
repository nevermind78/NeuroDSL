# ══════════════════════════════════════════════════════════════════════════════
# kv_cache_chat_timing_probe_warm.jl — teste directement l'hypothèse du
# coordinateur : le coût de 11.28s mesuré pour le passage avant batché du
# préfixe (36 tokens) est-il un coût de compilation JIT CUDA à froid,
# spécifique à CETTE forme (seqlen=36, chemin batched_attn=true), payé UNE
# SEULE fois par forme rencontrée -- comme déjà observé et isolé pour le
# premier pas de génération (2.628s, seqlen=1) ? Si oui, un DEUXIÈME passage
# avant batché sur un préfixe de longueur DIFFÉRENTE (donc une NOUVELLE
# forme, nouveau JIT) devrait ENCORE être lent, mais un DEUXIÈME passage sur
# la MÊME longueur devrait chuter vers les ~0.33-0.41s déjà mesurés ce soir
# pour seqlen=32/64 (tableau de temps du recalcul complet).
#
# Fait tourner DEUX vrais tours de conversation (`chat()`-like, même
# fonctions), DANS LE MÊME PROCESSUS (mêmes kernels CUDA déjà compilés après
# le premier tour), et rapporte le coût du passage avant batché du préfixe
# SÉPARÉMENT pour chacun -- pas une estimation, une mesure directe.
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

function run_turn!(label::String, history)
    _log("══ Tour: $label ══")
    ids0 = encode_chat(history)
    prefix = ids0 .+ 1
    logits_row, t_fwd, t_prime = batched_prefix_pass!(g, prefix)
    _log("  préfixe ($(length(prefix)) tokens) : passage batché=$(round(t_fwd,digits=3))s  amorçage=$(round(t_prime,digits=3))s")
    cur_step = length(prefix)
    gen0 = Int[]
    gen_times = Float64[]
    for step in 1:40
        nxt0 = argmax(logits_row) - 1
        nxt0 == EOS_ID && break
        push!(gen0, nxt0)
        cur_step += 1
        ts = time()
        logits_row = cached_step!(g, nxt0, cur_step)
        push!(gen_times, time() - ts)
    end
    reply = decode_ids(gen0)
    _log("  génération : $(length(gen0)) tokens, 1er pas=$(round(gen_times[1],digits=3))s, " *
         "moy. suivants=$(length(gen_times)>1 ? round(sum(gen_times[2:end])/length(gen_times[2:end]),digits=3) : NaN)s/pas")
    _log("  réponse : $reply")
    return (t_fwd=t_fwd, t_prime=t_prime, n_prefix=length(prefix), n_gen=length(gen0), gen_times=gen_times)
end

# -- Tour 1 : FROID pour le chemin batché du préfixe (seqlen=36, jamais rencontrée avant) --
h1 = Dict{String,Any}[]
push!(h1, Dict("role"=>"user", "content"=>"What is the capital of Egypt?"))
r1 = run_turn!("1 (froid, tout premier passage batché rencontré)", h1)

# -- Tour 2 : nouvelle conversation, question DIFFÉRENTE mais préfixe de longueur PROCHE --
# (teste si le JIT est vraiment amorti -- même ordre de grandeur de seqlen, nouvelle conversation)
h2 = Dict{String,Any}[]
push!(h2, Dict("role"=>"user", "content"=>"What is the capital of Japan?"))
r2 = run_turn!("2 (chaud, même ordre de grandeur de seqlen)", h2)

# -- Tour 3 : EXACTEMENT la même longueur de préfixe que le tour 1 (répéter la même question,
# nouvelle conversation à chaque fois donc même structure ChatML) --
h3 = Dict{String,Any}[]
push!(h3, Dict("role"=>"user", "content"=>"What is the capital of Egypt?"))
r3 = run_turn!("3 (chaud, MÊME longueur de préfixe que le tour 1)", h3)

println("\n", "═"^74)
println("RÉSUMÉ -- coût du passage avant batché du préfixe, froid vs chaud")
println("═"^74)
println("  Tour 1 (froid)  : préfixe=$(r1.n_prefix) tokens -- passage batché = $(round(r1.t_fwd,digits=3))s")
println("  Tour 2 (chaud)  : préfixe=$(r2.n_prefix) tokens -- passage batché = $(round(r2.t_fwd,digits=3))s")
println("  Tour 3 (chaud)  : préfixe=$(r3.n_prefix) tokens -- passage batché = $(round(r3.t_fwd,digits=3))s")
println()
if r2.t_fwd < r1.t_fwd / 3 || r3.t_fwd < r1.t_fwd / 3
    println("  CONFIRMÉ : le coût élevé du tour 1 était bien un coût de compilation JIT à froid --")
    println("  les tours suivants (même processus, kernels déjà compilés) sont nettement plus rapides.")
else
    println("  PAS CONFIRMÉ : les tours suivants restent proches du tour 1 -- le coût n'est PAS")
    println("  (uniquement) du JIT à froid, il reste une inefficacité réelle à investiguer dans")
    println("  le chemin du passage avant batché lui-même.")
end
flush(stdout)
