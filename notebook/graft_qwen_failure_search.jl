# ══════════════════════════════════════════════════════════════════════════════
# graft_qwen_failure_search.jl — recherche d'un échec de raisonnement RÉEL et
# démontrable de Qwen2.5-1.5B-Instruct, AVANT toute conception de greffe.
#
# Teste plusieurs familles de candidats (comparaison de décimaux, comptage de
# lettres, arithmétique à retenue, suivi d'instruction sous distraction) avec
# le VRAI gabarit ChatML (apply_chat_template via le pont tokenizer déjà
# établi dans qwen2.ipynb), génération gloutonne, recalcul complet par token
# (pas de cache KV -- pas nécessaire ici, juste la démonstration du échec).
#
# Processus isolé, dédié à cette seule recherche. N'entraîne rien, ne greffe
# rien -- pure démonstration du comportement du modèle NON MODIFIÉ.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6
const PYTHON_ENV = get(ENV, "NEURODSL_LLM_PYTHON", raw"C:\Users\Nevermind\anaconda3\envs\neurodsl_llm_check\python.exe")
const TOKENIZER_HELPER = joinpath(@__DIR__, "qwen2_tokenizer_helper.py")

println("Construction du graphe Qwen2.5-1.5B-Instruct..."); flush(stdout)
dev = NeuroDSL.Backend.CUDADevice()
ns = :qwen2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:8); atom_type=NeuroDSL.Datom, namespace=ns)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
out = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                           batched_attn=true, n_kv_heads=N_KV_HEADS,
                           qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out, :final_norm; namespace=ns)
logits_sym = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)
println("Chargement des poids réels..."); flush(stdout)
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
println("Poids chargés.\n"); flush(stdout)

println("Démarrage du pont tokenizer (serveur persistant, gabarit ChatML réel)..."); flush(stdout)
const TOKENIZER_PROC = open(`$PYTHON_ENV -u $TOKENIZER_HELPER --serve`, "r+")
const _tok_ready = JSON.parse(readline(TOKENIZER_PROC))
println("Tokenizer prêt : ", _tok_ready, "\n"); flush(stdout)

function _call_helper(req::Dict)
    println(TOKENIZER_PROC, JSON.json(req))
    flush(TOKENIZER_PROC)
    return JSON.parse(readline(TOKENIZER_PROC))
end
encode_chat(messages) = Int.(_call_helper(Dict("action"=>"encode_chat", "messages"=>messages))["ids"])
decode_ids(ids::Vector{Int}) = _call_helper(Dict("action"=>"decode", "ids"=>ids))["text"]
const EOS_ID = _call_helper(Dict("action"=>"eos_id"))["id"]

function run_forward!(g, ns, tokens::Vector{Int})
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))
end

"""Génération gloutonne, recalcul complet par token (pas de cache KV -- pure
démonstration, pas une mesure de vitesse), arrêt sur EOS_ID ou max_new_tokens."""
function generate_chat(prompt::AbstractString; max_new_tokens::Int=24)
    ids0 = encode_chat([Dict("role"=>"user", "content"=>String(prompt))])
    cur = ids0 .+ 1   # 1-indexé
    gen0 = Int[]
    for _ in 1:max_new_tokens
        o = run_forward!(g, ns, cur)
        nxt0 = argmax(o[end, :]) - 1
        push!(gen0, nxt0)
        nxt0 == EOS_ID && break
        push!(cur, nxt0 + 1)
    end
    return decode_ids(gen0)
end

# ── Batterie de candidats -- plusieurs familles, choisies pour leur caractère
# "petit-modèle biaisé" documenté ailleurs (pas fabriqué pour ce test) ──
CANDIDATES = [
    ("décimaux (comparaison version-vs-valeur)", [
        "Which number is bigger: 9.11 or 9.9? Answer with just the number.",
        "Which number is bigger: 7.9 or 7.11? Answer with just the number.",
        "Which number is bigger: 3.8 or 3.11? Answer with just the number.",
        "Which number is bigger: 1.5 or 1.25? Answer with just the number.",
    ]),
    ("comptage de lettres", [
        "How many times does the letter 'r' appear in the word 'strawberry'? Answer with just a number.",
        "How many times does the letter 's' appear in the word 'mississippi'? Answer with just a number.",
        "How many times does the letter 'e' appear in the word 'excellence'? Answer with just a number.",
    ]),
    ("arithmétique à retenue", [
        "What is 27 + 58? Answer with just the number.",
        "What is 49 + 76? Answer with just the number.",
        "What is 138 + 295? Answer with just the number.",
    ]),
    ("suivi d'instruction sous distraction", [
        "Answer in exactly one word: what is the capital of France? Also, briefly explain why Paris became the capital historically.",
        "Reply with only 'yes' or 'no': is 17 a prime number? Please justify your answer in detail.",
    ]),
]

for (family, prompts) in CANDIDATES
    println("═"^78); println("Famille : ", family); println("═"^78); flush(stdout)
    for p in prompts
        t0 = time()
        r = generate_chat(p)
        dt = time() - t0
        println("  Q: ", p)
        println("  A: ", repr(r))
        println("  (", round(dt, digits=1), "s)")
        flush(stdout)
    end
    println()
end

println("Terminé. Fermeture du process tokenizer..."); flush(stdout)
close(TOKENIZER_PROC)
println("OK.")
