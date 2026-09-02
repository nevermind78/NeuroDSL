# ══════════════════════════════════════════════════════════════════════════════
# diag_graft_determinism_probe_v3_warmup.jl — teste si un préchauffage GPU
# dédié, AVANT toute insertion de greffe/seed, stabilise la discontinuité de
# VALEUR (pas NaN) observée au pas 4-6 dans le probe v2
# (diag_graft_determinism_probe_v2.log) par rapport au probe v1 SANS phase
# d'éval-avant (diag_graft_determinism_probe_{A,B}.log, bit-identiques entre
# eux à 15 chiffres significatifs).
#
# HYPOTHÈSE TESTÉE : le probe v2 diffère de v1 UNIQUEMENT parce qu'il ajoute
# la phase d'éval AVANT entraînement (10 générations gloutonnes jusqu'à 40
# tokens), qui pousse la VRAM près de la saturation de la carte (16041-16074
# / 16384 Mio observé directement pendant `cleanA`) -- diagnostic le plus
# plausible : la sélection heuristique d'algorithme cuBLAS/CUDA dépend de
# l'état mémoire du device au moment du premier appel, donc un historique
# d'allocation différent (v1 vs v2) sélectionne un algorithme différent,
# produisant un résultat numériquement différent (pas faux, juste différent)
# dès que cet algorithme est exercé -- pas du bruit, déterministe une fois
# l'historique d'allocation fixé (confirmé : A et B, tous deux SANS eval-
# avant, sont bit-identiques entre eux ; le désaccord n'apparaît qu'entre v1
# et v2, pas entre deux lancements de la même variante).
#
# PRÉCÉDENT (qwen2.ipynb, cellule 16) : un appel factice (`chat("Hello";
# max_new_tokens=1)`) avant la vraie session paie le coût JIT une seule fois
# et stabilise le temps de réponse perçu -- mécanisme différent (compilation,
# pas allocation mémoire), mais même idée de "mettre le moteur dans un état
# stable avant de mesurer/entraîner pour de vrai".
#
# PROTOCOLE ICI : AVANT Random.seed!/insertion de greffe, on exécute un
# préchauffage dédié -- 10 générations gloutonnes jusqu'à 40 tokens sur le
# graphe NON MODIFIÉ (mêmes prompts que la phase d'éval réelle, résultats
# jetés) -- pour pousser la VRAM et l'historique d'allocation cuBLAS dans le
# MÊME état de saturation AVANT que quoi que ce soit de mesuré ne commence.
# Puis EXACTEMENT le protocole v2 (seed, greffe, gel, éval-avant réelle,
# 20 pas d'entraînement). Si l'hypothèse est correcte, cette variante
# devrait re-matcher v1 (A/B) au lieu de v2 -- ou au minimum donner un point
# de comparaison honnête, rapporté tel quel que l'effet soit présent ou non.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, Printf, Random

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6
const GRAFT_SITE = :layer_25_out
const GRAFT_PREFIX = :qwen_shadow_fix
const GRAFT_HEADS, GRAFT_HIDDEN = 4, 384
const LR = 3f-3
const N_STEPS = 20
const PYTHON_ENV = get(ENV, "NEURODSL_LLM_PYTHON", raw"C:\Users\Nevermind\anaconda3\envs\neurodsl_llm_check\python.exe")
const TOKENIZER_HELPER = joinpath(@__DIR__, "qwen2_tokenizer_helper.py")

dev = NeuroDSL.Backend.CUDADevice()
ns = :qwen2probev3
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:8); atom_type=NeuroDSL.Datom, namespace=ns)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
out = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                           batched_attn=true, n_kv_heads=N_KV_HEADS,
                           qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out, :final_norm; namespace=ns)
logits_sym = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)

const TOKENIZER_PROC = open(`$PYTHON_ENV -u $TOKENIZER_HELPER --serve`, "r+")
const _tok_ready = JSON.parse(readline(TOKENIZER_PROC))
function _call_helper(req::Dict)
    println(TOKENIZER_PROC, JSON.json(req))
    flush(TOKENIZER_PROC)
    return JSON.parse(readline(TOKENIZER_PROC))
end
encode_chat(messages) = Int.(_call_helper(Dict("action"=>"encode_chat", "messages"=>messages))["ids"])
encode_raw(text) = Int.(_call_helper(Dict("action"=>"encode", "text"=>text))["ids"])
decode_ids(ids::Vector{Int}) = _call_helper(Dict("action"=>"decode", "ids"=>ids))["text"]
const EOS_ID = _call_helper(Dict("action"=>"eos_id"))["id"]

oneword_prompt(q) = "Answer in exactly one word: $q Also, briefly explain your reasoning."
yesno_prompt(q)   = "Reply with only 'yes' or 'no': $q Please justify your answer in detail."
heldout_A = [
    (oneword_prompt("what is the capital of Japan?"), "Tokyo", :oneword),
    (oneword_prompt("what is the tallest mountain on Earth?"), "Everest", :oneword),
    (oneword_prompt("what is the chemical symbol for gold?"), "Au", :oneword),
    (oneword_prompt("how many continents are there on Earth?"), "Seven", :oneword),
]
heldout_B = [
    (yesno_prompt("is the Earth flat?"), "No", :yesno),
    (yesno_prompt("is water made of hydrogen and oxygen?"), "Yes", :yesno),
    (yesno_prompt("is Paris the capital of Germany?"), "No", :yesno),
]
neg_control = [
    "What is the capital of Italy? Also, briefly explain why.",
    "Is 12 an even number? Please justify your answer in detail.",
    "What is the largest ocean on Earth? Also, briefly explain your reasoning.",
]

function run_forward!(g, ns, tokens::Vector{Int})
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))
end
function generate_chat(prompt::AbstractString; max_new_tokens::Int=40)
    ids0 = encode_chat([Dict("role"=>"user", "content"=>String(prompt))])
    cur = ids0 .+ 1
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

# ── PRÉCHAUFFAGE DÉDIÉ (avant tout seed/greffe) -- 10 générations gloutonnes
# sur le graphe NON MODIFIÉ, résultats jetés. But : porter la VRAM/l'
# historique d'allocation cuBLAS dans le même état de saturation qu'aura la
# vraie phase d'éval-avant, AVANT que Random.seed!/la greffe n'entrent en jeu.
println("PRECHAUFFAGE : 10 generations gloutonnes sur le graphe NON MODIFIE (jetees)...")
flush(stdout)
t_warmup0 = time()
for (p, _t, _k) in vcat(heldout_A, heldout_B)
    generate_chat(p)
end
for p in neg_control
    generate_chat(p)
end
println("Prechauffage termine (", round(time() - t_warmup0, digits=1), "s)."); flush(stdout)

Random.seed!(4242)
NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.seed!(4242)
new_out, handle = NeuroDSL.graft_shadow_block!(g, ns, GRAFT_SITE, DIM, GRAFT_HEADS, GRAFT_HIDDEN;
                                                alpha0=0f0, zero_out_proj=false, prefix=GRAFT_PREFIX)

function freeze_backbone!(g::NeuroDSL.NeuroGraph, ns::Symbol, keep_prefix::Symbol)
    n_frozen = 0
    for (sym, nd) in g.nodes[ns]
        nd.is_param || continue
        startswith(String(sym), String(keep_prefix)) && continue
        nd.is_param = false
        n_frozen += 1
    end
    return n_frozen
end
freeze_backbone!(g, ns, GRAFT_PREFIX)
ps = NeuroDSL.params(g; namespace=ns)

train_set = [
    (oneword_prompt("what is the capital of France?"), "Paris"),
    (oneword_prompt("what color is the sky on a clear day?"), "Blue"),
    (oneword_prompt("what is the largest planet in the solar system?"), "Jupiter"),
    (oneword_prompt("who wrote the play Romeo and Juliet?"), "Shakespeare"),
    (oneword_prompt("what gas do humans need to breathe to survive?"), "Oxygen"),
    (oneword_prompt("what season comes after winter?"), "Spring"),
]

println("Phase eval AVANT entrainement (10 generations gloutonnes, comme le script reel)...")
flush(stdout)
for (p, _t, _k) in vcat(heldout_A, heldout_B)
    generate_chat(p)
end
for p in neg_control
    generate_chat(p)
end
println("Eval AVANT terminee."); flush(stdout)

NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [logits_sym, :labels], :cross_entropy; namespace=ns))
m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]

function train_example!(g, ns, prompt, target, t)
    prefix_ids = encode_chat([Dict("role"=>"user", "content"=>prompt)])
    answer_ids = encode_raw(target)
    full0 = vcat(prefix_ids, answer_ids, [EOS_ID])
    full1 = full0 .+ 1
    input_tokens = full1[1:end-1]
    label_tokens = full1[2:end]
    NeuroDSL.set!(g, :token_ids, input_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:length(input_tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :labels, label_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    loss_val = NeuroDSL.demand!(g, :loss; namespace=ns)
    NeuroDSL.backward_graph!(g, :loss; namespace=ns, prune_frozen=true)
    NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps], [p.gradient for p in ps],
                                  m1s, m2s, LR, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
    return Float64(sum(Array(loss_val)))
end

println("PROBE_V3_WARMUP_START run_id=", get(ENV, "PROBE_RUN_ID", "?"))
for t in 1:N_STEPS
    ex_idx = ((t - 1) % length(train_set)) + 1
    prompt, target = train_set[ex_idx]
    l = train_example!(g, ns, prompt, target, t)
    a = Float64(Array(NeuroDSL.node(g, handle.alpha_sym; namespace=ns).value)[1])
    finite = isfinite(l) && isfinite(a)
    @printf("STEP %3d ex=%d loss=%.15g alpha=%.15g finite=%s\n", t, ex_idx, l, a, finite)
    flush(stdout)
    if !finite
        println("DIVERGED at step ", t)
        break
    end
end
println("PROBE_V3_WARMUP_END")
close(TOKENIZER_PROC)
