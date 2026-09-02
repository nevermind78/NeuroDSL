# ══════════════════════════════════════════════════════════════════════════════
# diag_graft_determinism_probe.jl — isole la question : le pipeline
# graft_qwen_experiment_run.jl est-il bit-déterministe d'un lancement de
# process à l'autre pour la BOUCLE D'ENTRAÎNEMENT seule (pas de génération
# autoregressive coûteuse) ? Même seed (4242), même code, exécuté deux fois
# en SEQUENCE (jamais en parallèle -- on vérifie nvidia-smi propre avant
# chaque lancement). On imprime loss/alpha en pleine précision Float64 à
# CHAQUE pas (pas juste tous les 10) pour trouver le pas exact de première
# divergence et son ordre de grandeur.
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
ns = :qwen2probe
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
const EOS_ID = _call_helper(Dict("action"=>"eos_id"))["id"]

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

oneword_prompt(q) = "Answer in exactly one word: $q Also, briefly explain your reasoning."
train_set = [
    (oneword_prompt("what is the capital of France?"), "Paris"),
    (oneword_prompt("what color is the sky on a clear day?"), "Blue"),
    (oneword_prompt("what is the largest planet in the solar system?"), "Jupiter"),
    (oneword_prompt("who wrote the play Romeo and Juliet?"), "Shakespeare"),
    (oneword_prompt("what gas do humans need to breathe to survive?"), "Oxygen"),
    (oneword_prompt("what season comes after winter?"), "Spring"),
]

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

println("PROBE_START run_id=", get(ENV, "PROBE_RUN_ID", "?"))
for t in 1:N_STEPS
    ex_idx = ((t - 1) % length(train_set)) + 1
    prompt, target = train_set[ex_idx]
    l = train_example!(g, ns, prompt, target, t)
    a = Float64(Array(NeuroDSL.node(g, handle.alpha_sym; namespace=ns).value)[1])
    @printf("STEP %3d ex=%d loss=%.15g alpha=%.15g\n", t, ex_idx, l, a)
    flush(stdout)
end
println("PROBE_END")
close(TOKENIZER_PROC)
