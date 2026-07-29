using NeuroDSL, JSON

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6

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
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)

ref = JSON.parsefile(joinpath(MODEL_DIR, "reference_logits.json"))
tokens = Int.(ref[1]["token_ids"]) .+ 1
NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, logits_sym; namespace=ns)
cache = NeuroDSL.capture_activations(g, ns)

println("sk_h2 (avant softmax) valeurs brutes :")
display(Array(cache[:layer_1_mha_sk_h2]))
println("\npr_h2 (après softmax) valeurs brutes :")
display(Array(cache[:layer_1_mha_pr_h2]))
println("\nsk_h7 (avant softmax) valeurs brutes :")
display(Array(cache[:layer_1_mha_sk_h7]))
println("\npr_h7 (après softmax) valeurs brutes :")
display(Array(cache[:layer_1_mha_pr_h7]))
println("\nsk_h1 (référence propre, avant softmax) :")
display(Array(cache[:layer_1_mha_sk_h1]))
println()
