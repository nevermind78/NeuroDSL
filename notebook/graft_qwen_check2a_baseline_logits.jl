# ══════════════════════════════════════════════════════════════════════════════
# graft_qwen_check2a_baseline_logits.jl — Process A du Check 2
# (graft_qwen_correctness_preregistration.md) : logits de Qwen SANS greffe,
# sur le prompt réel, sauvegardés pour comparaison bit-exacte par Process B
# dans un processus julia séparé.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, Printf

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6

println("── Check 2, Process A : logits Qwen SANS greffe ──"); flush(stdout)

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
println("Poids chargés."); flush(stdout)

adhoc = JSON.parsefile(joinpath(MODEL_DIR, "adhoc_prompts.json"))
prompt_entry = adhoc[1]
prompt_text = prompt_entry["prompt"]
tokens0 = Int.(prompt_entry["token_ids"]) .+ 1
input_tokens = tokens0[1:end-1]
@printf("Prompt réel : %s (%d tokens d'entrée)\n", repr(prompt_text), length(input_tokens))
flush(stdout)

NeuroDSL.set!(g, :token_ids, input_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:length(input_tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
logits = Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))
@printf("logits shape = %s\n", size(logits))
flush(stdout)

open(joinpath(@__DIR__, "graft_qwen_check2_baseline_logits.json"), "w") do io
    JSON.print(io, Dict("prompt" => prompt_text, "shape" => collect(size(logits)),
                          "logits_flat" => vec(Float64.(logits))))
end
println("Logits baseline (sans greffe) sauvegardés -> notebook/graft_qwen_check2_baseline_logits.json")
