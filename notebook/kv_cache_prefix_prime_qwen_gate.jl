# ══════════════════════════════════════════════════════════════════════════════
# kv_cache_prefix_prime_qwen_gate.jl — Porte de correction pour
# `NeuroDSL.prime_kv_cache_from_prefix!` (src/layers.jl, 2026-07-29), sur le
# VRAI Qwen2.5-1.5B-Instruct (poids réels). Même protocole que
# kv_cache_prefix_prime_toy_gate.jl (PASS sur le jouet), refait ici sur le
# modèle réel avant de brancher ce mécanisme dans qwen2.ipynb -- exigence
# explicite du coordinateur : "byte-for-byte the same cache state... or the
# whole thing needs re-verifying from scratch."
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=1))s] ", msg); flush(stdout))

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6
const N_GEN_STEPS = 5

_log("── Porte de correction Qwen2.5-1.5B : remplissage batché vs séquentiel du préfixe ──")

dev = NeuroDSL.Backend.CUDADevice()
ns_load  = :qwen2_load2
ns_seq   = :qwen2_seq
ns_batch = :qwen2_batch
g = NeuroDSL.NeuroGraph(namespace=ns_load, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns_load)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns_load)
out_sym = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                               batched_attn=true, n_kv_heads=N_KV_HEADS,
                               qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns_load)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out_sym, :final_norm; namespace=ns_load)
logits_full = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns_load)
_log("Chargement des poids réels...")
NeuroDSL.load_graph!(g, ns_load, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
_log("Poids chargés.")

NeuroDSL.copy_params_to_namespace!(g, ns_load, ns_seq)
NeuroDSL.copy_params_to_namespace!(g, ns_load, ns_batch)
logits_seq = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns_seq)
logits_batch = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns_batch)
_log("Les deux graphes cachés construits.")

function cached_step!(g, ns, logits_sym, tok1::Int, cur_step::Int)
    NeuroDSL.set!(g, :dec_token_id, [tok1]; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :dec_cur_step, Float32[cur_step]; namespace=ns)
    NeuroDSL.set!(g, :dec_pos, Float32[cur_step-1]; namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))[1, :]
end

ref = JSON.parsefile(joinpath(MODEL_DIR, "reference_logits.json"))
r = ref[1]   # "The capital of France is" -- suffisant pour cette porte (mécanisme déjà générique, prouvé sur toy pour la diversité de formes/tailles)
tokens0 = Int.(r["token_ids"])
PROMPT = tokens0 .+ 1   # 1-indexé
_log("Prompt: $(repr(r["prompt"]))  ($(length(PROMPT)) tokens)")

_log("── A. Remplissage séquentiel (ns_seq) ──")
logits_seq_row = Float32[]
for (t, tok) in enumerate(PROMPT)
    global logits_seq_row = cached_step!(g, ns_seq, logits_seq, tok, t)
    (t % 5 == 0 || t == length(PROMPT)) && _log("  pas séquentiel $t/$(length(PROMPT))")
end
_log("  fait.")

_log("── B. Remplissage batché (ns_batch, via prime_kv_cache_from_prefix!) ──")
NeuroDSL.set!(g, :token_ids, PROMPT; atom_type=NeuroDSL.Datom, namespace=ns_load)
NeuroDSL.invalidate_all!(g; namespace=ns_load)
full_out = Array(NeuroDSL.demand!(g, logits_full; namespace=ns_load))
logits_batch_row = Float32.(full_out[end, :])
n_primed = NeuroDSL.prime_kv_cache_from_prefix!(g; src_ns=ns_load, dst_ns=ns_batch,
    n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, use_rope=true)
_log("  $n_primed tableaux K/V amorcés (attendu $(2*N_LAYERS*N_KV_HEADS)).")

_log("── (1) Comparaison directe de aux_data[:history] (A vs B), $(N_LAYERS)×$(N_KV_HEADS) têtes ──")
max_hist_err = 0f0
for i in 1:N_LAYERS, h in 1:N_KV_HEADS
    mha_prefix = Symbol(:layer_, i, :_mha)
    for dst in (Symbol(mha_prefix,:_kcache_h,h), Symbol(mha_prefix,:_vcache_h,h))
        hist_seq   = Array(g.nodes[ns_seq][dst].aux_data[:history])
        hist_batch = Array(g.nodes[ns_batch][dst].aux_data[:history])
        size(hist_seq) == size(hist_batch) || error("Formes différentes pour :$dst")
        d = maximum(abs.(hist_seq .- hist_batch))
        global max_hist_err = max(max_hist_err, d)
    end
end
_log("  écart absolu max sur tout l'historique K/V : $max_hist_err")

_log("── (2) Logits au premier pas de génération : A vs B vs recalcul complet ──")
d_AB = maximum(abs.(logits_seq_row .- logits_batch_row))
d_AC = maximum(abs.(logits_seq_row .- Float32.(full_out[end, :])))
_log("  écart abs max A vs B : $d_AB   A vs recalcul complet : $d_AC")
_log("  argmax A=$(argmax(logits_seq_row)-1)  argmax B=$(argmax(logits_batch_row)-1)  (0-indexé HF)")

_log("── (3) Continuation générée (cache seul), A vs B, $N_GEN_STEPS pas ──")
cur_step_seq = length(PROMPT); cur_step_batch = length(PROMPT)
row_seq = logits_seq_row; row_batch = logits_batch_row
all_match = true; max_gen_err = 0f0
for step in 1:N_GEN_STEPS
    am_seq = argmax(row_seq); am_batch = argmax(row_batch)
    match = am_seq == am_batch
    global all_match &= match
    d = maximum(abs.(row_seq .- row_batch))
    global max_gen_err = max(max_gen_err, d)
    _log("  pas $step : argmax_seq=$(am_seq-1)  argmax_batch=$(am_batch-1)  match=$match  écart=$d")
    global cur_step_seq += 1; global cur_step_batch += 1
    global row_seq = cached_step!(g, ns_seq, logits_seq, am_seq, cur_step_seq)
    global row_batch = cached_step!(g, ns_batch, logits_batch, am_batch, cur_step_batch)
end

println("\n══════════════ VERDICT ══════════════")
println("  écart max aux_data[:history] (A vs B) : ", max_hist_err)
println("  écart max logits continuation (A vs B) : ", max_gen_err)
println("  argmax identiques sur tous les pas : ", all_match)
if max_hist_err < 1f-2 && max_gen_err < 1f-2 && all_match
    println("  PASS -- le remplissage batché du préfixe produit un état de cache et une continuation équivalents au remplissage séquentiel, sur Qwen2.5-1.5B réel.")
else
    println("  FAIL -- divergence réelle, ne pas utiliser dans chat().")
end
flush(stdout)
