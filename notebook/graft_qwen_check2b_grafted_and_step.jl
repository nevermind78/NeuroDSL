# ══════════════════════════════════════════════════════════════════════════════
# graft_qwen_check2b_grafted_and_step.jl — Process B du Check 2
# (graft_qwen_correctness_preregistration.md) :
#   (i)  greffe Gradient Shadowing (alpha0=0) sur Qwen réel, logits AVANT tout
#        entraînement comparés bit-à-bit à Process A (graft_qwen_check2a_*.jl,
#        exécuté séparément, résultats relus depuis JSON).
#   (ii) backbone gelé, UN pas AdamW sur les seuls paramètres de la greffe,
#        alpha relu avant/après.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, Printf, Random

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6
const GRAFT_SITE = :layer_25_out
const GRAFT_PREFIX = :qwen_shadow_l25
const GRAFT_HEADS, GRAFT_HIDDEN = 4, 384

println("── Check 2, Process B : greffe (alpha0=0) + un pas AdamW ──"); flush(stdout)

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
label_tokens = tokens0[2:end]
@printf("Prompt réel : %s (%d tokens d'entrée)\n", repr(prompt_text), length(input_tokens))
flush(stdout)

Random.seed!(1234)
NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.seed!(1234)
println("Insertion de la greffe Gradient Shadowing à :$(GRAFT_SITE) (alpha0=0)..."); flush(stdout)
new_out, handle = NeuroDSL.graft_shadow_block!(g, ns, GRAFT_SITE, DIM, GRAFT_HEADS, GRAFT_HIDDEN;
                                                alpha0=0f0, zero_out_proj=false, prefix=GRAFT_PREFIX)
println("Greffe insérée -> $new_out ; alpha_sym=$(handle.alpha_sym)"); flush(stdout)

alpha_before = Float64(Array(NeuroDSL.node(g, handle.alpha_sym; namespace=ns).value)[1])
@printf("alpha AVANT tout entraînement = %.10f (attendu exactement 0.0)\n", alpha_before)
flush(stdout)

# ── (i) Logits AVANT entraînement, comparés bit-à-bit à Process A ──
NeuroDSL.set!(g, :token_ids, input_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:length(input_tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
logits_grafted = Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))
@printf("logits (greffé, alpha=0) shape = %s\n", size(logits_grafted))
flush(stdout)

baseline_path = joinpath(@__DIR__, "graft_qwen_check2_baseline_logits.json")
bitexact_ok = false
max_abs_diff = NaN
if isfile(baseline_path)
    ref = JSON.parsefile(baseline_path)
    ref_shape = Tuple(Int.(ref["shape"]))
    ref_logits = reshape(Float64.(ref["logits_flat"]), ref_shape)
    if size(logits_grafted) == ref_shape
        max_abs_diff = maximum(abs.(Float64.(logits_grafted) .- ref_logits))
        bitexact_ok = max_abs_diff == 0.0
        @printf("Comparaison vs Process A (baseline sans greffe) : écart absolu max = %.3e -- bit-exact = %s\n",
                max_abs_diff, bitexact_ok)
    else
        println("MISMATCH DE FORME entre logits greffés et baseline -- comparaison impossible.")
    end
else
    println("ATTENTION : ", baseline_path, " introuvable -- lancer graft_qwen_check2a_baseline_logits.jl d'abord.")
end
flush(stdout)

# ── (ii) Gel du backbone + un pas AdamW sur les seuls paramètres de la greffe ──
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
n_frozen = freeze_backbone!(g, ns, GRAFT_PREFIX)
@printf("Backbone gelé : %d nœuds -> is_param=false.\n", n_frozen)

ps = NeuroDSL.params(g; namespace=ns)
@printf("params(g;ns) après gel : %d tenseurs entraînables (devrait être uniquement la greffe).\n", length(ps))
flush(stdout)

NeuroDSL.set!(g, :labels, label_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [logits_sym, :labels], :cross_entropy; namespace=ns))
NeuroDSL.invalidate_all!(g; namespace=ns)
loss_val = NeuroDSL.demand!(g, :loss; namespace=ns)
@printf("loss (greffé, avant pas) = %.6f\n", Float64(sum(Array(loss_val))))
flush(stdout)

NeuroDSL.backward_graph!(g, :loss; namespace=ns, prune_frozen=true)

m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps], [p.gradient for p in ps],
                             m1s, m2s, 1f-3, 0.9f0, 0.999f0, 1f-8, 1, 1f0, 0f0)
println("Un pas AdamW effectué sur les ", length(ps), " tenseurs de la greffe."); flush(stdout)

alpha_after = Float64(Array(NeuroDSL.node(g, handle.alpha_sym; namespace=ns).value)[1])
@printf("alpha APRÈS un pas AdamW = %.10f (attendu != 0.0)\n", alpha_after)
flush(stdout)

pass_a = alpha_before == 0.0
pass_b = bitexact_ok
pass_c = alpha_after != 0.0
verdict = pass_a && pass_b && pass_c
println("\n", "="^80)
println("CHECK 2 VERDICT")
println("  (a) alpha exactement 0.0 avant entraînement : ", pass_a)
println("  (b) logits greffés (alpha=0) bit-exacts vs sans greffe : ", pass_b, "  (écart max=", max_abs_diff, ")")
println("  (c) alpha != 0.0 après UN pas AdamW : ", pass_c, "  (valeur=", alpha_after, ")")
println("  PASS global : ", verdict)
println("="^80)

open(joinpath(@__DIR__, "graft_qwen_check2_results.json"), "w") do io
    JSON.print(io, Dict(
        "alpha_before" => alpha_before, "alpha_after" => alpha_after,
        "bitexact_ok" => bitexact_ok, "max_abs_diff_logits" => max_abs_diff,
        "n_frozen" => n_frozen, "n_graft_params" => length(ps),
        "loss_before_step" => Float64(sum(Array(loss_val))),
        "pass_a" => pass_a, "pass_b" => pass_b, "pass_c" => pass_c, "verdict" => verdict,
    ))
end
println("Résultats écrits -> notebook/graft_qwen_check2_results.json")
