# Suite du diagnostic : ao3 (batched_pv) a des NaN alors que pr_h1 est propre
# -- vérifier TOUTES les têtes (pr_h1..pr_h12, sk_h1..sk_h12, vh1..vh2) pour
# localiser précisément où le NaN apparaît avant/dans batched_pv.
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

function nanrows(sym)
    haskey(cache, sym) || return "absent"
    v = Array(cache[sym])
    n = count(isnan, v)
    if n == 0
        return "propre  min=$(minimum(v)) max=$(maximum(v))"
    end
    # localise : lignes (positions de séquence) contenant au moins un NaN
    bad_rows = [i for i in axes(v,1) if any(isnan, v isa AbstractMatrix ? v[i,:] : v[i,:,:])]
    return "NAN=$n  lignes_affectées=$bad_rows  shape=$(size(v))"
end

println("-- vh (têtes KV de base, avant répétition) --")
for h in 1:N_KV_HEADS
    println("  layer_1_mha_v_h$h : ", nanrows(Symbol("layer_1_mha_v_h$h")))
end
println("-- kh_rope (têtes KV de base, après RoPE) --")
for h in 1:N_KV_HEADS
    println("  layer_1_mha_k_h$(h)_rope : ", nanrows(Symbol("layer_1_mha_k_h$(h)_rope")))
end
println("-- sc_h / sk_h / pr_h pour CHAQUE tête de requête --")
for h in 1:N_HEADS
    println("  h=$h  sc_h=", nanrows(Symbol("layer_1_mha_sc_h$h")),
            "  |  sk_h=", nanrows(Symbol("layer_1_mha_sk_h$h")),
            "  |  pr_h=", nanrows(Symbol("layer_1_mha_pr_h$h")))
end
println("-- ao3 (sortie batched_pv) et ao_h par tête --")
v = Array(cache[:layer_1_mha_ao3])
println("  ao3 shape=", size(v), "  nan total=", count(isnan,v))
for h in 1:N_HEADS
    slice = v[:,:,h]
    println("    tête $h : nan=", count(isnan,slice), "  (", count(isnan,slice)>0 ? "lignes=$([i for i in 1:size(slice,1) if any(isnan,slice[i,:])])" : "propre", ")")
end
