# Diagnostic : où apparaît le premier NaN ? (1) poids chargés, (2) activations
# passage avant, dans l'ordre où elles sont produites.
using NeuroDSL, JSON, LinearAlgebra

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

println("Chargement des poids sauvegardés...")
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)

println("\n", "═"^74); println("(1) Poids chargés -- recherche de NaN/Inf, AVANT tout passage avant"); println("═"^74)
nbad_w = 0
for (sym, nd) in g.nodes[ns]
    nd.is_param || continue
    nd.value === nothing && continue
    v = Array(nd.value)
    nnan = count(isnan, v); ninf = count(isinf, v)
    if nnan > 0 || ninf > 0
        global nbad_w += 1
        println("  PARAM SUSPECT : $sym  shape=$(size(v))  nan=$nnan  inf=$ninf  min=$(minimum(filter(isfinite,v); init=NaN))  max=$(maximum(filter(isfinite,v); init=NaN))")
    end
end
println("Total paramètres avec NaN/Inf : $nbad_w / $(count(nd.is_param for (_,nd) in g.nodes[ns]))")

# Quelques poids-clé inspectés explicitement, même s'ils sont "propres" selon
# le scan ci-dessus -- pour avoir des chiffres de référence (échelle typique).
for sym in (:tok_E, :layer_1_mha_q_W, :layer_1_mha_k_W, :layer_1_mha_v_W,
            :layer_1_norm1_gamma, :layer_1_mlp_w1, :final_norm_gamma, :lm_head_W)
    nd = NeuroDSL.node(g, sym; namespace=ns)
    v = Array(nd.value)
    println("  $sym : shape=$(size(v))  nan=$(count(isnan,v))  inf=$(count(isinf,v))  ",
            "min=$(minimum(v))  max=$(maximum(v))  mean=$(sum(v)/length(v))")
end

println("\n", "═"^74); println("(2) Passage avant sur les tokens du prompt 1 -- première activation avec NaN"); println("═"^74)
ref = JSON.parsefile(joinpath(MODEL_DIR, "reference_logits.json"))
tokens = Int.(ref[1]["token_ids"]) .+ 1
println("Tokens (1-idx pour NeuroDSL) : ", tokens)

NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, logits_sym; namespace=ns)  # force le calcul complet
cache = NeuroDSL.capture_activations(g, ns)

# Ordre topologique approximatif : on trie par nom de couche puis par ordre
# de construction connu (tok -> layer_1_* -> layer_2_* -> ... -> final_norm ->
# lm_head). Comme on n'a pas un vrai ordre topo exposé ici, on imprime
# `tok_out`, puis pour CHAQUE couche un sous-ensemble de nœuds clés dans
# l'ordre où `LlamaBlock`/`MultiHeadAttention` les construit.
function check(sym)
    haskey(cache, sym) || return nothing
    v = Array(cache[sym])
    nnan = count(isnan, v); ninf = count(isinf, v)
    return (; sym, shape=size(v), nnan, ninf,
             finite_min = nnan+ninf < length(v) ? minimum(filter(isfinite,v)) : NaN,
             finite_max = nnan+ninf < length(v) ? maximum(filter(isfinite,v)) : NaN)
end

function report(sym)
    r = check(sym)
    r === nothing && (println("  (absent du cache) ", sym); return false)
    bad = r.nnan > 0   # -Inf du masque causal est ATTENDU (10/25 pour 5x5) -- pas un bug
    println("  ", bad ? "❌" : "✅", " ", rpad(String(sym), 28), " shape=$(r.shape)  nan=$(r.nnan) inf=$(r.ninf)  min=$(r.finite_min) max=$(r.finite_max)")
    return bad
end

println("-- embedding --")
report(:tok_out)

println("-- couche 1, dans l'ordre de construction --")
for sym in (:layer_1_norm1_out, :layer_1_mha_q_out, :layer_1_mha_k_out, :layer_1_mha_v_out,
            :layer_1_mha_q_h1, :layer_1_mha_q_h1_rope, :layer_1_mha_k_h1, :layer_1_mha_k_h1_rope,
            :layer_1_mha_sc3, :layer_1_mha_sc_h1, :layer_1_mha_sk_h1, :layer_1_mha_pr_h1,
            :layer_1_mha_ao3, :layer_1_mha_ao_h1, :layer_1_mha_concat, :layer_1_mha_output_out,
            :layer_1_res1, :layer_1_norm2_out, :layer_1_mlp_out, :layer_1_out)
    bad = report(sym)
    bad && println("     ^^^ PREMIER NaN TROUVÉ ICI (couche 1) -- arrêt de l'inspection détaillée")
    bad && break
end

println("-- vérifs globales couches 2..28, norme finale, lm_head --")
for l in 2:N_LAYERS
    report(Symbol("layer_$(l)_out"))
end
report(:final_norm_out)
report(logits_sym)
