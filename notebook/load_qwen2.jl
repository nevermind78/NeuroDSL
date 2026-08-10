# ══════════════════════════════════════════════════════════════════════════════
# load_qwen2.jl — charge Qwen2.5-1.5B-Instruct (poids HuggingFace réels, jamais
# entraîné par NeuroDSL) dans un NeuroGraph, via safetensors_reader.jl (lecteur
# maison, aucune nouvelle dépendance) + le câblage GQA/RoPE additif de
# src/layers.jl (2026-07-28).
#
# Construit le graphe avec `LlamaModel(...; n_kv_heads=2, use_rope=true,
# rope_theta=1e6, qkv_bias=<vérifié>)`, PUIS écrase chaque poids créé
# aléatoirement par sa valeur réelle lue dans model.safetensors -- même
# discipline que `set_params!` déjà utilisé partout dans ce dépôt (écrit dans
# le nœud existant, ne change pas la structure du graphe).
#
# Table de correspondance noms HF -> symboles NeuroDSL, propre à l'architecture
# Qwen2 (`Qwen2ForCausalLM`) -- PAS un mécanisme générique, une table statique
# pour CETTE famille de modèles, comme prévu.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, LinearAlgebra, JSON, CUDA
include(joinpath(@__DIR__, "safetensors_reader.jl"))

# `MODEL_DIR` réglable sans éditer ce fichier -- via la variable d'environnement
# QWEN_MODEL_DIR (utile pour basculer 1.5B/7B/... sur le même dépôt), avec le
# 1.5B comme défaut historique inchangé si elle n'est pas définie.
const MODEL_DIR = get(ENV, "QWEN_MODEL_DIR", joinpath(@__DIR__, "qwen2.5-1.5b-instruct"))

# ── Config lue DYNAMIQUEMENT depuis config.json (fini le "à la main") --
# fonctionne pour n'importe quelle taille de la famille Qwen2/Qwen2.5 sans
# jamais recopier des valeurs devinées ou mal transcrites.
const HF_CONFIG  = JSON.parsefile(joinpath(MODEL_DIR, "config.json"))
const DIM        = HF_CONFIG["hidden_size"]
const N_LAYERS   = HF_CONFIG["num_hidden_layers"]
const N_HEADS    = HF_CONFIG["num_attention_heads"]
const N_KV_HEADS = get(HF_CONFIG, "num_key_value_heads", N_HEADS)
const D_HEAD     = DIM ÷ N_HEADS
const HIDDEN_DIM = HF_CONFIG["intermediate_size"]          # SwiGLU
const VOCAB_SIZE = HF_CONFIG["vocab_size"]
const ROPE_THETA = Float64(get(HF_CONFIG, "rope_theta", 1_000_000.0))
const RMS_EPS    = Float64(get(HF_CONFIG, "rms_norm_eps", 1e-6))
const TIE_EMBED  = get(HF_CONFIG, "tie_word_embeddings", true)
println("Config lue depuis $(joinpath(MODEL_DIR, "config.json")) :")
println("  dim=$DIM  n_layers=$N_LAYERS  n_heads=$N_HEADS  n_kv_heads=$N_KV_HEADS  d_head=$D_HEAD  hidden_dim=$HIDDEN_DIM  vocab=$VOCAB_SIZE")
println("  rope_theta=$ROPE_THETA  rms_eps=$RMS_EPS  tie_embed=$TIE_EMBED")

# `open_any_safetensors` détecte seul le format sharded (plusieurs fichiers +
# model.safetensors.index.json, ex. Qwen2.5-7B découpé en 4) ou fichier unique
# (ex. Qwen2.5-1.5B) -- voir safetensors_reader.jl.
st = open_any_safetensors(MODEL_DIR)
names = tensor_names(st)
println("Nombre de tenseurs : ", length(names))

# Détection : biais QKV présents ? (Qwen2 en a historiquement -- vérifié ici,
# pas supposé)
has_qkv_bias = any(n -> occursin("self_attn.q_proj.bias", n), names)
has_lm_head  = any(n -> n == "lm_head.weight", names)
println("qkv_bias détecté : ", has_qkv_bias, "   lm_head.weight présent : ", has_lm_head,
        " (tie_word_embeddings=", TIE_EMBED, " dans config.json)")

dev = NeuroDSL.Backend.CUDADevice()
ns = :qwen2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)

const SEQ_LEN_PLACEHOLDER = 8  # juste pour construire le graphe ; demand! retaillera
NeuroDSL.set!(g, :token_ids, ones(Int, SEQ_LEN_PLACEHOLDER); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN_PLACEHOLDER); atom_type=NeuroDSL.Datom, namespace=ns)

tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
# PAS d'embedding de position appris ici : RoPE remplace entièrement le
# schéma GPT-2/toy-tasks (Embedding(pos)+add) -- l'entrée du modèle est
# directement l'embedding de token.
out = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                           batched_attn=true, n_kv_heads=N_KV_HEADS,
                           qkv_bias=has_qkv_bias, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out, :final_norm; namespace=ns)
logits = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)

println("Graphe construit. Chargement des poids réels...")

# ── Table de correspondance HF -> NeuroDSL, couche par couche ───────────────
function load_layer!(g, ns, st, l_hf::Int, l_nd::Int; rms_eps_check::Bool)
    p = "model.layers.$(l_hf)."
    prefix = "layer_$(l_nd)"

    # Attention : q/k/v/o -- poids [out,in] en HF, MÊME convention que
    # NeuroDSL (Linear calcule x @ Wᵀ, W:[out,in]) -- AUCUNE transposition
    # supplémentaire au-delà de ce que `read_tensor` fait déjà pour corriger
    # le row-major PyTorch -> column-major Julia.
    NeuroDSL.set!(g, Symbol(prefix,"_mha_q_W"), read_tensor(st, p*"self_attn.q_proj.weight"); is_param=true, namespace=ns)
    NeuroDSL.set!(g, Symbol(prefix,"_mha_k_W"), read_tensor(st, p*"self_attn.k_proj.weight"); is_param=true, namespace=ns)
    NeuroDSL.set!(g, Symbol(prefix,"_mha_v_W"), read_tensor(st, p*"self_attn.v_proj.weight"); is_param=true, namespace=ns)
    NeuroDSL.set!(g, Symbol(prefix,"_mha_output_W"), read_tensor(st, p*"self_attn.o_proj.weight"); is_param=true, namespace=ns)
    if haskey_tensor(st, p*"self_attn.q_proj.bias")
        NeuroDSL.set!(g, Symbol(prefix,"_mha_q_b"), read_tensor(st, p*"self_attn.q_proj.bias"); is_param=true, namespace=ns)
        NeuroDSL.set!(g, Symbol(prefix,"_mha_k_b"), read_tensor(st, p*"self_attn.k_proj.bias"); is_param=true, namespace=ns)
        NeuroDSL.set!(g, Symbol(prefix,"_mha_v_b"), read_tensor(st, p*"self_attn.v_proj.bias"); is_param=true, namespace=ns)
    end

    # MLP : gate=w1, up=w3, down=w2 (voir LlamaBlock -- gt utilise w1, up
    # utilise w3, la sortie finale utilise w2)
    NeuroDSL.set!(g, Symbol(prefix,"_mlp_w1"), read_tensor(st, p*"mlp.gate_proj.weight"); is_param=true, namespace=ns)
    NeuroDSL.set!(g, Symbol(prefix,"_mlp_w3"), read_tensor(st, p*"mlp.up_proj.weight"); is_param=true, namespace=ns)
    NeuroDSL.set!(g, Symbol(prefix,"_mlp_w2"), read_tensor(st, p*"mlp.down_proj.weight"); is_param=true, namespace=ns)

    # Normes : gamma de RMSNorm, 1D, pas de transposition à faire
    NeuroDSL.set!(g, Symbol(prefix,"_norm1_gamma"), read_tensor(st, p*"input_layernorm.weight"); is_param=true, namespace=ns)
    NeuroDSL.set!(g, Symbol(prefix,"_norm2_gamma"), read_tensor(st, p*"post_attention_layernorm.weight"); is_param=true, namespace=ns)
end

for l in 1:N_LAYERS
    load_layer!(g, ns, st, l-1, l; rms_eps_check=true)
    # `set!` (src/graph_api.jl) remplace chaque paramètre par un NOUVEAU
    # GraphNode -- l'ancien buffer GPU (poids aléatoires de la construction
    # initiale du graphe, ou poids d'une couche précédente si ce script est
    # relancé) devient orphelin mais n'est PAS libéré synchronement (rien
    # n'appelle Backend.free! ici, contrairement aux correctifs apportés
    # cette session à backward_with_checkpointing!). Sans ce nettoyage
    # périodique, les anciens ET les nouveaux buffers coexistent en VRAM
    # jusqu'au bon vouloir du GC -- vérifié : pour un modèle ~30 Go réels,
    # ça peut transitoirement grimper à 90+ Go (2-3x), presque un OOM sur un
    # H100 à 96 Go. Toutes les quelques couches suffit (pas besoin à chaque
    # couche -- coût du GC lui-même sinon).
    if l % 4 == 0
        GC.gc()
        NeuroDSL.Backend.CUDA_AVAILABLE && CUDA.reclaim()
    end
end
GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && CUDA.reclaim()

# Embedding de token
tok_W = read_tensor(st, "model.embed_tokens.weight")   # [vocab, dim]
NeuroDSL.set!(g, :tok_E, tok_W; is_param=true, namespace=ns)

# Norme finale
NeuroDSL.set!(g, :final_norm_gamma, read_tensor(st, "model.norm.weight"); is_param=true, namespace=ns)

# LM head : tie_word_embeddings=true -- réutilise directement tok_W (même
# tableau, pas de tenseur lm_head.weight séparé si absent du fichier ; si
# présent malgré tout, on le charge et on vérifie qu'il est numériquement
# identique à tok_W, pour ne pas suppposer la coïncidence sans la vérifier).
if has_lm_head
    lm_W = read_tensor(st, "lm_head.weight")
    same = isapprox(lm_W, tok_W; atol=1e-6)
    println("lm_head.weight présent séparément -- identique à embed_tokens.weight (tying) : ", same)
    NeuroDSL.set!(g, :lm_head_W, lm_W; is_param=true, namespace=ns)
else
    NeuroDSL.set!(g, :lm_head_W, tok_W; is_param=true, namespace=ns)
end

println("Poids chargés. Sauvegarde au format NeuroDSL natif pour ne pas re-parser le safetensors...")
NeuroDSL.save_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"))
println("OK -> ", joinpath(MODEL_DIR, "qwen2_neurodsl"), ".json/.bin")
