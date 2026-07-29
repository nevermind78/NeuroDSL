# ══════════════════════════════════════════════════════════════════════════════
# kv_cache_toy_gate.jl — Porte de parité #1 (obligatoire avant Qwen) : logits
# du décodage incrémental avec cache KV (`build_cached_decode_graph!`,
# src/layers.jl) VS logits du recalcul complet existant (`LlamaModel`, chemin
# déjà validé toute la session), sur un petit modèle jouet CPU.
#
# RÉVISION 2026-07-29 -- namespaces séparés, PAS un namespace partagé : la
# toute première version de ce script (et de kv_cache_qwen_gate.jl) mettait
# les deux graphes dans le MÊME namespace, poids partagés via le MÊME
# GraphNode. Ça a semblé marcher ici (PASS, erreur 4.8e-7) mais UNIQUEMENT
# parce que ce script calculait TOUS les logits de référence d'abord, PUIS
# TOUS les logits cachés ensuite (deux boucles séquentielles, jamais
# entrelacées) -- sur Qwen, où les deux chemins sont entrelacés pas par pas
# (fidèle à l'usage réel de génération), le même partage de namespace a
# corrompu le cache KV et fait planter le script (voir l'erreur détaillée
# dans kv_cache_qwen_gate.jl et le commentaire de `copy_params_to_namespace!`,
# src/graph_api.jl, pour le mécanisme exact : `demand!` exécute tout nœud
# invalide qu'il rencontre dans SON namespace avant sa cible, pas seulement
# les ancêtres de la cible -- un nœud à état comme `:kv_cache_append` peut
# donc être ré-exécuté comme effet de bord d'un `demand!` qui ne le
# concerne pas). Ce script est maintenant réécrit pour (a) utiliser deux
# namespaces séparés + `copy_params_to_namespace!` pour partager les poids
# PAR VALEUR, et (b) entrelacer référence et cache pas par pas -- fidèle à
# l'usage réel et au script Qwen -- pour ne plus dépendre d'un ordre
# d'exécution qui a pu accidentellement éviter le bug la première fois.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL

println("── Porte de parité jouet : cache KV vs recalcul complet ──"); flush(stdout)

const DIM        = 32
const N_LAYERS    = 2
const N_HEADS     = 4
const N_KV_HEADS  = 2      # GQA réel (group_size=2), pas juste MHA (n_kv_heads=n_heads)
const HIDDEN_DIM  = 64
const VOCAB_SIZE  = 50
const ROPE_THETA  = 10000f0
const QKV_BIAS    = true   # exerce aussi le chemin :linear (pas seulement :matmul)

dev = NeuroDSL.Backend.CPUDevice()
ns_full   = :toy_full     # graphe de référence (recalcul complet)
ns_cached = :toy_cached   # graphe de décodage incrémental (cache KV), namespace SÉPARÉ
g = NeuroDSL.NeuroGraph(namespace=ns_full, device=dev)

const PROMPT = [3, 17, 8, 41, 2, 26]   # 6 tokens, 1-indexés (convention embedding NeuroDSL)
const SEQ_LEN = length(PROMPT)

println("Construction du graphe recalcul complet (namespace :$ns_full)..."); flush(stdout)
NeuroDSL.set!(g, :token_ids, ones(Int, SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns_full)

tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns_full)
out = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                           batched_attn=false, n_kv_heads=N_KV_HEADS,
                           qkv_bias=QKV_BIAS, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns_full)
final_norm = NeuroDSL.LayerNorm(DIM)(g, out, :final_norm; namespace=ns_full)
logits_full = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns_full)
println("Graphe recalcul complet construit. Poids créés (aléatoires, fixes pour ce test)."); flush(stdout)

println("Copie des poids vers le namespace caché (:$ns_cached), PAR VALEUR..."); flush(stdout)
n_copied = NeuroDSL.copy_params_to_namespace!(g, ns_full, ns_cached)
println("  $n_copied paramètres copiés."); flush(stdout)

println("Construction du graphe caché (namespace :$ns_cached, poids copiés ci-dessus)..."); flush(stdout)
logits_cached = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=QKV_BIAS, use_rope=true, rope_theta=ROPE_THETA, namespace=ns_cached)
println("Graphe caché construit.\n"); flush(stdout)

function run_forward!(g, ns, tokens::Vector{Int})
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, logits_full; namespace=ns))
end

# ── Comparaison ENTRELACÉE pas par pas -- référence puis cache à CHAQUE t,
# comme sur Qwen, pour exercer exactement le scénario qui a révélé le bug
# de partage de namespace (et vérifier qu'il ne se reproduit plus).
println("── Décodage entrelacé (référence puis cache, à chaque pas) ──"); flush(stdout)
max_abs_err = 0f0
all_argmax_match = true
for t in 1:SEQ_LEN
    prefix = PROMPT[1:t]
    ref_out = run_forward!(g, ns_full, prefix)
    ref_row = Float32.(ref_out[t, :])

    NeuroDSL.set!(g, :dec_token_id, [PROMPT[t]]; atom_type=NeuroDSL.Datom, namespace=ns_cached)
    NeuroDSL.set!(g, :dec_cur_step, Float32[t]; namespace=ns_cached)
    NeuroDSL.set!(g, :dec_pos, Float32[t-1]; namespace=ns_cached)
    NeuroDSL.invalidate_all!(g; namespace=ns_cached)
    cached_row = Float32.(Array(NeuroDSL.demand!(g, logits_cached; namespace=ns_cached))[1, :])

    d = maximum(abs.(ref_row .- cached_row))
    global max_abs_err = max(max_abs_err, d)
    am_ref = argmax(ref_row); am_cached = argmax(cached_row)
    match = am_ref == am_cached
    global all_argmax_match &= match
    println("  pas $t : erreur abs max = $(d)   argmax_ref=$am_ref  argmax_cache=$am_cached  match=$match")
    flush(stdout)
end

println("\n══════════════ RÉSULTAT PORTE DE PARITÉ JOUET ══════════════")
println("Erreur absolue maximale (sur tous les pas, tous les logits) : ", max_abs_err)
println("Tous les argmax identiques : ", all_argmax_match)
if max_abs_err < 1f-3 && all_argmax_match
    println("VERDICT : PASS -- le chemin caché reproduit le recalcul complet.")
else
    println("VERDICT : FAIL -- divergence réelle, à ne PAS utiliser sur Qwen tel quel.")
end
flush(stdout)
