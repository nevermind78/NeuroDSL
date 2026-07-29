# ══════════════════════════════════════════════════════════════════════════════
# kv_cache_prefix_prime_toy_gate.jl — Porte de correction pour
# `NeuroDSL.prime_kv_cache_from_prefix!` (src/layers.jl, 2026-07-29), sur un
# modèle jouet CPU. Motif : remplir le cache KV un token à la fois pour tout
# le PRÉFIXE coûtait ~0.5s/token (overhead fixe de `demand!` répété), pour un
# gain quasi nul par rapport au recalcul complet -- corrigé en remplissant le
# préfixe en UN SEUL passage avant batché (comme le ferait n'importe quel
# forward pass), puis en copiant directement le K/V résultant dans
# `aux_data[:history]` des nœuds de cache. Exigence explicite avant de faire
# confiance à ce raccourci : le résultat doit être IDENTIQUE, pas seulement
# plausible, à un remplissage séquentiel token par token.
#
# Compare TROIS chemins sur le MÊME préfixe + continuation :
#   A. remplissage séquentiel (`cached_step!` pour CHAQUE token du préfixe,
#      comme avant le correctif) -- référence "ancienne méthode", déjà
#      validée par kv_cache_toy_gate.jl.
#   B. remplissage batché (`prime_kv_cache_from_prefix!`) -- nouvelle méthode.
#   C. recalcul complet à chaque longueur de préfixe -- référence indépendante
#      des deux (déjà validée toute la session).
# Vérifie : (1) aux_data[:history] de B == aux_data[:history] de A, couche par
# couche, tête par tête, élément par élément ; (2) les logits du premier token
# généré sont identiques entre A, B et C ; (3) la continuation générée à partir
# de B est identique à celle de A sur plusieurs pas.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL

println("── Porte de correction : remplissage batché du préfixe vs séquentiel ──"); flush(stdout)

const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 32, 2, 4, 2, 64, 50
const ROPE_THETA, QKV_BIAS = 10000f0, true
const PROMPT = [3, 17, 8, 41, 2, 26]
const N_GEN_STEPS = 5   # pas de génération supplémentaires à comparer après le préfixe

dev = NeuroDSL.Backend.CPUDevice()
ns_load = :toy_load
ns_seq  = :toy_seq     # cache rempli SÉQUENTIELLEMENT (ancienne méthode)
ns_batch = :toy_batch   # cache rempli PAR LOT (nouvelle méthode, prime_kv_cache_from_prefix!)
g = NeuroDSL.NeuroGraph(namespace=ns_load, device=dev)

println("Construction du graphe de référence (recalcul complet)..."); flush(stdout)
NeuroDSL.set!(g, :token_ids, ones(Int, length(PROMPT)); atom_type=NeuroDSL.Datom, namespace=ns_load)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns_load)
out = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                           batched_attn=false, n_kv_heads=N_KV_HEADS,
                           qkv_bias=QKV_BIAS, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns_load)
final_norm = NeuroDSL.LayerNorm(DIM)(g, out, :final_norm; namespace=ns_load)
logits_full = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns_load)

println("Copie des poids + construction des deux graphes cachés..."); flush(stdout)
NeuroDSL.copy_params_to_namespace!(g, ns_load, ns_seq)
NeuroDSL.copy_params_to_namespace!(g, ns_load, ns_batch)
logits_seq = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=QKV_BIAS, use_rope=true, rope_theta=ROPE_THETA, namespace=ns_seq)
logits_batch = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=QKV_BIAS, use_rope=true, rope_theta=ROPE_THETA, namespace=ns_batch)
println("Graphes construits.\n"); flush(stdout)

function cached_step!(g, ns, logits_sym, tok1::Int, cur_step::Int)   # tok1 : 1-indexé
    NeuroDSL.set!(g, :dec_token_id, [tok1]; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :dec_cur_step, Float32[cur_step]; namespace=ns)
    NeuroDSL.set!(g, :dec_pos, Float32[cur_step-1]; namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))[1, :]
end

# -- A. Remplissage SÉQUENTIEL du préfixe (namespace ns_seq) --
println("── A. Remplissage séquentiel (ns_seq) ──"); flush(stdout)
logits_seq_row = Float32[]   # liaison globale normale -- PAS `local`, voir kv_cache_chat_timing_probe.jl
for (t, tok) in enumerate(PROMPT)
    global logits_seq_row = cached_step!(g, ns_seq, logits_seq, tok, t)
end
println("  fait -- logits[1:5] = ", round.(logits_seq_row[1:5], digits=4)); flush(stdout)

# -- B. Remplissage BATCHÉ du préfixe (namespace ns_batch, via ns_load) --
println("── B. Remplissage batché (ns_batch, via prime_kv_cache_from_prefix!) ──"); flush(stdout)
NeuroDSL.set!(g, :token_ids, PROMPT; atom_type=NeuroDSL.Datom, namespace=ns_load)
NeuroDSL.invalidate_all!(g; namespace=ns_load)
full_out = Array(NeuroDSL.demand!(g, logits_full; namespace=ns_load))
logits_batch_row = Float32.(full_out[end, :])   # dernière position du préfixe
n_primed = NeuroDSL.prime_kv_cache_from_prefix!(g; src_ns=ns_load, dst_ns=ns_batch,
    n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, use_rope=true)
println("  $n_primed tableaux K/V amorcés (attendu $(2*N_LAYERS*N_KV_HEADS))."); flush(stdout)
println("  logits[1:5] = ", round.(logits_batch_row[1:5], digits=4)); flush(stdout)

# -- (1) aux_data[:history] : A vs B, élément par élément --
println("\n── (1) Comparaison directe de aux_data[:history] (A vs B) ──"); flush(stdout)
max_hist_err = 0f0
for i in 1:N_LAYERS, h in 1:N_KV_HEADS
    mha_prefix = Symbol(:layer_, i, :_mha)
    for (kind, dst) in (("K", Symbol(mha_prefix,:_kcache_h,h)), ("V", Symbol(mha_prefix,:_vcache_h,h)))
        hist_seq   = g.nodes[ns_seq][dst].aux_data[:history]
        hist_batch = g.nodes[ns_batch][dst].aux_data[:history]
        size(hist_seq) == size(hist_batch) || error("Formes différentes pour :$dst -- $(size(hist_seq)) vs $(size(hist_batch))")
        d = maximum(abs.(hist_seq .- hist_batch))
        global max_hist_err = max(max_hist_err, d)
    end
end
println("  écart absolu max sur TOUT l'historique K/V, toutes couches/têtes : ", max_hist_err); flush(stdout)

# -- (2) logits du premier token généré : A vs B vs C(=recalcul complet) --
println("\n── (2) Logits au premier pas de génération : A vs B vs référence recalcul complet ──"); flush(stdout)
d_AB = maximum(abs.(logits_seq_row .- logits_batch_row))
d_AC = maximum(abs.(logits_seq_row .- Float32.(full_out[end, :])))
println("  écart abs max A vs B : ", d_AB, "   A vs recalcul complet (=C, doit être ~0) : ", d_AC); flush(stdout)
println("  argmax A=", argmax(logits_seq_row), "  argmax B=", argmax(logits_batch_row)); flush(stdout)

# -- (3) continuation générée : A vs B sur plusieurs pas --
println("\n── (3) Continuation générée (cache seul), A vs B, $N_GEN_STEPS pas ──"); flush(stdout)
cur_step_seq = length(PROMPT)
cur_step_batch = length(PROMPT)
row_seq = logits_seq_row
row_batch = logits_batch_row
all_match = true
max_gen_err = 0f0
for step in 1:N_GEN_STEPS
    am_seq = argmax(row_seq); am_batch = argmax(row_batch)
    match = am_seq == am_batch
    global all_match &= match
    d = maximum(abs.(row_seq .- row_batch))
    global max_gen_err = max(max_gen_err, d)
    println("  pas $step : argmax_seq=$am_seq  argmax_batch=$am_batch  match=$match  écart=$d"); flush(stdout)
    global cur_step_seq += 1; global cur_step_batch += 1
    global row_seq = cached_step!(g, ns_seq, logits_seq, am_seq, cur_step_seq)
    global row_batch = cached_step!(g, ns_batch, logits_batch, am_batch, cur_step_batch)
end

println("\n══════════════ VERDICT ══════════════")
println("  écart max aux_data[:history] (A vs B) : ", max_hist_err)
println("  écart max logits continuation (A vs B) : ", max_gen_err)
println("  argmax identiques sur tous les pas : ", all_match)
if max_hist_err < 1f-5 && max_gen_err < 1f-3 && all_match
    println("  PASS -- le remplissage batché du préfixe produit un état de cache et une continuation équivalents au remplissage séquentiel.")
else
    println("  FAIL -- divergence réelle, ne pas utiliser dans chat().")
end
flush(stdout)
