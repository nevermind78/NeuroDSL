# ══════════════════════════════════════════════════════════════════════════════
# Étape 2 (suite) — profil de recovery PAR COUCHE (pas un balayage multi-sites
# cumulatif). Constat empirique (calibration précédente) : avec max_sites=8 sur
# 24 candidats, la recherche gloutonne sature systématiquement >0.98 dès le pas
# 500, y compris quand query_acc est proche du hasard -- parce que toute
# l'influence de la corruption (position 2, différente de q_pos) ne peut voyager
# vers q_pos QUE via l'attention (les embeddings des autres positions sont
# inchangés), et 8/24 sites suffisent presque toujours à la reconstituer,
# indépendamment de si le modèle a appris le bon algorithme. La recovery
# multi-sites globale n'est donc PAS le bon indicateur de calibration.
#
# Ce qui compte pour le protocole goulot-vs-témoin (Étape 3) : la recovery
# obtenue en patchant TOUTES les têtes d'UNE SEULE couche à la fois (analogue
# exact du "ceiling" du protocole texte réel, mais couche par couche, pas
# seulement couche 1) -- si le profil varie franchement d'une couche à l'autre
# sur un modèle BIEN entraîné, on a un vrai goulot (couche au profil le plus
# haut) et un vrai témoin (couche au profil le plus bas, à profondeur comparable
# si possible) pour la greffe dirigée.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, LinearAlgebra, Printf

dev = NeuroDSL.Backend.CUDADevice()
vocab_size, dim, n_heads, hidden_dim, n_layers, n_hops = 30, 64, 4, 128, 6, 3
n_steps = 10_000

Random.seed!(n_hops)
ns = :layer_profile
g, logits_sym = NeuroDSL.build_multihop_graph(dev, ns; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                               hidden_dim=hidden_dim, n_layers=n_layers, n_hops=n_hops)

rng_train = MersenneTwister(123)
ps = NeuroDSL.params(g; namespace=ns)
m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
for t in 1:n_steps
    tokens, labels, _ = NeuroDSL.sample_multihop_sequence(rng_train, vocab_size, n_hops)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, :loss; namespace=ns)
    NeuroDSL.backward_graph!(g, :loss; namespace=ns)
    for (i, p) in enumerate(ps)
        NeuroDSL.adamw_step!(dev, p.value, p.gradient, m1s[i], m2s[i], 3f-3, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
    end
    NeuroDSL.invalidate_all!(g; namespace=ns)
end
acc = NeuroDSL.evaluate_multihop(g, logits_sym, ns; vocab_size=vocab_size, n_hops=n_hops, n_eval=300)
@printf("Entraînement terminé (n_hops=%d, %d pas) : query_acc=%.3f  body_acc=%.3f\n",
        n_hops, n_steps, acc.query_acc, acc.body_acc)

# ── Corruption fixe (position 2, comme la calibration précédente) ──────────
rng_eval = MersenneTwister(999)
tokens, labels, q_pos = NeuroDSL.sample_multihop_sequence(rng_eval, vocab_size, n_hops)
tokens_corrupt = copy(tokens)
orig = tokens_corrupt[2]
new_id = orig
while new_id == orig || new_id in tokens
    global new_id = rand(rng_eval, 1:vocab_size)
end
tokens_corrupt[2] = new_id

NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
clean_output = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))
clean_cache = NeuroDSL.capture_activations(g, ns)

NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
corrupted_output = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))
corrupted_cache = NeuroDSL.capture_activations(g, ns)

row_metric(out) = NeuroDSL.recovery_metric(Array(out)[q_pos:q_pos, :], clean_output[q_pos:q_pos, :], corrupted_output[q_pos:q_pos, :])

println("\nProfil de recovery par couche (toutes les têtes de la couche patchées ensemble) :")
per_layer = NamedTuple[]
for L in 1:n_layers
    heads = [Symbol(:layer_, L, :_mha_ao_h, h) for h in 1:n_heads]
    NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, logits_sym; namespace=ns)
    NeuroDSL.patch_nodes!(g, heads, clean_cache; namespace=ns)
    out = NeuroDSL.demand!(g, logits_sym; namespace=ns)
    r = row_metric(out)
    push!(per_layer, (; layer=L, recovery=r))
    @printf("  couche %d : recovery = %.4f\n", L, r)
end

vals = [x.recovery for x in per_layer]
@printf("\némax=%.4f (couche %d)  émin=%.4f (couche %d)  écart=%.4f\n",
        maximum(vals), per_layer[argmax(vals)].layer, minimum(vals), per_layer[argmin(vals)].layer,
        maximum(vals) - minimum(vals))
