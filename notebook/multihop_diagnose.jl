using NeuroDSL, Random, LinearAlgebra, Printf

dev = NeuroDSL.Backend.CUDADevice()
vocab_size, dim, n_heads, hidden_dim, n_layers, n_hops = 30, 64, 4, 128, 6, 2

Random.seed!(2)
ns = :diag
g, logits_sym = NeuroDSL.build_multihop_graph(dev, ns; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                               hidden_dim=hidden_dim, n_layers=n_layers, n_hops=n_hops)

# Entraîner un peu (10000 pas, comme le dernier point de calibration où query_acc=0.97).
rng_train = MersenneTwister(123)
ps = NeuroDSL.params(g; namespace=ns)
m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
for t in 1:10000
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
println("Entraînement terminé.")

acc = NeuroDSL.evaluate_multihop(g, logits_sym, ns; vocab_size=vocab_size, n_hops=n_hops, n_eval=200)
println("query_acc=", acc.query_acc, "  body_acc=", acc.body_acc)

# ── Diagnostic manuel, pas à pas ──────────────────────────────────────────────
rng_eval = MersenneTwister(999)
tokens, labels, q_pos = NeuroDSL.sample_multihop_sequence(rng_eval, vocab_size, n_hops)
tokens_corrupt = copy(tokens)
orig = tokens_corrupt[2]
new_id = orig
while new_id == orig || new_id in tokens
    global new_id = rand(rng_eval, 1:vocab_size)
end
tokens_corrupt[2] = new_id
println("tokens        = ", tokens)
println("tokens_corrupt= ", tokens_corrupt)
println("q_pos=", q_pos, "  label attendu=", labels[q_pos])

NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
clean_output = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))
clean_cache = NeuroDSL.capture_activations(g, ns)
clean_pred = argmax(clean_output[q_pos, :])
println("clean : pred au q_pos = ", clean_pred, "  (attendu=", labels[q_pos], ")")

NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
corrupted_output = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))
corrupted_cache = NeuroDSL.capture_activations(g, ns)
corrupted_pred = argmax(corrupted_output[q_pos, :])
println("corrupted : pred au q_pos = ", corrupted_pred)

diff_norm = norm(clean_output[q_pos, :] .- corrupted_output[q_pos, :])
println("‖clean[q_pos] - corrupted[q_pos]‖ = ", diff_norm)
println("clean_output[q_pos,1:5]     = ", clean_output[q_pos, 1:5])
println("corrupted_output[q_pos,1:5] = ", corrupted_output[q_pos, 1:5])

# Patch de TOUTES les têtes de la couche 1 d'un coup -- devrait avoir un effet
# massif si la corruption (position 2, très en amont) passe par la couche 1.
layer1_heads = [Symbol(:layer_1_mha_ao_h, h) for h in 1:n_heads]
NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, logits_sym; namespace=ns)
NeuroDSL.patch_nodes!(g, layer1_heads, clean_cache; namespace=ns)
patched_output = Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))
diff_patched = norm(patched_output[q_pos, :] .- clean_output[q_pos, :])
println("\nAprès patch de TOUTES les têtes couche 1 :")
println("‖patched[q_pos] - clean[q_pos]‖ = ", diff_patched, "  (0 = recovery totale)")
println("pred = ", argmax(patched_output[q_pos, :]))

recov = NeuroDSL.recovery_metric(patched_output[q_pos:q_pos, :], clean_output[q_pos:q_pos, :], corrupted_output[q_pos:q_pos, :])
println("recovery_metric (toutes têtes couche 1) = ", recov)
