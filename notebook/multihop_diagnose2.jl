using NeuroDSL, Random, LinearAlgebra, Printf

dev = NeuroDSL.Backend.CUDADevice()
vocab_size, dim, n_heads, hidden_dim, n_layers, n_hops = 30, 64, 4, 128, 6, 2

Random.seed!(2)
ns = :diag2
g, logits_sym = NeuroDSL.build_multihop_graph(dev, ns; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                               hidden_dim=hidden_dim, n_layers=n_layers, n_hops=n_hops)

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
candidates = sort(collect(filter(s -> occursin(r"_mha_ao_h\d+$", String(s)), keys(g.nodes[ns]))))
println("n_candidates = ", length(candidates))

NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, logits_sym; namespace=ns)

selected, trajectory = NeuroDSL.greedy_patch_search!(g, logits_sym, candidates, clean_cache, corrupted_cache,
                                                       clean_output, corrupted_output;
                                                       namespace=ns, max_sites=8, metric=row_metric)
println("selected = ", selected)
println("trajectoire (recovery cumulée après chaque site ajouté) :")
for (i, t) in enumerate(trajectory)
    println("  ", i, ": site=", t.site, "  cumulative_recovery=", t.cumulative_recovery)
end

# Vérification indépendante immédiate (sans backward_prune!) : patch de `selected`
# recalculé depuis un état frais.
NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, logits_sym; namespace=ns)
isempty(selected) || NeuroDSL.patch_nodes!(g, selected, clean_cache; namespace=ns)
out_check_selected = NeuroDSL.demand!(g, logits_sym; namespace=ns)
r_check_selected = row_metric(out_check_selected)
println("\nr_check (patch direct de `selected`, état frais) = ", r_check_selected)

remaining, pruned = isempty(selected) ? (Symbol[], Symbol[]) :
    NeuroDSL.backward_prune!(g, logits_sym, selected, clean_cache, corrupted_cache,
                              clean_output, corrupted_output; namespace=ns, metric=row_metric)
println("remaining (après backward_prune!) = ", remaining)
println("pruned = ", pruned)

NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, logits_sym; namespace=ns)
isempty(remaining) || NeuroDSL.patch_nodes!(g, remaining, clean_cache; namespace=ns)
r_check_remaining = row_metric(NeuroDSL.demand!(g, logits_sym; namespace=ns))
println("r_check (patch direct de `remaining`, état frais) = ", r_check_remaining)

println("\n--- Hypothèse : cache périmé après usage par greedy_patch_search!/backward_prune! ---")
NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
clean_output2 = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))
clean_cache2 = NeuroDSL.capture_activations(g, ns)   # RE-capturé, frais

NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
corrupted_output2 = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))
NeuroDSL.demand!(g, logits_sym; namespace=ns)
isempty(selected) || NeuroDSL.patch_nodes!(g, selected, clean_cache2; namespace=ns)
out_recap = NeuroDSL.demand!(g, logits_sym; namespace=ns)
r_recap = NeuroDSL.recovery_metric(Array(out_recap)[q_pos:q_pos,:], clean_output2[q_pos:q_pos,:], corrupted_output2[q_pos:q_pos,:])
println("r_check avec cache RE-capturé = ", r_recap)
println("clean_output2 == clean_output (avant re-capture) ? ", clean_output2 == clean_output)
println("‖clean_cache2[:layer_1_mha_ao_h4] - clean_cache[:layer_1_mha_ao_h4]‖ = ",
        norm(Array(clean_cache2[:layer_1_mha_ao_h4]) .- Array(clean_cache[:layer_1_mha_ao_h4])))
