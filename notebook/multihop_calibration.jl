# ══════════════════════════════════════════════════════════════════════════════
# Étape 2 (redesign chirurgie causale dirigée) — calibration empirique de la
# tâche d'induction à plusieurs sauts (src/synthetic_circuits.jl,
# sample_multihop_sequence/build_multihop_graph/train_multihop!) : trouver un
# (n_hops, checkpoint d'entraînement) qui donne une recovery de circuit
# CAUSAL (pas juste une précision de tâche) dans la bande [0.6, 0.8] sur un
# modèle à 6 couches -- ni trivial (déjà saturé comme l'induction à 1 saut sur
# le char-LM réel), ni impossible (aucun circuit à trouver du tout).
#
# Corruption : un seul token remplacé à la position v_1 (le premier maillon
# de la chaîne, PAS adjacent à la requête) -- force le modèle à propager
# correctement le changement à travers TOUS les sauts restants pour que le
# patch d'un site fasse une vraie différence à la position finale, exactement
# l'analogue du protocole texte réel (corruption près du PREMIER maillon, pas
# près de la cible).
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics, Printf, LinearAlgebra

dev = NeuroDSL.Backend.CUDADevice()

vocab_size, dim, n_heads, hidden_dim, n_layers = 30, 64, 4, 128, 6

function find_circuit_multihop!(g, ns, logits_sym, tokens_clean, tokens_corrupt, q_pos; max_sites=8)
    NeuroDSL.set!(g, :token_ids, tokens_clean; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens_clean)); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    clean_output = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))
    clean_cache  = NeuroDSL.capture_activations(g, ns)

    NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    corrupted_output = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))
    corrupted_cache  = NeuroDSL.capture_activations(g, ns)

    row_metric(out) = NeuroDSL.recovery_metric(Array(out)[q_pos:q_pos, :], clean_output[q_pos:q_pos, :], corrupted_output[q_pos:q_pos, :])

    candidates = sort(collect(filter(s -> occursin(r"_mha_ao_h\d+$", String(s)), keys(g.nodes[ns]))))

    NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, logits_sym; namespace=ns)

    selected, trajectory = NeuroDSL.greedy_patch_search!(g, logits_sym, candidates, clean_cache, corrupted_cache,
                                                           clean_output, corrupted_output;
                                                           namespace=ns, max_sites=max_sites, metric=row_metric)
    remaining, pruned = isempty(selected) ? (Symbol[], Symbol[]) :
        NeuroDSL.backward_prune!(g, logits_sym, selected, clean_cache, corrupted_cache,
                                  clean_output, corrupted_output; namespace=ns, metric=row_metric)

    NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, logits_sym; namespace=ns)
    isempty(remaining) || NeuroDSL.patch_nodes!(g, remaining, clean_cache; namespace=ns)
    r_check = row_metric(NeuroDSL.demand!(g, logits_sym; namespace=ns))

    return (; selected, remaining, r_check, n_candidates=length(candidates))
end

function corrupted_chain(rng, vocab_size, n_hops)
    tokens, labels, q_pos = NeuroDSL.sample_multihop_sequence(rng, vocab_size, n_hops)
    tokens_corrupt = copy(tokens)
    orig = tokens_corrupt[2]   # v_1 -- premier maillon, pas adjacent à la requête
    new_id = orig
    while new_id == orig || new_id in tokens
        new_id = rand(rng, 1:vocab_size)
    end
    tokens_corrupt[2] = new_id
    return tokens, tokens_corrupt, labels, q_pos
end

for n_hops in (2, 3, 4)
    println("\n", "="^70)
    println("n_hops = ", n_hops, "  (seq_len = ", 2n_hops+1, ")")
    println("="^70)

    Random.seed!(n_hops)
    ns = Symbol(:calib_hops_, n_hops)
    g, logits_sym = NeuroDSL.build_multihop_graph(dev, ns; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                                   hidden_dim=hidden_dim, n_layers=n_layers, n_hops=n_hops)

    rng_train = MersenneTwister(123)
    ps = NeuroDSL.params(g; namespace=ns)
    m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]

    checkpoints = (500, 1500, 3000, 6000, 10000)
    t_done = 0
    rng_eval_fixed = MersenneTwister(999)
    tokens_c, tokens_x, labels_c, q_pos = corrupted_chain(rng_eval_fixed, vocab_size, n_hops)

    for ckpt in checkpoints
        n_steps = ckpt - t_done
        for t in (t_done+1):ckpt
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
        t_done = ckpt

        eval_acc = NeuroDSL.evaluate_multihop(g, logits_sym, ns; vocab_size=vocab_size, n_hops=n_hops, n_eval=200)
        circuit = find_circuit_multihop!(g, ns, logits_sym, tokens_c, tokens_x, q_pos; max_sites=8)
        band = 0.6 <= circuit.r_check <= 0.8 ? "  <-- DANS LA BANDE [0.6,0.8]" : ""
        @printf("  pas=%5d  query_acc=%.3f  body_acc=%.3f  recovery_causale=%.4f  sites=%d%s\n",
                ckpt, eval_acc.query_acc, eval_acc.body_acc, circuit.r_check, length(circuit.remaining), band)
    end
end
