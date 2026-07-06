# Item 3 (revue d'architecture) : patching à l'échelle, poids NeuroDSL uniquement
# (décision utilisateur confirmée -- pas de checkpoint externe réel).
#
# Deux volets complémentaires :
#   1. Circuit construit à la main dans une seule tête d'attention, dont l'effet
#      a une prédiction EXACTE en forme close -- vérité terrain indépendante de
#      tout apprentissage, réutilise save_graph!/load_graph! (item 4).
#   2. Tâche d'induction (Olsson et al. 2022) à plus grande échelle que
#      notebook/induction.ipynb (dim=64/3 couches -> dim=128/8 têtes ici),
#      greedy_patch_search!/backward_prune! réutilisés SANS AUCUNE modification.

@testset "Circuit construit à la main (tête unique, prédiction en forme close)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    dim, n_heads, seq_len = 32, 4, 8
    target_head, query_pos, key_pos = 2, 5, 2
    qk_scale = 8f0
    ns = :selection_circuit

    g, out_sym, prefix, vt = NeuroDSL.build_selection_circuit(dev, dim, n_heads, seq_len,
                                                               target_head, query_pos, key_pos;
                                                               qk_scale=qk_scale, namespace=ns)
    out = Array(NeuroDSL.demand!(g, out_sym; namespace=ns))
    d_head = dim ÷ n_heads
    s = (target_head - 1) * d_head + 1
    e = target_head * d_head

    pr_sym = Symbol(prefix, :_pr_h, target_head)
    pr = Array(NeuroDSL.node(g, pr_sym; namespace=ns).value)

    M = qk_scale^2 / sqrt(Float32(d_head))
    pred_weight = Float32(exp(M) / (exp(M) + (query_pos - 1)))

    @test isapprox(pr[query_pos, key_pos], pred_weight; atol=1f-6)
    @test isapprox(sum(pr[query_pos, 1:query_pos]), 1f0; atol=1f-6)

    predicted_slice = pred_weight .* vt
    @test isapprox(out[query_pos, s:e], predicted_slice; atol=1f-5)

    # Ronde-trip via save_graph!/load_graph! (item 4) -- réutilisation directe, sert
    # aussi de test d'intégration croisé entre les deux items.
    tmpdir = mktempdir()
    prefix_path = joinpath(tmpdir, "circuit")
    NeuroDSL.save_graph!(g, ns, prefix_path)
    g2 = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.load_graph!(g2, ns, prefix_path)
    out2 = Array(NeuroDSL.demand!(g2, out_sym; namespace=ns))
    @test isapprox(out2[query_pos, s:e], predicted_slice; atol=1f-5)

    @testset "Patching sur le circuit exact (patch_node!/recovery_metric, prédiction exacte)" begin
        clean_input = copy(Array(NeuroDSL.node(g, :input; namespace=ns).value))
        clean_output = copy(Array(NeuroDSL.demand!(g, out_sym; namespace=ns)))
        clean_cache = NeuroDSL.capture_activations(g, ns)

        corrupted_input = copy(clean_input)
        corrupted_input[key_pos, :] .= 0f0   # supprime le one-hot à key_pos
        NeuroDSL.set!(g, :input, corrupted_input; namespace=ns)
        corrupted_output = copy(Array(NeuroDSL.demand!(g, out_sym; namespace=ns)))
        corrupted_cache = NeuroDSL.capture_activations(g, ns)

        # Prédiction en forme close du run corrompu : V_h(0) = 0 quel que soit le poids
        # d'attention -- la tranche de la tête cible à query_pos doit être EXACTEMENT nulle.
        @test all(iszero, corrupted_output[query_pos, s:e])

        clean_row = clean_output[query_pos:query_pos, :]
        corrupted_row = corrupted_output[query_pos:query_pos, :]

        hybrid = NeuroDSL.position_patch_cache(corrupted_cache, :input, clean_cache, key_pos)
        NeuroDSL.patch_node!(g, :input, hybrid; namespace=ns)
        patched_output = Array(NeuroDSL.demand!(g, out_sym; namespace=ns))
        recovery = NeuroDSL.recovery_metric(patched_output[query_pos:query_pos, :], clean_row, corrupted_row)

        @test isapprox(recovery, 1.0; atol=1e-5)
        @test isapprox(patched_output[query_pos, s:e], predicted_slice; atol=1f-5)
    end
end

@testset "Induction à plus grande échelle (LlamaModel entraîné, greedy_patch_search!/backward_prune!)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    ns = :induction_scale_test
    vocab_size, dim, n_heads, hidden_dim, n_layers, prefix_len = 20, 128, 8, 256, 3, 8

    Random.seed!(42)
    g, logits = NeuroDSL.build_induction_graph(dev, ns; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                                hidden_dim=hidden_dim, n_layers=n_layers, prefix_len=prefix_len)
    NeuroDSL.train_induction!(g, ns; vocab_size=vocab_size, prefix_len=prefix_len, n_steps=1500)

    # Généralisation AVANT toute analyse causale (même discipline que induction.ipynb) :
    # sinon le modèle a pu mémoriser plutôt qu'apprendre l'algorithme d'induction.
    acc = NeuroDSL.evaluate_induction(g, logits, ns; vocab_size=vocab_size, prefix_len=prefix_len)
    @test acc.second_half_acc > 0.9

    test_rng = MersenneTwister(555)
    clean_tokens, clean_labels = NeuroDSL.sample_induction_sequence(test_rng, vocab_size, prefix_len)
    j = prefix_len + 1        # position cible : premier point de la 2e moitié
    i_src = 2                 # position source : suit la 1ère occurrence du déclencheur
    corrupted_tokens = copy(clean_tokens)
    corrupted_tokens[i_src] = mod(clean_tokens[i_src] + 7, vocab_size) + 1

    NeuroDSL.set!(g, :token_ids, clean_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    clean_output = copy(NeuroDSL.demand!(g, logits; namespace=ns))
    clean_cache = NeuroDSL.capture_activations(g, ns)

    NeuroDSL.set!(g, :token_ids, corrupted_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    corrupted_output = copy(NeuroDSL.demand!(g, logits; namespace=ns))
    corrupted_cache = NeuroDSL.capture_activations(g, ns)

    @test argmax(clean_output[j, :]) != argmax(corrupted_output[j, :])  # contraste exploitable

    clean_row = clean_output[j:j, :]
    corrupted_row = corrupted_output[j:j, :]
    row_recovery(full) = NeuroDSL.recovery_metric(full[j:j, :], clean_row, corrupted_row)

    function greedy_patch_search_row!(candidates; max_sites::Int=length(candidates))
        selected = Symbol[]; remaining = collect(candidates)
        trajectory = NamedTuple[]; best_so_far = 0.0
        selected_cone_union = Set{Symbol}()
        for _ in 1:max_sites
            best_site = nothing; best_recovery = best_so_far
            for cand in remaining
                cand_cone = NeuroDSL._downstream_nodes(g, cand, ns)
                NeuroDSL.patch_node!(g, cand, clean_cache; namespace=ns)
                out = NeuroDSL.demand!(g, logits; namespace=ns)
                r = row_recovery(out)
                NeuroDSL._adaptive_restore!(g, ns, cand, cand_cone, selected_cone_union, corrupted_cache, logits)
                if r > best_recovery
                    best_recovery = r; best_site = cand
                end
            end
            best_site === nothing && break
            best_cone = NeuroDSL._downstream_nodes(g, best_site, ns)
            NeuroDSL.patch_node!(g, best_site, clean_cache; namespace=ns)
            NeuroDSL.demand!(g, logits; namespace=ns)
            push!(selected, best_site); filter!(sy -> sy != best_site, remaining)
            union!(selected_cone_union, best_cone); best_so_far = best_recovery
            push!(trajectory, (; site=best_site, cumulative_recovery=best_recovery))
        end
        return selected, trajectory
    end

    candidates = [Symbol(:layer_, l, :_mha_ao_h, h) for l in 1:n_layers for h in 1:n_heads]
    selected, trajectory = greedy_patch_search_row!(candidates)

    @test !isempty(selected)
    @test trajectory[end].cumulative_recovery > 0.8   # circuit réellement retrouvé, pas du bruit

    # Vérification indépendante de la recovery finale (même discipline qu'Invariant 5).
    NeuroDSL.patch_nodes!(g, selected, clean_cache; namespace=ns)
    out_final = NeuroDSL.demand!(g, logits; namespace=ns)
    r_final = row_recovery(out_final)
    NeuroDSL.restore_nodes_from_cache!(g, ns, corrupted_cache, selected)
    NeuroDSL.demand!(g, logits; namespace=ns)
    @test isapprox(r_final, trajectory[end].cumulative_recovery; atol=1f-5)

    # Signature indépendante : la tête retenue en premier doit concentrer son attention,
    # depuis la position cible j, sur la position source i_src ou la position précédente
    # (previous-token head, brique standard d'un circuit d'induction à 2 têtes) -- pas
    # seulement "la recovery est haute".
    NeuroDSL.set!(g, :token_ids, clean_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, logits; namespace=ns)

    top_site = selected[1]
    mm = match(r"layer_(\d+)_mha_ao_h(\d+)", string(top_site))
    l, h = parse(Int, mm[1]), parse(Int, mm[2])
    pr_sym = Symbol(:layer_, l, :_mha_pr_h, h)
    pattern = Array(NeuroDSL.node(g, pr_sym; namespace=ns).value)
    top_attended = argmax(pattern[j, :])
    @test top_attended in (i_src, j - 1)

    # Élagage arrière : ne doit jamais faire chuter la recovery (pleine sortie, même
    # métrique que backward_prune! utilise en interne) en dessous de son seuil de
    # tolérance -- le test de correction sur un cas ancêtre-descendant explicite est
    # déjà couvert dans test_patching.jl ; ici on exerce seulement l'usage réel.
    full_cone = union((NeuroDSL._downstream_nodes(g, sy, ns) for sy in selected)...)
    NeuroDSL.restore_from_cache!(g, ns, corrupted_cache, full_cone)
    NeuroDSL.patch_nodes!(g, selected, clean_cache; namespace=ns)
    out_full_selected = NeuroDSL.demand!(g, logits; namespace=ns)
    recovery_full_selected = NeuroDSL.recovery_metric(out_full_selected, clean_output, corrupted_output)

    remaining, pruned = NeuroDSL.backward_prune!(g, logits, selected, clean_cache, corrupted_cache,
                                                  clean_output, corrupted_output; namespace=ns)
    @test !isempty(remaining)

    NeuroDSL.restore_from_cache!(g, ns, corrupted_cache, full_cone)
    NeuroDSL.patch_nodes!(g, remaining, clean_cache; namespace=ns)
    out_full_remaining = NeuroDSL.demand!(g, logits; namespace=ns)
    recovery_full_remaining = NeuroDSL.recovery_metric(out_full_remaining, clean_output, corrupted_output)
    @test recovery_full_remaining >= recovery_full_selected - 1f-4
end
