@testset "Activation Patching" begin
    dev = NeuroDSL.Backend.CPUDevice()
    dim, n_heads, hidden_dim, n_layers, seq_len = 32, 4, 64, 4, 8

    function build_model(ns::Symbol)
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :input, randn(Float32, seq_len, dim); namespace=ns)
        output_sym = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim)(g, :input; namespace=ns)
        return g, output_sym
    end

    function clean_and_corrupted_runs(g, output_sym, ns)
        X_clean = randn(Float32, seq_len, dim)
        NeuroDSL.set!(g, :input, X_clean; namespace=ns)
        clean_output = copy(NeuroDSL.demand!(g, output_sym; namespace=ns))
        clean_cache = NeuroDSL.capture_activations(g, ns)

        X_corrupted = copy(X_clean)
        X_corrupted[1, :] .= randn(Float32, dim)   # un seul token corrompu
        NeuroDSL.set!(g, :input, X_corrupted; namespace=ns)
        corrupted_output = copy(NeuroDSL.demand!(g, output_sym; namespace=ns))
        corrupted_cache = NeuroDSL.capture_activations(g, ns)

        return clean_cache, corrupted_cache, clean_output, corrupted_output
    end

    @testset "Le nœud patché garde sa valeur après demand! (ne doit pas être recalculé)" begin
        # Régression directe : `set!` marque le nœud CIBLE lui-même invalide (pas
        # seulement ses successeurs) -- un `patch_node!` naïf basé sur `set!`
        # laisserait `demand!` recalculer le nœud patché à partir de sa propre
        # règle, avec ses VRAIES entrées (toujours corrompues), écrasant
        # silencieusement la valeur propre qu'on vient d'imposer. Vérifié ici en
        # comparant explicitement la valeur du nœud avant et après `demand!`.
        ns = :patch_value_survives
        g, output_sym = build_model(ns)
        clean_cache, corrupted_cache, clean_output, corrupted_output =
            clean_and_corrupted_runs(g, output_sym, ns)

        patch_sym = Symbol(:layer_, 2, :_out)
        NeuroDSL.patch_node!(g, patch_sym, clean_cache; namespace=ns)
        value_right_after_patch = copy(NeuroDSL.node(g, patch_sym; namespace=ns).value)
        NeuroDSL.demand!(g, output_sym; namespace=ns)
        value_after_demand = NeuroDSL.node(g, patch_sym; namespace=ns).value

        @test isapprox(Array(value_right_after_patch), Array(clean_cache[patch_sym]); atol=Float32(1e-6))
        @test isapprox(Array(value_after_demand), Array(clean_cache[patch_sym]); atol=Float32(1e-6))
    end

    @testset "Patcher la dernière couche équivaut à patcher la sortie" begin
        # `output_sym` EST `layer_{n_layers}_out` dans ce modèle -- patcher l'un
        # ou l'autre est la même action sur le même nœud, donc la récupération
        # doit être EXACTEMENT 1.0, sans dépendre d'une seconde implémentation
        # de référence (identité mathématique, pas juste une comparaison entre
        # deux chemins de code qui pourraient partager le même bug).
        ns = :patch_last_layer
        g, output_sym = build_model(ns)
        clean_cache, corrupted_cache, clean_output, corrupted_output =
            clean_and_corrupted_runs(g, output_sym, ns)

        last_layer_sym = Symbol(:layer_, n_layers, :_out)
        @test last_layer_sym == output_sym

        result = NeuroDSL.patch_and_measure!(g, output_sym, last_layer_sym, clean_cache, corrupted_cache,
                                              clean_output, corrupted_output; namespace=ns)
        @test isapprox(result.recovery, 1.0; atol=1e-5)
    end

    @testset "Patcher deux fois le même site donne le même résultat (déterminisme)" begin
        ns = :patch_determinism
        g, output_sym = build_model(ns)
        clean_cache, corrupted_cache, clean_output, corrupted_output =
            clean_and_corrupted_runs(g, output_sym, ns)

        patch_sym = Symbol(:layer_, 2, :_out)
        r1 = NeuroDSL.patch_and_measure!(g, output_sym, patch_sym, clean_cache, corrupted_cache,
                                          clean_output, corrupted_output; namespace=ns)
        r2 = NeuroDSL.patch_and_measure!(g, output_sym, patch_sym, clean_cache, corrupted_cache,
                                          clean_output, corrupted_output; namespace=ns)
        @test isapprox(r1.recovery, r2.recovery; atol=Float32(1e-6))
        @test r1.recovery > 0.0   # le patch doit avoir un effet réel, pas être un no-op
    end

    @testset "Cas limites : patcher :input directement, ou ne rien patcher" begin
        ns = :patch_edge
        g, output_sym = build_model(ns)
        clean_cache, corrupted_cache, clean_output, corrupted_output =
            clean_and_corrupted_runs(g, output_sym, ns)

        # Patcher :input lui-même doit redonner EXACTEMENT la sortie propre.
        result_input = NeuroDSL.patch_and_measure!(g, output_sym, :input, clean_cache, corrupted_cache,
                                                     clean_output, corrupted_output; namespace=ns)
        @test isapprox(result_input.recovery, 1.0; atol=1e-5)

        # Ne rien patcher (état corrompu tel quel) doit donner recovery == 0.
        current_output = NeuroDSL.demand!(g, output_sym; namespace=ns)
        r0 = NeuroDSL.recovery_metric(current_output, clean_output, corrupted_output)
        @test isapprox(r0, 0.0; atol=1e-6)
    end

    @testset "Restauration exacte après patch_and_measure!" begin
        ns = :patch_restore
        g, output_sym = build_model(ns)
        clean_cache, corrupted_cache, clean_output, corrupted_output =
            clean_and_corrupted_runs(g, output_sym, ns)

        before = NeuroDSL.capture_activations(g, ns)
        patch_sym = Symbol(:layer_, 3, :_out)
        NeuroDSL.patch_and_measure!(g, output_sym, patch_sym, clean_cache, corrupted_cache,
                                     clean_output, corrupted_output; namespace=ns)
        after = NeuroDSL.capture_activations(g, ns)

        @test Set(keys(before)) == Set(keys(after))
        @test all(isapprox(Array(before[k]), Array(after[k]); atol=Float32(1e-6)) for k in keys(before))
    end

    @testset "Restauration par cache == restauration par recalcul" begin
        # Invariant central du balayage amorti : remplacer patch_node!+demand!
        # par une copie directe depuis le cache doit laisser le graphe dans un
        # état RIGOUREUSEMENT identique (égalité exacte, le calcul étant
        # déterministe et le cache ayant été capturé sur ce même état).
        ns_recompute = :sweep_vs_recompute_a
        ns_cached    = :sweep_vs_recompute_b
        patch_sym = Symbol(:layer_, 2, :_out)

        # Même graine AVANT la construction du modèle : les poids aléatoires de
        # LlamaModel sont aussi tirés du RNG global, donc les deux graphes
        # doivent partir de la même graine avant build_model, pas seulement
        # avant les données d'entrée -- sinon les poids diffèrent entre g1/g2.
        Random.seed!(7)
        g1, output_sym1 = build_model(ns_recompute)
        clean_cache1, corrupted_cache1, clean_output1, corrupted_output1 =
            clean_and_corrupted_runs(g1, output_sym1, ns_recompute)

        Random.seed!(7)
        g2, output_sym2 = build_model(ns_cached)
        clean_cache2, corrupted_cache2, clean_output2, corrupted_output2 =
            clean_and_corrupted_runs(g2, output_sym2, ns_cached)

        # Graphe 1 : patch puis restauration par recalcul (chemin existant).
        NeuroDSL.patch_and_measure!(g1, output_sym1, patch_sym, clean_cache1, corrupted_cache1,
                                     clean_output1, corrupted_output1; namespace=ns_recompute)

        # Graphe 2 : patch puis restauration par cache (nouveau chemin).
        affected = NeuroDSL._downstream_nodes(g2, patch_sym, ns_cached)
        NeuroDSL.patch_node!(g2, patch_sym, clean_cache2; namespace=ns_cached)
        NeuroDSL.demand!(g2, output_sym2; namespace=ns_cached)
        NeuroDSL.restore_from_cache!(g2, ns_cached, corrupted_cache2, affected)

        state1 = NeuroDSL.capture_activations(g1, ns_recompute)
        state2 = NeuroDSL.capture_activations(g2, ns_cached)
        @test Set(keys(state1)) == Set(keys(state2))
        @test all(isapprox(Array(state1[k]), Array(state2[k]); atol=Float32(1e-6)) for k in keys(state1))
    end

    @testset "sweep_patch_sites! donne les mêmes recovery que patch_and_measure!" begin
        ns_loop  = :sweep_loop
        ns_sweep = :sweep_fast
        sites = [Symbol(:layer_, i, :_out) for i in 1:n_layers]

        Random.seed!(11)
        g1, output_sym1 = build_model(ns_loop)
        clean_cache1, corrupted_cache1, clean_output1, corrupted_output1 =
            clean_and_corrupted_runs(g1, output_sym1, ns_loop)

        Random.seed!(11)
        g2, output_sym2 = build_model(ns_sweep)
        clean_cache2, corrupted_cache2, clean_output2, corrupted_output2 =
            clean_and_corrupted_runs(g2, output_sym2, ns_sweep)

        recov_loop = [NeuroDSL.patch_and_measure!(g1, output_sym1, s, clean_cache1, corrupted_cache1,
                                                   clean_output1, corrupted_output1; namespace=ns_loop).recovery
                      for s in sites]
        sweep_results = NeuroDSL.sweep_patch_sites!(g2, output_sym2, sites, clean_cache2, corrupted_cache2,
                                                     clean_output2, corrupted_output2; namespace=ns_sweep)
        recov_sweep = [r.recovery for r in sweep_results]

        @test length(sweep_results) == n_layers
        @test all(isapprox(a, b; atol=Float32(1e-6)) for (a, b) in zip(recov_loop, recov_sweep))

        # Après le balayage, le graphe doit être exactement dans l'état corrompu d'origine.
        final_state = NeuroDSL.capture_activations(g2, ns_sweep)
        @test all(isapprox(Array(final_state[k]), Array(corrupted_cache2[k]); atol=Float32(1e-6))
                  for k in keys(corrupted_cache2))
    end

    @testset "patch_nodes! : composition commutative de deux patches" begin
        # layer_2_mha_ao_h1 et layer_2_mha_ao_h2 sont deux têtes d'attention
        # SŒURS : ni l'une ni l'autre n'est en amont de l'autre (toutes deux
        # calculées depuis les mêmes Q/K/V, fusionnées ensuite par
        # hcat_heads). C'est le cas où l'ordre d'application ne doit
        # structurellement jamais avoir d'importance -- contrairement à deux
        # nœuds en relation ancêtre-descendant (ex. deux couches différentes),
        # où patcher l'ancêtre APRÈS le descendant réinvaliderait le patch du
        # descendant (le patch le plus tardif dans l'ordre d'application
        # l'emporte aux points de recouvrement -- un comportement défini, mais
        # qui dépend de l'ordre). Les têtes sœurs isolent la propriété
        # d'indépendance qu'on veut vérifier ici.
        ns = :multi_patch_commute
        Random.seed!(13)
        g, output_sym = build_model(ns)
        clean_cache, corrupted_cache, clean_output, corrupted_output =
            clean_and_corrupted_runs(g, output_sym, ns)

        site_a = Symbol(:layer_2_mha_ao_h1)
        site_b = Symbol(:layer_2_mha_ao_h2)

        NeuroDSL.patch_nodes!(g, [site_a, site_b], clean_cache; namespace=ns)
        out_ab = copy(NeuroDSL.demand!(g, output_sym; namespace=ns))
        NeuroDSL.restore_nodes_from_cache!(g, ns, corrupted_cache, [site_a, site_b])
        NeuroDSL.demand!(g, output_sym; namespace=ns)

        NeuroDSL.patch_node!(g, site_b, clean_cache; namespace=ns)
        NeuroDSL.patch_node!(g, site_a, clean_cache; namespace=ns)
        out_ba = copy(NeuroDSL.demand!(g, output_sym; namespace=ns))

        @test isapprox(Array(out_ab), Array(out_ba); atol=Float32(1e-6))
    end

    @testset "Récupération combinée != somme des récupérations individuelles" begin
        # Patcher {A, B} ensemble (deux têtes sœurs, cf. test précédent)
        # capture un vrai effet joint, pas la simple superposition de deux
        # effets indépendants -- preuve que l'invalidation combinée calcule
        # l'effet réel du sous-graphe partagé, pas une approximation additive.
        ns = :multi_patch_nonadditive
        Random.seed!(17)
        g, output_sym = build_model(ns)
        clean_cache, corrupted_cache, clean_output, corrupted_output =
            clean_and_corrupted_runs(g, output_sym, ns)

        site_a = Symbol(:layer_2_mha_ao_h1)
        site_b = Symbol(:layer_2_mha_ao_h2)

        r_a  = NeuroDSL.patch_and_measure!(g, output_sym, site_a, clean_cache, corrupted_cache,
                                            clean_output, corrupted_output; namespace=ns).recovery
        r_b  = NeuroDSL.patch_and_measure!(g, output_sym, site_b, clean_cache, corrupted_cache,
                                            clean_output, corrupted_output; namespace=ns).recovery

        NeuroDSL.patch_nodes!(g, [site_a, site_b], clean_cache; namespace=ns)
        out_ab = NeuroDSL.demand!(g, output_sym; namespace=ns)
        r_ab = NeuroDSL.recovery_metric(out_ab, clean_output, corrupted_output)
        NeuroDSL.restore_nodes_from_cache!(g, ns, corrupted_cache, [site_a, site_b])

        @test !isapprox(r_ab, r_a + r_b; atol=Float32(1e-4))
    end

    @testset "Le cône combiné est borné par l'union, pas par la somme" begin
        # Preuve déterministe (pas de chronométrage, cf. discipline de cette
        # session) que patcher deux sites dont les cônes se recoupent coûte
        # moins cher que les patcher indépendamment : l'union des nœuds
        # affectés est strictement plus petite que la somme de leurs tailles
        # individuelles dès qu'il y a chevauchement.
        ns = :multi_patch_union_cost
        g, output_sym = build_model(ns)
        NeuroDSL.demand!(g, output_sym; namespace=ns)   # tous les nœuds valides avant de mesurer les cônes

        site_a = Symbol(:layer_2_mha_ao_h1)
        site_b = Symbol(:layer_2_mha_ao_h2)
        cone_a = NeuroDSL._downstream_nodes(g, site_a, ns)
        cone_b = NeuroDSL._downstream_nodes(g, site_b, ns)
        cone_union = union(cone_a, cone_b)

        @test !isempty(intersect(cone_a, cone_b))   # les cônes se recoupent bien
        @test length(cone_union) < length(cone_a) + length(cone_b)
    end

    @testset "Restauration adaptative == restauration par recalcul (cônes qui se recoupent)" begin
        # site_a est déjà retenu (comme le ferait greedy_patch_search! après une
        # première étape) ; site_b est testé PAR-DESSUS, avec un cône qui
        # recoupe celui de site_a (convergence via hcat_heads). La
        # restauration adaptative doit alors tomber sur le chemin par
        # recalcul (jamais sur le cache, qui effacerait l'effet actif de
        # site_a) -- et donner un état identique à un recalcul direct, tout
        # en laissant site_a actif.
        ns = :adaptive_restore_overlap
        Random.seed!(23)
        g, output_sym = build_model(ns)
        clean_cache, corrupted_cache, clean_output, corrupted_output =
            clean_and_corrupted_runs(g, output_sym, ns)

        site_a = Symbol(:layer_2_mha_ao_h1)
        site_b = Symbol(:layer_2_mha_ao_h2)

        cone_a = NeuroDSL._downstream_nodes(g, site_a, ns)
        NeuroDSL.patch_node!(g, site_a, clean_cache; namespace=ns)
        NeuroDSL.demand!(g, output_sym; namespace=ns)
        selected_cone_union = cone_a

        cone_b = NeuroDSL._downstream_nodes(g, site_b, ns)
        @test !isempty(intersect(cone_a, cone_b))   # précondition : les cônes se recoupent

        # Chemin 1 : restauration adaptative
        NeuroDSL.patch_node!(g, site_b, clean_cache; namespace=ns)
        NeuroDSL.demand!(g, output_sym; namespace=ns)
        NeuroDSL._adaptive_restore!(g, ns, site_b, cone_b, selected_cone_union, corrupted_cache, output_sym)
        state_adaptive = NeuroDSL.capture_activations(g, ns)

        # Chemin 2 : restauration par recalcul directe (référence)
        NeuroDSL.patch_node!(g, site_b, clean_cache; namespace=ns)
        NeuroDSL.demand!(g, output_sym; namespace=ns)
        NeuroDSL.patch_node!(g, site_b, corrupted_cache; namespace=ns)
        NeuroDSL.demand!(g, output_sym; namespace=ns)
        state_recompute = NeuroDSL.capture_activations(g, ns)

        @test all(isapprox(Array(state_adaptive[k]), Array(state_recompute[k]); atol=Float32(1e-6))
                  for k in keys(state_recompute))
        # site_a doit rester actif après la restauration adaptative de site_b
        @test isapprox(Array(g.nodes[ns][site_a].value), Array(clean_cache[site_a]); atol=Float32(1e-6))
    end

    @testset "greedy_patch_search! : récupération finale vérifiable indépendamment" begin
        ns = :greedy_verify
        Random.seed!(29)
        g, output_sym = build_model(ns)
        clean_cache, corrupted_cache, clean_output, corrupted_output =
            clean_and_corrupted_runs(g, output_sym, ns)

        candidates = [Symbol(:layer_2_mha_ao_h, h) for h in 1:n_heads]
        selected, trajectory = NeuroDSL.greedy_patch_search!(
            g, output_sym, candidates, clean_cache, corrupted_cache,
            clean_output, corrupted_output; max_sites=3, namespace=ns)

        @test length(selected) <= 3
        @test !isempty(selected)
        @test length(trajectory) == length(selected)

        # Recalcul indépendant de la récupération finale : patcher tous les
        # sites retenus d'un coup via patch_nodes! (déjà existant), sans
        # passer par la logique de recherche.
        NeuroDSL.patch_nodes!(g, selected, clean_cache; namespace=ns)
        out_final = NeuroDSL.demand!(g, output_sym; namespace=ns)
        r_final_independent = NeuroDSL.recovery_metric(out_final, clean_output, corrupted_output)
        NeuroDSL.restore_nodes_from_cache!(g, ns, corrupted_cache, selected)

        @test isapprox(r_final_independent, trajectory[end].cumulative_recovery; atol=Float32(1e-5))
    end

    @testset "backward_prune! : jamais de corruption silencieuse (cas ancêtre-descendant)" begin
        # layer_1_mha_ao_h1 / h2 (couche 1, têtes sœurs) sont en amont de
        # layer_2_out (couche 2) via le flux résiduel -- exactement la
        # relation rencontrée sur la tâche d'induction (têtes de couche 1 en
        # amont d'une tête de couche 2 retenue par la recherche gloutonne).
        ns = :backward_prune_ancestor
        Random.seed!(37)
        g, output_sym = build_model(ns)
        clean_cache, corrupted_cache, clean_output, corrupted_output =
            clean_and_corrupted_runs(g, output_sym, ns)

        site_up1 = Symbol(:layer_1_mha_ao_h1)
        site_up2 = Symbol(:layer_1_mha_ao_h2)
        site_down = Symbol(:layer_2_out)

        # Précondition : site_down est bien en aval de site_up1 (relation
        # ancêtre-descendant, pas seulement des sites frères).
        cone_up1 = NeuroDSL._downstream_nodes(g, site_up1, ns)
        @test site_down ∈ cone_up1

        selected = [site_up1, site_up2, site_down]
        full_cone = union(NeuroDSL._downstream_nodes(g, site_up1, ns),
                           NeuroDSL._downstream_nodes(g, site_up2, ns),
                           NeuroDSL._downstream_nodes(g, site_down, ns))

        # Récupération de référence de l'ensemble complet, obtenue
        # indépendamment de backward_prune! (reset total + patch_nodes!).
        NeuroDSL.restore_from_cache!(g, ns, corrupted_cache, full_cone)
        NeuroDSL.patch_nodes!(g, selected, clean_cache; namespace=ns)
        out_full = NeuroDSL.demand!(g, output_sym; namespace=ns)
        full_recovery_independent = NeuroDSL.recovery_metric(out_full, clean_output, corrupted_output)

        remaining, pruned = NeuroDSL.backward_prune!(g, output_sym, selected, clean_cache, corrupted_cache,
                                                       clean_output, corrupted_output; namespace=ns)

        @test Set(remaining) ∪ Set(pruned) == Set(selected)
        @test isempty(intersect(Set(remaining), Set(pruned)))

        # La récupération du sous-ensemble retenu, recalculée INDÉPENDAMMENT
        # depuis un état frais, doit rester au moins aussi bonne que celle de
        # l'ensemble complet moins la tolérance -- exactement la garantie que
        # backward_prune! est censé fournir.
        NeuroDSL.restore_from_cache!(g, ns, corrupted_cache, full_cone)
        NeuroDSL.patch_nodes!(g, remaining, clean_cache; namespace=ns)
        out_remaining = NeuroDSL.demand!(g, output_sym; namespace=ns)
        r_remaining_independent = NeuroDSL.recovery_metric(out_remaining, clean_output, corrupted_output)
        @test r_remaining_independent >= full_recovery_independent - 1f-5

        # Si site_down a été conservé, sa valeur active doit être EXACTEMENT
        # la valeur propre -- jamais corrompue par le retrait d'un site en
        # amont (le piège que backward_prune! est construit pour éviter).
        if site_down ∈ remaining
            @test isapprox(Array(NeuroDSL.node(g, site_down; namespace=ns).value),
                            Array(clean_cache[site_down]); atol=Float32(1e-6))
        end

        NeuroDSL.restore_from_cache!(g, ns, corrupted_cache, full_cone)

        # Démonstration explicite du piège que backward_prune! évite : retirer
        # naïvement site_up1 (patch_node! vers le corrompu + demand!) alors
        # que site_down est actif EFFACE silencieusement le patch de
        # site_down -- exactement ce que backward_prune! ne fait jamais.
        NeuroDSL.patch_nodes!(g, [site_up1, site_up2, site_down], clean_cache; namespace=ns)
        NeuroDSL.demand!(g, output_sym; namespace=ns)
        NeuroDSL.patch_node!(g, site_up1, corrupted_cache; namespace=ns)   # retrait naïf
        NeuroDSL.demand!(g, output_sym; namespace=ns)
        value_after_naive_removal = Array(NeuroDSL.node(g, site_down; namespace=ns).value)
        @test !isapprox(value_after_naive_removal, Array(clean_cache[site_down]); atol=Float32(1e-6))

        NeuroDSL.restore_from_cache!(g, ns, corrupted_cache, full_cone)
    end

    @testset "restore_from_cache_batched! (GPU) == restore_from_cache! -- égalité exacte" begin
        if NeuroDSL.Backend.CUDA_AVAILABLE
            cuda_dev = NeuroDSL.Backend.CUDADevice()
            ns = :batched_restore_gpu
            Random.seed!(31)
            g = NeuroDSL.NeuroGraph(namespace=ns, device=cuda_dev)
            NeuroDSL.set!(g, :input, randn(Float32, seq_len, dim); namespace=ns)
            output_sym = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim)(g, :input; namespace=ns)

            X_clean = randn(Float32, seq_len, dim)
            NeuroDSL.set!(g, :input, X_clean; namespace=ns)
            clean_output = copy(NeuroDSL.demand!(g, output_sym; namespace=ns))
            clean_cache = NeuroDSL.capture_activations(g, ns)

            X_corrupted = copy(X_clean)
            X_corrupted[1, :] .= randn(Float32, dim)
            NeuroDSL.set!(g, :input, X_corrupted; namespace=ns)
            corrupted_output = copy(NeuroDSL.demand!(g, output_sym; namespace=ns))
            corrupted_cache = NeuroDSL.capture_activations(g, ns)

            patch_sym = Symbol(:layer_, n_layers ÷ 2, :_out)
            affected = NeuroDSL._downstream_nodes(g, patch_sym, ns)

            # restore_from_cache_batched! doit donner un état EXACTEMENT identique
            # à restore_from_cache! (même copie, juste un chemin d'exécution différent).
            NeuroDSL.patch_node!(g, patch_sym, clean_cache; namespace=ns)
            NeuroDSL.demand!(g, output_sym; namespace=ns)
            NeuroDSL.restore_from_cache!(g, ns, corrupted_cache, affected)
            state_ref = NeuroDSL.capture_activations(g, ns)

            NeuroDSL.patch_node!(g, patch_sym, clean_cache; namespace=ns)
            NeuroDSL.demand!(g, output_sym; namespace=ns)
            NeuroDSL.restore_from_cache_batched!(g, ns, corrupted_cache, affected)
            state_batched = NeuroDSL.capture_activations(g, ns)

            @test all(Array(state_ref[k]) == Array(state_batched[k]) for k in keys(state_ref))

            # Vérification supplémentaire contre la restauration par recalcul
            # (même esprit que l'Invariant 3 déjà établi, avec la variante groupée).
            NeuroDSL.patch_node!(g, patch_sym, clean_cache; namespace=ns)
            NeuroDSL.demand!(g, output_sym; namespace=ns)
            NeuroDSL.patch_node!(g, patch_sym, corrupted_cache; namespace=ns)
            NeuroDSL.demand!(g, output_sym; namespace=ns)
            state_recompute = NeuroDSL.capture_activations(g, ns)

            @test all(isapprox(Array(state_batched[k]), Array(state_recompute[k]); atol=Float32(1e-5))
                      for k in keys(state_recompute))

            # sweep_patch_sites! avec restore_fn=restore_from_cache_batched! doit
            # donner les mêmes recovery que la variante par défaut.
            sites = [Symbol(:layer_, i, :_out) for i in 1:n_layers]
            results_default = NeuroDSL.sweep_patch_sites!(g, output_sym, sites, clean_cache, corrupted_cache,
                                                            clean_output, corrupted_output; namespace=ns)
            results_batched = NeuroDSL.sweep_patch_sites!(g, output_sym, sites, clean_cache, corrupted_cache,
                                                            clean_output, corrupted_output; namespace=ns,
                                                            restore_fn=NeuroDSL.restore_from_cache_batched!)
            @test all(isapprox(a.recovery, b.recovery; atol=Float32(1e-5))
                      for (a, b) in zip(results_default, results_batched))
        else
            println("⚠️  GPU non disponible — test restore_from_cache_batched! ignoré.")
            @test true
        end
    end
end
