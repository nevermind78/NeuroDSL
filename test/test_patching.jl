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
end
