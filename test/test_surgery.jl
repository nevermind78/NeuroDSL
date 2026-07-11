# Chirurgie à chaud (src/graph_surgery.jl) : insertion de couche en cours d'entraînement
# avec préservation exacte de la fonction calculée.
#
# F1 (identité exacte + falsifiabilité), F2 (localité de l'invalidation), F4 (le modèle
# greffé réapprend réellement l'induction) sont couverts ici -- adaptés à Pkg.test()
# (un seul processus, déterministe, rapide). F3 (continuité à travers un vrai redémarrage
# de processus Julia) est vérifié séparément par deux scripts manuels, hors suite
# automatisée (spawner un sous-processus julia depuis un test serait fragile/lent) -- voir
# le rapport de session pour les résultats de cette vérification.

@testset "Chirurgie à chaud (insert_block!)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    vocab_size, dim, n_heads, hidden_dim, n_layers, prefix_len = 20, 64, 4, 128, 4, 8

    # ── Oracles indépendants (même style que test_property_invariants.jl) ──────────────
    function _gt_consumers(g::NeuroDSL.NeuroGraph, ns::Symbol)
        idx = Dict{Symbol,Vector{Symbol}}()
        for (out_sym, rule) in g.rules[ns]
            for inp in rule.inputs
                push!(get!(idx, inp, Symbol[]), out_sym)
            end
        end
        return idx
    end
    function _gt_reachable_downstream(g::NeuroDSL.NeuroGraph, ns::Symbol, target::Symbol)
        consumers = _gt_consumers(g, ns)
        visited = Set{Symbol}([target])
        queue = Symbol[target]
        while !isempty(queue)
            cur = pop!(queue)
            for out_sym in get(consumers, cur, Symbol[])
                out_sym in visited || (push!(visited, out_sym); push!(queue, out_sym))
            end
        end
        return visited
    end

    @testset "F1 : identité exacte + falsifiabilité" begin
        ns = :surgery_f1
        Random.seed!(7)
        g, logits = NeuroDSL.build_induction_graph(dev, ns; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                                    hidden_dim=hidden_dim, n_layers=n_layers, prefix_len=prefix_len)
        tokens, labels = NeuroDSL.sample_induction_sequence(MersenneTwister(1), vocab_size, prefix_len)
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        ref_output = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))

        # 1a. insert_block! (zero-init) : sortie identique bit-à-bit, pas approximative.
        new_out = NeuroDSL.insert_block!(g, ns, :layer_2_out, dim, n_heads, hidden_dim)
        post_output = Array(NeuroDSL.demand!(g, logits; namespace=ns))
        @test ref_output == post_output

        # 1b. Falsifiabilité -- reproduire la réécriture À LA MAIN (pas insert_block!) avec des
        # poids ALÉATOIRES (pas zéro) pour le nouveau bloc, afin de garantir une vraie sortie
        # différente de la référence si la réécriture est bien prise en compte.
        #
        # `addrule!` réinvalide maintenant AUTOMATIQUEMENT un nœud existant dont on redéfinit la
        # règle (corrigé à la source dans `src/graph_api.jl` -- ceci était auparavant LE bug :
        # sans un appel explicite supplémentaire à `_invalidate_downstream!`, la valeur périmée
        # d'avant la réécriture était servie silencieusement). Ce test confirme que `addrule!`
        # SEUL, sans aucun appel supplémentaire, suffit maintenant -- et qu'un appel supplémentaire
        # (redondant) donne exactement le même résultat, preuve qu'il n'apporte plus rien de neuf.
        ns2 = :surgery_f1_falsif
        Random.seed!(7)
        g2, logits2 = NeuroDSL.build_induction_graph(dev, ns2; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                                      hidden_dim=hidden_dim, n_layers=n_layers, prefix_len=prefix_len)
        NeuroDSL.set!(g2, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns2)
        NeuroDSL.invalidate_all!(g2; namespace=ns2)
        ref_output2 = copy(Array(NeuroDSL.demand!(g2, logits2; namespace=ns2)))

        after_sym = :layer_2_out
        old_consumers = collect(get(NeuroDSL._consumers_index!(g2, ns2), after_sym, Symbol[]))
        new_out2 = NeuroDSL.LlamaBlock(dim, n_heads, hidden_dim)(g2, after_sym, :surgery_bad; namespace=ns2)
        # PAS de mise à zéro des poids -- le nouveau bloc garde ses poids aléatoires, donc son
        # effet sur le flux résiduel n'est PAS l'identité : la vraie sortie DOIT changer.
        rewired = Dict{Symbol,Vector{Symbol}}()
        for c in old_consumers
            rule = g2.rules[ns2][c]
            new_inputs = [inp == after_sym ? new_out2 : inp for inp in rule.inputs]
            rewired[c] = new_inputs
        end

        # -- addrule! SEUL, sans aucun appel supplémentaire à _invalidate_downstream! :
        for c in old_consumers
            rule = g2.rules[ns2][c]
            NeuroDSL.addrule!(g2, NeuroDSL.GraphRule(c, rewired[c], rule.op; attrs=rule.attrs,
                                                      namespace=ns2, atom_type=rule.atom_type))
        end
        output_addrule_alone = Array(NeuroDSL.demand!(g2, logits2; namespace=ns2))
        @test output_addrule_alone != ref_output2   # le correctif fonctionne : addrule! seul suffit

        # -- Même point de départ, mais avec un appel supplémentaire (désormais redondant) :
        for c in old_consumers
            NeuroDSL._invalidate_downstream!(g2, c, ns2)
        end
        output_with_redundant_call = Array(NeuroDSL.demand!(g2, logits2; namespace=ns2))
        @test output_with_redundant_call == output_addrule_alone   # l'appel supplémentaire n'apporte plus rien

        # 1c. Perturbation epsilon du zero-init réel (via insert_block! normal) -- confirme que la
        # vérification bit-à-bit de 1a n'est pas vide (un epsilon non nul DOIT casser l'égalité).
        ns3 = :surgery_f1_eps
        Random.seed!(7)
        g3, logits3 = NeuroDSL.build_induction_graph(dev, ns3; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                                      hidden_dim=hidden_dim, n_layers=n_layers, prefix_len=prefix_len)
        NeuroDSL.set!(g3, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns3)
        NeuroDSL.invalidate_all!(g3; namespace=ns3)
        ref_output3 = copy(Array(NeuroDSL.demand!(g3, logits3; namespace=ns3)))
        new_out3 = NeuroDSL.insert_block!(g3, ns3, :layer_2_out, dim, n_heads, hidden_dim)
        eps_prefix = Symbol(:surgery_, :layer_2_out)
        eps_W_sym = Symbol(eps_prefix, :_mha, :_output_W)
        eps_W = copy(Array(NeuroDSL.node(g3, eps_W_sym; namespace=ns3).value))
        eps_W[1, 1] = 1f-3   # epsilon non nul, plus la matrice n'est plus exactement zéro
        NeuroDSL.set_params!(g3, ns3, Dict(eps_W_sym => eps_W))
        eps_output = Array(NeuroDSL.demand!(g3, logits3; namespace=ns3))
        @test eps_output != ref_output3
    end

    @testset "F2 : localité de l'invalidation (cône exact, rien hors-cône touché)" begin
        ns = :surgery_f2
        Random.seed!(13)
        g, logits = NeuroDSL.build_induction_graph(dev, ns; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                                    hidden_dim=hidden_dim, n_layers=n_layers, prefix_len=prefix_len)
        tokens, labels = NeuroDSL.sample_induction_sequence(MersenneTwister(2), vocab_size, prefix_len)
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.demand!(g, logits; namespace=ns)   # tout valide avant la greffe

        before_valid = Dict(sym => nd.valid for (sym, nd) in g.nodes[ns])

        new_out = NeuroDSL.insert_block!(g, ns, :layer_2_out, dim, n_heads, hidden_dim)
        reachable = _gt_reachable_downstream(g, ns, new_out)

        for (sym, nd) in g.nodes[ns]
            if sym in reachable
                # Atteignable depuis le nouveau bloc : soit un nœud neuf (jamais valide),
                # soit un consommateur réécrit -- dans les deux cas doit être invalide maintenant.
                @test !nd.valid
            elseif haskey(before_valid, sym)
                # Hors du cône : le flag valid ne doit JAMAIS avoir bougé.
                @test nd.valid == before_valid[sym]
            end
        end

        # La sortie reste correcte après ce remaniement (identité exacte, comme F1).
        post_output = Array(NeuroDSL.demand!(g, logits; namespace=ns))
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)  # no-op, même valeur
        @test Array(NeuroDSL.demand!(g, logits; namespace=ns)) == post_output
    end

    @testset "F4 : le modèle greffé continue d'apprendre l'induction" begin
        ns = :surgery_f4
        vs, d, nh, hd, nl, pl = 20, 64, 4, 128, 3, 8
        Random.seed!(11)
        g, logits = NeuroDSL.build_induction_graph(dev, ns; vocab_size=vs, dim=d, n_heads=nh,
                                                    hidden_dim=hd, n_layers=nl, prefix_len=pl)
        ps = NeuroDSL.params(g; namespace=ns)
        m1s = Dict(p.name => zeros(Float32, size(p.value)...) for p in ps)
        m2s = Dict(p.name => zeros(Float32, size(p.value)...) for p in ps)
        rng = MersenneTwister(123)
        t = 0
        function _train_step!(param_list)
            t_ = t
            tokens, labels = NeuroDSL.sample_induction_sequence(rng, vs, pl)
            NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.invalidate_all!(g; namespace=ns)
            NeuroDSL.demand!(g, :loss; namespace=ns)
            NeuroDSL.backward_graph!(g, :loss; namespace=ns)
            for p in param_list
                p.gradient === nothing && continue
                NeuroDSL.adamw_step!(dev, p.value, p.gradient, m1s[p.name], m2s[p.name],
                                      3f-3, 0.9f0, 0.999f0, 1f-8, t_, 1f0, 0f0)
            end
            NeuroDSL.invalidate_all!(g; namespace=ns)
        end

        for _ in 1:750
            t += 1
            _train_step!(ps)
        end
        acc_mid = NeuroDSL.evaluate_induction(g, logits, ns; vocab_size=vs, prefix_len=pl)
        @test acc_mid.second_half_acc > 0.9   # doit déjà bien fonctionner avant la greffe

        new_out = NeuroDSL.insert_block!(g, ns, :layer_2_out, d, nh, hd)
        new_ps = [p for p in NeuroDSL.params(g; namespace=ns) if !haskey(m1s, p.name)]
        @test !isempty(new_ps)   # le nouveau bloc a bien apporté de nouveaux paramètres
        for p in new_ps
            m1s[p.name] = zeros(Float32, size(p.value)...)
            m2s[p.name] = zeros(Float32, size(p.value)...)
        end
        NeuroDSL.invalidate_all!(g; namespace=ns)

        ps2 = NeuroDSL.params(g; namespace=ns)
        for _ in 1:750
            t += 1
            _train_step!(ps2)
        end
        acc_final = NeuroDSL.evaluate_induction(g, logits, ns; vocab_size=vs, prefix_len=pl)
        @test acc_final.second_half_acc > 0.9   # continue d'apprendre l'induction après la greffe
    end

    @testset "F1 (extension) : insert_block!(...; batched_attn=true) -- identité exacte, marqueur, patchable" begin
        # `batched_attn` a été ajouté à `insert_block!` (2026-07-10) pour que le
        # bloc greffé reste homogène avec un modèle dont les autres couches
        # utilisent déjà l'attention batchée (gemm_strided_batched!, voir
        # src/layers.jl). L'identité exacte ne dépend que des poids de sortie
        # mis à zéro (mécanisme indépendant du mode d'attention) -- doit donc
        # tenir à l'identique.
        ns = :surgery_f1_batched
        Random.seed!(7)
        g, logits = NeuroDSL.build_induction_graph(dev, ns; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                                    hidden_dim=hidden_dim, n_layers=n_layers, prefix_len=prefix_len)
        tokens, labels = NeuroDSL.sample_induction_sequence(MersenneTwister(1), vocab_size, prefix_len)
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        ref_output = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))

        new_out = NeuroDSL.insert_block!(g, ns, :layer_2_out, dim, n_heads, hidden_dim; batched_attn=true)
        post_output = Array(NeuroDSL.demand!(g, logits; namespace=ns))
        @test ref_output == post_output   # identité exacte, indépendante du mode d'attention

        # Marqueur du mode batché : le nœud du tenseur groupé Q·Kᵀ existe.
        marker_sym = :surgery_layer_2_out_mha_sc3
        @test haskey(g.nodes[ns], marker_sym)

        # Une tête du bloc greffé reste individuellement patchable comme une
        # tête ordinaire (vue zero-copy sur le tenseur groupé, mais un nœud du
        # graphe comme un autre du point de vue de patch_node!).
        head_sym = :surgery_layer_2_out_mha_ao_h1
        @test haskey(g.nodes[ns], head_sym)
        cache = NeuroDSL.capture_activations(g, ns)
        patched_val = Array(cache[head_sym]) .+ 1f0
        NeuroDSL.patch_node!(g, head_sym, Dict(head_sym => patched_val); namespace=ns)
        NeuroDSL.demand!(g, logits; namespace=ns)
        @test Array(NeuroDSL.node(g, head_sym; namespace=ns).value) == patched_val
    end
end
