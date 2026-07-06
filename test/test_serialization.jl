@testset "Sérialisation (save_graph!/load_graph!)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    dim, n_heads, hidden_dim, n_layers, seq_len = 32, 4, 64, 4, 8

    @testset "Round-trip complet : structure + poids + demand! bit-identique" begin
        tmpdir = mktempdir()
        ns = :ser_test
        Random.seed!(41)
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :input, randn(Float32, seq_len, dim); namespace=ns)
        output_sym = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim)(g, :input; namespace=ns)
        ref_output = copy(NeuroDSL.demand!(g, output_sym; namespace=ns))

        # Snapshot des feuilles avant sauvegarde (comparaison exacte après chargement).
        leaf_values_before = Dict{Symbol,Array{Float32}}(
            sym => copy(nd.value) for (sym, nd) in g.nodes[ns] if !haskey(g.rules[ns], sym)
        )

        prefix = joinpath(tmpdir, "graph")
        NeuroDSL.save_graph!(g, ns, prefix)

        # Graphe totalement neuf -- ne rappelle PAS LlamaModel, preuve que la structure
        # vient bien du fichier.
        g2 = NeuroDSL.NeuroGraph(device=dev)
        NeuroDSL.load_graph!(g2, ns, prefix)

        @test Set(keys(g2.nodes[ns])) == Set(keys(g.nodes[ns]))
        @test Set(keys(g2.rules[ns])) == Set(keys(g.rules[ns]))

        for (sym, val) in leaf_values_before
            @test g2.nodes[ns][sym].value == val   # égalité exacte, pas isapprox
            @test g2.nodes[ns][sym].is_param == g.nodes[ns][sym].is_param
        end

        # Décisif : les RÈGLES ont bien survécu, pas seulement les poids bruts.
        out2 = NeuroDSL.demand!(g2, output_sym; namespace=ns)
        @test Array(out2) == Array(ref_output)
    end

    @testset "Round-trip AdamWState (keyé par symbole, pas par position)" begin
        tmpdir = mktempdir()
        ns = :ser_opt_test
        Random.seed!(43)
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :input, randn(Float32, seq_len, dim); namespace=ns)
        output_sym = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim)(g, :input; namespace=ns)
        NeuroDSL.set!(g, :labels, fill(0.0f0, size(NeuroDSL.demand!(g, output_sym; namespace=ns))...);
                      atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [output_sym, :labels], :mse_loss; namespace=ns))

        ps = NeuroDSL.params(g; namespace=ns)
        m1 = Dict{Symbol,Array{Float32}}(); m2 = Dict{Symbol,Array{Float32}}()
        for p in ps
            m1[p.name] = randn(Float32, size(p.value)...) .* 0.01f0
            m2[p.name] = abs.(randn(Float32, size(p.value)...)) .* 0.01f0
        end

        # Vrais appels adamw_step! (pas des tableaux inventés) pour obtenir un état réaliste.
        NeuroDSL.demand!(g, :loss; namespace=ns)
        NeuroDSL.backward_graph!(g, :loss; namespace=ns)
        t = 7
        for p in ps
            p.gradient === nothing && continue
            NeuroDSL.adamw_step!(dev, p.value, p.gradient, m1[p.name], m2[p.name],
                                  1f-3, 0.9f0, 0.999f0, 1f-8, t, 1f0, 1f-2)
        end
        opt_state = NeuroDSL.AdamWState(t, m1, m2)

        prefix = joinpath(tmpdir, "graph_opt")
        NeuroDSL.save_graph!(g, ns, prefix; opt_state=opt_state)

        g2 = NeuroDSL.NeuroGraph(device=dev)
        loaded_opt = NeuroDSL.load_graph!(g2, ns, prefix)

        @test loaded_opt !== nothing
        @test loaded_opt.t == t
        @test Set(keys(loaded_opt.m1)) == Set(keys(m1))
        for sym in keys(m1)
            @test loaded_opt.m1[sym] == m1[sym]
            @test loaded_opt.m2[sym] == m2[sym]
        end
    end

    @testset "save_all_graph!/load_all_graph! (multi-namespace + active_ns)" begin
        tmpdir = mktempdir()
        g = NeuroDSL.NeuroGraph(namespace=:multi_a, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 4, 4); namespace=:multi_a)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:y, [:x], :relu; namespace=:multi_a))
        NeuroDSL.activate!(g, :multi_b)
        NeuroDSL.set!(g, :x, randn(Float32, 3, 3); namespace=:multi_b)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:y, [:x], :relu; namespace=:multi_b))
        NeuroDSL.activate!(g, :multi_a)  # namespace actif au moment de la sauvegarde

        dir_prefix = joinpath(tmpdir, "all")
        NeuroDSL.save_all_graph!(g, dir_prefix)

        g2 = NeuroDSL.NeuroGraph(device=dev)
        NeuroDSL.load_all_graph!(g2, dir_prefix)

        # NeuroGraph() crée toujours un namespace :default vide -- on vérifie seulement
        # que les deux namespaces chargés sont bien présents, pas l'ensemble exact.
        @test issubset(Set([:multi_a, :multi_b]), Set(NeuroDSL.namespaces(g2)))
        @test g2.active_ns == :multi_a
        @test g2.nodes[:multi_a][:x].value == g.nodes[:multi_a][:x].value
        @test g2.nodes[:multi_b][:x].value == g.nodes[:multi_b][:x].value
    end

    @testset "Chemins d'erreur exercés réellement" begin
        tmpdir = mktempdir()
        ns = :err_test
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 4, 4); namespace=ns)
        prefix = joinpath(tmpdir, "graph_err")
        NeuroDSL.save_graph!(g, ns, prefix)

        # Collision de namespace sans overwrite=true.
        g2 = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g2, :x, randn(Float32, 4, 4); namespace=ns)
        @test_throws ErrorException NeuroDSL.load_graph!(g2, ns, prefix)
        # Avec overwrite=true, ça doit réussir.
        NeuroDSL.load_graph!(g2, ns, prefix; overwrite=true)
        @test g2.nodes[ns][:x].value == g.nodes[ns][:x].value

        # Namespace absent du répertoire multi-namespace.
        dir_prefix = joinpath(tmpdir, "all_err")
        NeuroDSL.save_all_graph!(g, dir_prefix)
        g3 = NeuroDSL.NeuroGraph(device=dev)
        @test_throws Exception NeuroDSL.load_graph!(g3, :namespace_absent, joinpath(dir_prefix, "namespace_absent"))
    end

    @testset "on_change non-nothing refusé explicitement à la sauvegarde" begin
        tmpdir = mktempdir()
        ns = :onchange_test
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 4, 4); namespace=ns)
        g.nodes[ns][:x].on_change = (gg, sym, nns) -> nothing
        prefix = joinpath(tmpdir, "graph_onchange")
        @test_throws ErrorException NeuroDSL.save_graph!(g, ns, prefix)
    end
end
