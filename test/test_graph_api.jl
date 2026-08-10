


@testset "Graph API" begin
    g = NeuroDSL.NeuroGraph(namespace=:t)

    @testset "set! / node" begin
        NeuroDSL.set!(g, :x, ones(Float32,2,2); is_param=true, namespace=:t)
        NeuroDSL.set!(g, :d, [1,2]; atom_type=NeuroDSL.Datom, namespace=:t)
        @test  NeuroDSL.is_backpropable(NeuroDSL.node(g,:x; namespace=:t))
        @test !NeuroDSL.is_backpropable(NeuroDSL.node(g,:d; namespace=:t))
        @test  length(NeuroDSL.params(g; namespace=:t)) == 1
    end

    @testset "addrule! / topo_order!" begin
        NeuroDSL.set!(g, :W, ones(Float32,2,2); is_param=true, namespace=:t)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:out, [:x,:W], :matmul;
            attrs=Dict{Symbol,Any}(:trans_b=>false), namespace=:t))
        order = NeuroDSL.topo_order!(g; namespace=:t)
        @test :out in order
        xi = findfirst(==(:x),   order)
        oi = findfirst(==(:out), order)
        @test xi < oi
    end

    @testset "invalidation" begin
        NeuroDSL.set!(g, :x, zeros(Float32,2,2); namespace=:t)
        @test !NeuroDSL.node(g,:out; namespace=:t).valid
    end

    @testset "namespace isolation" begin
        NeuroDSL.activate!(g, :other)
        NeuroDSL.set!(g, :x, ones(Float32,2,2); namespace=:other)
        @test !haskey(g.nodes[:other], :out)
    end

    @testset "demand! restreint aux vrais ancêtres (pas le préfixe topologique complet)" begin
        # Régression du bug historique 2026-07-29 (copy_params_to_namespace!,
        # cache KV corrompu) : `demand!` parcourait tout le préfixe
        # topologique jusqu'à la cible au lieu de ses seuls ancêtres réels --
        # demander un PARAMÈTRE (jamais périmé en pratique, mais marqué
        # `valid=false` par `_invalidate_downstream!` dans `set!`, puisque le
        # nœud lui-même est mis en file avant ses consommateurs) ressuscitait
        # par accident tout nœud invalide rencontré en chemin, même sans
        # rapport avec la cible. Vérifié empiriquement avant ce correctif :
        # `demand!(g, :W1)` recalculait `h2`, deux nœuds plus loin.
        ns = :anc_test
        g2 = NeuroDSL.NeuroGraph(namespace=ns)
        NeuroDSL.set!(g2, :x, rand(Float32,4,4); namespace=ns)
        NeuroDSL.set!(g2, :W1, rand(Float32,4,4); is_param=true, namespace=ns)
        NeuroDSL.addrule!(g2, NeuroDSL.GraphRule(:h1, [:x,:W1], :matmul; namespace=ns))
        NeuroDSL.set!(g2, :W2, rand(Float32,4,4); is_param=true, namespace=ns)
        NeuroDSL.addrule!(g2, NeuroDSL.GraphRule(:h2, [:h1,:W2], :matmul; namespace=ns))

        NeuroDSL.demand!(g2, :h2; namespace=ns)
        NeuroDSL.invalidate_all!(g2; namespace=ns)
        @test !NeuroDSL.node(g2,:h1; namespace=ns).valid
        @test !NeuroDSL.node(g2,:h2; namespace=ns).valid

        # demand! sur un paramètre : aucune règle, doit être un NO-OP total.
        NeuroDSL.demand!(g2, :W1; namespace=ns)
        @test !NeuroDSL.node(g2,:h1; namespace=ns).valid   # PAS ressuscité
        @test !NeuroDSL.node(g2,:h2; namespace=ns).valid   # PAS ressuscité (encore plus loin que h1)

        # Contrôle positif -- le mécanisme fonctionne bien quand demandé sur
        # un vrai ancêtre, pour ne pas rendre ce test vide de sens.
        NeuroDSL.demand!(g2, :h1; namespace=ns)
        @test NeuroDSL.node(g2,:h1; namespace=ns).valid
        @test !NeuroDSL.node(g2,:h2; namespace=ns).valid   # h2 n'est pas un ancêtre de h1, toujours pas touché
    end
end
