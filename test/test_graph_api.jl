


@testset "Graph API" begin
    g = NeuroDSL.JuliusGraph(namespace=:t)

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
end
