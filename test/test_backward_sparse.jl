# test/test_backward_sparse.jl
using Test, NeuroDSL, Random, LinearAlgebra

@testset "Backward Sparse" begin
    
    @testset "Sequential network" begin
        dev = NeuroDSL.Backend.CPUDevice()
        g = NeuroDSL.NeuroGraph(device=dev, namespace=:main)
        D = 4
        
        NeuroDSL.set!(g, :x, randn(Float32, 2, D); namespace=:main)
        NeuroDSL.set!(g, :y, randn(Float32, 2, D); atom_type=NeuroDSL.Datom, namespace=:main)
        NeuroDSL.set!(g, :W1, randn(Float32, D, D).*0.1f0; is_param=true, namespace=:main)
        NeuroDSL.set!(g, :W2, randn(Float32, D, D).*0.1f0; is_param=true, namespace=:main)
        NeuroDSL.set!(g, :W3, randn(Float32, D, D).*0.1f0; is_param=true, namespace=:main)
        
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h1, [:x, :W1], :matmul; attrs=Dict(:trans_b=>true), namespace=:main))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h2, [:h1, :W2], :matmul; attrs=Dict(:trans_b=>true), namespace=:main))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h3, [:h2, :W3], :matmul; attrs=Dict(:trans_b=>true), namespace=:main))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:h3, :y], :mse_loss; namespace=:main))
        
        # Freeze W1 and W3
        g.nodes[:main][:W1].is_param = false
        g.nodes[:main][:W3].is_param = false
        
        ctx = NeuroDSL.CtxStore()
        NeuroDSL.demand!(g, :loss; ctx_store=ctx, namespace=:main)
        NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx, namespace=:main, sparse=true)

        @test g.nodes[:main][:W1].gradient === nothing  # Frozen → nothing
        @test g.nodes[:main][:W2].gradient !== nothing  # Trainable → present
        @test g.nodes[:main][:W3].gradient === nothing  # Frozen → nothing
    end

    @testset "Sequential network avec prune_frozen=true" begin
        dev = NeuroDSL.Backend.CPUDevice()
        g = NeuroDSL.NeuroGraph(device=dev, namespace=:main_pf)
        D = 4

        NeuroDSL.set!(g, :x, randn(Float32, 2, D); namespace=:main_pf)
        NeuroDSL.set!(g, :y, randn(Float32, 2, D); atom_type=NeuroDSL.Datom, namespace=:main_pf)
        NeuroDSL.set!(g, :W1, randn(Float32, D, D).*0.1f0; is_param=true, namespace=:main_pf)
        NeuroDSL.set!(g, :W2, randn(Float32, D, D).*0.1f0; is_param=true, namespace=:main_pf)
        NeuroDSL.set!(g, :W3, randn(Float32, D, D).*0.1f0; is_param=true, namespace=:main_pf)

        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h1, [:x, :W1], :matmul; attrs=Dict(:trans_b=>true), namespace=:main_pf))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h2, [:h1, :W2], :matmul; attrs=Dict(:trans_b=>true), namespace=:main_pf))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h3, [:h2, :W3], :matmul; attrs=Dict(:trans_b=>true), namespace=:main_pf))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:h3, :y], :mse_loss; namespace=:main_pf))

        # Freeze W1 and W3 (comme le test ci-dessus)
        g.nodes[:main_pf][:W1].is_param = false
        g.nodes[:main_pf][:W3].is_param = false

        ctx = NeuroDSL.CtxStore()
        NeuroDSL.demand!(g, :loss; ctx_store=ctx, namespace=:main_pf)
        NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx, namespace=:main_pf, sparse=true)
        grad_full = copy(Array(g.nodes[:main_pf][:W2].gradient))

        ctx2 = NeuroDSL.CtxStore()
        NeuroDSL.demand!(g, :loss; ctx_store=ctx2, namespace=:main_pf)
        NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx2, namespace=:main_pf, sparse=true, prune_frozen=true)
        grad_pruned = Array(g.nodes[:main_pf][:W2].gradient)

        @test isapprox(grad_full, grad_pruned; atol=Float32(1e-6))
        @test g.nodes[:main_pf][:W1].gradient === nothing
        @test g.nodes[:main_pf][:W2].gradient !== nothing
        @test g.nodes[:main_pf][:W3].gradient === nothing
    end

    @testset "Branched network" begin
        dev = NeuroDSL.Backend.CPUDevice()
        g = NeuroDSL.NeuroGraph(device=dev, namespace=:main)
        D = 4
        
        NeuroDSL.set!(g, :x, randn(Float32, 2, D); namespace=:main)
        NeuroDSL.set!(g, :y, randn(Float32, 2, D); atom_type=NeuroDSL.Datom, namespace=:main)
        NeuroDSL.set!(g, :W1, randn(Float32, D, D).*0.1f0; is_param=true, namespace=:main)
        NeuroDSL.set!(g, :W2a, randn(Float32, D, D).*0.1f0; is_param=true, namespace=:main)
        NeuroDSL.set!(g, :W2b, randn(Float32, D, D).*0.1f0; is_param=true, namespace=:main)
        
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h1, [:x, :W1], :matmul; attrs=Dict(:trans_b=>true), namespace=:main))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h2a, [:h1, :W2a], :matmul; attrs=Dict(:trans_b=>true), namespace=:main))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h2b, [:h1, :W2b], :matmul; attrs=Dict(:trans_b=>true), namespace=:main))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:sum, [:h2a, :h2b], :add; namespace=:main))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:sum, :y], :mse_loss; namespace=:main))
        
        g.nodes[:main][:W2a].is_param = false  # Freeze branch A
        
        ctx = NeuroDSL.CtxStore()
        NeuroDSL.demand!(g, :loss; ctx_store=ctx, namespace=:main)
        NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx, namespace=:main, sparse=true)
        
        @test g.nodes[:main][:W1].gradient !== nothing   # Trainable
        @test g.nodes[:main][:W2a].gradient === nothing  # Frozen
        @test g.nodes[:main][:W2b].gradient !== nothing  # Trainable
    end

    @testset "Branched network avec prune_frozen=true" begin
        dev = NeuroDSL.Backend.CPUDevice()
        g = NeuroDSL.NeuroGraph(device=dev, namespace=:main_pf2)
        D = 4

        NeuroDSL.set!(g, :x, randn(Float32, 2, D); namespace=:main_pf2)
        NeuroDSL.set!(g, :y, randn(Float32, 2, D); atom_type=NeuroDSL.Datom, namespace=:main_pf2)
        NeuroDSL.set!(g, :W1, randn(Float32, D, D).*0.1f0; is_param=true, namespace=:main_pf2)
        NeuroDSL.set!(g, :W2a, randn(Float32, D, D).*0.1f0; is_param=true, namespace=:main_pf2)
        NeuroDSL.set!(g, :W2b, randn(Float32, D, D).*0.1f0; is_param=true, namespace=:main_pf2)

        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h1, [:x, :W1], :matmul; attrs=Dict(:trans_b=>true), namespace=:main_pf2))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h2a, [:h1, :W2a], :matmul; attrs=Dict(:trans_b=>true), namespace=:main_pf2))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h2b, [:h1, :W2b], :matmul; attrs=Dict(:trans_b=>true), namespace=:main_pf2))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:sum, [:h2a, :h2b], :add; namespace=:main_pf2))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:sum, :y], :mse_loss; namespace=:main_pf2))

        g.nodes[:main_pf2][:W2a].is_param = false  # Freeze branch A

        ctx = NeuroDSL.CtxStore()
        NeuroDSL.demand!(g, :loss; ctx_store=ctx, namespace=:main_pf2)
        NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx, namespace=:main_pf2, sparse=true)
        grad_W1_full = copy(Array(g.nodes[:main_pf2][:W1].gradient))
        grad_W2b_full = copy(Array(g.nodes[:main_pf2][:W2b].gradient))

        ctx2 = NeuroDSL.CtxStore()
        NeuroDSL.demand!(g, :loss; ctx_store=ctx2, namespace=:main_pf2)
        NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx2, namespace=:main_pf2, sparse=true, prune_frozen=true)
        grad_W1_pruned = Array(g.nodes[:main_pf2][:W1].gradient)
        grad_W2b_pruned = Array(g.nodes[:main_pf2][:W2b].gradient)

        @test isapprox(grad_W1_full, grad_W1_pruned; atol=Float32(1e-6))
        @test isapprox(grad_W2b_full, grad_W2b_pruned; atol=Float32(1e-6))
        @test g.nodes[:main_pf2][:W1].gradient !== nothing   # Trainable
        @test g.nodes[:main_pf2][:W2a].gradient === nothing  # Frozen
        @test g.nodes[:main_pf2][:W2b].gradient !== nothing  # Trainable
        # h1 alimente la branche entraînable (h2b), ne doit jamais être élagué
        @test g.nodes[:main_pf2][:h1].backwarded == true
    end
end