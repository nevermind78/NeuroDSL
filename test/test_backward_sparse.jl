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
end