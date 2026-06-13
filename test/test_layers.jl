


@testset "Layers" begin
    dev = NeuroDSL.Backend.CPUDevice()

    @testset "LayerNorm shape" begin
        g = NeuroDSL.JuliusGraph(namespace=:t_ln, device=dev)
        NeuroDSL.set!(g,:x, randn(Float32,4,16))
        out = NeuroDSL.LayerNorm(16)(g,:x,:ln; namespace=:t_ln)
        val = NeuroDSL.demand!(g, out; namespace=:t_ln)
        @test size(val) == (4,16)
    end

    @testset "Linear shape" begin
        g = NeuroDSL.JuliusGraph(namespace=:t_li, device=dev)
        NeuroDSL.set!(g,:x, randn(Float32,3,8))
        out = NeuroDSL.Linear(8,16)(g,:x,:fc; namespace=:t_li)
        val = NeuroDSL.demand!(g, out; namespace=:t_li)
        @test size(val) == (3,16)
    end

    @testset "MultiHeadAttention shape" begin
        g = NeuroDSL.JuliusGraph(namespace=:t_mha, device=dev)
        NeuroDSL.set!(g,:x, randn(Float32,4,16))
        out = NeuroDSL.MultiHeadAttention(16,2)(g,:x,:mha; namespace=:t_mha)
        val = NeuroDSL.demand!(g, out; namespace=:t_mha)
        @test size(val) == (4,16)
    end

    @testset "LlamaBlock residual shape" begin
        g = NeuroDSL.JuliusGraph(namespace=:t_lb, device=dev)
        NeuroDSL.set!(g,:x, randn(Float32,4,16))
        out = NeuroDSL.LlamaBlock(16,2,32)(g,:x,:blk; namespace=:t_lb)
        val = NeuroDSL.demand!(g, out; namespace=:t_lb)
        @test size(val) == (4,16)
    end

    @testset "LlamaModel 2 layers" begin
        g = NeuroDSL.JuliusGraph(namespace=:t_lm, device=dev)
        NeuroDSL.set!(g,:x, randn(Float32,4,16))
        out = NeuroDSL.LlamaModel(2,16,2,32)(g,:x; namespace=:t_lm)
        val = NeuroDSL.demand!(g, out; namespace=:t_lm)
        @test size(val) == (4,16)
        @test length(NeuroDSL.params(g; namespace=:t_lm)) > 0
    end
end
