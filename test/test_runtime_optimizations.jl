
using Test

@testset "Runtime optimizations" begin
    @testset "GraphData / auto_graphdata" begin
        d = auto_graphdata()
        @test isa(d, GraphData)
        @test get_device(d) === Backend.active_device()
        @test !supports_checkpointing(d)
        @test !supports_mixed_precision(d)

        cd = CheckpointData(d; every=2)
        @test supports_checkpointing(cd)
        @test checkpoint_every(cd) == 2

        mpc = MixedPrecData(d)
        @test supports_mixed_precision(mpc)
    end

    @testset "Memory planning" begin
        g = JuliusGraph(namespace=:t)
        set!(g, :x, rand(Float32, 2, 2); is_param=true, namespace=:t)
        set!(g, :w, rand(Float32, 2, 2); is_param=true, namespace=:t)
        addrule!(g, GraphRule(:h, [:x, :w], :matmul; namespace=:t))
        addrule!(g, GraphRule(:out, [:h], :sum_matrix; namespace=:t))

        plan, pool = plan_memory!(g; namespace=:t)
        @test plan.n_slots <= length(plan.order)
        @test pool.n_alloc >= 0
        @test pool.n_hits == 0

        invalidate_all!(g; namespace=:t)
        val = demand_planned!(g, :out, plan, pool; namespace=:t)
        @test isa(val, AbstractArray)
        @test size(val) == (1,)
    end

    @testset "Checkpointing" begin
        g1 = JuliusGraph(namespace=:t)
        set!(g1, :x, rand(Float32, 2, 2); is_param=true, namespace=:t)
        set!(g1, :w, rand(Float32, 2, 2); is_param=true, namespace=:t)
        addrule!(g1, GraphRule(:h, [:x, :w], :matmul; namespace=:t))
        addrule!(g1, GraphRule(:out, [:h], :sum_matrix; namespace=:t))

        ctx = CtxStore()
        schedule = CheckpointSchedule(g1, CheckpointData(CPUTrainData(); every=2); namespace=:t)
        forward_with_checkpointing!(g1, :out, ctx, schedule; namespace=:t)

        @test :out in schedule.checkpoints
        @test g1.nodes[:t][:h].value === nothing || g1.nodes[:t][:h].valid == false

        @test backward_with_checkpointing!(g1, :out; ctx_store=ctx, schedule=schedule, namespace=:t) !== nothing
    end

    @testset "FlashAttention CPU reference" begin
        N, D, d_head = 8, 16, 8
        Q = rand(Float32, N, D)
        K = rand(Float32, N, D)
        V = rand(Float32, N, D)

        out1 = zeros(Float32, N, D)
        out2 = zeros(Float32, N, D)
        out1, l1, m1 = flash_attn_fwd_cpu!(out1, Q, K, V, d_head; causal=true)
        out2, l2, m2 = flash_attn_fwd_cpu_simple!(out2, Q, K, V, d_head; causal=true)

        @test isapprox(out1, out2; atol=1e-5, rtol=1e-6)
        @test isapprox(l1, l2; atol=1e-5, rtol=1e-6)
        @test isapprox(m1, m2; atol=1e-5, rtol=1e-6)
    end

    @testset "Mixed precision tracker" begin
        tracker = LossScaleTracker(scale=16f0, growth_interval=1)
        scale1, ok1 = update!(tracker, true)
        @test ok1
        @test scale1 == 32f0

        scale2, ok2 = update!(tracker, false)
        @test !ok2
        @test scale2 == 16f0
    end
end
