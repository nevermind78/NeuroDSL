


@testset "Kernels" begin
    dev = NeuroDSL.Backend.CPUDevice()
    M, N = 4, 8

    @testset "RMSNorm fwd/bwd CPU" begin
        x     = randn(Float32,M,N)
        gamma = ones(Float32,N)
        out   = zeros(Float32,M,N)
        rms   = zeros(Float32,M)
        NeuroDSL.rmsnorm_fwd!(dev, out, rms, x, gamma)
        @test size(out) == (M,N)
        @test all(isfinite, out)
        dx = similar(x); dg = similar(gamma)
        NeuroDSL.rmsnorm_bwd!(dev, dx, dg, out, x, gamma, reshape(rms,M,1))
        @test size(dx) == (M,N)
        @test all(isfinite, dx)
    end

    @testset "SwiGLU fwd/bwd CPU" begin
        gate = randn(Float32,M,N); up = randn(Float32,M,N)
        out  = similar(gate)
        NeuroDSL.swiglu_fwd!(dev, out, gate, up)
        @test all(isfinite, out)
        dg = similar(gate); du = similar(up)
        NeuroDSL.swiglu_bwd!(dev, dg, du, out, gate, up)
        @test all(isfinite, dg) && all(isfinite, du)
    end

    @testset "Softmax" begin
        out = zeros(Float32,M,N)
        NeuroDSL.softmax_fwd!(dev, out, randn(Float32,M,N))
        @test all(sum(out, dims=2) .≈ 1f0)
        dx = similar(out)
        NeuroDSL.softmax_bwd!(dev, dx, out, out)
        @test all(isfinite, dx)
    end

    @testset "MSE loss" begin
        pred   = randn(Float32,M,N)
        target = randn(Float32,M,N)
        loss   = NeuroDSL.mse_loss_fwd(pred, target)
        @test length(loss) == 1
        @test loss[1] >= 0f0
        grad_out, grad_target = NeuroDSL.mse_loss_bwd(pred, target, [1f0])
        @test size(grad_out) == (M,N)
        @test size(grad_target) == (M,N)
        @test grad_out ≈ -grad_target
    end

    @testset "Cross-entropy" begin
        logits = randn(Float32,M,N)
        labels = rand(1:N, M)
        @test NeuroDSL.cross_entropy_loss(logits, labels) > 0f0
        @test size(NeuroDSL.cross_entropy_grad(logits, labels)) == (M,N)
    end

    @testset "adamw_step_batched! == boucle adamw_step! -- bit-exact (CPU)" begin
        Random.seed!(41)
        shapes = [(100,256), (256,256), (256,3), (7,), (1,3), (300,)]
        Ws_ref  = [randn(Float32, s...) for s in shapes]
        Ws_bat  = [copy(w) for w in Ws_ref]
        m1_ref  = [0.01f0 .* randn(Float32, s...) for s in shapes]
        m1_bat  = [copy(m) for m in m1_ref]
        m2_ref  = [0.01f0 .* rand(Float32, s...) for s in shapes]  # m2 est une variance -- toujours >= 0
        m2_bat  = [copy(m) for m in m2_ref]
        lr, b1, b2, eps_v, clip, wd = 1f-3, 0.9f0, 0.999f0, 1f-8, 0.5f0, 0.1f0

        for t in 1:5
            G = [randn(Float32, s...) .* 2f0 for s in shapes]
            dW_ref = [copy(g) for g in G]
            dW_bat = [copy(g) for g in G]
            for i in eachindex(Ws_ref)
                NeuroDSL.adamw_step!(dev, Ws_ref[i], dW_ref[i], m1_ref[i], m2_ref[i], lr, b1, b2, eps_v, t, clip, wd)
            end
            NeuroDSL.adamw_step_batched!(dev, Ws_bat, dW_bat, m1_bat, m2_bat, lr, b1, b2, eps_v, t, clip, wd)
            @test all(Ws_ref[i] == Ws_bat[i] for i in eachindex(Ws_ref))
            @test all(m1_ref[i] == m1_bat[i] for i in eachindex(m1_ref))
            @test all(m2_ref[i] == m2_bat[i] for i in eachindex(m2_ref))
            @test all(all(iszero, dw) for dw in dW_bat)
        end

        # Cas vide : ne plante pas
        NeuroDSL.adamw_step_batched!(dev, [], [], [], [], lr, b1, b2, eps_v, 1, clip, wd)
    end

    @testset "adamw_step_batched! == boucle adamw_step! -- bit-exact (GPU)" begin
        if NeuroDSL.Backend.CUDA_AVAILABLE
            cuda_dev = NeuroDSL.Backend.CUDADevice()
            Random.seed!(42)
            shapes = [(100,256), (256,256), (256,688), (688,256), (256,), (256,), (257,), (7,), (1,3), (300,)]
            Ws_ref  = [NeuroDSL.CUDA.cu(randn(Float32, s...)) for s in shapes]
            Ws_bat  = [copy(w) for w in Ws_ref]
            m1_ref  = [NeuroDSL.CUDA.cu(0.01f0 .* randn(Float32, s...)) for s in shapes]
            m1_bat  = [copy(m) for m in m1_ref]
            m2_ref  = [NeuroDSL.CUDA.cu(0.01f0 .* rand(Float32, s...)) for s in shapes]  # m2 est une variance -- toujours >= 0
            m2_bat  = [copy(m) for m in m2_ref]
            lr, b1, b2, eps_v, clip, wd = 1f-3, 0.9f0, 0.999f0, 1f-8, 0.5f0, 0.1f0

            for t in 1:5
                G = [NeuroDSL.CUDA.cu(randn(Float32, s...) .* 2f0) for s in shapes]
                dW_ref = [copy(g) for g in G]
                dW_bat = [copy(g) for g in G]
                for i in eachindex(Ws_ref)
                    NeuroDSL.adamw_step!(cuda_dev, Ws_ref[i], dW_ref[i], m1_ref[i], m2_ref[i], lr, b1, b2, eps_v, t, clip, wd)
                end
                NeuroDSL.adamw_step_batched!(cuda_dev, Ws_bat, dW_bat, m1_bat, m2_bat, lr, b1, b2, eps_v, t, clip, wd)
                @test all(Array(Ws_ref[i]) == Array(Ws_bat[i]) for i in eachindex(Ws_ref))
                @test all(Array(m1_ref[i]) == Array(m1_bat[i]) for i in eachindex(m1_ref))
                @test all(Array(m2_ref[i]) == Array(m2_bat[i]) for i in eachindex(m2_ref))
                @test all(all(iszero, Array(dw)) for dw in dW_bat)
            end

            # Cas vide : ne plante pas
            NeuroDSL.adamw_step_batched!(cuda_dev, [], [], [], [], lr, b1, b2, eps_v, 1, clip, wd)
        else
            println("⚠️  GPU non disponible — test adamw_step_batched! (GPU) ignoré.")
            @test true
        end
    end
end
