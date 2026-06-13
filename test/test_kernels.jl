


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
end
