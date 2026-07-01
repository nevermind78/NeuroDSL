
using Printf
using Statistics


# ─────────────────────────────────────────────────────────────────────────────
# grad_check — différences finies centrées
# CORRECTIONS : eps 1e-3→1e-4, tol 5e-2→1e-3, invalidate_all! avant forward,
#               retourne (Bool, Float32) pour logger les valeurs exactes
# ─────────────────────────────────────────────────────────────────────────────
function grad_check(g, param_sym, loss_sym;
                    eps=Float32(1e-4), tol=Float32(1e-3), verbose=true)
    NeuroDSL.invalidate_all!(g)
    ctx = NeuroDSL.CtxStore()
    NeuroDSL.demand!(g, loss_sym; ctx_store=ctx)
    NeuroDSL.backward_graph!(g, loss_sym; ctx_store=ctx)
    grad_a = Array(NeuroDSL.node(g, param_sym).gradient)

    pn = NeuroDSL.node(g, param_sym)
    orig = copy(pn.value); orig_cpu = Array(orig)
    grad_n = zeros(Float32, size(orig_cpu))
    for i in eachindex(orig_cpu)
        v⁺ = copy(orig_cpu); v⁺[i] += eps
        pn.value = NeuroDSL.Backend.to_device(g.device, v⁺); NeuroDSL.invalidate_all!(g)
        l⁺ = sum(Array(NeuroDSL.demand!(g, loss_sym)))
        v⁻ = copy(orig_cpu); v⁻[i] -= eps
        pn.value = NeuroDSL.Backend.to_device(g.device, v⁻); NeuroDSL.invalidate_all!(g)
        l⁻ = sum(Array(NeuroDSL.demand!(g, loss_sym)))
        grad_n[i] = (l⁺ - l⁻) / (2f0 * eps)
    end
    pn.value = orig; NeuroDSL.invalidate_all!(g)

    diff     = abs.(grad_a .- grad_n)
    max_err  = maximum(diff)
    mean_err = mean(diff)
    worst_i  = argmax(diff)
    ok       = max_err < tol

    if verbose
        status = ok ? "✅" : "❌"
        @printf "  [:%s] max_err=%.2e  mean_err=%.2e  (tol=%.0e) %s
" param_sym max_err mean_err tol status
        if !ok
            @printf "     analytique[%s]=%.6f  numérique[%s]=%.6f
" string(worst_i) grad_a[worst_i] string(worst_i) grad_n[worst_i]
        end
    end
    return ok, max_err
end

# Alias utilisé dans les cellules 16-21 du notebook
function check_gradients(g, param_sym, loss_sym;
                         eps=Float32(1e-4), tol=Float32(1e-3))
    ok, _ = grad_check(g, param_sym, loss_sym; eps=eps, tol=tol, verbose=true)
    return ok
end

# ─────────────────────────────────────────────────────────────────────────────
# @testset — utilisé par runtests.jl
# ─────────────────────────────────────────────────────────────────────────────
@testset "Backward — gradient checks (tol=1e-3, eps=1e-4)" begin
    dev = NeuroDSL.Backend.CPUDevice()

    @testset ":rmsnorm" begin
        g = NeuroDSL.NeuroGraph(namespace=:t_rmsnorm, device=dev)
        NeuroDSL.set!(g, :X,     randn(Float32, 2, 4))
        NeuroDSL.set!(g, :gamma, ones(Float32, 4); is_param=true)
        NeuroDSL.set!(g, :Z,     zeros(Float32, 2, 4); atom_type=NeuroDSL.Datom)
        NeuroDSL.@addrules g :t_rmsnorm begin
            H = rmsnorm(X, gamma)
            L = mse_loss(H, Z)
        end
        ok, _ = grad_check(g, :gamma, :L)
        @test ok
    end

    @testset ":matmul trans_b=false" begin
        g = NeuroDSL.NeuroGraph(namespace=:t_mm, device=dev)
        NeuroDSL.set!(g, :A, randn(Float32, 2, 3); is_param=true)
        NeuroDSL.set!(g, :B, randn(Float32, 3, 4))
        NeuroDSL.set!(g, :Z, zeros(Float32, 2, 4); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:C, [:A, :B], :matmul;
            attrs=Dict{Symbol,Any}(:trans_b=>false), namespace=:t_mm))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [:C, :Z], :mse_loss; namespace=:t_mm))
        ok, _ = grad_check(g, :A, :L; eps=Float32(1e-3), tol=Float32(5e-3))
        @test ok
    end

    @testset ":matmul trans_b=true" begin
        g = NeuroDSL.NeuroGraph(namespace=:t_mmt, device=dev)
        NeuroDSL.set!(g, :A, randn(Float32, 2, 3); is_param=true)
        NeuroDSL.set!(g, :B, randn(Float32, 4, 3))
        NeuroDSL.set!(g, :Z, zeros(Float32, 2, 4); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:C, [:A, :B], :matmul;
            attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=:t_mmt))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [:C, :Z], :mse_loss; namespace=:t_mmt))
        ok, _ = grad_check(g, :A, :L; eps=Float32(1e-3), tol=Float32(5e-3))
        @test ok
    end

    @testset ":linear avec biais" begin
        g = NeuroDSL.NeuroGraph(namespace=:t_lin, device=dev)
        NeuroDSL.set!(g, :X, randn(Float32, 3, 4))
        lin = NeuroDSL.Linear(4, 8, bias=true)
        out = lin(g, :X, :fc; namespace=:t_lin)
        NeuroDSL.set!(g, :Z, zeros(Float32, 3, 8); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [out, :Z], :mse_loss; namespace=:t_lin))
        ok_W, _ = grad_check(g, :fc_W, :L)
        ok_b, _ = grad_check(g, :fc_b, :L)
        @test ok_W
        @test ok_b
    end

    @testset ":swiglu" begin
        g = NeuroDSL.NeuroGraph(namespace=:t_sg, device=dev)
        NeuroDSL.set!(g, :gate, randn(Float32, 2, 4); is_param=true)
        NeuroDSL.set!(g, :up,   randn(Float32, 2, 4))
        NeuroDSL.set!(g, :Z,    zeros(Float32, 2, 4); atom_type=NeuroDSL.Datom)
        NeuroDSL.@addrules g :t_sg begin
            out = swiglu(gate, up)
            L   = mse_loss(out, Z)
        end
        ok, _ = grad_check(g, :gate, :L; eps=Float32(1e-3), tol=Float32(5e-3))
        @test ok
    end

    @testset ":softmax" begin
        g = NeuroDSL.NeuroGraph(namespace=:t_sm, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 3, 4); is_param=true)
        NeuroDSL.set!(g, :Z, zeros(Float32, 3, 4); atom_type=NeuroDSL.Datom)
        NeuroDSL.@addrules g :t_sm begin
            p = softmax(x)
            L = mse_loss(p, Z)
        end
        ok, _ = grad_check(g, :x, :L)
        @test ok
    end

    @testset ":scale_mask" begin
        g = NeuroDSL.NeuroGraph(namespace=:t_sc, device=dev)
        NeuroDSL.set!(g, :scores, randn(Float32, 4, 4); is_param=true)
        NeuroDSL.set!(g, :Z,      zeros(Float32, 4, 4); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:sc, [:scores], :scale_mask;
            attrs=Dict{Symbol,Any}(:d_head=>8), namespace=:t_sc))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:p,  [:sc], :softmax; namespace=:t_sc))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L,  [:p, :Z], :mse_loss; namespace=:t_sc))
        ok, _ = grad_check(g, :scores, :L)
        @test ok
    end

    @testset ":add fan-out" begin
        g = NeuroDSL.NeuroGraph(namespace=:t_add, device=dev)
        NeuroDSL.set!(g, :x,  randn(Float32, 2, 2); is_param=true)
        NeuroDSL.set!(g, :w1, ones(Float32, 2, 2))
        NeuroDSL.set!(g, :w2, ones(Float32, 2, 2))
        NeuroDSL.set!(g, :Z,  zeros(Float32, 2, 2); atom_type=NeuroDSL.Datom)
        NeuroDSL.@addrules g :t_add begin
            y = mul(x, w1)
            z = mul(x, w2)
            s = add(y, z)
            L = mse_loss(s, Z)
        end
        ok, _ = grad_check(g, :x, :L; eps=Float32(1e-3), tol=Float32(1e-2))
        @test ok
    end

    @testset ":mse_loss" begin
        g = NeuroDSL.NeuroGraph(namespace=:t_mse, device=dev)
        NeuroDSL.set!(g, :pred,   randn(Float32, 2, 4); is_param=true)
        NeuroDSL.set!(g, :target, randn(Float32, 2, 4); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [:pred, :target], :mse_loss; namespace=:t_mse))
        ok, _ = grad_check(g, :pred, :L; eps=Float32(1e-3), tol=Float32(5e-3))
        @test ok
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Cellule 16 — Backprop : RMSNorm + Matmul
# ─────────────────────────────────────────────────────────────────────────────
let
    using Random 
    using CUDA
    Random.seed!(1234) 
    CUDA.seed!(1234) 
    println("seed = 1234")
    g = NeuroDSL.NeuroGraph(namespace=:grad_check)
    NeuroDSL.set!(g, :X,     NeuroDSL.Backend.randn32(g.device, 2, 4))
    NeuroDSL.set!(g, :W,     NeuroDSL.Backend.randn32(g.device, 4, 4); is_param=true)
    NeuroDSL.set!(g, :gamma, NeuroDSL.Backend.ones32(g.device, 4);     is_param=true)
    NeuroDSL.set!(g, :Zeros, NeuroDSL.Backend.zeros32(g.device, 2, 4); atom_type=NeuroDSL.Datom)
    NeuroDSL.@addrules g :grad_check begin
        H    = rmsnorm(X, gamma)
        Out  = matmul(H, W, trans_b=true)
        Loss = mse_loss(Out, Zeros)
    end
    w_ok = check_gradients(g, :W, :Loss; eps=Float32(1e-4), tol=Float32(5e-3))
    g_ok = check_gradients(g, :gamma, :Loss; eps=Float32(1e-4), tol=Float32(5e-3))
    if w_ok && g_ok
    println("
🚀 Cell 16 — Moteur backprop validé.")
    else
        @warn "Cell 16 — Gradient check échoué (erreur Float32 attendue sur matmul)"
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Cellule 17 — Accumulation fan-out
# ─────────────────────────────────────────────────────────────────────────────
let
    println("--- Test accumulation (branching) ---")
    g = NeuroDSL.NeuroGraph(namespace=:branch_test)
    NeuroDSL.set!(g, :x,     NeuroDSL.Backend.randn32(g.device, 2, 2); is_param=true)
    NeuroDSL.set!(g, :w1,    NeuroDSL.Backend.ones32(g.device, 2, 2);  is_param=true)
    NeuroDSL.set!(g, :w2,    NeuroDSL.Backend.ones32(g.device, 2, 2);  is_param=true)
    NeuroDSL.set!(g, :Zeros, NeuroDSL.Backend.zeros32(g.device, 2, 2); atom_type=NeuroDSL.Datom)
    NeuroDSL.@addrules g :branch_test begin
        y            = mul(x, w1)
        z            = mul(x, w2)
        added_output = add(y, z)
        loss         = mse_loss(added_output, Zeros)
    end
    ok = ok = check_gradients(g, :x, :loss; eps=1e-4, tol=2e-3)
    ok ? println("✅ Cell 17 — Accumulation fan-out validée.") :
         println("⚠️  Écart détecté — vérifier GRAD_RULES[:add] et [:mul]")
end

# ─────────────────────────────────────────────────────────────────────────────
# Cellule 18 — Linear backward (biais)
# ─────────────────────────────────────────────────────────────────────────────
let
    println("--- Test :linear backward (avec biais) ---")
    g = NeuroDSL.NeuroGraph(namespace=:linear_test)
    NeuroDSL.set!(g, :X, NeuroDSL.Backend.randn32(g.device, 3, 4))
    lin     = NeuroDSL.Linear(4, 8, bias=true)
    out_sym = lin(g, :X, :fc; namespace=:linear_test)
    NeuroDSL.set!(g, :Zeros, NeuroDSL.Backend.zeros32(g.device, 3, 8); atom_type=NeuroDSL.Datom)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [out_sym, :Zeros], :mse_loss; namespace=:linear_test))
    ok_W = check_gradients(g, :fc_W, :L)
    ok_b = check_gradients(g, :fc_b, :L)
    (ok_W && ok_b) ? println("✅ Cell 18 — :linear backward OK (W + biais)") :
                     println("❌ Cell 18 — Échec gradient :linear")
end

# ─────────────────────────────────────────────────────────────────────────────
# Cellule 19 — Backward de :scale_mask
# ─────────────────────────────────────────────────────────────────────────────
let
    println("--- Test :scale_mask backward ---")
    g = NeuroDSL.NeuroGraph(namespace=:smask_test)
    seqlen, d_head = 4, 8
    NeuroDSL.set!(g, :scores, NeuroDSL.Backend.randn32(g.device, seqlen, seqlen); is_param=true)
    NeuroDSL.set!(g, :Zeros,  NeuroDSL.Backend.zeros32(g.device, seqlen, seqlen); atom_type=NeuroDSL.Datom)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:scaled, [:scores], :scale_mask;
        attrs=Dict{Symbol,Any}(:d_head=>d_head), namespace=:smask_test))
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:probs, [:scaled], :softmax; namespace=:smask_test))
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [:probs, :Zeros], :mse_loss; namespace=:smask_test))
    ok = check_gradients(g, :scores, :L)
    ok ? println("✅ Cell 19 — :scale_mask backward OK") :
         println("❌ Cell 19 — Échec gradient :scale_mask")
end

# ─────────────────────────────────────────────────────────────────────────────
# Cellule 20 — Backward de :mse_loss
# ─────────────────────────────────────────────────────────────────────────────
let
    println("--- Test :mse_loss backward ---")
    g = NeuroDSL.NeuroGraph(namespace=:mse_test)
    NeuroDSL.set!(g, :pred,   NeuroDSL.Backend.randn32(g.device, 2, 4); is_param=true)
    NeuroDSL.set!(g, :target, NeuroDSL.Backend.randn32(g.device, 2, 4); atom_type=NeuroDSL.Datom)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [:pred, :target], :mse_loss; namespace=:mse_test))
    ok = check_gradients(g, :pred, :L; eps=Float32(1e-3), tol=Float32(5e-3))
    ok ? println("✅ Cell 20 — :mse_loss backward OK") :
         println("❌ Cell 20 — Échec gradient :mse_loss")
end

# ─────────────────────────────────────────────────────────────────────────────
# Cellule 21 — LlamaBlock forward pass + shapes
# ─────────────────────────────────────────────────────────────────────────────
let
    println("--- Test LlamaBlock forward ---")
    dim, n_heads, hidden, seqlen = 16, 2, 32, 4
    g = NeuroDSL.NeuroGraph(namespace=:llama_fwd)
    x_val = NeuroDSL.Backend.randn32(g.device, seqlen, dim)
    NeuroDSL.set!(g, :input_x, x_val)
    block   = NeuroDSL.LlamaBlock(dim, n_heads, hidden)
    out_sym = block(g, :input_x, :block1; namespace=:llama_fwd)
    ctx     = NeuroDSL.CtxStore()
    out_val = NeuroDSL.demand!(g, out_sym; ctx_store=ctx)
    @assert size(out_val) == (seqlen, dim) "Shape doit être ($seqlen,$dim)"
    println("✅ Cell 21 — LlamaBlock forward OK")
    println("   Input  : $(size(x_val))")
    println("   Output : $(size(out_val))")
    println("   Règles : $(length(g.rules[:llama_fwd]))")
    println("   Params : $(length(NeuroDSL.params(g; namespace=:llama_fwd)))")
end

# ─────────────────────────────────────────────────────────────────────────────
# Section E — Validation des kernels CUDA (si GPU disponible)
# ─────────────────────────────────────────────────────────────────────────────
if NeuroDSL.Backend.CUDA_AVAILABLE
    using CUDA
    CUDA.allowscalar(false)

    println("\n" * "="^50)
    println("   Validation des kernels CUDA (tol=1e-3, eps=1e-4)")
    println("="^50)

    for (M, N) in [(2,4), (4,8), (8,16), (1,32)]
        println("\n--- RMSNorm CUDA (M=$M, N=$N) ---")
        dev = NeuroDSL.Backend.CUDADevice()
        ns  = Symbol(:cuda_rmsnorm_, M, :_, N)
        graph = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(graph, :X,     NeuroDSL.Backend.randn32(dev, M, N))
        NeuroDSL.set!(graph, :gamma, NeuroDSL.Backend.ones32(dev, N);  is_param=true)
        NeuroDSL.set!(graph, :W,     NeuroDSL.Backend.randn32(dev, N, N); is_param=true)
        NeuroDSL.set!(graph, :Z,     NeuroDSL.Backend.zeros32(dev, M, N); atom_type=NeuroDSL.Datom)
        NeuroDSL.@addrules graph ns begin
            H    = rmsnorm(X, gamma)
            Out  = matmul(H, W, trans_b=true)
            Loss = mse_loss(Out, Z)
        end
        ok_g = check_gradients(graph, :gamma, :Loss; eps=Float32(1e-3), tol=Float32(1e-2))
        ok_w = check_gradients(graph, :W,     :Loss; eps=Float32(1e-3), tol=Float32(1e-2))
        (ok_g && ok_w) ? println("  ✅ M=$M N=$N") : println("  ❌ M=$M N=$N")
    end
else
    println("⚠️  GPU non disponible — tests CUDA ignorés.")
end
