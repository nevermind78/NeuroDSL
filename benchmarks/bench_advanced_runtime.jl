using Pkg
Pkg.activate(@__DIR__ * "/..")
using NeuroDSL
using BenchmarkTools

function bench_flash(N=64, D=64, d_head=16)
    Q = rand(Float32, N, D)
    K = rand(Float32, N, D)
    V = rand(Float32, N, D)
    out = zeros(Float32, N, D)

    println("--- FlashAttention forward CPU (tiled) ---")
    @btime flash_attn_fwd_cpu!($out, $Q, $K, $V, $d_head; causal=true)
    println("--- FlashAttention forward CPU (reference) ---")
    @btime flash_attn_fwd_cpu_simple!($out, $Q, $K, $V, $d_head; causal=true)
end

function bench_memory_plan()
    g = NeuroGraph(namespace=:bench)
    set!(g, :x, rand(Float32, 128, 128); is_param=true, namespace=:bench)
    set!(g, :w, rand(Float32, 128, 128); is_param=true, namespace=:bench)
    addrule!(g, GraphRule(:h, [:x, :w], :matmul; namespace=:bench))
    addrule!(g, GraphRule(:o, [:h], :sum_matrix; namespace=:bench))

    plan, pool = plan_memory!(g; namespace=:bench)
    invalidate_all!(g; namespace=:bench)

    println("--- Demand planned performance ---")
    @btime demand_planned!($g, :o, $plan, $pool; namespace=:bench)
end

println("=== Benchmarks NeuroDSL advanced runtime ===")
bench_flash()
bench_memory_plan()
