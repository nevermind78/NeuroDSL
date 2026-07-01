using Pkg
Pkg.activate(pwd())
using NeuroDSL, BenchmarkTools, Printf, Random, Statistics
Random.seed!(1234)

function mk_graph(D, depth, batch=1)
    g = NeuroGraph(device=Backend.CPUDevice())
    NeuroDSL.set!(g, :x, randn(Float32, batch, D))
    NeuroDSL.set!(g, :y, zeros(Float32, batch, D); atom_type=NeuroDSL.Datom)
    prev = :x
    for i in 1:depth
        w = Symbol(:W, i)
        NeuroDSL.set!(g, w, randn(Float32, D, D) .* 0.01f0; is_param=true)
        out = Symbol(:h, i)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out, [prev, w], :matmul; attrs=Dict(:trans_b=>true)))
        prev = out
    end
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [prev, :y], :mse_loss))
    return g
end

D, depth, batch = 128, 8, 8
g = mk_graph(D, depth, batch)
ctx = CtxStore()
for i in 1:10
    NeuroDSL.demand!(g, :loss; ctx_store=ctx)
end
GC.gc()
alloc_fwd = @allocated NeuroDSL.demand!(g, :loss; ctx_store=ctx)
@printf("NeuroDSL forward allocated = %d bytes\n", alloc_fwd)
b_fwd = @benchmark NeuroDSL.demand!($g, :loss; ctx_store=$ctx) samples=20 evals=3
println(b_fwd)

# Baseline forward equivalent compute
H = randn(Float32, batch, D)
W = randn(Float32, D, D)
y = zeros(Float32, batch, D)
alloc_base = @allocated begin
    Y = H * W
    sum((Y .- y).^2)
end
@printf("Baseline forward allocated = %d bytes\n", alloc_base)
b_base = @benchmark begin
    Y = $H * $W
    sum((Y .- $y).^2)
end samples=20 evals=3
println(b_base)

# Backward-only timing
D, depth, batch = 128, 8, 8
g = mk_graph(D, depth, batch)
ctx = CtxStore()
NeuroDSL.demand!(g, :loss; ctx_store=ctx)
for i in 1:10
    NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx)
end
GC.gc()
alloc_bwd = @allocated NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx)
@printf("NeuroDSL backward allocated = %d bytes\n", alloc_bwd)
b_bwd = @benchmark NeuroDSL.backward_graph!($g, :loss; ctx_store=$ctx) samples=20 evals=3
println(b_bwd)

# Incremental vs full backward
D, depth, batch = 128, 16, 8
g = mk_graph(D, depth, batch)
ctx = CtxStore()
NeuroDSL.demand!(g, :loss; ctx_store=ctx)
NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx; full=true)
for i in 1:10
    NeuroDSL.demand!(g, :loss; ctx_store=ctx)
    NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx; full=true)
end
GC.gc()
@printf("Full backward benchmark:\n")
b_full = @benchmark begin NeuroDSL.demand!($g, :loss; ctx_store=$ctx); NeuroDSL.backward_graph!($g, :loss; ctx_store=$ctx; full=true) end samples=15 evals=3
println(b_full)
@printf("Incremental backward benchmark:\n")
b_incr = @benchmark begin NeuroDSL.demand!($g, :loss; ctx_store=$ctx); NeuroDSL.backward_graph!($g, :loss; ctx_store=$ctx; full=false) end samples=15 evals=3
println(b_incr)
