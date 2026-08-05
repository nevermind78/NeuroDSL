using NeuroDSL, CUDA, Random, Printf
dev = NeuroDSL.Backend.CUDADevice()

function build_deep_network(depth::Int, dim::Int, batch::Int; ns::Symbol)
    g = NeuroDSL.NeuroGraph(device=dev, namespace=ns)
    Random.seed!(1234)
    NeuroDSL.set!(g, :x0, randn(Float32, batch, dim) .* 0.01f0; namespace=ns)
    h = :x0
    for i in 1:depth
        Wsym = Symbol(:W, i)
        NeuroDSL.set!(g, Wsym, randn(Float32, dim, dim) .* 0.01f0; is_param=true, namespace=ns)
        mm = Symbol(:h, i, :_mm)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(mm, [h, Wsym], :matmul; namespace=ns))
        rl = Symbol(:h, i, :_relu)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(rl, [mm], :relu; namespace=ns))
        h = rl
    end
    NeuroDSL.set!(g, :target, randn(Float32, batch, dim) .* 0.01f0; namespace=ns)
    loss_sym = :loss
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss_sym, [h, :target], :mse_loss; namespace=ns))
    return g, loss_sym
end

depth, dim, batch, every = 16, 512, 4096, 4
ns = :wa
g, loss_sym = build_deep_network(depth, dim, batch; ns=ns)
cd = NeuroDSL.CheckpointData(every=every)
sch = NeuroDSL.CheckpointSchedule(g, cd; namespace=ns)
ctx = NeuroDSL.CtxStore()

# position of each symbol in schedule.order, for cross-reference
pos = Dict(sym => i for (i,sym) in enumerate(sch.order))

println("recomputable set, in order of position:")
for sym in sort(collect(sch.recomputable); by = s -> pos[s])
    @printf "  pos=%3d  %s\n" pos[sym] String(sym)
end

NeuroDSL.forward_with_checkpointing!(g, loss_sym, ctx, sch; namespace=ns)
CUDA.synchronize()

NeuroDSL.backward_with_checkpointing!(g, loss_sym; ctx_store=ctx, schedule=sch, namespace=ns)
CUDA.synchronize()

println("\nAfter real backward_with_checkpointing!, recomputable nodes still alive (value != nothing):")
for sym in sort(collect(sch.recomputable); by = s -> pos[s])
    nd = g.nodes[ns][sym]
    alive = nd.value !== nothing
    @printf "  pos=%3d  %-12s alive=%-5s valid=%-5s has_ctx=%-5s grad=%s\n" pos[sym] String(sym) alive nd.valid haskey(ctx, sym) (nd.gradient === nothing ? "nothing" : "SET")
end
