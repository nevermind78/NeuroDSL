include(joinpath(@__DIR__, "surgery_f3_common.jl"))

dev = NeuroDSL.Backend.CPUDevice()
ns = :surgery_f3
graph_prefix = ARGS[1]
losses_path = ARGS[2]

Random.seed!(99)
g, logits = NeuroDSL.build_induction_graph(dev, ns; vocab_size=VS, dim=D, n_heads=NH,
                                            hidden_dim=HD, n_layers=NL, prefix_len=PL)
ps = NeuroDSL.params(g; namespace=ns)
m1 = Dict(p.name => zeros(Float32, size(p.value)...) for p in ps)
m2 = Dict(p.name => zeros(Float32, size(p.value)...) for p in ps)

losses = Float64[]
for t in 1:300
    push!(losses, train_step!(g, ns, logits, m1, m2, t, ps))
end

NeuroDSL.insert_block!(g, ns, :layer_2_out, D, NH, HD)
new_ps = [p for p in NeuroDSL.params(g; namespace=ns) if !haskey(m1, p.name)]
for p in new_ps
    m1[p.name] = zeros(Float32, size(p.value)...)
    m2[p.name] = zeros(Float32, size(p.value)...)
end
NeuroDSL.invalidate_all!(g; namespace=ns)
ps2 = NeuroDSL.params(g; namespace=ns)

for t in 301:400
    push!(losses, train_step!(g, ns, logits, m1, m2, t, ps2))
end

opt_state = NeuroDSL.AdamWState(400, m1, m2)
NeuroDSL.save_graph!(g, ns, graph_prefix; opt_state=opt_state)

open(losses_path, "w") do io
    for l in losses
        println(io, l)
    end
end
println("SCRIPT_A_OK  t=400")
