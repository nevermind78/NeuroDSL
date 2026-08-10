using NeuroDSL, Random

dev = NeuroDSL.Backend.CPUDevice()
ns = :oracle_replay_test
n_layers, dim, n_heads, hidden_dim, seq_len = 6, 32, 4, 64, 8

Random.seed!(1)
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :input, randn(Float32, seq_len, dim); namespace=ns)
output_sym = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim)(g, :input; namespace=ns)
NeuroDSL.demand!(g, output_sym; namespace=ns)
clean_cache = NeuroDSL.capture_activations(g, ns)

layer_sites = [Symbol(:layer_, i, :_out) for i in 1:n_layers]
head_sites  = [Symbol(:layer_, i, :_mha_ao_h, h) for i in 1:n_layers for h in 1:n_heads]
all_sites   = vcat(layer_sites, head_sites)

rng = MersenneTwister(2024)
n_ops = 300
action_kinds = [:patch, :restore, :dense_set, :demand_output, :demand_mid]

STOP_AT = 29
for step in 1:STOP_AT
    kind = action_kinds[rand(rng, 1:length(action_kinds))]
    site_used = nothing
    if kind == :patch
        site = all_sites[rand(rng, 1:length(all_sites))]
        site_used = site
        NeuroDSL.patch_node!(g, site, clean_cache; namespace=ns)
    elseif kind == :restore
        site = all_sites[rand(rng, 1:length(all_sites))]
        site_used = site
        affected = NeuroDSL._downstream_nodes(g, site, ns)
        NeuroDSL.restore_from_cache!(g, ns, clean_cache, affected)
    elseif kind == :dense_set
        NeuroDSL.set!(g, :input, randn(MersenneTwister(1000 + step), Float32, seq_len, dim); namespace=ns)
    elseif kind == :demand_output
        NeuroDSL.demand!(g, output_sym; namespace=ns)
    else # :demand_mid
        site = all_sites[rand(rng, 1:length(all_sites))]
        site_used = site
        NeuroDSL.demand!(g, site; namespace=ns)
    end
    println("step $step : kind=$kind  site=$site_used")
end

println("\n--- état final (step $STOP_AT) : symbole => (valid, value_hash_ou_nothing) ---")
open(joinpath(@__DIR__, "oracle_dump_$(ARGS[1]).txt"), "w") do io
    for sym in sort(collect(keys(g.nodes[ns])))
        nd = g.nodes[ns][sym]
        vh = nd.value === nothing ? "nothing" : string(hash(Array(nd.value)))
        println(io, "$sym  valid=$(nd.valid)  value_hash=$vh")
    end
end
println("Dump écrit -> oracle_dump_$(ARGS[1]).txt")
