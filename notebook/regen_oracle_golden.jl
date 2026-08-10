# Régénère test/fixtures/oracle_trace_golden_hashes.bin après le correctif
# demand!/_ancestors_of! -- la divergence trouvée au step 29 (voir
# diag_oracle_diff.jl) est un changement de comportement INTENTIONNEL et
# CORRECT : sous l'ancien demand!, demander layer_4_mha_ao_h3 (step 28,
# :demand_mid) "ressuscitait" par accident des têtes SŒURS (h1/h2/h4, pas de
# vrais ancêtres de h3) simplement parce qu'elles se trouvaient dans le
# préfixe topologique parcouru. Le nouveau demand! ne touche plus que les
# vrais ancêtres -- confirmé cohérent avec la suite d'invariants existante
# (1644/1644 toujours verts, qui vérifie déjà "seuls les nœuds du chemin
# topologique X->Y peuvent changer", propriété que ce correctif resserre
# sans la violer).
using NeuroDSL, Random

function _fnv1a_64(bytes::Vector{UInt8})::UInt64
    h = 0xcbf29ce484222325
    for b in bytes
        h = xor(h, UInt64(b))
        h *= 0x100000001b3
    end
    return h
end

function _oracle_snapshot_bytes(g::NeuroDSL.NeuroGraph, ns::Symbol)
    io = IOBuffer()
    for sym in sort(collect(keys(g.nodes[ns])))
        nd = g.nodes[ns][sym]
        write(io, UInt8(nd.valid ? 1 : 0))
        if nd.value === nothing
            write(io, Int32(-1))
        else
            arr = Array(nd.value)
            write(io, Int32(length(arr)))
            write(io, arr)
        end
    end
    return take!(io)
end

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

hashes = UInt64[_fnv1a_64(_oracle_snapshot_bytes(g, ns))]
for step in 1:n_ops
    kind = action_kinds[rand(rng, 1:length(action_kinds))]
    if kind == :patch
        site = all_sites[rand(rng, 1:length(all_sites))]
        NeuroDSL.patch_node!(g, site, clean_cache; namespace=ns)
    elseif kind == :restore
        site = all_sites[rand(rng, 1:length(all_sites))]
        affected = NeuroDSL._downstream_nodes(g, site, ns)
        NeuroDSL.restore_from_cache!(g, ns, clean_cache, affected)
    elseif kind == :dense_set
        NeuroDSL.set!(g, :input, randn(MersenneTwister(1000 + step), Float32, seq_len, dim); namespace=ns)
    elseif kind == :demand_output
        NeuroDSL.demand!(g, output_sym; namespace=ns)
    else
        site = all_sites[rand(rng, 1:length(all_sites))]
        NeuroDSL.demand!(g, site; namespace=ns)
    end
    push!(hashes, _fnv1a_64(_oracle_snapshot_bytes(g, ns)))
end

golden_path = joinpath(dirname(@__DIR__), "test", "fixtures", "oracle_trace_golden_hashes.bin")
open(golden_path, "w") do io
    write(io, hashes)
end
println("Écrit ", length(hashes), " hashes -> ", golden_path)
