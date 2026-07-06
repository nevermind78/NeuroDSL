# Oracle de correction différentielle : rejoue une séquence pseudo-aléatoire MAIS
# DÉTERMINISTE (graine fixée) de set!/patch_node!/demand!/restore_from_cache! sur un
# LlamaModel de taille modeste, et compare l'état complet du graphe après chaque
# opération à un hash de référence (test/fixtures/oracle_trace_golden_hashes.bin).
#
# Construit à partir du script scratch manuel utilisé pour valider le correctif
# _consumers_index! (Phase 1) : à l'époque, comparait deux fichiers de trace brute
# (valeurs Float32 complètes) entre deux révisions git — correct mais bien trop volumineux
# pour être committé (>100 Mo pour 300 opérations). Ici, seul un hash 64 bits par étape est
# stocké (301 × 8 octets), ce qui préserve la même sensibilité à toute divergence de
# comportement (bit-à-bit, pas seulement de vitesse) sans committer de gros binaire.
#
# Hash maison (FNV-1a 64 bits, pure arithmétique sur UInt64) plutôt que CRC32c/Base.hash :
# aucune dépendance supplémentaire dans Project.toml, et un résultat garanti stable dans le
# temps (Base.hash n'offre aucune garantie de stabilité inter-versions Julia, ce qui
# invaliderait le fixture sans aucune régression réelle du moteur).
#
# Toute divergence du hash à une étape donnée signale une régression de comportement du
# moteur de graphe (invalidation, patching, restauration par cache) — pas seulement une
# régression de vitesse, qui n'est jamais couverte par cette suite de tests.

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

@testset "Oracle de replay différentiel (300 opérations, hashes de référence)" begin
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
        else # :demand_mid
            site = all_sites[rand(rng, 1:length(all_sites))]
            NeuroDSL.demand!(g, site; namespace=ns)
        end
        push!(hashes, _fnv1a_64(_oracle_snapshot_bytes(g, ns)))
    end

    golden_path = joinpath(@__DIR__, "fixtures", "oracle_trace_golden_hashes.bin")
    @test isfile(golden_path)
    if isfile(golden_path)
        golden = reinterpret(UInt64, read(golden_path))
        @test length(hashes) == length(golden)
        if length(hashes) == length(golden)
            first_diff = findfirst(i -> hashes[i] != golden[i], eachindex(hashes))
            @test first_diff === nothing
        end
    end
end
