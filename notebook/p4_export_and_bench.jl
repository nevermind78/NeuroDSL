# ══════════════════════════════════════════════════════════════════════════════
# P4 — export d'un modèle NeuroDSL (5 couches effectives : 4 originales + 1
# greffée via insert_block!, même config que real_llm_surgery_v2) vers
# .bin+.json (save_graph!, src/serialization.jl, déjà existant, zéro nouveau
# code source), mesure du coût NeuroDSL (sweep_patch_sites!/cone) sur les 20
# sites tête d'attention, pour comparaison croisée avec un script PyTorch
# (notebook/real_llm_patch_bench.py) qui rejoue le même modèle avec les MÊMES
# poids exportés. Conçu avec Fable le 2026-07-11.
#
# Les valeurs des poids sont sans incidence sur le coût de calcul (seule la
# structure du graphe compte -- déjà établi et réutilisé tel quel dans P1-bis
# cette session) : poids aléatoires (seed fixe), pas de réentraînement.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics, Printf, LinearAlgebra, JSON

dev = NeuroDSL.Backend.CUDADevice()
dim, n_heads, hidden_dim, n_layers, seq_len = 256, 4, 512, 4, 256

Random.seed!(7)
ns = :p4_model
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :input, (NeuroDSL.Backend.rand32(dev, seq_len, dim) .- 0.5f0); namespace=ns)
out = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim; batched_attn=true)(g, :input; namespace=ns)
NeuroDSL.insert_block!(g, ns, :layer_1_out, dim, n_heads, hidden_dim; batched_attn=true)
# insert_block! rebranche les consommateurs de :layer_1_out -- `out` (le nom du
# nœud de sortie du modèle, ex. :layer_4_out) reste la bonne cible en aval,
# seules les RÈGLES intermédiaires changent (déjà vérifié F1, test/test_surgery.jl).

clean_out = Array(NeuroDSL.demand!(g, out; namespace=ns))
println("Sortie propre calculée, shape=", size(clean_out))

export_dir = joinpath(@__DIR__, "p4_export")
mkpath(export_dir)
NeuroDSL.save_graph!(g, ns, joinpath(export_dir, "model"))
open(joinpath(export_dir, "reference_output.bin"), "w") do io
    write(io, clean_out)
end

layer_prefixes = ["layer_1", "surgery_layer_1_out", "layer_2", "layer_3", "layer_4"]
sites = [Symbol(lp, "_mha_ao_h", h) for lp in layer_prefixes for h in 1:n_heads]
@assert length(sites) == 20
for s in sites
    @assert haskey(g.nodes[ns], s) "site manquant : $s"
end

open(joinpath(export_dir, "meta.json"), "w") do io
    JSON.print(io, Dict(
        "output_symbol" => string(out), "output_shape" => collect(size(clean_out)),
        "dim" => dim, "n_heads" => n_heads, "hidden_dim" => hidden_dim,
        "n_layers_orig" => n_layers, "seq_len" => seq_len,
        "layer_prefixes" => layer_prefixes,
        "graft_after" => "layer_1_out", "graft_prefix" => "surgery_layer_1_out",
        "sites" => string.(sites),
    ))
end
println("Export -> ", export_dir)

# ═══ Corruption : un seul token remplacé (protocole standard, déjà utilisé toute la session) ═══
clean_input = copy(Array(NeuroDSL.demand!(g, :input; namespace=ns)))
corrupted_input = copy(clean_input)
corrupt_row = 5
corrupted_input[corrupt_row, :] .= Array(NeuroDSL.Backend.rand32(dev, dim) .- 0.5f0)

NeuroDSL.set!(g, :input, clean_input; namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, out; namespace=ns)
clean_cache = NeuroDSL.capture_activations(g, ns)

NeuroDSL.set!(g, :input, corrupted_input; namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
corrupted_out_val = Array(NeuroDSL.demand!(g, out; namespace=ns))
corrupted_cache = NeuroDSL.capture_activations(g, ns)

function sync()
    NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.synchronize()
end

function timed_site!(g, ns, site, out_sym, clean_cache, corrupted_cache; reps=20, warmup=3)
    affected = NeuroDSL._downstream_nodes(g, site, ns)
    for _ in 1:warmup
        NeuroDSL.patch_node!(g, site, clean_cache; namespace=ns)
        Array(NeuroDSL.demand!(g, out_sym; namespace=ns)); sync()
        NeuroDSL.restore_from_cache!(g, ns, corrupted_cache, affected)
        Array(NeuroDSL.demand!(g, out_sym; namespace=ns)); sync()
    end
    times = Float64[]
    for _ in 1:reps
        t0 = time_ns()
        NeuroDSL.patch_node!(g, site, clean_cache; namespace=ns)
        Array(NeuroDSL.demand!(g, out_sym; namespace=ns))
        sync()
        push!(times, (time_ns() - t0) / 1e6)
        NeuroDSL.restore_from_cache!(g, ns, corrupted_cache, affected)
        Array(NeuroDSL.demand!(g, out_sym; namespace=ns)); sync()
    end
    s = sort(times); n = length(s)
    return (; median=s[n÷2+1], q25=s[max(1,n÷4)], q75=s[min(n,3n÷4)])
end

# Référence : forward complet propre->corrompu (coût "TransformerLens naïf")
function timed_full_forward!(g, ns, out_sym, clean_input, corrupted_input; reps=20, warmup=3)
    for _ in 1:warmup
        NeuroDSL.set!(g, :input, corrupted_input; namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        Array(NeuroDSL.demand!(g, out_sym; namespace=ns)); sync()
    end
    times = Float64[]
    for _ in 1:reps
        t0 = time_ns()
        NeuroDSL.set!(g, :input, corrupted_input; namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        Array(NeuroDSL.demand!(g, out_sym; namespace=ns))
        sync()
        push!(times, (time_ns() - t0) / 1e6)
    end
    s = sort(times); n = length(s)
    return (; median=s[n÷2+1], q25=s[max(1,n÷4)], q75=s[min(n,3n÷4)])
end

full_fwd = timed_full_forward!(g, ns, out, clean_input, corrupted_input)
@printf("Forward complet (référence) : médiane=%.4f ms [q25=%.4f q75=%.4f]\n",
        full_fwd.median, full_fwd.q25, full_fwd.q75)

# Repartir d'un état corrompu cohérent avant le balayage des sites
NeuroDSL.set!(g, :input, corrupted_input; namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, out; namespace=ns)

rows = NamedTuple[]
for site in sites
    cone = NeuroDSL._downstream_nodes(g, site, ns)
    t = timed_site!(g, ns, site, out, clean_cache, corrupted_cache)
    push!(rows, (; site=String(site), cone_size=length(cone),
                  med_ms=t.median, q25=t.q25, q75=t.q75))
    @printf("  %-32s cône=%-4d médiane=%.4f ms [q25=%.4f q75=%.4f]\n",
            String(site), length(cone), t.median, t.q25, t.q75)
end

results = Dict(
    "full_forward_ms" => Dict("median"=>full_fwd.median, "q25"=>full_fwd.q25, "q75"=>full_fwd.q75),
    "sites" => rows,
)
open(joinpath(export_dir, "neurodsl_bench_results.json"), "w") do io
    JSON.print(io, results)
end
println("\nRésultats NeuroDSL écrits -> ", joinpath(export_dir, "neurodsl_bench_results.json"))
