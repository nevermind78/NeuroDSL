# ══════════════════════════════════════════════════════════════════════════════
# Jalon 0b -- réplique jalon 0 (2026-07-12) à l'échelle GPT-2-medium (24 couches,
# 16 têtes, dim=1024, hidden=4096), pour tester si le ratio agrégé progresse
# vers/au-delà du seuil de 3x avec l'échelle, comme la tendance déjà observée
# le suggère (5 couches/dim=256 : 1.54x -> 12 couches/dim=768 : 2.12x).
#
# Rendu praticable par les deux correctifs VRAM du même jour
# (capture_activations exclut désormais les nœuds paramètres ; GC.gc(true)
# périodique évite la croissance du résidu avec la longueur du balayage) --
# sans eux, les caches à cette échelle auraient dupliqué ~1.6 Go de poids
# chacun, en plus d'un risque de croissance non bornée sur 156 sites.
#
# Même protocole, mêmes seuils, mêmes 156 sites (144 têtes + 12 MLP) que
# jalon 0 -- seule l'échelle change, pour rester directement comparable.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics, Printf, LinearAlgebra, JSON, CUDA

dev = NeuroDSL.Backend.CUDADevice()
dim, n_heads, hidden_dim, n_layers, seq_len = 1024, 16, 4096, 24, 20

# Sonde VRAM rapide avant de s'engager (même discipline que les scans GPU précédents)
const _POOL = CUDA.CUDACore.pool_create(CUDA.CUDACore.active_state().device)
pool_current_mb() = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_CURRENT)) / 1024^2
pool_high_mb()     = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH)) / 1024^2
reset_high!()      = CUDA.attribute!(_POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))

Random.seed!(1)
ns = :jalon0b
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :input, (NeuroDSL.Backend.rand32(dev, seq_len, dim) .- 0.5f0); namespace=ns)
out = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim; batched_attn=true)(g, :input; namespace=ns)

println("Graphe construit : dim=$dim, n_heads=$n_heads, hidden_dim=$hidden_dim, n_layers=$n_layers, seq_len=$seq_len")
println("Nombre total de nœuds : ", length(g.nodes[ns]))
n_params = length(NeuroDSL.params(g; namespace=ns))
println("Nombre de tenseurs paramètres : ", n_params)

function sync()
    NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.synchronize()
end

# ═══ Sites : 384 têtes + 24 MLP = 408 (comptage réel à cette échelle) ═══
layer_prefixes = ["layer_$i" for i in 1:n_layers]
head_sites = [Symbol(lp, "_mha_ao_h", h) for lp in layer_prefixes for h in 1:n_heads]
mlp_sites  = [Symbol(lp, "_mlp_out") for lp in layer_prefixes]
sites = vcat(head_sites, mlp_sites)
for s in sites
    @assert haskey(g.nodes[ns], s) "site manquant : $s"
end
println("$(length(sites)) sites confirmés ($(length(head_sites)) têtes + $(length(mlp_sites)) MLP).")

NeuroDSL.demand!(g, out; namespace=ns)  # premier calcul (compilation JIT), valeur jetée
GC.gc(true); sync()
weights_only_mb = pool_current_mb()
@printf("VRAM après warm-up (poids + 1 forward) : %.2f MB\n", weights_only_mb)

clean_input = copy(Array(NeuroDSL.node(g, :input; namespace=ns).value))
corrupted_input = copy(clean_input)
corrupt_row = 3
corrupted_input[corrupt_row, :] .= Array(NeuroDSL.Backend.rand32(dev, dim) .- 0.5f0)

NeuroDSL.set!(g, :input, clean_input; namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, out; namespace=ns)
clean_cache = NeuroDSL.capture_activations(g, ns)

NeuroDSL.set!(g, :input, corrupted_input; namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, out; namespace=ns)
corrupted_cache = NeuroDSL.capture_activations(g, ns)

GC.gc(true); sync()
@printf("VRAM après les 2 caches (correctif appliqué) : %.2f MB\n", pool_current_mb())
n_param_keys = count(k -> haskey(g.nodes[ns], k) && g.nodes[ns][k].is_param, keys(clean_cache))
@printf("Clés paramètres dans clean_cache : %d/%d (attendu 0)\n", n_param_keys, length(clean_cache))

# ═══ Coût d'un forward complet (référence "hooks naïfs" -- N x un forward) ═══
function timed_full_forward!(g, ns, out_sym, clean_input, corrupted_input; reps=15, warmup=3)
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
@printf("\nForward complet (référence, style hooks naïfs) : médiane=%.4f ms [q25=%.4f q75=%.4f]\n",
        full_fwd.median, full_fwd.q25, full_fwd.q75)

NeuroDSL.set!(g, :input, corrupted_input; namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, out; namespace=ns)

# ═══ Balayage complet des sites (côté NeuroDSL uniquement) ═══
function timed_site!(g, ns, site, out_sym, clean_cache, corrupted_cache; reps=15, warmup=3)
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
    return (; median=s[n÷2+1], cone_size=length(affected))
end

println("\nBalayage des $(length(sites)) sites en cours...")
reset_high!()
total_nd_ms = 0.0
rows = NamedTuple[]
t_start = time()
for (i, site) in enumerate(sites)
    r = timed_site!(g, ns, site, out, clean_cache, corrupted_cache)
    global total_nd_ms += r.median
    push!(rows, (; site=String(site), cone_size=r.cone_size, med_ms=r.median))
    if i % 5 == 0
        GC.gc(true)  # correctif du même jour -- évite la croissance du résidu VRAM avec la longueur du balayage
    end
    if i % 40 == 0
        @printf("  [%d/%d] %-32s cône=%-5d médiane=%.4f ms  (cumulé=%.1f ms, VRAM pic jusqu'ici=%.0f MB)\n",
                i, length(sites), String(site), r.cone_size, r.median, total_nd_ms, pool_high_mb())
    end
end
elapsed_total = time() - t_start
peak_vram = pool_high_mb()
@printf("\nBalayage terminé en %.1f s (mesure). Pic VRAM pendant le balayage : %.2f MB\n", elapsed_total, peak_vram)

# ═══ Le calcul décisif ═══
naive_total = length(sites) * full_fwd.median
ratio = naive_total / total_nd_ms

println("\n", "="^70)
println("RÉSULTAT DÉCISIF -- Jalon 0b (échelle GPT-2-medium)")
println("="^70)
@printf("Coût \"hooks naïfs\" (%d × un forward complet)      : %.2f ms\n", length(sites), naive_total)
@printf("Coût NeuroDSL (somme du balayage des %d sites)     : %.2f ms\n", length(sites), total_nd_ms)
@printf("Ratio agrégé (équivalents-full-forward économisés)  : %.2fx\n", ratio)
println()
if ratio >= 3.0
    println("✅ SEUIL FRANCHI (>= 3x) à cette échelle -- relance légitimement la question du portage GPT-2 réel (jalon 1).")
else
    println("⚠️  SEUIL TOUJOURS NON FRANCHI (< 3x) à cette échelle plus grande.")
end

open(joinpath(@__DIR__, "jalon0b_results.json"), "w") do io
    JSON.print(io, Dict(
        "full_forward_ms" => full_fwd.median,
        "total_naive_ms" => naive_total,
        "total_neurodsl_ms" => total_nd_ms,
        "ratio" => ratio,
        "n_sites" => length(sites),
        "peak_vram_mb" => peak_vram,
        "rows" => rows,
        "config" => Dict("dim"=>dim,"n_heads"=>n_heads,"hidden_dim"=>hidden_dim,"n_layers"=>n_layers,"seq_len"=>seq_len),
    ))
end
println("\nRésultats écrits -> notebook/jalon0b_results.json")
