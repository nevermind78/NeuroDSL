# ══════════════════════════════════════════════════════════════════════════════
# Jalon 0 (piste ACDC/AtP*, conçu avec Fable le 2026-07-12) — géométrie de coût
# PURE à l'échelle GPT-2 small (12 couches, 12 têtes, dim=768), poids ALÉATOIRES,
# AUCUN portage GPT-2 réel. La géométrie de coût d'un balayage ne dépend ni des
# poids ni de la non-linéarité exacte (GELU vs SwiGLU : même forme de graphe) --
# seule la taille/structure du graphe compte, déjà établi et réutilisé cette
# session (P1-bis, P4).
#
# Critère d'arrêt/continuation fixé par Fable AVANT toute mesure : ratio agrégé
# (156 × un forward complet) / (coût total du balayage des 156 sites) >= 3x
# pour justifier de porter GPT-2 réellement (jalon 1). En dessous, on pivote
# vers une revendication plus étroite (vérification des top-k d'AtP*) plutôt
# que d'abandonner.
#
# Sites : 144 sorties de tête (ao_h, 12 couches × 12 têtes) + 12 sorties MLP
# (mlp_out par couche) = 156, exactement le compte du jalon 1 réel -- pour que
# ce chiffre soit directement comparable, pas juste indicatif.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics, Printf, LinearAlgebra, JSON

dev = NeuroDSL.Backend.CUDADevice()
dim, n_heads, hidden_dim, n_layers, seq_len = 768, 12, 3072, 12, 20

Random.seed!(1)
ns = :jalon0
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

# ═══ Sites : 144 têtes + 12 MLP ═══
layer_prefixes = ["layer_$i" for i in 1:n_layers]
head_sites = [Symbol(lp, "_mha_ao_h", h) for lp in layer_prefixes for h in 1:n_heads]
mlp_sites  = [Symbol(lp, "_mlp_out") for lp in layer_prefixes]
sites = vcat(head_sites, mlp_sites)
@assert length(sites) == 156 "attendu 156 sites, obtenu $(length(sites))"
for s in sites
    @assert haskey(g.nodes[ns], s) "site manquant : $s"
end
println("156 sites confirmés (144 têtes + 12 MLP).")

# ═══ Corruption : un seul token remplacé (protocole standard de cette session) ═══
NeuroDSL.demand!(g, out; namespace=ns)  # premier calcul (compilation JIT), valeur jetée
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

# Repartir d'un état corrompu cohérent avant le balayage
NeuroDSL.set!(g, :input, corrupted_input; namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, out; namespace=ns)

# ═══ Balayage complet des 156 sites (côté NeuroDSL uniquement) ═══
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

println("\nBalayage des 156 sites en cours...")
total_nd_ms = 0.0
rows = NamedTuple[]
t_start = time()
for (i, site) in enumerate(sites)
    r = timed_site!(g, ns, site, out, clean_cache, corrupted_cache)
    global total_nd_ms += r.median
    push!(rows, (; site=String(site), cone_size=r.cone_size, med_ms=r.median))
    if i % 20 == 0
        @printf("  [%d/156] %-32s cône=%-5d médiane=%.4f ms  (cumulé=%.1f ms)\n",
                i, String(site), r.cone_size, r.median, total_nd_ms)
    end
end
elapsed_total = time() - t_start
@printf("\nBalayage terminé en %.1f s (mesure).\n", elapsed_total)

# ═══ Le calcul décisif ═══
naive_total = 156 * full_fwd.median
ratio = naive_total / total_nd_ms

println("\n", "="^70)
println("RÉSULTAT DÉCISIF -- Jalon 0")
println("="^70)
@printf("Coût \"hooks naïfs\" (156 × un forward complet)     : %.2f ms\n", naive_total)
@printf("Coût NeuroDSL (somme du balayage des 156 sites)    : %.2f ms\n", total_nd_ms)
@printf("Ratio agrégé (équivalents-full-forward économisés)  : %.2fx\n", ratio)
println()
if ratio >= 3.0
    println("✅ SEUIL FRANCHI (>= 3x) -- justifie de porter GPT-2 réellement (jalon 1).")
else
    println("⚠️  SEUIL NON FRANCHI (< 3x) -- pivoter vers une revendication plus étroite")
    println("    (vérification des top-k d'AtP*) plutôt que le portage complet de GPT-2.")
end

open(joinpath(@__DIR__, "jalon0_results.json"), "w") do io
    JSON.print(io, Dict(
        "full_forward_ms" => full_fwd.median,
        "total_naive_ms" => naive_total,
        "total_neurodsl_ms" => total_nd_ms,
        "ratio" => ratio,
        "n_sites" => length(sites),
        "rows" => rows,
        "config" => Dict("dim"=>dim,"n_heads"=>n_heads,"hidden_dim"=>hidden_dim,"n_layers"=>n_layers,"seq_len"=>seq_len),
    ))
end
println("\nRésultats écrits -> notebook/jalon0_results.json")
