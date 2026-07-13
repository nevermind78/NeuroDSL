#=
Étape 0b (suite directe d'Étape 0, après le correctif capture_activations du
2026-07-12) -- le pic pendant un balayage de 10 sites est tombé à 639.94 MB,
dont poids=448.96 MB + 2 caches=42.24 MB = 491.20 MB de résidence "fixe".
Reliquat observé : 639.94 - 491.20 ≈ 148.74 MB de mémoire "de travail" pendant
le balayage lui-même.

Question posée ici, AVANT d'envisager d'intégrer demand_release!/GradPool
dans la boucle de patching : ce reliquat est-il un coût FIXE (quelques
buffers transitoires réutilisés/récupérés par le pool CUDA, indépendant du
nombre de sites balayés), ou un coût qui CROÎT avec le nombre de sites
balayés (signe d'une vraie accumulation -- donc une cible réelle pour
demand_release!) ?

Méthode : mesurer le pic pour deux tailles de balayage très différentes
(10 sites vs 50 sites, sur les mêmes 156 sites que jalon 0) avec le MÊME
graphe/caches déjà construits. Si le pic ne bouge presque pas -> le
reliquat est fixe, la piste "budget mémoire" est essentiellement épuisée.
Si le pic croît avec le nombre de sites -> il reste une vraie cible.
=#
using NeuroDSL, CUDA, Random, Printf

dev = NeuroDSL.Backend.CUDADevice()
ns = :etape0b_vram

const _POOL = CUDA.CUDACore.pool_create(CUDA.CUDACore.active_state().device)
pool_current_mb() = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_CURRENT)) / 1024^2
pool_high_mb()     = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH)) / 1024^2
reset_high!()      = CUDA.attribute!(_POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
function quiesce()
    CUDA.synchronize(); GC.gc(true); GC.gc(true); CUDA.synchronize()
end

dim, n_heads, hidden_dim, n_layers, seq_len = 768, 12, 3072, 12, 20

Random.seed!(1)
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :input, (NeuroDSL.Backend.rand32(dev, seq_len, dim) .- 0.5f0); namespace=ns)
out = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim; batched_attn=true)(g, :input; namespace=ns)

println("\n=== Warm-up (3 forwards, compilation JIT CUDA) ===")
for _ in 1:3
    NeuroDSL.invalidate_all!(g; namespace=ns)
    Array(NeuroDSL.demand!(g, out; namespace=ns))
end
quiesce()

baseline_weights = pool_current_mb()
@printf("[1] Poids seuls résidents : %.2f MB\n", baseline_weights)

clean_cache = NeuroDSL.capture_activations(g, ns)
CUDA.synchronize()
after_clean = pool_current_mb()
@printf("[2] + clean_cache : %.2f MB (delta %.2f)\n", after_clean, after_clean - baseline_weights)

corrupted_input = copy(Array(NeuroDSL.node(g, :input; namespace=ns).value))
corrupted_input[3, :] .= Array(NeuroDSL.Backend.rand32(dev, dim) .- 0.5f0)
NeuroDSL.set!(g, :input, corrupted_input; namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
Array(NeuroDSL.demand!(g, out; namespace=ns))
corrupted_cache = NeuroDSL.capture_activations(g, ns)
CUDA.synchronize()
after_both = pool_current_mb()
@printf("[3] + corrupted_cache : %.2f MB (delta %.2f)\n", after_both, after_both - after_clean)
fixed_residency = after_both

layer_prefixes = ["layer_$i" for i in 1:n_layers]
all_sites = vcat(
    [Symbol(lp, "_mha_ao_h", h) for lp in layer_prefixes for h in 1:n_heads],
    [Symbol(lp, "_mlp_out") for lp in layer_prefixes],
)
@printf("\n%d sites disponibles au total.\n", length(all_sites))

function run_sweep!(sites; gc_every::Int=0, gc_full::Bool=false)
    reset_high!()
    for (i, site) in enumerate(sites)
        affected = NeuroDSL._downstream_nodes(g, site, ns)
        NeuroDSL.patch_node!(g, site, clean_cache; namespace=ns)
        Array(NeuroDSL.demand!(g, out; namespace=ns))
        NeuroDSL.restore_from_cache!(g, ns, corrupted_cache, affected)
        Array(NeuroDSL.demand!(g, out; namespace=ns))
        if gc_every > 0 && i % gc_every == 0
            GC.gc(gc_full)
        end
    end
    CUDA.synchronize()
    return pool_high_mb()
end

# Warm-up dédié au chemin patch_node!/demand!/restore_from_cache! (compilation JIT séparée)
run_sweep!(all_sites[1:3])
quiesce()

peak_10  = run_sweep!(all_sites[1:10])
quiesce()
peak_50  = run_sweep!(all_sites[1:50])
quiesce()
peak_156 = run_sweep!(all_sites)
quiesce()

println("\n", "="^70)
println("RÉSIDU DE TRAVAIL PENDANT LE BALAYAGE -- Étape 0b")
println("="^70)
@printf("Résidence fixe (poids + 2 caches)     : %8.2f MB\n", fixed_residency)
@printf("Pic balayage  10 sites                : %8.2f MB (résidu %.2f MB)\n", peak_10,  peak_10  - fixed_residency)
@printf("Pic balayage  50 sites                : %8.2f MB (résidu %.2f MB)\n", peak_50,  peak_50  - fixed_residency)
@printf("Pic balayage 156 sites                 : %8.2f MB (résidu %.2f MB)\n", peak_156, peak_156 - fixed_residency)

r10, r50, r156 = peak_10 - fixed_residency, peak_50 - fixed_residency, peak_156 - fixed_residency
println()
if abs(r156 - r10) < 0.1 * max(r10, 1.0)
    println("✅ Résidu STABLE quel que soit le nombre de sites balayés : coût FIXE (buffers transitoires")
    println("   récupérés par le pool CUDA entre sites), pas une accumulation.")
else
    println("⚠️  Résidu CROISSANT avec le nombre de sites balayés : accumulation détectée.")
end

# ── Test décisif : est-ce un artefact de timing du GC, ou un vrai besoin résident ? ──
println("\n", "="^70)
println("DIAGNOSTIC : artefact GC vs. vrai besoin résident")
println("="^70)
peak_156_gc1_minor  = run_sweep!(all_sites; gc_every=1, gc_full=false)
quiesce()
peak_156_gc1_full   = run_sweep!(all_sites; gc_every=1, gc_full=true)
quiesce()
peak_156_gc10_minor = run_sweep!(all_sites; gc_every=10, gc_full=false)
quiesce()
peak_156_gc10_full  = run_sweep!(all_sites; gc_every=10, gc_full=true)
quiesce()
@printf("Pic 156 sites, SANS GC périodique              : %8.2f MB (résidu %.2f MB)\n", peak_156, peak_156 - fixed_residency)
@printf("Pic 156 sites, GC mineur chaque site            : %8.2f MB (résidu %.2f MB)\n", peak_156_gc1_minor, peak_156_gc1_minor - fixed_residency)
@printf("Pic 156 sites, GC COMPLET chaque site            : %8.2f MB (résidu %.2f MB)\n", peak_156_gc1_full, peak_156_gc1_full - fixed_residency)
@printf("Pic 156 sites, GC mineur tous les 10 sites       : %8.2f MB (résidu %.2f MB)\n", peak_156_gc10_minor, peak_156_gc10_minor - fixed_residency)
@printf("Pic 156 sites, GC COMPLET tous les 10 sites      : %8.2f MB (résidu %.2f MB)\n", peak_156_gc10_full, peak_156_gc10_full - fixed_residency)

println()
best = min(peak_156_gc1_full, peak_156_gc10_minor, peak_156_gc10_full) - fixed_residency
if best < 0.15 * (peak_156 - fixed_residency)
    println("✅ ARTEFACT GC CONFIRMÉ : au moins une politique de GC périodique fait quasiment disparaître")
    println("   le résidu. Pas besoin d'intégrer demand_release!/keep= dans sweep_patch_sites! -- un appel")
    println("   GC.gc() périodique bien choisi dans la boucle suffit. Correctif trivial, risque quasi nul.")
else
    println("⚠️  Le résidu PERSISTE même avec GC.gc() complet fréquent : vrai besoin résident, pas un artefact.")
    println("   C'est une cible légitime pour l'intégration demand_release!/keep= dans la boucle de")
    println("   patching -- à creuser dans une prochaine étape.")
end
