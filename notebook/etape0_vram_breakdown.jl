#=
Étape 0 (piste "interprétabilité sous budget mémoire", conçue avec Fable le
2026-07-12) -- décomposition du pic VRAM pendant un balayage de patching
d'activation, AVANT d'écrire la moindre ligne d'intégration
patching.jl/demand_release.jl. Hypothèse à vérifier : `capture_activations`
(src/patching.jl) copie la valeur de CHAQUE nœud, y compris les paramètres
(poids) -- donc `clean_cache` ET `corrupted_cache` dupliquent chacun TOUT le
modèle, pas seulement les activations intermédiaires. Si confirmé, c'est le
poste de gaspillage dominant, pas les activations elles-mêmes.

Méthodologie identique à `real_llm_vram_probe.jl` (déjà validée cette
session) : watermark CU_MEMPOOL_ATTR_USED_MEM_HIGH du pool CUDA
stream-ordered, synchrone, immunisé au délai du GC Julia. Processus Julia
ISOLÉ obligatoire (rien d'autre en mémoire GPU).

Même config que le jalon 0 (12 couches, dim=768, 12 têtes, hidden=3072,
seq_len=20) pour rester directement comparable.
=#
using NeuroDSL, CUDA, Random, Printf

dev = NeuroDSL.Backend.CUDADevice()
ns = :etape0_vram

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

n_nodes = length(g.nodes[ns])
n_params = length(NeuroDSL.params(g; namespace=ns))
println("Graphe : $n_nodes nœuds au total, dont $n_params tenseurs paramètres.")

println("\n=== Warm-up (3 forwards, compilation JIT CUDA) ===")
for _ in 1:3
    NeuroDSL.invalidate_all!(g; namespace=ns)
    Array(NeuroDSL.demand!(g, out; namespace=ns))
end
quiesce()

# ── 1. Baseline : poids seuls résidents ──────────────────────────────────────
baseline_weights = pool_current_mb()
@printf("\n[1] Poids seuls résidents (après warm-up, avant tout cache) : %.2f MB\n", baseline_weights)

# ── 2. Un seul forward complet (poids + un jeu d'activations) ───────────────
reset_high!()
NeuroDSL.invalidate_all!(g; namespace=ns)
Array(NeuroDSL.demand!(g, out; namespace=ns))
CUDA.synchronize()
peak_one_forward = pool_high_mb()
current_after_forward = pool_current_mb()
@printf("[2] Après UN forward complet : pic=%.2f MB | résident après=%.2f MB (delta activations=%.2f MB)\n",
        peak_one_forward, current_after_forward, current_after_forward - baseline_weights)

# ── 3. Capture de clean_cache (capture_activations -- copie TOUT nœud, poids inclus ?) ──
reset_high!()
clean_cache = NeuroDSL.capture_activations(g, ns)
CUDA.synchronize()
peak_clean_capture = pool_high_mb()
current_after_clean = pool_current_mb()
@printf("[3] Après capture_activations (clean_cache) : pic=%.2f MB | résident après=%.2f MB (delta cache=%.2f MB)\n",
        peak_clean_capture, current_after_clean, current_after_clean - current_after_forward)

n_param_keys_in_cache = count(k -> haskey(g.nodes[ns], k) && g.nodes[ns][k].is_param, keys(clean_cache))
@printf("    -> %d des %d clés de clean_cache correspondent à des nœuds paramètres (poids).\n",
        n_param_keys_in_cache, length(clean_cache))

# ── 4. Corruption + capture de corrupted_cache (SECONDE copie complète) ─────
corrupted_input = copy(Array(NeuroDSL.node(g, :input; namespace=ns).value))
corrupted_input[3, :] .= Array(NeuroDSL.Backend.rand32(dev, dim) .- 0.5f0)
NeuroDSL.set!(g, :input, corrupted_input; namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
Array(NeuroDSL.demand!(g, out; namespace=ns))

reset_high!()
corrupted_cache = NeuroDSL.capture_activations(g, ns)
CUDA.synchronize()
peak_corrupted_capture = pool_high_mb()
current_after_both = pool_current_mb()
@printf("[4] Après capture_activations (corrupted_cache, EN PLUS de clean_cache) : pic=%.2f MB | résident après=%.2f MB\n",
        peak_corrupted_capture, current_after_both)
@printf("    -> delta pour ce 2e cache : %.2f MB (attendu ≈ même taille que le premier si symétrique)\n",
        current_after_both - current_after_clean)

# ── 5. Pic pendant un vrai balayage (échantillon de 10 sites, résidence actuelle par défaut) ──
layer_prefixes = ["layer_$i" for i in 1:n_layers]
sample_sites = [Symbol(lp, "_mha_ao_h", h) for lp in layer_prefixes[1:3] for h in 1:n_heads][1:10]

reset_high!()
for site in sample_sites
    affected = NeuroDSL._downstream_nodes(g, site, ns)
    NeuroDSL.patch_node!(g, site, clean_cache; namespace=ns)
    Array(NeuroDSL.demand!(g, out; namespace=ns))
    NeuroDSL.restore_from_cache!(g, ns, corrupted_cache, affected)
    Array(NeuroDSL.demand!(g, out; namespace=ns))
end
CUDA.synchronize()
peak_sweep = pool_high_mb()
@printf("\n[5] Pic pendant un balayage de 10 sites (résidence par défaut, les 2 caches restent en mémoire) : %.2f MB\n",
        peak_sweep)

# ── Résumé ────────────────────────────────────────────────────────────────────
println("\n", "="^70)
println("DÉCOMPOSITION DU PIC VRAM -- Étape 0")
println("="^70)
@printf("Poids seuls                                   : %8.2f MB\n", baseline_weights)
@printf("+ un jeu d'activations (1 forward)             : %8.2f MB (delta %.2f)\n",
        current_after_forward, current_after_forward - baseline_weights)
@printf("+ clean_cache (capture_activations complète)   : %8.2f MB (delta %.2f)\n",
        current_after_clean, current_after_clean - current_after_forward)
@printf("+ corrupted_cache (2e capture complète)        : %8.2f MB (delta %.2f)\n",
        current_after_both, current_after_both - current_after_clean)
@printf("Pic pendant le balayage (10 sites)             : %8.2f MB\n", peak_sweep)
println()
frac_caches = (current_after_both - current_after_forward) / current_after_both * 100
@printf("Part des DEUX caches dans la mémoire résidente totale : %.1f%%\n", frac_caches)
if n_param_keys_in_cache > 0
    weight_bytes_per_cache = sum(length(clean_cache[k]) for k in keys(clean_cache)
                                   if haskey(g.nodes[ns],k) && g.nodes[ns][k].is_param) * 4 / 1024^2
    @printf("Estimation : %.2f MB par cache sont des POIDS dupliqués inutilement (%.1f%% du delta d'un cache)\n",
            weight_bytes_per_cache, weight_bytes_per_cache / (current_after_clean - current_after_forward) * 100)
    println("\n⚠️  HYPOTHÈSE CONFIRMÉE : capture_activations copie aussi les poids -- chaque cache duplique")
    println("    inutilement tout le modèle. Exclure les nœuds is_param de la capture est un gain")
    println("    immédiat, sans changement de comportement fonctionnel.")
else
    println("\nHypothèse infirmée : aucune clé paramètre trouvée dans le cache (à ré-examiner).")
end
