#=
Étape 0c -- confirmation finale via l'API réelle (sweep_patch_sites!(...;
gc_every=...)) plutôt que la boucle manuelle d'étape 0b, sur les 156 sites
complets à l'échelle GPT-2 small (jalon 0).
=#
using NeuroDSL, CUDA, Random, Printf

dev = NeuroDSL.Backend.CUDADevice()
ns = :etape0c_vram

const _POOL = CUDA.CUDACore.pool_create(CUDA.CUDACore.active_state().device)
pool_high_mb()     = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH)) / 1024^2
reset_high!()      = CUDA.attribute!(_POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
quiesce() = (CUDA.synchronize(); GC.gc(true); GC.gc(true); CUDA.synchronize())

dim, n_heads, hidden_dim, n_layers, seq_len = 768, 12, 3072, 12, 20
Random.seed!(1)
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :input, (NeuroDSL.Backend.rand32(dev, seq_len, dim) .- 0.5f0); namespace=ns)
out = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim; batched_attn=true)(g, :input; namespace=ns)
for _ in 1:3
    NeuroDSL.invalidate_all!(g; namespace=ns)
    Array(NeuroDSL.demand!(g, out; namespace=ns))
end
quiesce()

clean_output = copy(Array(NeuroDSL.demand!(g, out; namespace=ns)))
clean_cache = NeuroDSL.capture_activations(g, ns)
corrupted_input = copy(Array(NeuroDSL.node(g, :input; namespace=ns).value))
corrupted_input[3, :] .= Array(NeuroDSL.Backend.rand32(dev, dim) .- 0.5f0)
NeuroDSL.set!(g, :input, corrupted_input; namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
corrupted_output = copy(Array(NeuroDSL.demand!(g, out; namespace=ns)))
corrupted_cache = NeuroDSL.capture_activations(g, ns)

layer_prefixes = ["layer_$i" for i in 1:n_layers]
all_sites = vcat(
    [Symbol(lp, "_mha_ao_h", h) for lp in layer_prefixes for h in 1:n_heads],
    [Symbol(lp, "_mlp_out") for lp in layer_prefixes],
)

reset_high!()
NeuroDSL.sweep_patch_sites!(g, out, all_sites[1:3], clean_cache, corrupted_cache, clean_output, corrupted_output; namespace=ns, gc_every=1)
quiesce()  # warm-up dédié à sweep_patch_sites!+GC.gc(true)

reset_high!()
res_off = NeuroDSL.sweep_patch_sites!(g, out, all_sites, clean_cache, corrupted_cache, clean_output, corrupted_output; namespace=ns)
peak_off = pool_high_mb()
quiesce()

reset_high!()
res_on = NeuroDSL.sweep_patch_sites!(g, out, all_sites, clean_cache, corrupted_cache, clean_output, corrupted_output; namespace=ns, gc_every=5)
peak_on = pool_high_mb()
quiesce()

@printf("Pic 156 sites, gc_every=0 (défaut) : %8.2f MB\n", peak_off)
@printf("Pic 156 sites, gc_every=5           : %8.2f MB\n", peak_on)
@printf("Réduction : %.1f%%\n", (1 - peak_on/peak_off) * 100)

max_recov_diff = maximum(abs(a.recovery - b.recovery) for (a, b) in zip(res_off, res_on))
@printf("Écart max de recovery (off vs on)   : %.2e (doit être ~0)\n", max_recov_diff)
