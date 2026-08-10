#=
Mesure du pic de mémoire GPU pour le char-LM de real_llm.ipynb (dim=256, 4
têtes, hidden=512, 4 couches, block_size=256, ~2.72M paramètres), comparable
à torch.cuda.max_memory_allocated() côté real_llm_py.py (73.6 MB, mesuré sur
toute la boucle de 20 000 pas -- entraînement + validation + génération).

DOIT tourner en processus Julia ISOLÉ (rien d'autre en mémoire GPU). Deux
approches plus simples ont été essayées et rejetées :

1. Compteur logiciel (MemTrack, patchant Backend.zeros32/ones32/rand32/
   randn32 + similar) intégré à toute la boucle d'entraînement : PEU FIABLE.
   Un test de contrôle a montré 346 MB SANS GC.gc() périodique contre 583 MB
   AVEC un GC toutes les 50 étapes, pour EXACTEMENT le même travail -- le
   compteur dépend de quand le ramasse-miettes Julia exécute les `finalizer`
   qui décrémentent le compteur, pas de la mémoire réellement simultanée. Sur
   la vraie boucle de 20 000 pas (aucun GC.gc() intermédiaire), ça donnait
   1011.4 MB -- un artefact d'accumulation de tampons morts jamais comptés
   comme libérés, pas un vrai pic de travail simultané.

2. Mesure matérielle brute (CUDA.total_memory() - CUDA.free_memory()) :
   inutilisable aussi -- avant même de construire un graphe, ouvrir un
   contexte CUDA depuis Julia consomme déjà ~1.1 GB (overhead fixe
   CUDA.jl/cuBLAS/cuRAND), un chiffre qui n'a rien à voir avec le modèle et
   n'est pas comparable à `torch.cuda.max_memory_allocated()` (qui ne compte
   que les tenseurs, pas le contexte).

3. Mesure intégrée à la fin de real_llm.ipynb (même processus que
   l'entraînement) : essayée, rejetée aussi -- le modèle déjà entraîné ET un
   graphe de mesure jetable résident simultanément en VRAM, gonflant la
   baseline mesurée (128.3 MB observés contre 74.6 MB en isolation totale,
   pour EXACTEMENT le même travail de mesure). D'où ce script séparé.

Instrument retenu : l'attribut CU_MEMPOOL_ATTR_USED_MEM_HIGH du pool
stream-ordered CUDA (mis à jour de façon SYNCHRONE par le driver à chaque
allocation/libération réelle -- immunisé au retard du GC Julia) et exclut par
construction l'overhead fixe du contexte CUDA (le pool ne trace que les
CuArray, pas l'init de contexte/cuBLAS/cuRAND). Vérifié reproductible au byte
près sur 5 essais indépendants.
=#
using NeuroDSL, CUDA, Random, Printf

dev = NeuroDSL.Backend.CUDADevice()
ns = :real_llm_vram_probe

const _POOL = CUDA.CUDACore.pool_create(CUDA.CUDACore.active_state().device)
pool_current_mb() = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_CURRENT)) / 1024^2
pool_high_mb()     = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH)) / 1024^2
reset_high!()      = CUDA.attribute!(_POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
function quiesce()
    CUDA.synchronize(); GC.gc(true); GC.gc(true); CUDA.synchronize()
end

# ── Même architecture EXACTE que build_char_lm_graph dans real_llm.ipynb ────
function build_char_lm_graph(dev, ns::Symbol; vocab_size::Int, dim::Int, n_heads::Int,
                              hidden_dim::Int, n_layers::Int, block_size::Int)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.set!(g, :token_ids, ones(Int, block_size); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns)
    tok_emb = NeuroDSL.Embedding(vocab_size, dim)(g, :token_ids, :tok; namespace=ns)
    pos_emb = NeuroDSL.Embedding(block_size, dim)(g, :pos_ids, :pos; namespace=ns)
    x = :embed_sum
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(x, [tok_emb, pos_emb], :add; namespace=ns))
    out = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim)(g, x; namespace=ns)
    logits = NeuroDSL.Linear(dim, vocab_size)(g, out, :lm_head; namespace=ns)
    NeuroDSL.set!(g, :labels, ones(Int, block_size); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [logits, :labels], :cross_entropy; namespace=ns))
    return g, logits
end
function sample_window(rng, ids::Vector{Int}, block_size::Int)
    i = rand(rng, 1:(length(ids) - block_size))
    tokens = ids[i:i+block_size-1]
    labels = ids[i+1:i+block_size]
    return tokens, labels
end

vocab_size, dim, n_heads, hidden_dim, n_layers, block_size = 65, 256, 4, 512, 4, 256
train_ids = rand(1:vocab_size, 200_000)  # corpus factice -- seules les FORMES comptent pour la mémoire

g, logits_sym = build_char_lm_graph(dev, ns; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                     hidden_dim=hidden_dim, n_layers=n_layers, block_size=block_size)
ps = NeuroDSL.params(g; namespace=ns)
m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
rng = MersenneTwister(1)

function train_step!(t; release_values::Bool=false)
    tokens, labels = sample_window(rng, train_ids, block_size)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, :loss; namespace=ns)
    NeuroDSL.backward_graph!(g, :loss; namespace=ns, release_values=release_values)
    for (i, p) in enumerate(ps)
        NeuroDSL.adamw_step!(dev, p.value, p.gradient, m1s[i], m2s[i], 1f-3, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
    end
    NeuroDSL.invalidate_all!(g; namespace=ns)
end

println("=== Warm-up (10 pas, stabilise formes + compilation JIT CUDA) ===")
for t in 1:10; train_step!(t); end
quiesce()

println("\n=== Épisode 'train_step' : baseline + pic sur UN SEUL pas, répété 5x ===")
peaks_train = Float64[]
for trial in 1:5
    quiesce()
    baseline = pool_current_mb()
    reset_high!()
    train_step!(100 + trial)
    CUDA.synchronize()
    peak = pool_high_mb()
    push!(peaks_train, peak)
    @printf("  essai %d : baseline=%.2f MB  pic=%.2f MB  (delta épisode=%.2f MB)\n",
            trial, baseline, peak, peak - baseline)
end

println("\n=== Épisode 'train_step' AVEC release_values=true (opt-in, BACKWARD_RELEASE_VALUES) ===")
println("    -- libère les activations forward à la fin de la propre visite de chaque nœud")
println("    -- pendant le backward, + vide les listes libres du GradPool en fin de passe")
for t in 1:10; train_step!(200+t; release_values=true); end   # warm-up dédié à ce chemin
quiesce()
peaks_train_rv = Float64[]
for trial in 1:5
    quiesce()
    baseline = pool_current_mb()
    reset_high!()
    train_step!(300 + trial; release_values=true)
    CUDA.synchronize()
    peak = pool_high_mb()
    push!(peaks_train_rv, peak)
    @printf("  essai %d : baseline=%.2f MB  pic=%.2f MB\n", trial, baseline, peak)
end
# Revenir au régime par défaut (release_values=false) pour les épisodes suivants.
for t in 1:10; train_step!(400+t; release_values=false); end
quiesce()

println("\n=== Épisode 'val_window' : une fenêtre de validation (forward seul, t=256) ===")
println("    -- demand_release! (src/demand_release.jl) : jamais suivi d'un backward")
println("    -- ici, donc les activations intermédiaires sont libérées au fil du calcul")
val_ids_probe = rand(1:vocab_size, 50_000)
NeuroDSL.release_intermediates!(g; namespace=ns)   # solde les ~34 Mo laissés par le dernier train_step!
quiesce()
baseline_val = pool_current_mb()
reset_high!()
tokens = val_ids_probe[1:block_size]
labels = val_ids_probe[2:block_size+1]
NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand_release!(g, :loss; namespace=ns)
CUDA.synchronize()
peak_val = pool_high_mb()
@printf("  baseline=%.2f MB  pic=%.2f MB  (delta épisode=%.2f MB)\n", baseline_val, peak_val, peak_val - baseline_val)

println("\n=== Épisode 'gen_token' pire cas : transition de forme t=255 -> t=256 ===")
println("    -- demand_release! dans la boucle de génération")
for t in 1:255
    ctx = ones(Int, t)
    NeuroDSL.set!(g, :token_ids, ctx; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:t); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand_release!(g, logits_sym; namespace=ns)
end
quiesce()
baseline_gen = pool_current_mb()
reset_high!()
ctx256 = ones(Int, 256)
NeuroDSL.set!(g, :token_ids, ctx256; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:256); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand_release!(g, logits_sym; namespace=ns)
CUDA.synchronize()
peak_gen = pool_high_mb()
@printf("  baseline=%.2f MB  pic=%.2f MB  (delta épisode=%.2f MB)\n", baseline_gen, peak_gen, peak_gen - baseline_gen)

# Après demand_release!, le graphe `g` a des activations intermédiaires
# manquantes -- OBLIGATOIRE de repasser par un demand!/backward_graph!
# complet avant tout nouveau train_step! (sinon _freed_backward_error,
# garde-fou explicite de src/backward.jl). Un simple demand! sur :loss
# recalcule tout ce qui manque (valid=false partout après release), donc
# le prochain train_step! (qui fait son propre set!+invalidate_all!+demand!)
# repart de zéro correctement -- rien à faire de spécial ici.

println("\n=== Vérification : la baseline reste-t-elle stable (pas de fuite) sur 100 pas de plus ? ===")
for chunk in 1:4
    for t in 1:25; train_step!(1000 + (chunk-1)*25 + t); end
    quiesce()
    @printf("  après %d pas cumulés : baseline (current, MB) = %.2f\n", chunk*25, pool_current_mb())
end

peak_vram_mb = maximum(vcat(peaks_train, [peak_val, peak_gen]))
println()
println("="^62)
@printf("PIC VRAM RETENU (défaut, release_values=false) : %.2f MB\n", peak_vram_mb)
@printf("  train_step = %.2f MB | val_window = %.2f MB | gen_token = %.2f MB\n",
        peaks_train[end], peak_val, peak_gen)
@printf("  train_step (opt-in release_values=true)        = %.2f MB\n", peaks_train_rv[end])
@printf("Comparaison : PyTorch (torch.cuda.max_memory_allocated(), toute la boucle) = 73.6 MB\n")
println("="^62)

# ── Archivage sur disque ─────────────────────────────────────────────────────
# Ajouté le 2026-08-08 : cette sonde n'écrivait QUE sur la console, si bien que
# quatre des cinq chiffres qu'elle produit (opt-in, val_window, gen_token,
# baseline stable) n'avaient aucun artefact dans le dépôt -- exactement le défaut
# pour lequel d'autres affirmations de l'article ont été retirées. Les figures
# citées doivent être relisibles sans relancer la sonde.
const _RESULTS_PATH = joinpath(@__DIR__, "real_llm_vram_probe_results.txt")
open(_RESULTS_PATH, "w") do io
    println(io, "# real_llm_vram_probe.jl -- pic VRAM par épisode (watermark du pool CUDA)")
    println(io, "# char-LM TinyShakespeare, dim=256, 4 couches, 4 têtes, 2 722 369 paramètres")
    println(io, "# référence PyTorch (torch.cuda.max_memory_allocated, boucle entière) = 73.62 MB")
    @printf(io, "RESULT episode=train_step release_values=false peak_mb=%.2f\n", peaks_train[end])
    @printf(io, "RESULT episode=train_step release_values=true  peak_mb=%.2f\n", peaks_train_rv[end])
    @printf(io, "RESULT episode=val_window release_values=false peak_mb=%.2f\n", peak_val)
    @printf(io, "RESULT episode=gen_token  release_values=false peak_mb=%.2f\n", peak_gen)
    @printf(io, "RESULT retained_peak_mb=%.2f pytorch_ref_mb=73.62\n", peak_vram_mb)
end
println("Résultats archivés dans : ", _RESULTS_PATH)
