#=
LE PLAN MÉMOIRE SE TRADUIT-IL EN OCTETS ? -- bras GPU, forward seul.

Suite de `bench_liveness_slots.jl` (201 slots -> 40) et `bench_planned_exec.jl`
(164 allocations -> 13, taux de réutilisation 92.1%, sortie bit-exacte). Ces deux
bancs comptent des slots et des allocations, PAS des octets. Ce banc-ci mesure la
VRAM réelle, avec le même instrument que partout ailleurs dans ce dépôt : le
watermark du pool CUDA (CU_MEMPOOL_ATTR_USED_MEM_HIGH), pic ABSOLU.

On ne convertit PAS un compte de slots en mégaoctets à la main : c'est
exactement l'erreur du vieux `PMLJULIA/notebook.ipynb` (cellule 83), qui
multipliait `n_slots` par une taille supposée, oubliait que les poids valent 16x
plus, et annonçait 42.9% d'économie pour un calcul qui n'avait rien exécuté.

MODÈLE : identique à `real_llm_vram_probe.jl` -- char-LM, vocab=65, dim=256,
4 têtes, hidden=512, 4 couches, block_size=256, ~2.72M paramètres -- pour que la
comparaison aux 51.44 MB (`val_window`) et 51.80 MB (`gen_token`) déjà mesurés
soit légitime. Seules les FORMES comptent pour la mémoire, le corpus est factice.

RÉGIME : forward seul. C'est le seul où le plan peut agir -- `for_backward=false`
coupe l'extension de durée de vie des activations, ce qui n'est correct que si
aucun backward ne suit. C'est aussi le régime où NeuroDSL bat déjà PyTorch
(0.70x), donc celui où un gain supplémentaire compte.

TROIS BRAS, un par exécution (`ARGS[1]`), processus neuf :
  demand      : `demand!` nu -- référence naïve, tout reste résident
  release     : `demand_release!` -- ce qui est livré aujourd'hui (51.44 MB)
  planned     : `demand_planned!` + plan `for_backward=false` -- le candidat

PORTE DE CORRECTION : chaque bras compare sa sortie à celle de `demand!` du même
processus, en égalité BIT-À-BIT. Un bras qui économise en changeant le résultat
est disqualifié, pas optimisé.

CONTRAINTE DÉCOUVERTE, à dire : `acquire!` arme `_POOLED_EXECUTION_SEEN[]`, ce
qui DÉSARME définitivement CTX_REBUILD pour tout le processus (src/backward.jl).
Or CTX_REBUILD valait 142 -> 125 MB sur `train_step`. Exécution planifiée et
backward d'entraînement efficace sont donc mutuellement exclusifs dans un même
processus -- une raison de plus de n'utiliser le plan qu'en forward seul.

USAGE : julia --project=. notebook/bench_planned_vram.jl [demand|release|planned]
=#
using NeuroDSL, CUDA, Printf

const ARM = length(ARGS) >= 1 ? ARGS[1] : "demand"
ARM in ("demand","release","planned") || error("bras inconnu : $ARM")

dev = NeuroDSL.Backend.CUDADevice()
const _POOL = CUDA.CUDACore.pool_create(CUDA.CUDACore.active_state().device)
pool_cur_mb()  = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_CURRENT))/1024^2
pool_high_mb() = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH))/1024^2
reset_high!()  = CUDA.attribute!(_POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
quiesce()      = (CUDA.synchronize(); GC.gc(true); GC.gc(true); CUDA.synchronize())

const VOCAB, DIM, HEADS, HIDDEN, NLAYERS, BLOCK = 65, 256, 4, 512, 4, 256

function build_char_lm(ns::Symbol)
    g = NeuroDSL.NeuroGraph(device=dev, namespace=ns)
    NeuroDSL.set!(g, :token_ids, ones(Int, BLOCK); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:BLOCK); atom_type=NeuroDSL.Datom, namespace=ns)
    tok = NeuroDSL.Embedding(VOCAB, DIM)(g, :token_ids, :tok; namespace=ns)
    pos = NeuroDSL.Embedding(BLOCK, DIM)(g, :pos_ids, :pos; namespace=ns)
    x = :x_emb
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(x, [tok, pos], :add; namespace=ns))
    out = NeuroDSL.LlamaModel(NLAYERS, DIM, HEADS, HIDDEN)(g, x; namespace=ns)
    logits = NeuroDSL.Linear(DIM, VOCAB)(g, out, :lm_head; namespace=ns)
    return g, logits
end

ns = Symbol(:pv_, ARM)
g, logits = build_char_lm(ns)

# ── référence de correction : demand! nu, dans CE processus ───────────────────
ref = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))
n_nodes = length(NeuroDSL.topo_order!(g; namespace=ns))

# plan construit une fois (hors mesure) pour le bras planifié
plan, pool = ARM == "planned" ?
    NeuroDSL.plan_memory!(g; namespace=ns, for_backward=false) : (nothing, nothing)

function fwd!()
    NeuroDSL.invalidate_all!(g; namespace=ns)
    if ARM == "demand"
        NeuroDSL.demand!(g, logits; namespace=ns)
    elseif ARM == "release"
        NeuroDSL.demand_release!(g, logits; namespace=ns)
    else
        NeuroDSL.demand_planned!(g, logits, plan, pool; namespace=ns)
    end
end

for _ in 1:5; fwd!(); end          # warm-up : formes stabilisées, JIT amorti
quiesce()

# ── mesure : pic ABSOLU, 3 essais, doit être stable ──────────────────────────
peaks = Float64[]; bases = Float64[]
for _ in 1:3
    quiesce(); b = pool_cur_mb(); reset_high!()
    fwd!(); CUDA.synchronize()
    push!(bases, b); push!(peaks, pool_high_mb())
end
stable = all(==(peaks[1]), peaks)

got = Array(NeuroDSL.demand!(g, logits; namespace=ns))
exact = size(got) == size(ref) && got == ref

nslots = plan === nothing ? -1 : plan.n_slots
hits   = pool === nothing ? -1 : NeuroDSL.pool_stats(pool).hits

line = @sprintf("RESULT arm=%s n_nodes=%d n_slots=%d pool_hits=%d baseline_mb=%.2f peak_mb=%.2f stable=%s bitexact=%s",
                ARM, n_nodes, nslots, hits, bases[1], peaks[1], stable, exact)
println(line)

# Archivage. Un bras par processus, donc on AJOUTE à la suite ; l'appelant vide
# le fichier avant la série. Sans ça ces chiffres ne vivraient que sur la
# console -- exactement le défaut pour lequel plusieurs affirmations de l'article
# ont été retirées le 2026-08-08.
open(joinpath(@__DIR__, "bench_planned_vram_results.txt"), "a") do io
    println(io, line)
end
