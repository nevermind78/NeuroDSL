#=
Benchmark du claim "moins de recalcul au backward grâce à `prune_frozen`" (article 1).

CONTEXTE : l'article annonçait "up to 86.4% less recomputation on a fine-tuning
workload when frozen parameters are contiguous with the graph's inputs (and no
gain otherwise)". Audit du 2026-08-08 : AUCUN artefact ne produisait ces chiffres
dans le dépôt (les seules occurrences de "86.4" étaient des coordonnées de tracés
SVG). Ce script est l'artefact manquant.

DEUX QUANTITÉS DISTINCTES, l'article les confondait ("less recomputation" vs
"speedups") :
  (1) TRAVAIL ÉLAGUÉ (structurel, déterministe, SANS chronomètre) : fraction des
      nœuds recevant un traitement backward (`backwarded=true`) avec
      `prune_frozen=true` vs sans. C'est le sens littéral de "less recomputation".
  (2) TEMPS D'HORLOGE : médiane du backward, bras alternés dans le processus.

PROTOCOLE -- corrigé le 2026-08-08 après qu'une première version ait produit un
résultat NON REPRODUCTIBLE sur le bras contrôle (+1.0% puis +22.8% entre deux
lancements). Deux causes identifiées et corrigées ici :
  (a) les deux bras partageaient un processus, le contrôle tournant toujours en
      second -- il héritait de l'état mémoire/pool laissé par le premier bras.
      => ce script ne fait plus QU'UN SEUL BRAS par exécution (ARGS[1]), pour une
         isolation complète de l'état.
  (b) un seul lancement par point. => le pilote (`bench_prune_frozen_driver.sh`)
      relance ce script N fois par bras et agrège min/médiane/p90 ENTRE
      lancements, jamais un pourcentage isolé.
Le pourcentage du bras contrôle est structurellement fragile : c'est un rapport
entre deux temps presque égaux, donc son erreur relative explose dès que la base
bouge un peu. L'agrégation entre lancements est là pour l'exposer, pas la cacher.

Chronométrage valide seulement horloge GPU VERROUILLÉE (`nvidia-smi -lgc
1402,1402`, terminal administrateur). Le volet structurel (1) n'en dépend pas.

SÉMANTIQUE DE "GELÉ", vérifiée dans `_compute_needs_bwd` (src/backward.jl:657) :
un nœud "a besoin" du backward s'il est `is_param=true` OU si une de ses entrées
en a besoin. Donc un poids GELÉ est une feuille `is_param=false`. Reproduit le
montage de `test/test_backward.jl` ("Préfixe séquentiel entièrement gelé").

USAGE : julia --project=. notebook/bench_prune_frozen.jl [favorable|controle]
=#
using NeuroDSL, CUDA, Statistics, Printf

const DIM     = 512
const ROWS    = 64
const NLAYERS = 8
const REPS    = 30
const WARMUP  = 10

const ARM = length(ARGS) >= 1 ? ARGS[1] : "favorable"
ARM in ("favorable", "controle") ||
    error("bras inconnu : $ARM (attendu 'favorable' ou 'controle')")

# FAVORABLE : W1..W7 gelés, seul W8 entraînable -> tout le préfixe est élagable
# CONTROLE  : W1 entraînable, W2..W8 gelés       -> rien n'est élagable en amont
const TRAINABLE = ARM == "favorable" ? Set(NLAYERS) : Set(1)

dev = NeuroDSL.Backend.CUDADevice()

"Chaîne h_i = relu(h_{i-1} * W_i) puis MSE. `trainable` = indices des W entraînables."
function build_chain(ns::Symbol, trainable::Set{Int})
    g = NeuroDSL.NeuroGraph(device=dev, namespace=ns)
    NeuroDSL.set!(g, :h0, NeuroDSL.Backend.randn32(dev, ROWS, DIM); namespace=ns)
    prev = :h0
    for i in 1:NLAYERS
        W = Symbol(:W, i)
        NeuroDSL.set!(g, W, randn(Float32, DIM, DIM) .* 0.02f0;
                      is_param = (i in trainable), namespace=ns)
        z = Symbol(:z, i); h = Symbol(:h, i)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(z, [prev, W], :matmul; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(h, [z], :relu; namespace=ns))
        prev = h
    end
    NeuroDSL.set!(g, :y, NeuroDSL.Backend.zeros32(dev, ROWS, DIM);
                  atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [prev, :y], :mse_loss; namespace=ns))
    return g
end

backwarded_count(g, ns) = count(nd -> nd.backwarded, values(g.nodes[ns]))

function one_pass!(g, ns; prune::Bool)
    ctx = NeuroDSL.CtxStore()
    NeuroDSL.demand!(g, :loss; ctx_store=ctx, namespace=ns)
    NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx, namespace=ns, prune_frozen=prune)
    CUDA.synchronize()
end

ns = Symbol(:bench_pf_, ARM)
g  = build_chain(ns, TRAINABLE)

# ── (1) travail élagué : structurel, déterministe ────────────────────────────
one_pass!(g, ns; prune=false); n_full   = backwarded_count(g, ns)
one_pass!(g, ns; prune=true);  n_pruned = backwarded_count(g, ns)

# ── correction avant vitesse : gradients identiques élagage on/off ───────────
tsym = Symbol(:W, maximum(TRAINABLE))
ctx = NeuroDSL.CtxStore()
NeuroDSL.demand!(g, :loss; ctx_store=ctx, namespace=ns)
NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx, namespace=ns, prune_frozen=false)
grad_full = copy(Array(g.nodes[ns][tsym].gradient))
ctx2 = NeuroDSL.CtxStore()
NeuroDSL.demand!(g, :loss; ctx_store=ctx2, namespace=ns)
NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx2, namespace=ns, prune_frozen=true)
max_err = maximum(abs.(grad_full .- Array(g.nodes[ns][tsym].gradient)))

# ── (2) temps d'horloge, bras alternés DANS le processus ─────────────────────
for _ in 1:WARMUP
    one_pass!(g, ns; prune=false); one_pass!(g, ns; prune=true)
end
t_full = Float64[]; t_pruned = Float64[]
for _ in 1:REPS
    t0 = time_ns(); one_pass!(g, ns; prune=false); push!(t_full,   (time_ns()-t0)/1e6)
    t1 = time_ns(); one_pass!(g, ns; prune=true);  push!(t_pruned, (time_ns()-t1)/1e6)
end
mf, mp = median(t_full), median(t_pruned)

# Ligne unique, parsable par le pilote. Un seul bras, un seul processus.
@printf("RESULT arm=%s nodes=%d bw_full=%d bw_pruned=%d pruned_pct=%.4f max_err=%.3e t_full_ms=%.4f t_pruned_ms=%.4f gain_pct=%.4f\n",
        ARM, length(g.nodes[ns]), n_full, n_pruned,
        100*(n_full-n_pruned)/n_full, max_err, mf, mp, 100*(mf-mp)/mf)
