#=
Benchmark du claim "37% speedup in fused operator execution" (article 1).

CONTEXTE / AUDIT du 2026-08-08 :
  - L'abstract annonçait "37% speedup in fused operator execution" sans préciser
    de quoi ce pourcentage était le gain.
  - Seuls QUATRE ops fusionnés sont réellement exécutables
    (`_DISPATCH_EXECUTABLE_FUSED_OPS`, src/dispatch.jl:722) :
    `:fused_matmul_relu`, `:fused_matmul_add`, `:fused_matmul_add_relu`,
    `:fused_qkv_projection` -- et ce dernier est un simple `mul!`, aucune
    batchification 3-en-1 (fausse affirmation trouvée et supprimée du code le
    2026-07-05). Les ~13 autres ops "fusionnés" déclarés dans les jeux de règles
    n'ont pas de branche d'exécution et sont silencieusement ignorés.
  - Ce que la fusion fait RÉELLEMENT (src/dispatch.jl:340) : elle appelle le GEMM
    du vendeur (`mul!` -> OpenBLAS/cuBLAS) puis applique l'épilogue élémentaire
    en broadcast. Elle N'ÉCONOMISE DONC PAS DE FLOPs. Elle économise un nœud de
    graphe, son dispatch, et son buffer intermédiaire.
    Le commentaire du code est explicite : un kernel matmul maison avait été
    1.9-4.2x PLUS LENT que cuBLAS et a été abandonné.

CONSÉQUENCE MÉTHODOLOGIQUE : puisque le gain est du surcoût d'interpréteur et non
du calcul, il doit être GRAND quand les tenseurs sont petits (le dispatch domine)
et s'ÉVANOUIR quand ils grossissent (BLAS domine). Un pourcentage unique n'a donc
aucun sens hors de son régime -- ce banc balaie plusieurs tailles sur CPU ET GPU
pour délimiter le régime, au lieu de citer un chiffre isolé.

PROTOCOLE : un seul (device, dim) par exécution -- processus neuf, aucune
contamination d'état (leçon tirée de `bench_prune_frozen.jl`, dont une première
version partageait un processus entre bras et produisait un résultat non
reproductible). Le pilote `bench_fusion_driver.sh` relance et agrège.
Chronométrage GPU valide seulement horloge verrouillée (`nvidia-smi -lgc 1402,1402`).

CORRECTION AVANT VITESSE : la sortie fusionnée est comparée à la sortie non
fusionnée à chaque exécution ; l'écart max est rapporté.

USAGE : julia --project=. notebook/bench_fusion.jl [cpu|gpu] [dim]
=#
using NeuroDSL, LinearAlgebra, Statistics, Printf
using CUDA

const DEVARG = length(ARGS) >= 1 ? ARGS[1] : "cpu"
const DIM    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 128
const ROWS   = 64
const NPAIRS = 8      # nombre de motifs matmul->relu dans la chaîne
const REPS   = 50
const WARMUP = 10

const dev = DEVARG == "gpu" ? NeuroDSL.Backend.CUDADevice() : NeuroDSL.Backend.CPUDevice()
sync() = DEVARG == "gpu" ? CUDA.synchronize() : nothing

"Chaîne de NPAIRS motifs (matmul -> relu). Retourne (graphe, symbole de sortie)."
function build_chain(ns::Symbol)
    g = NeuroDSL.NeuroGraph(device=dev, namespace=ns)
    NeuroDSL.set!(g, :h0, NeuroDSL.Backend.randn32(dev, ROWS, DIM); namespace=ns)
    prev = :h0
    for i in 1:NPAIRS
        W = Symbol(:W, i)
        NeuroDSL.set!(g, W, randn(Float32, DIM, DIM) .* 0.05f0; is_param=true, namespace=ns)
        z = Symbol(:z, i); h = Symbol(:h, i)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(z, [prev, W], :matmul; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(h, [z], :relu; namespace=ns))
        prev = h
    end
    return g, prev
end

fwd!(g, out, ns) = begin
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, out; namespace=ns)
    sync()
end

# ── graphe NON fusionné ──────────────────────────────────────────────────────
ns_u = Symbol(:fus_u_, DEVARG, :_, DIM)
g_u, out_u = build_chain(ns_u)
n_rules_unfused = length(g_u.rules[ns_u])
fwd!(g_u, out_u, ns_u)
ref = Array(NeuroDSL.demand!(g_u, out_u; namespace=ns_u))

# ── graphe FUSIONNÉ (mêmes poids, copiés, pour comparer les sorties) ─────────
ns_f = Symbol(:fus_f_, DEVARG, :_, DIM)
g_f, out_f = build_chain(ns_f)
for i in 1:NPAIRS       # aligner les poids sur le graphe non fusionné
    W = Symbol(:W, i)
    NeuroDSL.set!(g_f, W, Array(g_u.nodes[ns_u][W].value); is_param=true, namespace=ns_f)
end
NeuroDSL.set!(g_f, :h0, Array(g_u.nodes[ns_u][:h0].value); namespace=ns_f)

rule = NeuroDSL.RewriteRule(:matmul_relu_fusion, (:matmul, :relu), :fused_matmul_relu;
                            cost_delta=0.3f0)
plan = NeuroDSL.compile(g_f, NeuroDSL.CompilerConfig(rules=[rule]); namespace=ns_f)
n_rules_fused = length(g_f.rules[ns_f])
n_fused_applied = length(plan.fused_ops)

fwd!(g_f, out_f, ns_f)
got = Array(NeuroDSL.demand!(g_f, out_f; namespace=ns_f))
max_err = size(got) == size(ref) ? maximum(abs.(got .- ref)) : NaN

# ── chronométrage, bras alternés ─────────────────────────────────────────────
for _ in 1:WARMUP; fwd!(g_u, out_u, ns_u); fwd!(g_f, out_f, ns_f); end
t_u = Float64[]; t_f = Float64[]
for _ in 1:REPS
    t0 = time_ns(); fwd!(g_u, out_u, ns_u); push!(t_u, (time_ns()-t0)/1e6)
    t1 = time_ns(); fwd!(g_f, out_f, ns_f); push!(t_f, (time_ns()-t1)/1e6)
end
mu, mf = median(t_u), median(t_f)

@printf("RESULT dev=%s dim=%d rules_unfused=%d rules_fused=%d fused_applied=%d max_err=%.3e t_unfused_ms=%.4f t_fused_ms=%.4f gain_pct=%.4f\n",
        DEVARG, DIM, n_rules_unfused, n_rules_fused, n_fused_applied,
        max_err, mu, mf, 100*(mu-mf)/mu)
