#=
Banc MÉMOIRE de la fusion de nœuds (article 1).

POURQUOI CE BANC EXISTE : `notebook/bench_fusion.jl` a montré que la fusion
n'apporte quasiment rien en VITESSE aux tailles réalistes (0-2% à dim 512,
10-12% seulement à dim 32). C'est structurel : la fusion appelle le GEMM du
vendeur -- qui écrit toute sa sortie en mémoire -- puis relit/réécrit cette
sortie dans une passe de broadcast séparée. Le TRAFIC mémoire est donc identique
à la version non fusionnée. Ce n'est pas de la fusion de NOYAU (cuBLASLt,
CUTLASS, Triton : épilogue appliqué depuis les registres, avant écriture en
mémoire globale), c'est de la fusion de NŒUDS au niveau du graphe.

Ce qu'elle économise réellement, et que la vitesse ne mesure pas : un TENSEUR
INTERMÉDIAIRE par motif. Sans fusion, une paire matmul->relu détient deux
buffers (`z` la sortie du matmul, `h` celle du relu) ; fusionnée, un seul.
Sur NPAIRS motifs, ça fait NPAIRS buffers de moins à garder résidents -- ce qui
est l'axe réel de la thèse de l'article (mémoire et localité), pas la vitesse.

DEUX QUANTITÉS, séparées comme dans `bench_prune_frozen.jl` :
  (1) OCTETS RÉSIDENTS D'ACTIVATIONS (structurel, déterministe, sans chronomètre
      ni horloge) : somme des tailles de toutes les valeurs de nœuds non
      paramètres. C'est ce que le graphe persistant garde vivant.
  (2) PIC ABSOLU (GPU seulement) : watermark du pool CUDA
      (`CU_MEMPOOL_ATTR_USED_MEM_HIGH`), même instrument que
      `article_benchmark_vram_probe.jl` -- pic absolu, baseline incluse, jamais
      un delta comparé à un absolu (piège déjà documenté là-bas).

PROTOCOLE : UN SEUL BRAS par exécution (`ARGS[3]`), processus neuf. Nécessaire
ici plus qu'ailleurs : les deux graphes coexistant dans un même processus,
les buffers du premier bras gonfleraient la baseline du second. Leçon tirée de
`bench_prune_frozen.jl`, dont une première version partageait un processus entre
bras et produisait un résultat non reproductible.

USAGE : julia --project=. notebook/bench_fusion_memory.jl [cpu|gpu] [dim] [unfused|fused]
=#
using NeuroDSL, Printf
using CUDA

const DEVARG = length(ARGS) >= 1 ? ARGS[1] : "gpu"
const DIM    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 512
const ARM    = length(ARGS) >= 3 ? ARGS[3] : "unfused"
ARM in ("unfused","fused") || error("bras inconnu : $ARM")
const ROWS   = 64
const NPAIRS = 8

const dev = DEVARG == "gpu" ? NeuroDSL.Backend.CUDADevice() : NeuroDSL.Backend.CPUDevice()

# ── watermark du pool CUDA (GPU seulement) ───────────────────────────────────
if DEVARG == "gpu"
    const _POOL = CUDA.CUDACore.pool_create(CUDA.CUDACore.active_state().device)
    pool_cur_mb() = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_CURRENT))/1024^2
    pool_high_mb() = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH))/1024^2
    reset_high!() = CUDA.attribute!(_POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
    quiesce() = (CUDA.synchronize(); GC.gc(true); GC.gc(true); CUDA.synchronize())
else
    pool_cur_mb() = 0.0; pool_high_mb() = 0.0; reset_high!() = nothing
    quiesce() = (GC.gc(true); GC.gc(true))
end

function build_chain(ns::Symbol)
    g = NeuroDSL.NeuroGraph(device=dev, namespace=ns)
    NeuroDSL.set!(g, :h0, NeuroDSL.Backend.randn32(dev, ROWS, DIM); namespace=ns)
    prev = :h0
    for i in 1:NPAIRS
        W = Symbol(:W, i)
        NeuroDSL.set!(g, W, randn(Float32, DIM, DIM).*0.05f0; is_param=true, namespace=ns)
        z = Symbol(:z, i); h = Symbol(:h, i)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(z, [prev, W], :matmul; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(h, [z], :relu; namespace=ns))
        prev = h
    end
    return g, prev
end

"Octets résidents des valeurs de nœuds NON paramètres (les activations)."
function activation_bytes(g, ns)
    tot = 0
    for (sym, nd) in g.nodes[ns]
        nd.is_param && continue
        nd.value === nothing && continue
        tot += length(nd.value) * sizeof(eltype(nd.value))
    end
    return tot
end

ns = Symbol(:fmem_, DEVARG, :_, DIM, :_, ARM)
g, out = build_chain(ns)

if ARM == "fused"
    rule = NeuroDSL.RewriteRule(:matmul_relu_fusion, (:matmul, :relu), :fused_matmul_relu;
                                cost_delta=0.3f0)
    NeuroDSL.compile(g, NeuroDSL.CompilerConfig(rules=[rule]); namespace=ns)
end
n_rules = length(g.rules[ns])

fwd!() = (NeuroDSL.invalidate_all!(g; namespace=ns);
          NeuroDSL.demand!(g, out; namespace=ns);
          DEVARG == "gpu" && CUDA.synchronize())

for _ in 1:5; fwd!(); end          # stabilise formes et buffers
quiesce()

# (1) structurel, déterministe
act_bytes = activation_bytes(g, ns)
n_act = count(nd -> !nd.is_param && nd.value !== nothing, values(g.nodes[ns]))

# (2) pic absolu, 3 essais -- doit être stable, sinon fuite/non-déterminisme
peaks = Float64[]; bases = Float64[]
for _ in 1:3
    quiesce(); b = pool_cur_mb(); reset_high!(); fwd!()
    push!(bases, b); push!(peaks, pool_high_mb())
end
stable = all(==(peaks[1]), peaks)

@printf("RESULT dev=%s dim=%d arm=%s rules=%d n_act=%d act_bytes=%d baseline_mb=%.3f peak_mb=%.3f stable=%s\n",
        DEVARG, DIM, ARM, n_rules, n_act, act_bytes, bases[1], peaks[1], stable)
