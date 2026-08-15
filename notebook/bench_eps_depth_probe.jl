# =============================================================================
# SONDE DE COÛT AVANT L'ÉCHELLE EN PROFONDEUR (à lancer AVANT tout entraînement
# long). Mesure, pour L = 4 et L = 64 à largeur fixe :
#   - le temps d'un pas d'entraînement (batch = 1, le moteur n'a pas de
#     dimension de lot) ;
#   - le temps d'une passe arrière avec portes eps, donc le coût des B+1 = 2L+1
#     passes de l'identité d'ablation exacte ;
#   - le pic de VRAM.
# N'ENTRAÎNE RIEN. Sert uniquement à choisir un budget de pas défendable.
#
# Usage : julia --project=. notebook/bench_eps_depth_probe.jl
# =============================================================================

using NeuroDSL, Random, Printf, Statistics

const DIM, NH, HID, SEQ = 384, 6, 1024, 64
const OUT = joinpath(@__DIR__, "bench_eps_depth_probe_results.txt")

corpus = read(joinpath(@__DIR__, "data", "tinyshakespeare", "input.txt"), String)
chars  = sort(collect(Set(corpus)))
stoi   = Dict(c => i for (i, c) in enumerate(chars))
data   = [stoi[c] for c in corpus]
V      = length(chars)

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())
vram_used() = NeuroDSL.Backend.CUDA_AVAILABLE ?
    (NeuroDSL.CUDA.total_memory() - NeuroDSL.CUDA.free_memory())/2^30 : 0.0

function build(dev, ns::Symbol, L::Int)
    g = NeuroGraph(namespace=ns, device=dev)
    set!(g, :token_ids, ones(Int, SEQ); atom_type=Datom, namespace=ns)
    set!(g, :pos_ids, collect(1:SEQ); atom_type=Datom, namespace=ns)
    te = NeuroDSL.Embedding(V, DIM)(g, :token_ids, :tok; namespace=ns)
    pe = NeuroDSL.Embedding(SEQ, DIM)(g, :pos_ids, :pos; namespace=ns)
    addrule!(g, GraphRule(:embed_sum, [te, pe], :add; namespace=ns))
    h  = NeuroDSL.LlamaModel(L, DIM, NH, HID)(g, :embed_sum; namespace=ns)
    hn = NeuroDSL.LayerNorm(DIM)(g, h, :final_norm; namespace=ns)
    lg = NeuroDSL.Linear(DIM, V; bias=false)(g, hn, :lm_head; namespace=ns)
    set!(g, :labels, ones(Int, SEQ); atom_type=Datom, namespace=ns)
    addrule!(g, GraphRule(:loss, [lg, :labels], :cross_entropy; namespace=ns))
    return g, lg
end

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("SONDE DE COÛT -- échelle en profondeur à largeur fixe")
    emit("dim=$DIM  n_heads=$NH  hidden=$HID  seq=$SEQ  vocab=$V (tinyshakespeare)")
    emit("batch = 1 (le moteur n'a pas de dimension de lot)")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("")
    emit(@sprintf("%5s %12s %10s %12s %12s %12s", "L", "params", "VRAM Go",
                  "s/pas", "s/passe arr", "s pour 2L+1"))

    dev = NeuroDSL.Backend.CUDADevice()
    rng = MersenneTwister(1)
    for L in (4, 64)
        ns = Symbol("probe", L)
        reclaim(); v0 = vram_used()
        g, lg = build(dev, ns, L)
        ps = params(g; namespace=ns)
        npar = sum(length(p.value) for p in ps)
        m1 = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
        m2 = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]

        step!(t) = begin
            i = rand(rng, 1:(length(data) - SEQ - 1))
            set!(g, :token_ids, data[i:i+SEQ-1]; atom_type=Datom, namespace=ns)
            set!(g, :labels, data[i+1:i+SEQ]; atom_type=Datom, namespace=ns)
            invalidate_all!(g; namespace=ns)
            l = demand!(g, :loss; namespace=ns)
            backward_graph!(g, :loss; namespace=ns)
            for (k, p) in enumerate(ps)
                adamw_step!(dev, p.value, p.gradient, m1[k], m2[k],
                            3f-4, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
            end
            invalidate_all!(g; namespace=ns)
            Float64(sum(Array(l)))
        end

        step!(1); step!(2)                       # préchauffage / compilation
        t_step = median([(@elapsed step!(t)) for t in 3:12])

        # coût d'une passe arrière seule (ce que paient les 2L+1 ablations)
        invalidate_all!(g; namespace=ns)
        demand!(g, :loss; namespace=ns)
        backward_graph!(g, :loss; namespace=ns)
        t_bwd = median([(@elapsed begin
            invalidate_all!(g; namespace=ns)
            demand!(g, :loss; namespace=ns)
            backward_graph!(g, :loss; namespace=ns)
        end) for _ in 1:5])

        vpk = vram_used() - v0
        emit(@sprintf("%5d %12d %10.2f %12.4f %12.4f %12.1f",
                      L, npar, vpk, t_step, t_bwd, (2L+1)*t_bwd))
        for s in keys(g.nodes[ns]); end
        g = nothing; ps = nothing; m1 = nothing; m2 = nothing
        reclaim()
    end
    emit("")
    emit("Lire : le budget de pas doit tenir dans un temps raisonnable pour")
    emit("5 profondeurs x 3 graines, et la mesure exacte coûte (2L+1) passes")
    emit("arrière par modèle, une seule fois par modèle.")
end
println("\nÉcrit : ", OUT)
