# =============================================================================
# n_bar À L'INITIALISATION, SUR LA MÊME ARCHITECTURE QUE L'ÉCHELLE ENTRAÎNÉE
#
# POURQUOI CE SCRIPT EXISTE : UN CONFONDANT QUE PERSONNE N'AVAIT NOMMÉ
# ---------------------------------------------------------------------
# On veut opposer « à l'initialisation » et « entraîné ». Les deux mesures
# disponibles jusqu'ici ne diffèrent PAS seulement par l'entraînement :
#   - bench_eps_depth_synthetic : pile pré-norm SYNTHÉTIQUE, branches
#     tanh-MLP, PAS d'attention, largeur 64, double précision ;
#   - bench_eps_depth_trained   : vraie attention + SwiGLU, largeur 384,
#     Float32, corpus caractère.
# Attribuer à l'ENTRAÎNEMENT l'écart entre les deux serait donc confondre
# entraînement et architecture. Ce script supprime le confondant : MÊME
# architecture, MÊMES profondeurs, MÊMES graines que l'échelle entraînée, mais
# ZÉRO pas d'entraînement. Coût : (2L+1) passes arrière par modèle, aucune
# passe d'entraînement.
#
# MESURE AUSSI l'indice d'annulation somme|q_j| / |somme q_j|. Une affirmation
# de convergence pour n_bar demande de contrôler somme|q_j|, pas seulement
# somme q_j : des q_j signés peuvent converger conditionnellement, et n_bar
# n'est alors pas robuste. Cette quantité sort des MÊMES passes.
#
# N'ÉCRIT QUE bench_eps_depth_init_samearch_results.txt. Ne touche à aucun
# artefact existant.
# Usage : julia --project=. notebook/bench_eps_depth_init_samearch.jl
# =============================================================================

using NeuroDSL, Random, Printf, Statistics

const DIM, NH, HID, SEQ = 384, 6, 1024, 64
const L_LIST = [4, 8, 16, 32, 64]
const SEEDS  = [1, 2, 3]
const NVAL   = 24
const OUT = joinpath(@__DIR__, "bench_eps_depth_init_samearch_results.txt")

corpus = read(joinpath(@__DIR__, "data", "tinyshakespeare", "input.txt"), String)
chars  = sort(collect(Set(corpus)))
stoi   = Dict(c => i for (i, c) in enumerate(chars))
alld   = [stoi[c] for c in corpus]
V      = length(chars)
ntr    = floor(Int, 0.9 * length(alld))
valid  = alld[ntr+1:end]
val_start = [1 + (k-1) * div(length(valid) - SEQ - 1, NVAL) for k in 1:NVAL]

const EPSV     = Dict{Symbol,Float32}()
const CAPTURED = Dict{Symbol,Array{Float32}}()
const CAPTURE  = Ref(false)

function ensure_eps_op!(branch::Symbol)
    op = Symbol("epsop_", branch)
    EPSV[branch] = 1.0f0
    haskey(NeuroDSL.CUSTOM_OPS, op) && return op
    NeuroDSL.register_op!(op,
        (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> (out_buf .= inputs[1]))
    NeuroDSL.CUSTOM_SHAPE_RULES[op] = (inputs, attrs) -> size(inputs[1])
    NeuroDSL.GRAD_RULES[op] = (dev, dy, ctx, inputs) -> begin
        gr = EPSV[branch] .* dy
        CAPTURE[] && (CAPTURED[branch] = copy(Array(gr)))
        return (gr,)
    end
    return op
end

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())

function build(dev, ns, L)
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
    return g
end

function install_gates!(g, ns, L)
    br = Symbol[]
    for i in 1:L
        push!(br, Symbol("layer_", i, "_mha_output_out"))
        push!(br, Symbol("layer_", i, "_mlp_out"))
    end
    for i in 1:L
        for (js, b) in ((Symbol("layer_", i, "_res1"), Symbol("layer_", i, "_mha_output_out")),
                        (Symbol("layer_", i, "_out"),  Symbol("layer_", i, "_mlp_out")))
            r = g.rules[ns][js]
            r.inputs[2] == b || error("$js : branche inattendue")
            es = Symbol("eps_", b)
            addrule!(g, GraphRule(es, [b], ensure_eps_op!(b); namespace=ns))
            addrule!(g, GraphRule(js, [r.inputs[1], es], r.op;
                                  attrs=r.attrs, namespace=ns, atom_type=r.atom_type))
            NeuroDSL._invalidate_downstream!(g, js, ns)
        end
    end
    br
end

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("n_bar À L'INITIALISATION -- MÊME ARCHITECTURE QUE L'ÉCHELLE ENTRAÎNÉE")
    emit("Supprime le confondant architecture : mêmes L, mêmes graines, mêmes")
    emit("largeurs que bench_eps_depth_trained, mais ZÉRO pas d'entraînement.")
    emit("dim=$DIM n_heads=$NH hidden=$HID seq=$SEQ vocab=$V  Float32  batch=1")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("")
    emit(@sprintf("%5s %5s %5s %10s %10s %10s %9s %9s %11s %11s",
         "L", "gr", "B", "n_bar", "excès", "exc/logL", "p_L1", "p_deep",
         "somme|q|/|s|", "% neg"))

    dev = NeuroDSL.Backend.CUDADevice()
    ACC = Dict{Int,Vector{NamedTuple}}()
    for L in L_LIST, sd in SEEDS
        ns = Symbol("i", L, "_s", sd)
        reclaim()
        NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.seed!(2000 + sd)
        Random.seed!(2000 + sd)
        g = build(dev, ns, L)
        branches = install_gates!(g, ns, L)
        pos = Dict(b => k for (k, b) in enumerate(branches))
        B_of(b) = length(branches) - pos[b] + 1
        set!(g, :token_ids, valid[val_start[1]:val_start[1]+SEQ-1]; atom_type=Datom, namespace=ns)
        set!(g, :labels,    valid[val_start[1]+1:val_start[1]+SEQ]; atom_type=Datom, namespace=ns)
        invalidate_all!(g; namespace=ns)
        bwd!() = begin
            empty!(CAPTURED)
            demand!(g, :loss; namespace=ns)
            backward_graph!(g, :loss; namespace=ns)
            copy(CAPTURED)
        end
        CAPTURE[] = true
        for b in branches; EPSV[b] = 1.0f0; end
        base = bwd!()
        nrm2 = Dict(b => Float64(sum(abs2, base[b])) for b in branches)
        Q = Dict{Tuple{Symbol,Symbol},Float64}()
        for bj in branches
            EPSV[bj] = 0.0f0; abl = bwd!(); EPSV[bj] = 1.0f0
            for si in branches
                g0, gg = abl[si], base[si]
                Q[(si,bj)] = 1.0 - Float64(sum(g0 .* gg))/nrm2[si]
            end
        end
        CAPTURE[] = false; empty!(CAPTURED)
        s1 = Symbol("layer_1_mha_output_out")
        nbar = Dict(si => sum(Q[(si,bj)] for bj in branches if pos[bj] >= pos[si])
                    for si in branches)
        ratios = [nbar[Symbol("layer_", i, "_mha_output_out")] /
                  B_of(Symbol("layer_", i, "_mha_output_out")) for i in 1:L]
        qs1 = [Q[(s1,bj)] for bj in branches if pos[bj] >= pos[s1]]
        canc = sum(abs, qs1) / abs(sum(qs1))
        ins = [(si,bj) for si in branches for bj in branches if pos[bj] >= pos[si]]
        pneg = 100 * count(k -> Q[k] < 0, ins) / length(ins)
        r = (nbar = nbar[s1], exc = nbar[s1] - 1, p1 = ratios[1],
             pdeep = mean(ratios[1:max(1, L ÷ 2)]), canc = canc, pneg = pneg)
        push!(get!(ACC, L, NamedTuple[]), r)
        emit(@sprintf("%5d %5d %5d %10.4f %10.4f %10.4f %9.5f %9.5f %11.3f %11.1f",
             L, sd, 2L, r.nbar, r.exc, r.exc/log(L), r.p1, r.pdeep, r.canc, r.pneg))
        g = nothing; reclaim()
    end

    emit(""); emit("="^78)
    emit("MÉDIANES, ET COMPARAISON AU RÉGIME ENTRAÎNÉ")
    emit("="^78); emit("")
    emit(@sprintf("%5s %10s %10s %10s %11s %9s", "L", "n_bar méd", "excès méd",
                  "exc/logL", "somme|q|/|s|", "p_deep*L"))
    for L in L_LIST
        rs = ACC[L]
        e = median([r.exc for r in rs])
        emit(@sprintf("%5d %10.4f %10.4f %10.4f %11.3f %9.4f",
             L, median([r.nbar for r in rs]), e, e/log(L),
             median([r.canc for r in rs]), median([r.pdeep for r in rs])*L))
    end
    emit("")
    e4  = median([r.exc for r in ACC[4]]); e64 = median([r.exc for r in ACC[64]])
    emit(@sprintf("  croissance de l'excès L=4 -> 64 : x%.2f", e64/e4))
    emit(@sprintf("  une somme LOG divergente prédirait x%.2f (log 64/log 4)",
                  log(64)/log(4)))
    emit("  Lire : excès/log L PLAT => q_j ~ 1/j (somme log divergente).")
    emit("  excès/log L DÉCROISSANT => décroissance plus raide que 1/j.")
end
println("\nÉcrit : ", OUT)
