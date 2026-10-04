# ══════════════════════════════════════════════════════════════════════════════
# WIND-3 -- HH5 (descriptif, pré-enregistré) : rang du canal horizontal.
# Esquisse aléatoire de J_{(i,s)→(n,28)} pour TOUTES les sources (i,s) à la fois :
# K backwards de sonde P = <x_{n,28}, u_r>, u_r gaussien -> Y_r = (J^T u_r) pour tout (i,s).
# erank₂ calculé sur les valeurs singulières de Y (K×D). Porte GS : bloc vertical (i=n)
# vs erank₂ EXACT des produits de Jacobiennes DF de WIND-1, s ∈ {1,7,14}, facteur 1.5.
# Écrit aussi X (activations complètes, n×D×29) pour le descriptif gradient×entrée.
# USAGE : julia --project=. notebook/wind3_rank_sketch.jl
# ══════════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "wind3_common.jl"))

const OUTDIR = joinpath(@__DIR__, "wind_data")
const RES = joinpath(@__DIR__, "wind3_rank_sketch_results.txt")
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND3_PROMPTS", "5,6,12,17,22"), ","))
const K = parse(Int, get(ENV, "WIND3_K", "128"))
const SLIST = [1, 7, 14, 21]

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-3 -- HH5 : esquisse du rang du canal horizontal (K = $K backwards de sonde par variante)")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    g, dev, orig, W_U = load_instrumented()
    prompts = JSON.parsefile(joinpath(@__DIR__, "qwen_sweep_prompts.json"))
    results = Dict{String,Any}()
    for pidx in PROMPT_IDXS
        ids = Int.(prompts[pidx]["token_ids"]) .+ 1
        swap_norms!(g, orig, false); SG_ON[] = false
        n = set_prompt!(g, ids)
        meta1 = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(pidx).json"))
        set_readout!(dev, Float64.(meta1["w_read"]))
        NeuroDSL.demand!(g, :lm_head_out; namespace=NS)
        cache = capture_rule_nodes(g)
        for s in NORM_SYMS
            FROZEN_RMS[s] = copy(NeuroDSL.node(g, s; namespace=NS).aux_data[:rms_inv])
        end
        Xs = Array{Float32}(undef, n, D, L + 1)
        for k in 0:L; Xs[:, :, k+1] .= Array(cache[res_sym(k)]); end
        r0 = all(Float64.(Xs[n, :, k+1]) == Float64.(meta1["X"][k+1]) for k in 0:L)
        open(joinpath(OUTDIR, "wind3_X_p$(pidx).bin"), "w") do f; write(f, Xs); end
        emit("\nPROMPT $pidx (n=$n) ; porte R0 (x_{n,k} == WIND-1) : $(r0 ? "OK" : "ÉCHEC")")
        rng = MersenneTwister(5000 + pidx)
        U = [randn(rng, D) for _ in 1:K]
        pres = Dict{String,Any}()
        for V in (:T, :A, :AN)
            t0 = time()
            Y = Array{Float32}(undef, K, D, n, L)          # s = 0..27
            for r in 1:K
                set_probe!(dev, U[r])
                Gx, _ = variant_grads!(g, orig, V; seed=:w3_P)
                for s in 0:L-1, i in 1:n
                    Y[r, :, i, s+1] .= Float32.(Gx[s+1][i, :])
                end
            end
            er = zeros(n, L); erh = zeros(L)
            for s in 0:L-1
                for i in 1:n
                    er[i, s+1] = erank2(svdvals(Float64.(Y[:, :, i, s+1])))
                end
                erh[s+1] = erank2(svdvals(Float64.(reshape(Y[:, :, 1:n-1, s+1], K, :))))   # toutes sources i<n
            end
            emit(@sprintf("  [%-2s] %.0f s ; erank₂ esquissé (s = %s) :", V, time() - t0, join(SLIST, ",")))
            emit("        vertical i=n     : " * join([@sprintf("%7.1f", er[n, s+1]) for s in SLIST], " "))
            emit("        puits i=1        : " * join([@sprintf("%7.1f", er[1, s+1]) for s in SLIST], " "))
            emit("        max_{i<n}        : " * join([@sprintf("%7.1f", maximum(er[1:n-1, s+1])) for s in SLIST], " "))
            emit("        médiane_{2≤i<n}  : " * join([@sprintf("%7.1f", median(er[2:n-1, s+1])) for s in SLIST], " "))
            emit("        horizontal joint : " * join([@sprintf("%7.1f", erh[s+1]) for s in SLIST], " "))
            # porte GS
            Js = Array{Float32}(undef, D, D, L)
            open(joinpath(OUTDIR, "wind_J_p$(pidx)_$(V).bin"), "r") do f; read!(f, Js); end
            gs = Float64[]
            for s in (1, 7, 14)
                P = Matrix{Float64}(I, D, D)
                for k in s:L-1; P = Float64.(Js[:, :, k+1]) * P; end
                ex = erank2(svdvals(P)); push!(gs, er[n, s+1] / ex)
                emit(@sprintf("        GS s=%2d : erank₂ exact (WIND-1) %.1f ; esquissé %.1f ; rapport %.2f", s, ex, er[n, s+1], er[n, s+1] / ex))
            end
            Js = nothing; GC.gc()
            ok = all(1 / 1.5 .<= gs .<= 1.5)
            emit("        Porte GS [$V] : " * (ok ? "OK" : "ÉCHEC (comparaisons relatives seulement)"))
            pres[string(V)] = Dict("erank" => [er[:, s+1] for s in 0:L-1], "erank_hor" => erh, "GS" => gs, "GS_ok" => ok)
            Y = nothing; GC.gc()
        end
        results[string(pidx)] = pres
        cache = nothing; reclaim()
    end
    open(joinpath(OUTDIR, "wind3_rank_sketch.json"), "w") do f; JSON.print(f, results); end
end
println("\nÉcrit : ", RES)
