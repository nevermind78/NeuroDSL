# ══════════════════════════════════════════════════════════════════════════════
# WIND-2b -- CONTRÔLE POST-HOC (NON pré-enregistré) : l'enrichissement E_amp des paires
# adjacentes est-il un couplage LOCAL ou une anisotropie GLOBALE du flux résiduel ?
#
# On compare la sortie amplifiée de J_k (U_k) à l'entrée amplifiée de J_{k+j} (V_{k+j}) dans
# la base résiduelle commune, pour plusieurs distances j. Courbe plate en j -> anisotropie
# partagée par toutes les couches ; décroissante -> couplage local entre couches proches.
# Nul de Haar analytique : E[E_amp] = 1, Var = 2 (n−r_a)(n−r_b) / (r_a r_b (n−1)(n+2)).
#
# USAGE : julia --project=. notebook/wind_alignment_lag.jl
# ══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra, Statistics, Printf, JSON, Dates

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const RES  = joinpath(@__DIR__, "wind_alignment_lag_results.txt")
const JOUT = joinpath(@__DIR__, "wind_alignment_lag_results.json")
const L, D = 28, 1536
const LAGS = [1, 2, 3, 5, 8, 12, 20]
BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ 2))

loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 1:L])
null_sd(ra, rb, n = D) = sqrt(2 * (n - ra) * (n - rb) / (ra * rb * (n - 1) * (n + 2)))

res = Dict(j => Float64[] for j in LAGS); zs = Dict(j => Float64[] for j in LAGS)
xres = Dict(j => Float64[] for j in LAGS)      # contrôle inter-prompts : U_k(p) vs V_{k+j}(q), q ≠ p
Fs_all = Dict{Int,Any}()

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-2b -- CONTRÔLE POST-HOC (non pré-enregistré) : E_amp en fonction de la distance entre couches")
    emit("Date : " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PROMPT_IDXS
        Js = loadJ(p, "T")
        Fs_all[p] = [(F = svd(Js[k]); r = count(>(1.5), F.S);
                      (U = F.U[:, 1:r], V = Matrix(F.V[:, 1:r]), S = F.S)) for k in 1:L]   # colonnes amplifiées seulement
        Js = nothing; GC.gc()
        emit("prompt $p : SVD faites")
    end
    for p in PROMPT_IDXS, j in LAGS, k in 1:(L - 1 - j)          # J_k indexé k (Fs[k+1]), k ≥ 1 (hors embedding)
        Fa = Fs_all[p][k+1]; Fb = Fs_all[p][k+j+1]
        ra = count(>(1.5), Fa.S); rb = count(>(1.5), Fb.S)
        (ra == 0 || rb == 0) && continue
        cap = sum(abs2, Fb.V' * Fa.U) / ra / (rb / D)
        push!(res[j], cap); push!(zs[j], (cap - 1) / null_sd(ra, rb))
        q = PROMPT_IDXS[mod1(findfirst(==(p), PROMPT_IDXS) + 1, length(PROMPT_IDXS))]
        Fq = Fs_all[q][k+j+1]; rq = count(>(1.5), Fq.S)
        rq > 0 && push!(xres[j], sum(abs2, Fq.V' * Fa.U) / ra / (rq / D))
    end
    emit("\n   j   médiane E_amp(k,k+j)  [Q1, Q3]        z médian (nul de Haar)   inter-prompts E_amp médiane")
    for j in LAGS
        emit(@sprintf("  %2d        %.3f          [%.3f, %.3f]        %7.1f                  %.3f",
                      j, median(res[j]), quantile(res[j], 0.25), quantile(res[j], 0.75), median(zs[j]), median(xres[j])))
    end
    emit("\nLecture : courbe plate en j et inter-prompts ≈ intra-prompt -> anisotropie globale ;")
    emit("          décroissance en j et intra > inter -> couplage local, spécifique au prompt.")
end
open(JOUT, "w") do f
    JSON.print(f, Dict("lags" => LAGS, "Eamp" => Dict(string(j) => res[j] for j in LAGS),
                       "z" => Dict(string(j) => zs[j] for j in LAGS),
                       "Eamp_crossprompt" => Dict(string(j) => xres[j] for j in LAGS)))
end
println("\nÉcrit : ", RES, " et ", JOUT)
