# WIND-T5c (POST HOC) -- l'effondrement de AN est-il alimenté par les termes de rang 1 des normes figées ?
# R_k = 2 vecteurs singuliers droits de ΔN_k = J_AN − J_A ; frac_k = ‖R_kᵀ v₁(P_k)‖², P_k = J_27···J_k.
using LinearAlgebra, Statistics, Printf, Dates
const OUTDIR = joinpath(@__DIR__, "wind_data"); const PROMPT_IDXS = [5, 6, 12, 17, 22]
const RES = joinpath(@__DIR__, "wind_collapse_normmode_results.txt"); const L, D = 28, 1536; const KMAX = 10
BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ 2))
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 1:L])
mx = Dict{Tuple{Int,String},Float64}()
open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T5c (POST HOC) -- frac_k = ‖R_kᵀ v₁(P_k)‖², R_k = sous-espace d'entrée (rang 2) de J_AN − J_A ; hasard 2/1536 = 0.0013")
    emit("Date : " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PROMPT_IDXS
        JA = loadJ(p, "A"); JN = loadJ(p, "AN")
        R = [svd(JN[k+1] - JA[k+1]).V[:, 1:2] for k in 1:KMAX]
        JA = nothing; GC.gc()
        fr = Dict{String,Vector{Float64}}()
        for V in ("T", "AN")
            Js = V == "AN" ? JN : loadJ(p, "T")
            P = Matrix{Float64}(I, D, D); f = zeros(KMAX)
            for k in 27:-1:1
                P = P * Js[k+1]
                k <= KMAX && (f[k] = sum(abs2, R[k]' * svd(P).V[:, 1]))
            end
            fr[V] = f; mx[(p, V)] = maximum(f)
        end
        JN = nothing; GC.gc()
        emit(@sprintf("\nPROMPT %d    k : ", p) * join([@sprintf("%6d", k) for k in 1:KMAX]))
        for V in ("T", "AN")
            emit(@sprintf("   %-3s frac_k : ", V) * join([@sprintf("%6.3f", x) for x in fr[V]]))
        end
    end
    p2 = all(mx[(p, "AN")] >= 0.5 for p in (12, 22)) && median(mx[(p, "T")] for p in PROMPT_IDXS) < 0.1
    emit(@sprintf("\nP2' : %s  (AN max : p12 %.3f, p22 %.3f ; T médiane des max %.3f ; AN autres : %s)",
                  p2 ? "VRAI" : "FAUX", mx[(12, "AN")], mx[(22, "AN")], median(mx[(p, "T")] for p in PROMPT_IDXS),
                  join([@sprintf("p%d %.3f", p, mx[(p, "AN")]) for p in (5, 6, 17)], ", ")))
end
