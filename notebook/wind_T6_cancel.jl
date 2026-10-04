# ══════════════════════════════════════════════════════════════════════════════
# WIND-T6 (EXPLORATOIRE) -- compensation par les termes de rétroaction (QK et normes).
# Télescopages exacts :
#   P_T − P_A  = Σ_k P^T_{k+1} E_k Q^A_{k−1},   E_k = J_k^T − J_k^A   (terme QK de la couche k)
#   P_AN − P_A = Σ_k P^AN_{k+1} R_k Q^A_{k−1},  R_k = J_k^AN − J_k^A  (gel des normes, rang ≤ 2)
# Le long de la direction dominante v = v₁(N·P_V) : part de chaque couche dans la projection
#   c_k = ⟨N P_A v, N T_k v⟩ / ‖N P_A v‖²   (somme sur k = ⟨N P_A v, N(P_T − P_A) v⟩/‖N P_A v‖²)
# c < 0 : le terme QK compense (retire) la direction dominante de A.
# USAGE : julia --project=. notebook/wind_T6_cancel.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = joinpath(@__DIR__, "wind_data")
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const RES = joinpath(@__DIR__, get(ENV, "WIND_RESNAME", "wind_T6_cancel_results.txt"))
const L, D = 28, 1536
BLAS.set_num_threads(16)
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
prodall(J) = (P = J[1]; for k in 2:27; P = J[k] * P; end; P)
fmt(v) = join([@sprintf("%6.3f", x) for x in v])

function telescope(N, Jdown, JA, v, Q)
    # renvoie les vecteurs N T_k v, T_k = Pdown_{k+1} (Jdown_k − JA_k) Q^A_{k−1}
    S = Matrix{Float64}(I, D, D); out = Vector{Vector{Float64}}(undef, 27)
    for k in 27:-1:1
        out[k] = N * (S * ((Jdown[k] - JA[k]) * (Q[k] * v)))
        S = S * Jdown[k]
    end
    out
end

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T6 compensation (télescopage le long de la direction dominante) -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PROMPT_IDXS
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x = Float64.(meta["X"][end])
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        JA = loadJ(p, "A"); JT = loadJ(p, "T"); JN = loadJ(p, "AN")
        Q = Vector{Matrix{Float64}}(undef, 27); Q[1] = Matrix{Float64}(I, D, D)
        for k in 2:27; Q[k] = JA[k-1] * Q[k-1]; end
        PA = JA[27] * Q[27]; PT = prodall(JT); PN = prodall(JN)
        emit(@sprintf("\n==================== PROMPT %d ====================", p))
        for (lab, v) in (("v₁(N·P_A)", svd(N * PA).V[:, 1]), ("v₁(N·P_AN)", svd(N * PN).V[:, 1]),
                         ("v₁(N·P_T)", svd(N * PT).V[:, 1]))
            a = N * (PA * v); t = N * (PT * v); n_ = N * (PN * v)
            emit(@sprintf("  direction %-11s : ‖N P_T v‖ = %9.3e  ‖N P_A v‖ = %9.3e  ‖N P_AN v‖ = %9.3e  | σ₁ T/A/AN = %9.3e %9.3e %9.3e",
                          lab, norm(t), norm(a), norm(n_), opnorm(N * PT), opnorm(N * PA), opnorm(N * PN)))
            emit(@sprintf("     cos(N P_A v, N(P_T−P_A)v) = %+.3f   ‖N(P_T−P_A)v‖/‖N P_A v‖ = %.3f   | cos(N P_A v, N(P_AN−P_A)v) = %+.3f   ‖N(P_AN−P_A)v‖/‖N P_A v‖ = %.3f",
                          dot(a, t - a) / (norm(a) * norm(t - a)), norm(t - a) / norm(a),
                          dot(a, n_ - a) / (norm(a) * norm(n_ - a)), norm(n_ - a) / norm(a)))
            tq = telescope(N, JT, JA, v, Q); tn = telescope(N, JN, JA, v, Q)
            gq = norm(sum(tq) - (t - a)) / norm(t - a); gn = norm(sum(tn) - (n_ - a)) / norm(n_ - a)
            cq = [dot(a, y) / dot(a, a) for y in tq]; cn = [dot(a, y) / dot(a, a) for y in tn]
            emit(@sprintf("     portes télescopage : QK %.1e  normes %.1e ;  Σc_QK = %+.3f   Σc_norm = %+.3f", gq, gn, sum(cq), sum(cn)))
            emit("     k      : " * join([@sprintf("%6d", k) for k in 1:27]))
            emit("     c_QK   : " * fmt(cq))
            emit("     c_norm : " * fmt(cn))
        end
        JA = nothing; JT = nothing; JN = nothing; Q = nothing; GC.gc()
    end
end
println("Écrit : ", RES)
