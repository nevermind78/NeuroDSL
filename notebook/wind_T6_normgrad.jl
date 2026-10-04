# ══════════════════════════════════════════════════════════════════════════════
# WIND-T6 (EXPLORATOIRE) -- structure du terme de gel des normes par télescopage exact.
#   P_AN − P_A = Σ_k P^AN_{k+1} R_k Q^A_{k−1},   R_k = J_k^AN − J_k^A (rang ≤ 2, F1),
#   P^AN_{k+1} = J^AN_27···J^AN_{k+1},  Q^A_{k−1} = J^A_{k−1}···J^A_1.
# Espace des lignes ⊆ S = span_k { Q^A_{k−1}ᵀ V_k }, V_k = 2 vecteurs singuliers droits de R_k ;
# (Q^A_{k−1}ᵀ x̂_k = gradient de ‖x_k‖ par rapport à x_1 le long de la chaîne A, normes vivantes).
# Mesures : porte (Δ hors de S), fraction de v₁(N·P_V) dans S (V = T, A, AN), poids par couche
# ‖N T_k‖_F des termes du télescopage, bornes de rang faible.
# USAGE : julia --project=. notebook/wind_T6_normgrad.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = joinpath(@__DIR__, "wind_data")
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const RES = joinpath(@__DIR__, get(ENV, "WIND_RESNAME", "wind_T6_normgrad_results.txt"))
const L, D = 28, 1536
BLAS.set_num_threads(16)
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])      # J[k] = J_k, k = 1..27
erank2(σ) = (q = σ .^ 2 ./ sum(abs2, σ); q = q[q .> 0]; exp(-sum(q .* log.(q))))
prodall(J) = (P = J[1]; for k in 2:27; P = J[k] * P; end; P)

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T6 gel des normes : télescopage exact -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PROMPT_IDXS
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        X = [Float64.(v) for v in meta["X"]]
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x = X[29]
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        JA = loadJ(p, "A"); JN = loadJ(p, "AN")
        Vk = Vector{Matrix{Float64}}(undef, 27); cxk = zeros(27)
        for k in 1:27
            F = svd(JN[k] - JA[k]); Vk[k] = F.V[:, 1:2]
            xh = X[k+1] ./ norm(X[k+1]); cxk[k] = norm(Vk[k]' * xh)     # x̂_k ∈ span V_k ?
        end
        # préfixes A et suffixes AN
        Q = Vector{Matrix{Float64}}(undef, 27); Q[1] = Matrix{Float64}(I, D, D)
        for k in 2:27; Q[k] = JA[k-1] * Q[k-1]; end                        # Q[k] = Q^A_{k−1}
        G = hcat([Q[k]' * Vk[k] for k in 1:27]...)                         # D × 54
        Fg = svd(G); rk = count(Fg.S .> 1e-10 * Fg.S[1]); Sb = Fg.U[:, 1:rk]
        PA = JA[27] * Q[27]; PAN = prodall(JN)
        Δ = PAN - PA
        gate = norm(Δ - (Δ * Sb) * Sb') / norm(Δ)
        sΔ = svdvals(N * Δ)
        emit(@sprintf("\nPROMPT %d : rang numérique de S = %d ; porte ‖Δ(I−Π_S)‖_F/‖Δ‖_F = %.2e ; x̂_k ∈ span V_k : min %.4f médiane %.4f",
                      p, rk, gate, minimum(cxk), median(cxk)))
        emit(@sprintf("   ‖N·P_A‖_F = %.3e  σ₁(N·P_A) = %.3e | σ₁(N·Δ) = %.3e  σ₂(N·Δ) = %.3e  ‖N·Δ‖_F = %.3e | σ₁(N·P_AN) = %.3e",
                      norm(N * PA), opnorm(N * PA), sΔ[1], sΔ[2], norm(sΔ), opnorm(N * PAN)))
        JT = loadJ(p, "T"); PT = prodall(JT); JT = nothing; GC.gc()
        for (lab, P) in (("T", PT), ("A", PA), ("AN", PAN))
            F = svd(N * P)
            fr = [sum(abs2, Sb' * F.V[:, i]) for i in 1:3]
            emit(@sprintf("   %-3s ern = %6.2f  p₁ = %.4f  fraction de v₁,v₂,v₃(N·P) dans S : %.3f %.3f %.3f   (hasard %.3f)",
                          lab, erank2(F.S), F.S[1]^2 / sum(abs2, F.S), fr..., rk / D))
        end
        # poids des termes du télescopage
        S = Matrix{Float64}(I, D, D); w = zeros(27); v1 = svd(N * PAN).V[:, 1]; a = zeros(27)
        for k in 27:-1:1
            Tk = S * (JN[k] - JA[k]) * Q[k]
            NT = N * Tk
            w[k] = norm(NT); a[k] = norm(NT * v1)
            S = S * JN[k]
        end
        emit("   poids ‖N·T_k‖_F / ‖N·Δ‖_F et ‖N·T_k v₁‖ / σ₁(N·P_AN) par couche :")
        emit("   k   : " * join([@sprintf("%6d", k) for k in 1:27]))
        emit("   ‖·‖ : " * join([@sprintf("%6.3f", v / norm(sΔ)) for v in w]))
        emit("   v₁  : " * join([@sprintf("%6.3f", v / opnorm(N * PAN)) for v in a]))
        JA = nothing; JN = nothing; Q = nothing; G = nothing; GC.gc()
    end
end
println("Écrit : ", RES)
