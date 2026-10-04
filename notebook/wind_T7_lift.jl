# ══════════════════════════════════════════════════════════════════════════════
# WIND-T7 (EXPLORATOIRE) -- relèvement exact de la chaîne à normes figées en boucle fermée.
#   R_k = J_k^AN − J_k^A = U_k V_kᵀ (rang 2, U_k = U·Σ, V_k orthonormé)
#   P_AN = P_A + 𝒰 (I − W)⁻¹ 𝒱,   W_{jk} = V_jᵀ A_{j−1}···A_{k+1} U_k (j > k), nilpotent
#   𝒰_k = N·A_27···A_{k+1} U_k (n×2),  𝒱_k = V_kᵀ A_{k−1}···A_1 (2×n)
# Mesures : porte de l'identité, poids premier ordre (G = I) vs résonance (G − I), spectre de G,
# semi-séparabilité de W, gains d'état λ_A, λ_AN et identité λ_AN − λ_A = x̂_{k+1}ᵀ R_k x̂_k.
# USAGE : julia --project=. notebook/wind_T7_lift.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = joinpath(@__DIR__, "wind_data")
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const RES = joinpath(@__DIR__, get(ENV, "WIND_RESNAME", "wind_T7_lift_results.txt"))
const JOUT = joinpath(OUTDIR, "wind_T7_lift.json")
const L, D = 28, 1536
BLAS.set_num_threads(16)
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
τf(s) = sqrt(sum(abs2, s[2:end]))
out = Dict{String,Any}()

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T7 relèvement exact (boucle fermée des normes) -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PROMPT_IDXS
        t0 = time()
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        X = [Float64.(v) for v in meta["X"]]; xh(k) = X[k+1] ./ norm(X[k+1])
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x = X[29]
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        JA = loadJ(p, "A"); JN = loadJ(p, "AN")
        Uk = Vector{Matrix{Float64}}(undef, 27); Vk = Vector{Matrix{Float64}}(undef, 27); s3 = zeros(27)
        for k in 1:27
            F = svd(JN[k] - JA[k]); Uk[k] = F.U[:, 1:2] * Diagonal(F.S[1:2]); Vk[k] = F.V[:, 1:2]; s3[k] = F.S[3] / F.S[1]
        end
        # 𝒱 (avant) et P_A
        𝒱 = zeros(54, D); Q = Matrix{Float64}(I, D, D)
        for k in 1:27
            𝒱[2k-1:2k, :] = Vk[k]' * Q; Q = JA[k] * Q
        end
        PA = Q; Q = nothing
        # 𝒰 (arrière)
        𝒰 = zeros(D, 54); S = copy(N)
        for k in 27:-1:1
            𝒰[:, 2k-1:2k] = S * Uk[k]; S = S * JA[k]
        end
        S = nothing
        # W
        W = zeros(54, 54)
        for k in 1:26
            y = Uk[k]
            for j in k+1:27
                W[2j-1:2j, 2k-1:2k] = Vk[j]' * y
                j < 27 && (y = JA[j] * y)
            end
        end
        G = inv(I - W)
        PN = JN[1]; for k in 2:27; PN = JN[k] * PN; end
        MA = N * PA; MN = N * PN
        Mlift = MA + 𝒰 * G * 𝒱
        gate = norm(Mlift - MN) / norm(MN)
        F1 = 𝒰 * 𝒱; Fres = 𝒰 * (G - I) * 𝒱
        sA = svdvals(MA); sN = svdvals(MN); s1 = svdvals(MA + F1); sF = svdvals(F1); sR = svdvals(Fres); sX = svdvals(MN - MA)
        sG = svdvals(G); sW = svdvals(W)
        emit(@sprintf("\n==================== PROMPT %d  (σ₃/σ₁(R_k) max %.1e) ====================", p, maximum(s3)))
        emit(@sprintf("  porte : ‖N P_A + 𝒰G𝒱 − N P_AN‖/‖N P_AN‖ = %.2e", gate))
        emit(@sprintf("  N·P_A : σ₁ = %.3e τ = %.2f ern = %.2f | N·P_AN : σ₁ = %.3e τ = %.2f ern = %.2f", sA[1], τf(sA), ent(sA), sN[1], τf(sN), ent(sN)))
        emit(@sprintf("  ordre 1 (G = I) : σ₁(𝒰𝒱) = %.3e  σ₂/σ₁ = %.3f ; ern(N P_A + 𝒰𝒱) = %.2f", sF[1], sF[2] / sF[1], ent(s1)))
        emit(@sprintf("  résonance 𝒰(G−I)𝒱 : σ₁ = %.3e  σ₂/σ₁ = %.3f ; extra total σ₁ = %.3e σ₂/σ₁ = %.3f", sR[1], sR[2] / sR[1], sX[1], sX[2] / sX[1]))
        emit(@sprintf("  ‖W‖ = %.3f  ‖G‖ = %.3f  σ₂(G) = %.3f  σ₃(G) = %.3f  ‖G − I − W‖ = %.3f", sW[1], sG[1], sG[2], sG[3], opnorm(G - I - W)))
        # semi-séparabilité de W : blocs hors-diagonale W[2i+1:end, 1:2i]
        ss = Float64[]
        for i in 2:25
            sb = svdvals(W[2i+1:end, 1:2i]); push!(ss, sb[2] / sb[1])
        end
        emit(@sprintf("  semi-séparabilité de W : σ₂/σ₁ des blocs W[>i, ≤i], i=2..25 : médiane %.3f [min %.3f, max %.3f]", median(ss), minimum(ss), maximum(ss)))
        # gains d'état
        λA = [dot(xh(k + 1), JA[k] * xh(k)) for k in 1:27]
        λN = [dot(xh(k + 1), JN[k] * xh(k)) for k in 1:27]
        idn = maximum(abs(λN[k] - λA[k] - dot(xh(k + 1), (JN[k] - JA[k]) * xh(k))) for k in 1:27)
        emit(@sprintf("  λ_A médian %.3f, λ_AN médian %.3f ; Σlog10 λ_A = %.2f, Σlog10 λ_AN = %.2f ; identité λ_AN−λ_A : err %.1e",
                      median(λA), median(λN), sum(log10.(abs.(λA))), sum(log10.(abs.(λN))), idn))
        emit("  λ_AN k=1..27 : " * join([@sprintf("%.2f", v) for v in λN], " "))
        # bloc diagonal le plus fort de G par distance
        gd = [maximum(opnorm(G[2j-1:2j, 2k-1:2k]) for j in 1:27, k in 1:27 if j - k == d; init = 0.0) for d in 1:26]
        emit("  max_‖G_{j,j−d}‖ d=1..26 : " * join([@sprintf("%.2g", v) for v in gd], " "))
        out["p$p"] = Dict("W" => W, "G_svals" => sG, "lamA" => λA, "lamN" => λN, "gate" => gate,
                          "s1A" => sA[1], "tauA" => τf(sA), "s1N" => sN[1], "tauN" => τf(sN),
                          "s1F1" => sF[1], "s1Res" => sR[1], "s1X" => sX[1])
        emit(@sprintf("  (%.0f s)", time() - t0))
        JA = nothing; JN = nothing; GC.gc()
    end
end
open(JOUT, "w") do f; JSON.print(f, out); end
println("Écrit : ", RES)
