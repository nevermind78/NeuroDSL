# WIND-T7 -- contrôle synthétique du théorème 2 (limite stationnaire, constante exacte) et du théorème 1 (encadrement).
using LinearAlgebra, Random, Printf
rng = MersenneTwister(7)
p = 2
T = randn(rng, 2, 2); M = T * Diagonal([1.25, -0.55]) / T          # λ_u = 1,25, λ_s = −0,55
C = randn(rng, p, 2); B = randn(rng, 2, p)
Fe = eigen(M); iu = argmax(abs.(Fe.values)); e = real(Fe.vectors[:, iu]); e ./= norm(e)
s = real(Fe.vectors[:, 3-iu]); s ./= norm(s); FG = inv(hcat(e, s)); f = FG[1, :]
λ = real(Fe.values[iu])
const_th = norm(C * e) * norm(f' * B) / (λ^2 - 1)
println(@sprintf("constante prédite ‖Ce‖‖fᵀB‖/(λ_u²−1) = %.6f", const_th))
for Lb in (20, 40, 60, 80)   # L ≤ 80 : zone Float64 fiable (σ₁ ≤ 1e8)
    G = Matrix{Float64}(I, p * Lb, p * Lb)
    for k in 1:Lb-1
        ζ = B
        for j in k+1:Lb
            G[p*(j-1)+1:p*j, p*(k-1)+1:p*k] = C * ζ; ζ = M * ζ
        end
    end
    sv = svdvals(G)
    println(@sprintf("L = %3d : σ₁·|λ_u|^{−L} = %.6f   σ₂ = %.4f   σ₃ = %.4f", Lb, sv[1] / abs(λ)^Lb, sv[2], sv[3]))
end
# β exact, borne de Schur explicite (théorème 1(c)) et σ₂, pour L croissant
schur_u = norm(C * e) * norm(f' * B) / (abs(λ) - 1)
λs = real(Fe.values[3-iu]); gs = FG[2, :]
schur_s = norm(C * s) * norm(gs' * B) / (1 - abs(λs))
println(@sprintf("borne uniforme de Schur : β ≤ 1 + %.4f + %.4f = %.4f", schur_u, schur_s, 1 + schur_u + schur_s))
for Lb in (20, 40, 60, 80)  # au-delà, σ₁ > 1e8 : σ₂ par SVD Float64 non fiable
    n = p * Lb
    ĉ = zeros(n); b̂ = zeros(n); Ks = zeros(n, n); K = zeros(n, n)
    for j in 2:Lb; ĉ[p*(j-1)+1:p*j] = C * e * λ^(j-1); end
    for k in 1:Lb-1; b̂[p*(k-1)+1:p*k] = vec(f' * B) .* λ^(-k); end
    for k in 1:Lb-1
        ζ = B; ζs = s * (gs' * B)
        for j in k+1:Lb
            K[p*(j-1)+1:p*j, p*(k-1)+1:p*k] = C * ζ; Ks[p*(j-1)+1:p*j, p*(k-1)+1:p*k] = C * ζs
            ζ = M * ζ; ζs = M * ζs
        end
    end
    H = ĉ * b̂'; blk(j) = div(j - 1, p) + 1
    Uup = [blk(j) <= blk(k) ? H[j, k] : 0.0 for j in 1:n, k in 1:n]
    G = I + K; Y = I - Uup + Ks
    β = opnorm(Y); sv = svdvals(G)
    println(@sprintf("L = %3d : porte ‖G − H − Y‖ = %.1e ; σ₂ = %.4f ≤ β = %.4f ; |σ₁ − ‖ĉ‖‖b̂‖| = %.4f", Lb, norm(G - H - Y), sv[2], β, abs(sv[1] - norm(ĉ) * norm(b̂))))
end
