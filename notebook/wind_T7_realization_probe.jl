# WIND-T7 (EXPLORATOIRE) -- réalisation d'ordre r de W (Ho–Kalman variant dans le temps) et boucle fermée réduite.
using LinearAlgebra, Statistics, Printf, JSON
d = JSON.parsefile(joinpath(@__DIR__, "wind_data", "wind_T7_lift.json"))
function loadW(v)
    M = reduce(hcat, [Float64.(c) for c in v]); norm(triu(M, 1)) > norm(tril(M, -1)) ? Matrix(M') : M
end
const Lb = 27
rows(i) = 2i+1:2Lb          # couches i+1..27
cols(i) = 1:2i              # couches 1..i
function realize(W, r)
    O = Dict{Int,Matrix{Float64}}(); R = Dict{Int,Matrix{Float64}}()
    for i in 1:Lb-1
        F = svd(W[rows(i), cols(i)]); q = min(r, length(F.S))
        O[i] = F.U[:, 1:q] * Diagonal(sqrt.(F.S[1:q])); R[i] = Diagonal(sqrt.(F.S[1:q])) * F.Vt[1:q, :]
    end
    C = Dict{Int,Matrix{Float64}}(); B = Dict{Int,Matrix{Float64}}(); Φ = Dict{Int,Matrix{Float64}}()
    for i in 1:Lb-1
        C[i+1] = O[i][1:2, :]                  # lit l'état au temps i+1 (base de la coupure i)
        B[i] = R[i][:, end-1:end]              # écrit dans l'état au temps i+1
        i <= Lb-2 && (Φ[i+1] = pinv(O[i+1]) * O[i][3:end, :])   # temps i+1 -> i+2
    end
    Wf = zeros(2Lb, 2Lb)
    for k in 1:Lb-1
        ζ = B[k]
        for j in k+1:Lb
            Wf[2j-1:2j, 2k-1:2k] = C[j] * ζ
            j <= Lb-1 && (ζ = Φ[j] * ζ)
        end
    end
    Wf, C, B, Φ
end
for p in (12, 22, 17, 5, 6)
    W = loadW(d["p$p"]["W"]); G = inv(I - W); sG = svdvals(G)
    line = @sprintf("p%-3d vrai σ₁(G) = %7.2f σ₂ = %5.2f |", p, sG[1], sG[2])
    for r in (1, 2, 3, 4, 6)
        Wf, C, B, Φ = realize(W, r)
        Gf = inv(I - Wf); sf = svdvals(Gf)
        line *= @sprintf(" r=%d: errW %.2f σ₁ %7.2f σ₂ %5.2f |", r, norm(Wf - W) / norm(W), sf[1], sf[2])
    end
    println(line)
end
