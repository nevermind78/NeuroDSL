# ══════════════════════════════════════════════════════════════════════════════
# WIND-T7 (POST HOC, descriptif) -- théorème 1 appliqué au modèle d'ordre 2 avec le scindement invariant
# ADAPTÉ : pour chaque segment [a, b] du cocycle fermé 2×2, e_a = vecteur singulier droit dominant de M(b, a),
# propagé en avant (M) et en arrière (M⁻¹) ; on garde le scindement qui minimise β. Bornes de Schur explicites.
# USAGE : julia --project=. notebook/wind_T7_splitting.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates
const OUTDIR = joinpath(@__DIR__, "wind_data"); const D, Lb = 1536, 27
const RES = joinpath(@__DIR__, "wind_T7_splitting_results.txt")
rows(i) = 2i+1:2Lb; cols(i) = 1:2i
function realize(W, r)
    O = Dict{Int,Matrix{Float64}}(); R = Dict{Int,Matrix{Float64}}()
    for i in 1:Lb-1
        F = svd(W[rows(i), cols(i)]); q = min(r, length(F.S))
        O[i] = F.U[:, 1:q] * Diagonal(sqrt.(F.S[1:q])); R[i] = Diagonal(sqrt.(F.S[1:q])) * F.Vt[1:q, :]
    end
    C = Dict(i + 1 => O[i][1:2, :] for i in 1:Lb-1); B = Dict(i => R[i][:, end-1:end] for i in 1:Lb-1)
    Φ = Dict(i + 1 => pinv(O[i+1]) * O[i][3:end, :] for i in 1:Lb-2)
    C, B, Φ
end
blkmask(f) = [f(div(j + 1, 2), div(k + 1, 2)) ? 1.0 : 0.0 for j in 1:2Lb, k in 1:2Lb]
const LOW = blkmask(>); const UPD = blkmask(<=)
function schur(Z)        # borne de Schur sur les normes de blocs
    z = [opnorm(Z[2j-1:2j, 2k-1:2k]) for j in 1:Lb, k in 1:Lb]
    sqrt(maximum(sum(z, dims = 2)) * maximum(sum(z, dims = 1)))
end
function theorem1(C, B, M, a, b)
    P = Matrix{Float64}(I, 2, 2); for i in a:b-1; P = M[i] * P; end
    F = svd(P); e = Dict{Int,Vector{Float64}}(); s = Dict{Int,Vector{Float64}}()
    e[a] = F.V[:, 1]; s[a] = F.V[:, 2]
    for i in a:Lb-1; y = M[i] * e[i]; e[i+1] = y / norm(y); y = M[i] * s[i]; s[i+1] = y / norm(y); end
    for i in a-1:-1:2; y = M[i] \ e[i+1]; e[i] = y / norm(y); y = M[i] \ s[i+1]; s[i] = y / norm(y); end
    μ = Dict(i => dot(e[i+1], M[i] * e[i]) for i in 2:Lb-1)
    m = Dict(2 => 1.0); for j in 3:Lb; m[j] = m[j-1] * μ[j-1]; end
    ĉ = zeros(2Lb); b̂ = zeros(2Lb)
    for j in 2:Lb; ĉ[2j-1:2j] = C[j] * e[j] * m[j]; end
    for k in 1:Lb-1; Fd = inv(hcat(e[k+1], s[k+1])); b̂[2k-1:2k] = vec(Fd[1:1, :] * B[k]) ./ m[k+1]; end
    K = zeros(2Lb, 2Lb)
    for k in 1:Lb-1
        ζ = B[k]
        for j in k+1:Lb
            K[2j-1:2j, 2k-1:2k] = C[j] * ζ; j <= Lb-1 && (ζ = M[j] * ζ)
        end
    end
    H = ĉ * b̂'; U = H .* UPD; Ks = K - H .* LOW
    G = I + K
    gate = norm(G - (H + I - U + Ks)) / norm(G)
    β = opnorm(I - U + Ks); A = norm(ĉ) * norm(b̂)
    (A = A, β = β, βschur = 1 + schur(U) + schur(Ks), gate = gate, sG = svdvals(G), μ = μ)
end
open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T7 théorème 1, scindement adapté (POST HOC)")
    for p in (12, 22, 17, 5, 6)
        buf = Array{Float64}(undef, D * 54 * 2 + D * D + 54 * 54)
        open(joinpath(OUTDIR, "wind_T7_lift_p$(p).bin"), "r") do f; read!(f, buf); end
        W = reshape(buf[end-54*54+1:end], 54, 54)
        C, B, Φ = realize(W, 2)
        M = Dict(i => Φ[i] + B[i] * C[i] for i in 2:Lb-1)
        best = nothing; ab = (0, 0)
        for a in 2:Lb-2, b in a+2:Lb
            r = theorem1(C, B, M, a, b)
            if best === nothing || r.β < best.β; best = r; ab = (a, b); end
        end
        r = best
        emit(@sprintf("p%-3d segment [%d, %d] : A = %.2f  β = %.2f (Schur ≤ %.2f)  A/β = %.2f | σ₁(G₂) = %.2f ∈ [%.2f, %.2f] ; σ₂(G₂) = %.2f ≤ %.2f ; porte %.1e",
                      p, ab..., r.A, r.β, r.βschur, r.A / r.β, r.sG[1], r.A - r.β, r.A + r.β, r.sG[2], r.β, r.gate))
        emit("      μ_i : " * join([@sprintf("%.2f", r.μ[i]) for i in 2:Lb-1], " "))
    end
end
println("Écrit : ", RES)
