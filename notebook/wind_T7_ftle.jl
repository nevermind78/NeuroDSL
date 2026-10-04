# WIND-T7 (POST HOC) -- exposants de Lyapunov en temps fini du cocycle fermé 2×2 M_i(θ) = Φ_i + θ B_i C_i,
# maximisés sur les segments [a, b] (b − a ≥ 4) ; et nombre de valeurs singulières de G(θ) vraie > 5.
using LinearAlgebra, Printf
const OUTDIR = joinpath(@__DIR__, "wind_data"); const D, Lb = 1536, 27
rows(i) = 2i+1:2Lb; cols(i) = 1:2i
function realize(W, r)
    O = Dict{Int,Matrix{Float64}}(); R = Dict{Int,Matrix{Float64}}()
    for i in 1:Lb-1
        F = svd(W[rows(i), cols(i)]); q = min(r, length(F.S))
        O[i] = F.U[:, 1:q] * Diagonal(sqrt.(F.S[1:q])); R[i] = Diagonal(sqrt.(F.S[1:q])) * F.Vt[1:q, :]
    end
    Dict(i + 1 => O[i][1:2, :] for i in 1:Lb-1), Dict(i => R[i][:, end-1:end] for i in 1:Lb-1),
    Dict(i + 1 => pinv(O[i+1]) * O[i][3:end, :] for i in 1:Lb-2)
end
open(joinpath(@__DIR__, "wind_T7_ftle_results.txt"), "w") do io
    emit(s) = (println(io, s); println(s))
    emit("WIND-T7 (POST HOC) : max sur segments [a,b] (b−a ≥ 4) de ln σ₁/(b−a) et ln σ₂/(b−a) du cocycle fermé 2×2 ; #σ(G_vraie(θ)) > 5")
    for p in (12, 22, 17, 5, 6)
        buf = Array{Float64}(undef, D * 54 * 2 + D * D + 54 * 54)
        open(joinpath(OUTDIR, "wind_T7_lift_p$(p).bin"), "r") do f; read!(f, buf); end
        W = reshape(buf[end-54*54+1:end], 54, 54)
        C, B, Φ = realize(W, 2)
        line = @sprintf("p%-3d", p)
        for θ in (1.0, 1.5, 2.0)
            M = Dict(i => Φ[i] + θ .* (B[i] * C[i]) for i in 2:Lb-1)
            l1 = -Inf; l2 = -Inf
            for a in 2:Lb-5, b in a+4:Lb
                P = Matrix{Float64}(I, 2, 2); for i in a:b-1; P = M[i] * P; end
                s = svdvals(P); l1 = max(l1, log(s[1]) / (b - a)); l2 = max(l2, log(s[2]) / (b - a))
            end
            nbig = count(>(5), svdvals(inv(I - θ .* W)))
            line *= @sprintf(" | θ=%.1f : λ₁* = %+.3f λ₂* = %+.3f  #σ(G)>5 = %d", θ, l1, l2, nbig)
        end
        emit(line)
    end
end
