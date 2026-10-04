# ══════════════════════════════════════════════════════════════════════════════
# WIND-T11 (EXPLORATOIRE) -- pourquoi le seuil θ_c est-il robuste à la troncature ?
# Identité exacte (Hellmann–Feynman pour un produit, σ₁ simple) sur la VRAIE chaîne J_l(θ) = A_l + θR_l :
#   d log σ₁(N·P(θ))/dθ = Σ_l r_l,   r_l = (w_lᵀ R_l f_{l−1}) / (w_lᵀ J_l f_{l−1}),
#   f_{l} = J_l···J_1 v (f_0 = v), w_l = (N J_27···J_{l+1})ᵀ u ; (u, v) = vecteurs singuliers dominants.
# Porte : Σ r_l vs différence finie centrée de log σ₁. Puis : répartition des r_l sur les couches.
# USAGE : WIND_PROMPTS=… julia --project=. notebook/wind_T11_slope.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = joinpath(@__DIR__, "wind_data_T8")
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", "8,13,25,30,43,1,6,16,19,34"), ","))
const RES = joinpath(@__DIR__, "wind_T11_slope_results.txt")
const L, D = 28, 1536
BLAS.set_num_threads(16)
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
truth = JSON.parsefile(joinpath(@__DIR__, "wind_T8_truth.json"))
prod_at(JA, R, θ) = (P = JA[1] + θ .* R[1]; for k in 2:27; P = (JA[k] + θ .* R[k]) * P; end; P)

open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T11 (EXPLORATOIRE) pente du mode dominant = somme des gains locaux -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PIDX
        t0 = time()
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x = Float64.(meta["X"][end])
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        JA = loadJ(p, "A"); JN = loadJ(p, "AN"); R = [JN[k] - JA[k] for k in 1:27]; JN = nothing; GC.gc()
        θ = min(truth[string(p)]["theta_c_true"], 2.5)
        J = [JA[k] + θ .* R[k] for k in 1:27]
        F = svd(N * prod_at(JA, R, θ)); u = F.U[:, 1]; v = F.V[:, 1]; s1 = F.S[1]
        f = Vector{Vector{Float64}}(undef, 28); f[1] = v
        for l in 1:27; f[l+1] = J[l] * f[l]; end
        w = Vector{Vector{Float64}}(undef, 27); y = N' * u
        for l in 27:-1:1; w[l] = y; y = J[l]' * y; end
        r = [dot(w[l], R[l] * f[l]) / dot(w[l], J[l] * f[l]) for l in 1:27]
        cons = maximum(abs(dot(w[l], J[l] * f[l]) - s1) / s1 for l in 1:27)      # w_lᵀ J_l f_{l−1} = σ₁ ∀l
        h = 1e-3
        fd = (log(svdvals(N * prod_at(JA, R, θ + h))[1]) - log(svdvals(N * prod_at(JA, R, θ - h))[1])) / (2h)
        tot = sum(r); pos = sum(max.(r, 0));
        srt = sort(abs.(r), rev = true); k80 = findfirst(>=(0.8 * sum(abs.(r))), cumsum(srt))
        emit(@sprintf("  p%-3d θ_c %.3f | Σr_l %.3f vs DF %.3f (porte %.1e ; conservation %.1e) | couches 1–10 : %.3f, 11–27 : %.3f | %d couches portent 80 %% de Σ|r| | max r %.3f @l=%d | %.0f s",
                      p, θ, tot, fd, abs(tot - fd) / abs(fd), cons, sum(r[1:10]), sum(r[11:27]), k80, maximum(r), argmax(r), time() - t0))
        emit("        r_l : " * join([@sprintf("%.2f", x) for x in r], " "))
        JA = nothing; R = nothing; J = nothing; GC.gc()
    end
end
println("Écrit : ", RES)
