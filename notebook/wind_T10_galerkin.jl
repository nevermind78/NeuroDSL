# ══════════════════════════════════════════════════════════════════════════════
# WIND-T10 (EXPLORATOIRE) -- la boucle des normes figées est-elle d'ordre 2 parce que son état est la lecture
# des deux « capteurs » de norme ? Réduction de Galerkin : W_G[j,k] = Φ_{j−1}···Φ_{k+1} B_k,
# Φ_l = V_{l+1}ᵀ A_l V_l, B_k = V_{k+1}ᵀ U_k (2×2), exacte si A_lᵀ span V_{l+1} ⊆ span V_l.
# Mesures : spectre de Hankel de W par coupure ; écart ‖W − W_G‖/‖W‖ ; fermeture adjointe
# κ_l = ‖Π_{V_l} A_lᵀ V_{l+1}‖²/‖A_lᵀ V_{l+1}‖² ; θ_c du modèle de Galerkin vs vérité WIND-T8.
# USAGE : WIND_PROMPTS=… julia --project=. notebook/wind_T10_galerkin.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = joinpath(@__DIR__, "wind_data_T8")
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", "8,13,25,30,43,1,6,16,19,34"), ","))
const RES = get(ENV, "WIND_T10_RES", joinpath(@__DIR__, "wind_T10_galerkin_results.txt"))
const JOUT = get(ENV, "WIND_T10_OUT", joinpath(@__DIR__, "wind_T10_galerkin.json"))
const L, D, Lb = 28, 1536, 27
const THETAS = collect(0.25:0.05:2.5)
BLAS.set_num_threads(parse(Int, get(ENV, "WIND_BLAS", "16")))
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
θcross(θs, e) = (i = findfirst(<=(3.0), e); i === nothing ? Inf : i == 1 ? θs[1] :
                 θs[i-1] + (θs[i] - θs[i-1]) * (e[i-1] - 3) / (e[i-1] - e[i]))
rows(i) = 2i+1:2Lb; cols(i) = 1:2i

truth = JSON.parsefile(joinpath(@__DIR__, "wind_T8_truth.json"))
pred = JSON.parsefile(joinpath(@__DIR__, "wind_T8_predictions.json"))["predictions"]
out = isfile(JOUT) ? JSON.parsefile(JOUT) : Dict{String,Any}()
open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T10 (EXPLORATOIRE) réduction de Galerkin sur les plans-capteurs -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PIDX
        haskey(out, string(p)) && continue
        t0 = time()
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x = Float64.(meta["X"][end])
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        X = [Float64.(v) for v in meta["X"]]                       # X[k+1] = x_k
        JA = loadJ(p, "A"); JN = loadJ(p, "AN")
        Uk = Vector{Matrix{Float64}}(undef, 27); Vk = Vector{Matrix{Float64}}(undef, 27)
        for k in 1:27
            F = svd(JN[k] - JA[k]); Uk[k] = F.U[:, 1:2] * Diagonal(F.S[1:2]); Vk[k] = F.V[:, 1:2]
        end
        JN = nothing; GC.gc()
        xin = [abs(dot(Vk[k]' * (X[k+1] ./ norm(X[k+1])), Vk[k]' * (X[k+1] ./ norm(X[k+1])))) for k in 1:27]  # ‖Π_V x̂_k‖²
        # relèvement exact (comme wind_T8_predict.jl)
        𝒱 = zeros(54, D); Q = Matrix{Float64}(I, D, D)
        for k in 1:27; 𝒱[2k-1:2k, :] = Vk[k]' * Q; Q = JA[k] * Q; end
        MA = N * Q; Q = nothing
        𝒰 = zeros(D, 54); S = copy(N)
        for k in 27:-1:1; 𝒰[:, 2k-1:2k] = S * Uk[k]; S = S * JA[k]; end
        S = nothing
        W = zeros(54, 54)
        for k in 1:26
            y = Uk[k]
            for j in k+1:27
                W[2j-1:2j, 2k-1:2k] = Vk[j]' * y
                j < 27 && (y = JA[j] * y)
            end
        end
        # Galerkin 2×2 et fermeture adjointe
        Φ = [Vk[l+1]' * JA[l] * Vk[l] for l in 1:26]
        κ = [(Z = JA[l]' * Vk[l+1]; sum(abs2, Vk[l]' * Z) / sum(abs2, Z)) for l in 1:26]
        JA = nothing; GC.gc()
        B = [Vk[k+1]' * Uk[k] for k in 1:26]
        WG = zeros(54, 54)
        for k in 1:26
            y = B[k]
            for j in k+1:27
                WG[2j-1:2j, 2k-1:2k] = y
                j < 27 && (y = Φ[j] * y)
            end
        end
        relG = norm(W - WG) / norm(W)
        hk = [svdvals(W[rows(i), cols(i)]) for i in 1:Lb-1]
        h3 = median([length(s) >= 3 ? s[3] / s[1] : 0.0 for s in hk]); h2 = median([s[2] / s[1] for s in hk])
        curve = [ent(svdvals(MA + θ .* (𝒰 * inv(I - θ .* WG) * 𝒱))) for θ in THETAS]
        θG = θcross(THETAS, curve)
        tT = truth[string(p)]["theta_c_true"]; tM = pred[string(p)]["theta_c_mod2"]
        out[string(p)] = Dict("theta_c_galerkin" => θG, "theta_c_true" => tT, "theta_c_HK2" => tM, "relG" => relG,
                              "kappa" => κ, "x_in_V" => xin, "hankel_s2_s1_med" => h2, "hankel_s3_s1_med" => h3,
                              "hankel" => [s[1:min(4, end)] for s in hk])
        open(JOUT, "w") do f; JSON.print(f, out); end
        emit(@sprintf("  p%-3d θ_c vrai %.3f | Ho–Kalman ordre 2 %.3f | Galerkin capteurs %s | ‖W−W_G‖/‖W‖ %.3f | κ médian %.3f [min %.3f] | x̂∈V %.4f | Hankel σ₂/σ₁ %.3f σ₃/σ₁ %.3f | %.0f s",
                      p, tT, tM, isfinite(θG) ? @sprintf("%.3f", θG) : "> 2,5", relG, median(κ), minimum(κ), minimum(xin), h2, h3, time() - t0))
        GC.gc()
    end
end
println("Écrit : ", RES)
