# ══════════════════════════════════════════════════════════════════════════════
# WIND-T12 (EXPLORATOIRE) -- caractériser les deux résonances (précoce : prompts effondrés ; tardive : θ_c ≈ 1,5).
# Au seuil θ_c (vraie chaîne J_l(θ) = A_l + θR_l), avec le chemin dominant (f, w) et r_l (WIND-T11) :
#  (1) coupure par capteur : R_l = c_x x̂_lᵀ + c_h ĥ_lᵀ (base duale de (x̂_l, ĥ_l) dans l'espace des lignes de R_l)
#      r_l = r_l^x + r_l^h ; x̂ = capteur de la norme d'entrée de l'ATTENTION, ĥ = celle du MLP.
#      Contrôle : ρ_l = ‖R_l (I − Π_{x̂,ĥ})‖/‖R_l‖ (doit être petit pour que la coupure ait un sens).
#  (2) réponse d'échelle : cos(u₁, z_E) et cos(u₁, z_T), z = N Σ_k J_27···J_{k+1} Δ_k sur k ≤ 10 (E) ou 11..26 (T).
# USAGE : WIND_PROMPTS=… julia --project=. notebook/wind_T12_resonances.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = joinpath(@__DIR__, "wind_data_T8")
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", "8,13,25,30,43,1,6,16,19,34"), ","))
const RES = joinpath(@__DIR__, "wind_T12_resonances_results.txt")
const L, D = 28, 1536
BLAS.set_num_threads(16)
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])                       # J_1 .. J_27
truth = JSON.parsefile(joinpath(@__DIR__, "wind_T8_truth.json"))

open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T12 (EXPLORATOIRE) deux résonances : capteur attention (x̂) vs capteur MLP (ĥ), réponse d'échelle -- " *
         Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    emit("  r^x = part du capteur de la norme d'ATTENTION, r^h = part du capteur de la norme du MLP ; E = couches 1–10, T = 11–27")
    for p in PIDX
        t0 = time()
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; X = [Float64.(v) for v in meta["X"]]
        H = [Float64.(v) for v in meta["H"]]                               # H[l] = h de la couche l (bloc k = l − 1)
        x = X[end]; N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        JA = loadJ(p, "A"); JN = loadJ(p, "AN"); R = [JN[k] - JA[k] for k in 1:27]; JN = nothing; GC.gc()
        θ = min(truth[string(p)]["theta_c_true"], 2.5)
        J = [JA[k] + θ .* R[k] for k in 1:27]; JA = nothing; GC.gc()
        P = J[1]; for k in 2:27; P = J[k] * P; end
        F = svd(N * P); u = F.U[:, 1]; v = F.V[:, 1]; s1 = F.S[1]; P = nothing
        f = Vector{Vector{Float64}}(undef, 28); f[1] = v
        for l in 1:27; f[l+1] = J[l] * f[l]; end
        w = Vector{Vector{Float64}}(undef, 27); y = N' * u
        for l in 27:-1:1; w[l] = y; y = J[l]' * y; end
        rx = zeros(27); rh = zeros(27); r = zeros(27); ρ = zeros(27)
        for k in 1:27
            xh = X[k+1] ./ norm(X[k+1]); hh = H[k+1] ./ norm(H[k+1])           # bloc k = couche k+1
            B = hcat(xh, hh); E = B / (B' * B)                                    # base duale : Bᵀ E = I
            Q = Matrix(qr(B).Q)[:, 1:2]
            ρ[k] = norm(R[k] - (R[k] * Q) * Q') / norm(R[k])
            C = R[k] * E                                                          # R ≈ C Bᵀ
            rx[k] = dot(w[k], C[:, 1]) * dot(xh, f[k]) / s1
            rh[k] = dot(w[k], C[:, 2]) * dot(hh, f[k]) / s1
            r[k] = dot(w[k], R[k] * f[k]) / s1
        end
        # réponse d'échelle des mises à jour précoces / tardives, propagée par la vraie chaîne à θ_c
        zE = zeros(D); zT = zeros(D)
        for k in 1:26
            d = X[k+2] .- X[k+1]                                                  # Δ_k = x_{k+1} − x_k
            for j in k+1:27; d = J[j] * d; end
            k <= 10 ? (zE .+= d) : (zT .+= d)
        end
        zE = N * zE; zT = N * zT
        cE = abs(dot(u, zE)) / norm(zE); cT = abs(dot(u, zT)) / norm(zT); cA = abs(dot(u, zE .+ zT)) / norm(zE .+ zT)
        emit(@sprintf("  p%-3d θ_c %.3f | Σr %.2f (= r^x + r^h + reste %.2f) | ρ médian %.3f | E : r^x %.2f r^h %.2f | T : r^x %.2f r^h %.2f | cos(u₁,z_E) %.3f cos(u₁,z_T) %.3f cos(u₁,z_tout) %.3f | %.0f s",
                      p, θ, sum(r), sum(r) - sum(rx) - sum(rh), median(ρ), sum(rx[1:10]), sum(rh[1:10]), sum(rx[11:27]), sum(rh[11:27]),
                      cE, cT, cA, time() - t0))
        emit("        r^x : " * join([@sprintf("%.2f", a) for a in rx], " "))
        emit("        r^h : " * join([@sprintf("%.2f", a) for a in rh], " "))
        J = nothing; R = nothing; GC.gc()
    end
end
println("Écrit : ", RES)
