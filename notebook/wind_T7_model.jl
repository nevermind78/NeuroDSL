# ══════════════════════════════════════════════════════════════════════════════
# WIND-T7 -- côté MODÈLE uniquement (aucun produit vrai à θ ≠ 1 n'est calculé ici).
#  1. relèvement : 𝒰 (n×54), 𝒱 (54×n), M_A = N·P_A, W (54×54) ; sauvegarde pour le balayage.
#  2. réalisation d'ordre 2 de W (Ho–Kalman variant dans le temps) : C_j, Φ_i, B_k ; cocycle fermé M_i = Φ_i + B_i C_i.
#  3. théorème de dédoublement : scindement dominé fini (SVD de M(27,2)), μ_i, ĉ, b̂, A = ‖ĉ‖‖b̂‖,
#     β = ‖I − U(ĉb̂) + K^s‖ ; bornes |σ₁(G) − A| ≤ β, σ₂(G) ≤ β (exactes pour le modèle).
#  4. courbe modèle θ ↦ ern(M_A + θ𝒰(I − θW₂)⁻¹𝒱) et gain critique θ_c^mod (ern ≤ 3).
# USAGE : julia --project=. notebook/wind_T7_model.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = joinpath(@__DIR__, "wind_data")
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "12,22,17,5,6"), ","))
const RES = joinpath(@__DIR__, "wind_T7_model_results.txt")
const L, D, Lb = 28, 1536, 27
const THETAS = collect(0.25:0.05:3.0)
BLAS.set_num_threads(16)
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
rows(i) = 2i+1:2Lb; cols(i) = 1:2i

function realize(W, r)
    O = Dict{Int,Matrix{Float64}}(); R = Dict{Int,Matrix{Float64}}()
    for i in 1:Lb-1
        F = svd(W[rows(i), cols(i)]); q = min(r, length(F.S))
        O[i] = F.U[:, 1:q] * Diagonal(sqrt.(F.S[1:q])); R[i] = Diagonal(sqrt.(F.S[1:q])) * F.Vt[1:q, :]
    end
    C = Dict{Int,Matrix{Float64}}(); B = Dict{Int,Matrix{Float64}}(); Φ = Dict{Int,Matrix{Float64}}()
    for i in 1:Lb-1
        C[i+1] = O[i][1:2, :]; B[i] = R[i][:, end-1:end]
        i <= Lb-2 && (Φ[i+1] = pinv(O[i+1]) * O[i][3:end, :])
    end
    C, B, Φ
end
function transfer(C, B, M)        # K_{jk} = C_j M_{j−1}···M_{k+1} B_k (j > k)
    K = zeros(2Lb, 2Lb)
    for k in 1:Lb-1
        ζ = B[k]
        for j in k+1:Lb
            K[2j-1:2j, 2k-1:2k] = C[j] * ζ
            j <= Lb-1 && (ζ = M[j] * ζ)
        end
    end
    K
end

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T7 côté modèle (ordre 2) -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PROMPT_IDXS
        t0 = time()
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x = Float64.(meta["X"][end])
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        JA = loadJ(p, "A"); JN = loadJ(p, "AN")
        Uk = Vector{Matrix{Float64}}(undef, 27); Vk = Vector{Matrix{Float64}}(undef, 27)
        for k in 1:27
            F = svd(JN[k] - JA[k]); Uk[k] = F.U[:, 1:2] * Diagonal(F.S[1:2]); Vk[k] = F.V[:, 1:2]
        end
        JN = nothing; GC.gc()
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
        JA = nothing; GC.gc()
        open(joinpath(OUTDIR, "wind_T7_lift_p$(p).bin"), "w") do f; write(f, 𝒰); write(f, 𝒱); write(f, MA); write(f, W); end
        G = inv(I - W); sG = svdvals(G)
        # ── réalisation d'ordre 2 et cocycle fermé ─────────────────────────────
        C, B, Φ = realize(W, 2)
        M = Dict(i => Φ[i] + B[i] * C[i] for i in 2:Lb-1)
        Gm = I + transfer(C, B, M)
        Wf = transfer(C, B, Φ)
        gchk = norm(Gm - inv(I - Wf)) / norm(Gm)
        sGm = svdvals(Gm)
        # scindement dominé fini : e_i = M(i,2)v₁, s_i = M(i,2)v₂
        P = Matrix{Float64}(I, 2, 2); for i in 2:Lb-1; P = M[i] * P; end
        Fv = svd(P); v1 = Fv.V[:, 1]; v2 = Fv.V[:, 2]
        e = Dict{Int,Vector{Float64}}(); s = Dict{Int,Vector{Float64}}(); μ = Dict{Int,Float64}(); ν = Dict{Int,Float64}()
        a = v1; b = v2
        for i in 2:Lb
            e[i] = a / norm(a); s[i] = b / norm(b)
            if i <= Lb-1
                an = M[i] * e[i]; bn = M[i] * s[i]
                μ[i] = norm(an) * sign(dot(an, an)); ν[i] = norm(bn)
                a = an; b = bn
            end
        end
        # signe de μ_i : M_i e_i = μ_i e_{i+1} avec e_{i+1} = M_i e_i/‖M_i e_i‖ -> μ_i > 0 par construction
        m = Dict{Int,Float64}(2 => 1.0); for j in 3:Lb; m[j] = m[j-1] * μ[j-1]; end
        ĉ = zeros(54); b̂ = zeros(54)
        for j in 2:Lb
            ĉ[2j-1:2j] = C[j] * e[j] * m[j]
        end
        for k in 1:Lb-1
            E = hcat(e[k+1], s[k+1]); Fd = inv(E)          # lignes : f_{k+1}ᵀ, g_{k+1}ᵀ
            b̂[2k-1:2k] = vec(Fd[1:1, :] * B[k]) ./ m[k+1]
        end
        H = ĉ * b̂'
        Lmask = [j > k ? 1.0 : 0.0 for j in 1:54, k in 1:54] .* [div(j + 1, 2) > div(k + 1, 2) ? 1.0 : 0.0 for j in 1:54, k in 1:54]
        Ku = H .* Lmask
        Ks = (Gm - I) - Ku
        Uup = H .* (1 .- Lmask)
        β = opnorm(I - Uup + Ks); Aamp = norm(ĉ) * norm(b̂)
        lyap = (log(Fv.S[1]) / 25, log(Fv.S[2]) / 25)
        emit(@sprintf("\n==================== PROMPT %d ====================", p))
        emit(@sprintf("  vrai : σ₁(G) = %.2f σ₂(G) = %.2f | modèle ordre 2 : σ₁ = %.2f σ₂ = %.2f (contrôle boucle fermée %.1e)",
                      sG[1], sG[2], sGm[1], sGm[2], gchk))
        emit(@sprintf("  cocycle fermé 2×2 : σ(M(27,2)) = %.3e, %.3e ; exposants de Lyapunov en temps fini (par couche) %.3f, %.3f",
                      Fv.S[1], Fv.S[2], lyap...))
        emit("  μ_i (i=2..26) : " * join([@sprintf("%.2f", μ[i]) for i in 2:Lb-1], " "))
        emit("  ν_i (i=2..26) : " * join([@sprintf("%.2f", ν[i]) for i in 2:Lb-1], " "))
        emit(@sprintf("  dédoublement (modèle) : A = ‖ĉ‖‖b̂‖ = %.2f ; β = %.2f ; A/β = %.2f ; bornes : |σ₁ − A| ≤ β -> %s ; σ₂ ≤ β -> %s",
                      Aamp, β, Aamp / β, abs(sGm[1] - Aamp) <= β + 1e-9 ? "OK" : "VIOLÉ", sGm[2] <= β + 1e-9 ? "OK" : "VIOLÉ"))
        # ── courbe θ du modèle ─────────────────────────────────────────────────
        ern_m = Float64[]; b_m = Float64[]; nG = Float64[]
        for θ in THETAS
            Gθ = inv(I - θ * Wf)
            Mθ = MA + θ * (𝒰 * Gθ * 𝒱)
            sv = svdvals(Mθ)
            push!(ern_m, ent(sv)); push!(b_m, sv[1] / sqrt(sum(abs2, sv[2:end]))); push!(nG, opnorm(Gθ))
        end
        ic = findfirst(<=(3.0), ern_m)
        θc = ic === nothing ? Inf : (ic == 1 ? THETAS[1] :
             THETAS[ic-1] + (THETAS[ic] - THETAS[ic-1]) * (ern_m[ic-1] - 3) / (ern_m[ic-1] - ern_m[ic]))
        emit(@sprintf("  θ_c^mod (ern ≤ 3, interpolé) = %s", isfinite(θc) ? @sprintf("%.3f", θc) : "> 3"))
        emit("  θ      : " * join([@sprintf("%6.2f", t) for t in THETAS[1:5:end]], " "))
        emit("  ern_mod: " * join([@sprintf("%6.2f", v) for v in ern_m[1:5:end]], " "))
        emit("  ‖G(θ)‖ : " * join([@sprintf("%6.1f", v) for v in nG[1:5:end]], " "))
        open(joinpath(OUTDIR, "wind_T7_model_p$(p).json"), "w") do f
            JSON.print(f, Dict("thetas" => THETAS, "ern_mod" => ern_m, "b_mod" => b_m, "normG" => nG, "theta_c_mod" => θc,
                               "A" => Aamp, "beta" => β, "sG" => sG[1:5], "sGm" => sGm[1:5], "mu" => [μ[i] for i in 2:Lb-1],
                               "nu" => [ν[i] for i in 2:Lb-1]))
        end
        emit(@sprintf("  (%.0f s)", time() - t0))
        GC.gc()
    end
end
println("Écrit : ", RES)
