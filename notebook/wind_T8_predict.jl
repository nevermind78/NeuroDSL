# ══════════════════════════════════════════════════════════════════════════════
# WIND-T8 -- PRÉDICTIONS figées (pré-enregistré : notebook/wind_T8_preregistration.md, H1).
# Côté MODÈLE uniquement, code repris de wind_T7_model.jl / wind_T7_sweep.jl :
#   relèvement N·Π(A_k + θR_k) = M_A + θ𝒰(I − θW)⁻¹𝒱 ; réalisation d'ordre r de W (Ho–Kalman variant
#   dans le temps) ; θ_c^mod(r) = premier franchissement de ern = 3 sur θ = 0,25 : 0,05 : 2,5.
# N'évalue AUCUN produit vrai à θ ≠ 0 ni AUCUNE résolvante complète (I − θW)⁻¹.
# Refuse d'écraser un fichier de prédictions existant.
#
# USAGE : WIND_PROMPTS=1,...,50 julia --project=. notebook/wind_T8_predict.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data_T8"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const PRED = get(ENV, "WIND_PRED", joinpath(@__DIR__, "wind_T8_predictions.json"))
const RES = get(ENV, "WIND_PRED_RES", joinpath(@__DIR__, "wind_T8_predict_results.txt"))
const L, D, Lb = 28, 1536, 27
const THETAS = collect(0.25:0.05:2.5)
const ORDERS = (1, 2, 3)
BLAS.set_num_threads(16)
isfile(PRED) && error("prédictions déjà figées : $PRED -- refus d'écraser")

loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])                     # J_1 .. J_27
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
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
function θcross(θs, e)
    i = findfirst(<=(3.0), e)
    i === nothing && return Inf
    i == 1 && return θs[1]
    θs[i-1] + (θs[i] - θs[i-1]) * (e[i-1] - 3) / (e[i-1] - e[i])
end

preds = Dict{String,Any}()
open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T8 PRÉDICTIONS (modèle seulement) -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ") *
         "   prompts : $(PROMPT_IDXS)   sortie : $(PRED)")
    for p in PROMPT_IDXS
        t0 = time()
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x = Float64.(meta["X"][end])
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        JA = loadJ(p, "A"); JN = loadJ(p, "AN")
        Uk = Vector{Matrix{Float64}}(undef, 27); Vk = Vector{Matrix{Float64}}(undef, 27); r3 = 0.0
        for k in 1:27
            F = svd(JN[k] - JA[k]); Uk[k] = F.U[:, 1:2] * Diagonal(F.S[1:2]); Vk[k] = F.V[:, 1:2]
            r3 = max(r3, F.S[3] / F.S[1])
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
        open(joinpath(OUTDIR, "wind_T8_lift_p$(p).bin"), "w") do f; write(f, 𝒰); write(f, 𝒱); write(f, MA); write(f, W); end
        rec = Dict{String,Any}("ern_A" => ent(svdvals(MA)), "rank2_sigma3_over_sigma1_max" => r3, "thetas" => THETAS)
        for r in ORDERS
            C, B, Φ = realize(W, r)
            Wf = transfer(C, B, Φ)
            curve = [ent(svdvals(MA + θ .* (𝒰 * inv(I - θ .* Wf) * 𝒱))) for θ in THETAS]
            rec["ern_mod$(r)"] = curve; rec["theta_c_mod$(r)"] = θcross(THETAS, curve)
            if r == 2
                M = Dict(i => Φ[i] + B[i] * C[i] for i in 2:Lb-1)
                P = Matrix{Float64}(I, size(M[2], 2), size(M[2], 2)); for i in 2:Lb-1; P = M[i] * P; end
                sv = svdvals(P); rec["lyap2"] = log.(sv[1:min(2, end)]) ./ (Lb - 2)
            end
        end
        preds[string(p)] = rec
        f(v) = isfinite(v) ? @sprintf("%.3f", v) : "> 2,5"
        emit(@sprintf("  p%-3d ern_A %6.2f | θ_c^mod r=1 %s  r=2 %s  r=3 %s | Lyapunov 2×2 %.3f %.3f | rang-2 σ₃/σ₁ %.1e | %.0f s",
                      p, rec["ern_A"], f(rec["theta_c_mod1"]), f(rec["theta_c_mod2"]), f(rec["theta_c_mod3"]),
                      rec["lyap2"]..., r3, time() - t0))
        GC.gc()
    end
end
open(PRED, "w") do f
    JSON.print(f, Dict("date" => Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"), "script" => "wind_T8_predict.jl",
                       "grid" => "0.25:0.05:2.5", "threshold_ern" => 3.0, "predictions" => preds))
end
println("Prédictions figées : ", PRED)
