# WIND-T20 (CPU, EXPLORATOIRE) -- relèvement exact de la correction des normes figées, pour les 3 modèles.
# R_k = J_k^AN − J_k^A = U_k V_kᵀ (rang p : 2 pour Qwen/GPT-2, 4 pour Gemma ; V_k orthonormée = capteurs).
# Lemme A (T7) : N·P^AN = N·P^A + 𝒰 (I − W)⁻¹ 𝒱, avec 𝒰_k = N Φ_A(L,k+1) U_k, 𝒱_k = V_kᵀ Φ_A(k,1),
# W_jk = V_jᵀ Φ_A(j,k+1) U_k (j > k). Décomposition boucle ouverte / boucle fermée :
#   Δ¹ = 𝒰𝒱 (une seule injection), Δ = 𝒰 (I − W)⁻¹ 𝒱 ; facteur de boucle = σ₁(Δ)/σ₁(Δ¹).
# USAGE : WIND_T20_MODEL=… WIND_PROMPTS=… WIND_BLAS=4 julia --project=. notebook/wind_T20_lift.jl
using LinearAlgebra, Statistics, Printf, JSON
const MODEL = get(ENV, "WIND_T20_MODEL", "qwen")
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", "1"), ","))
const DIR = joinpath(@__DIR__, Dict("qwen" => "wind_data_T8", "gpt2" => "wind_data_GPT2", "gemma" => "wind_data_GEMMA")[MODEL])
const L, D = Dict("qwen" => (28, 1536), "gpt2" => (12, 768), "gemma" => (26, 2304))[MODEL]
const PR = MODEL == "gemma" ? 4 : 2
const OUT = get(ENV, "WIND_T20_LIFT_OUT", joinpath(@__DIR__, "wind_T20_lift_$(MODEL).json"))
BLAS.set_num_threads(parse(Int, get(ENV, "WIND_BLAS", "4")))
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L); open(joinpath(DIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
function finalN(meta)
    x = Float64.(meta["X"][end]); γ = Float64.(meta["gamma_final"]); r = meta["rms_inv_final"]
    if MODEL == "gpt2"
        xc = x .- mean(x); Pm = Matrix{Float64}(I, D, D) .- 1 / D
        return r .* (Diagonal(γ) * (Pm .- (xc * xc') .* (r^2 / D)))
    else
        return r .* (Diagonal(γ) * (I - (x * x') .* (r^2 / D)))
    end
end
out = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
for p in PIDX
    haskey(out, string(p)) && continue
    isfile(joinpath(DIR, "wind_J_p$(p)_AN.bin")) && isfile(joinpath(DIR, "wind_meta_p$(p).json")) || (println("p$p absent"); continue)
    t0 = time(); meta = JSON.parsefile(joinpath(DIR, "wind_meta_p$(p).json")); N = finalN(meta)
    X = [Float64.(v) for v in meta["X"]]; H = [Float64.(v) for v in meta["H"]]
    JA = loadJ(p, "A"); JN = loadJ(p, "AN"); K = L - 1
    U = Vector{Matrix{Float64}}(undef, K); V = Vector{Matrix{Float64}}(undef, K); tail = zeros(K)
    for k in 1:K
        F = svd(JN[k] - JA[k]); V[k] = F.V[:, 1:PR]; U[k] = F.U[:, 1:PR] * Diagonal(F.S[1:PR])
        tail[k] = F.S[PR+1] / F.S[1]
    end
    PN = JN[1]; for k in 2:K; PN = JN[k] * PN; end; MN = N * PN; JN = nothing; GC.gc()
    # transports vivants : 𝒱_k = V_kᵀ Φ_A(k,1) (lignes) ; colonnes U_k propagées vers l'avant
    n = PR * K; W = zeros(n, n); 𝒰 = zeros(D, n); 𝒱 = zeros(n, D)
    Φ = Matrix{Float64}(I, D, D)                       # Φ_A(k,1) = J_{k-1}···J_1
    for k in 1:K
        𝒱[PR*(k-1)+1:PR*k, :] = V[k]' * Φ
        Φ = JA[k] * Φ
    end
    MA = N * Φ
    for k in 1:K
        Y = U[k]                                         # au point x_{k+1}
        for j in k+1:K
            W[PR*(j-1)+1:PR*j, PR*(k-1)+1:PR*k] = V[j]' * Y
            Y = JA[j] * Y
        end
        𝒰[:, PR*(k-1)+1:PR*k] = N * Y
    end
    G = inv(I - W)
    Δ = 𝒰 * G * 𝒱; Δ1 = 𝒰 * 𝒱
    gate = norm(MA + Δ - MN) / norm(MN)
    sA = svdvals(MA); sN = svdvals(MN); sΔ = svdvals(Δ); sΔ1 = svdvals(Δ1)
    # facteurs de boucle par portée : W local (j = k+1) vs longue portée (j ≥ k+2)
    Wloc = zeros(n, n); for k in 1:K-1; Wloc[PR*k+1:PR*(k+1), PR*(k-1)+1:PR*k] = W[PR*k+1:PR*(k+1), PR*(k-1)+1:PR*k]; end
    Gloc = inv(I - Wloc)
    rec = Dict("gate_lift" => gate, "rank_tail_max" => maximum(tail), "sigma1_A" => sA[1], "sigma1_AN" => sN[1],
        "tau_A" => sqrt(sum(abs2, sA[2:end])), "tau_AN" => sqrt(sum(abs2, sN[2:end])), "ern_A" => ent(sA), "ern_AN" => ent(sN),
        "sigma1_Delta" => sΔ[1], "sigma2_Delta" => sΔ[2], "sigma1_Delta1" => sΔ1[1], "loop_factor" => sΔ[1] / sΔ1[1],
        "normG" => opnorm(G), "normW" => opnorm(W), "normGloc" => opnorm(Gloc),
        "cos_Delta_A" => abs(dot(svd(Δ).U[:, 1], svd(MA).U[:, 1])),
        "W" => [W[i, :] for i in 1:n])
    out[string(p)] = rec
    open(OUT, "w") do f; JSON.print(f, out); end
    @printf("%s p%-3d porte %.1e queue %.1e | σ₁ A %.2f AN %.2f | σ₁(Δ¹) %.2f σ₁(Δ) %.2f boucle ×%.2f | ‖G‖ %.2f ‖G_loc‖ %.2f ‖W‖ %.2f | cos(Δ,A) %.2f | %.0f s\n",
            MODEL, p, gate, maximum(tail), sA[1], sN[1], sΔ1[1], sΔ[1], sΔ[1] / sΔ1[1], opnorm(G), opnorm(Gloc), opnorm(W), rec["cos_Delta_A"], time() - t0)
    JA = nothing; GC.gc()
end
