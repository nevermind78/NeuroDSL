# WIND-T20 (CPU, EXPLORATOIRE) -- descriptif du « mode d'échelle » des linéarisations figées, 3 modèles.
# Échantillon de DÉCOUVERTE seulement (les prédictions T20 seront testées sur d'autres prompts).
# Par prompt et variante V : gains radiaux relatifs ρ_k = ⟨x̂_{k+1}, J_k x_k⟩/‖x_{k+1}‖ ; défauts d'Euler
# e_k = J_k x_k − x_{k+1} ; degré d'Euler de bout en bout D_j = ⟨x̂_j, P(j←1) x_1⟩/‖x_j‖ ; σ₁, ern de N·P ;
# fuite non radiale du mode d'échelle ‖N P x_1‖ ; fractions d'écriture radiales μ_k (MLP), ν_k (attention).
# USAGE : WIND_T20_MODEL=qwen|gpt2|gemma WIND_PROMPTS=1,2 WIND_BLAS=4 julia --project=. notebook/wind_T20_explore.jl
using LinearAlgebra, Statistics, Printf, JSON
const MODEL = get(ENV, "WIND_T20_MODEL", "qwen")
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", "1"), ","))
const DIR = joinpath(@__DIR__, Dict("qwen" => "wind_data_T8", "gpt2" => "wind_data_GPT2", "gemma" => "wind_data_GEMMA")[MODEL])
const L, D = Dict("qwen" => (28, 1536), "gpt2" => (12, 768), "gemma" => (26, 2304))[MODEL]
const VARS = Dict("qwen" => ["T", "A", "AN", "AH"], "gpt2" => ["T", "A", "AN"], "gemma" => ["A", "AN"])[MODEL]
const OUT = get(ENV, "WIND_T20_OUT", joinpath(@__DIR__, "wind_T20_explore_$(MODEL).json"))
BLAS.set_num_threads(parse(Int, get(ENV, "WIND_BLAS", "4")))
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L); open(joinpath(DIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])                     # J_1 .. J_{L-1}
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
    t0 = time()
    meta = JSON.parsefile(joinpath(DIR, "wind_meta_p$(p).json")); N = finalN(meta)
    X = [Float64.(v) for v in meta["X"]]; H = [Float64.(v) for v in meta["H"]]
    MLP = [Float64.(v) for v in meta["MLP"]]; MHA = [Float64.(v) for v in meta["MHA"]]
    # bloc k (1..L-1) : x_k = X[k+1] -> x_{k+1} = X[k+2] ; sorties MHA[k+1], MLP[k+1]
    xs(k) = X[k+1]; xh(k) = X[k+1] ./ norm(X[k+1])
    xL = X[L+1]; xLc = MODEL == "gpt2" ? xL .- mean(xL) : xL
    rec = Dict{String,Any}()
    rec["xnorm"] = [norm(X[k+1]) for k in 0:L]
    rec["mu"] = [dot(xh(k + 1), MLP[k+1]) / norm(xs(k + 1)) for k in 1:L-1]      # écriture radiale MLP
    rec["nu"] = [dot(xh(k + 1), MHA[k+1]) / norm(xs(k + 1)) for k in 1:L-1]      # écriture radiale attention
    rec["mfrac"] = [norm(MLP[k+1]) / norm(xs(k + 1)) for k in 1:L-1]
    if haskey(meta, "euler_d"); rec["euler_d"] = meta["euler_d"]; end
    for V in VARS
        isfile(joinpath(DIR, "wind_J_p$(p)_$(V).bin")) || continue
        Js = loadJ(p, V)
        ρ = [dot(xh(k + 1), Js[k] * xs(k)) / norm(xs(k + 1)) for k in 1:L-1]
        e = [Js[k] * xs(k) .- xs(k + 1) for k in 1:L-1]
        # régression du défaut d'Euler sur (m_k, a_k, x_k) : coefficients et résidu relatif
        coef = Vector{Vector{Float64}}(); resid = Float64[]
        for k in 1:L-1
            B = hcat(MLP[k+1], MHA[k+1], xs(k)); c = B \ e[k]
            push!(coef, c); push!(resid, norm(e[k] - B * c) / max(norm(e[k]), 1e-30))
        end
        # chaîne du mode d'échelle et degré d'Euler de bout en bout
        s = copy(xs(1)); Dj = Float64[1.0]
        for k in 1:L-1; s = Js[k] * s; push!(Dj, dot(xh(k + 1), s) / norm(xs(k + 1))); end
        P = Js[1]; for k in 2:L-1; P = Js[k] * P; end
        Fr = svd(P); FN = svd(N * P)
        sN = N * s
        rec[V] = Dict("rho" => ρ, "e_rel" => [norm(e[k]) / norm(xs(k + 1)) for k in 1:L-1],
            "e_coef" => coef, "e_resid" => resid, "Dj" => Dj,
            "sigma1_raw" => Fr.S[1], "sigma1_N" => FN.S[1], "ern_N" => ent(FN.S), "p1_N" => FN.S[1]^2 / sum(abs2, FN.S),
            "tau_N" => sqrt(sum(abs2, FN.S[2:end])),
            "c_rad_raw" => abs(dot(Fr.U[:, 1], xLc ./ norm(xLc))),
            "v1_scale_raw" => abs(dot(Fr.V[:, 1], xh(1))), "v1_scale_N" => abs(dot(FN.V[:, 1], xh(1))),
            "scale_out_norm" => norm(s) / norm(xL), "scale_cos_xL" => dot(s, xLc) / (norm(s) * norm(xLc)),
            "scale_N_over_sigma1N" => norm(sN) / FN.S[1], "scale_N_cos_u1" => abs(dot(sN, FN.U[:, 1])) / norm(sN),
            "U1N" => FN.U[:, 1], "V1N" => FN.V[:, 1])
        @printf("%s p%-3d %-2s σ₁raw %8.2f σ₁N %8.3f ern %6.2f | Π ρ %9.3g  D_L %9.3g | ‖NPx₁‖/σ₁ %.3f cos(u₁) %.3f | c_rad %.3f\n",
                MODEL, p, V, Fr.S[1], FN.S[1], ent(FN.S), prod(ρ), Dj[end], norm(sN) / FN.S[1], abs(dot(sN, FN.U[:, 1])) / norm(sN),
                abs(dot(Fr.U[:, 1], xLc ./ norm(xLc))))
        Js = nothing; GC.gc()
    end
    out[string(p)] = rec
    open(OUT, "w") do f; JSON.print(f, out); end
    @printf("%s p%d fait en %.0f s\n", MODEL, p, time() - t0)
end
