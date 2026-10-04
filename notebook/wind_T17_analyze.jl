# WIND-T17 (CPU) -- métriques pour L2–L4 (pré-enregistrement wind_T17_preregistration.md).
# GPT-2 : ern(N·Π J^V) pour V = T, A, AN ; θ_c sur J^A + θ(J^AN − J^A) ; gains d'état λ_k (T, AN).
# Qwen : gains d'état λ_k (T, AN) et contrôle de profondeur Qwen-12 (J^AN couches 1..12, J^A ensuite).
# USAGE : WIND_T17_MODEL=gpt2|qwen WIND_PROMPTS=… julia --project=. notebook/wind_T17_analyze.jl
using LinearAlgebra, Statistics, Printf, JSON
const MODEL = get(ENV, "WIND_T17_MODEL", "gpt2")
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const DIR = joinpath(@__DIR__, MODEL == "gpt2" ? "wind_data_GPT2" : "wind_data_T8")
const L, D = MODEL == "gpt2" ? (12, 768) : (28, 1536)
const OUT = get(ENV, "WIND_T17_OUT", joinpath(@__DIR__, "wind_T17_analysis_$(MODEL).json"))
const THETAS = collect(0.25:0.125:2.5)
BLAS.set_num_threads(parse(Int, get(ENV, "WIND_BLAS", "16")))
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(DIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])                     # J_1 .. J_{L-1}
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
θcross(θs, e) = (i = findfirst(<=(3.0), e); i === nothing ? 99.0 : i == 1 ? θs[1] :
                 θs[i-1] + (θs[i] - θs[i-1]) * (e[i-1] - 3) / (e[i-1] - e[i]))
chain(Js) = (P = Js[1]; for k in 2:length(Js); P = Js[k] * P; end; P)
function finalN(meta)
    x = Float64.(meta["X"][end]); γ = Float64.(meta["gamma_final"]); r = meta["rms_inv_final"]
    if MODEL == "gpt2"                       # LayerNorm : r·diag(γ)·(P − r² x̃x̃ᵀ/D), P = I − 11ᵀ/D
        xc = x .- mean(x); Pm = Matrix{Float64}(I, D, D) .- 1 / D
        return r .* (Diagonal(γ) * (Pm .- (xc * xc') .* (r^2 / D)))
    else
        return r .* (Diagonal(γ) * (I - (x * x') .* (r^2 / D)))
    end
end
out = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
for p in PIDX
    haskey(out, string(p)) && continue
    t0 = time()
    meta = JSON.parsefile(joinpath(DIR, "wind_meta_p$(p).json")); N = finalN(meta)
    X = [Float64.(v) for v in meta["X"]]; xh(k) = X[k+1] ./ norm(X[k+1])
    JT = loadJ(p, "T"); JA = loadJ(p, "A"); JN = loadJ(p, "AN")
    lam(Js) = [dot(xh(k + 1), Js[k] * xh(k)) for k in 1:L-1]
    rec = Dict{String,Any}("lam_T" => lam(JT), "lam_AN" => lam(JN))
    if MODEL == "gpt2"
        rec["ern_T"] = ent(svdvals(N * chain(JT))); rec["ern_A"] = ent(svdvals(N * chain(JA))); rec["ern_AN"] = ent(svdvals(N * chain(JN)))
        e = [ent(svdvals(N * chain([JA[k] + θ .* (JN[k] - JA[k]) for k in 1:L-1]))) for θ in THETAS]
        rec["ern_path"] = e; rec["theta_c_AN"] = θcross(THETAS, e); rec["euler_d"] = meta["euler_d"]
        @printf("gpt2 p%-3d ern T %.2f A %.2f AN %.2f θ_c %.3f | %.0f s\n", p, rec["ern_T"], rec["ern_A"], rec["ern_AN"], rec["theta_c_AN"], time() - t0)
    else
        rec["ern_Q12"] = ent(svdvals(N * chain([k <= 12 ? JN[k] : JA[k] for k in 1:L-1])))
        @printf("qwen p%-3d ern Qwen-12 %.2f | %.0f s\n", p, rec["ern_Q12"], time() - t0)
    end
    out[string(p)] = rec
    open(OUT, "w") do f; JSON.print(f, out); end
    JT = nothing; JA = nothing; JN = nothing; GC.gc()
end
