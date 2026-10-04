# WIND-T19 (CPU) -- Gemma-2-2B : critère G1 + descriptif (pré-enregistrement wind_T19_preregistration.md).
# Par prompt (collecte complète A, AN) : ern(N·P_A), ern(N·P_AN), rapport ; chemin θ (rapport ern(θ)/ern(0)) ;
# métriques WIND-T18 (g_raw, g_N, c_rad, c_dom, p1_A) ; gains d'état λ_k (A, AN) ; degré d'Euler (méta).
# USAGE : WIND_PROMPTS=1,...,15 julia --project=. notebook/wind_T19_analyze.jl
using LinearAlgebra, Statistics, Printf, JSON
const DIR = joinpath(@__DIR__, "wind_data_GEMMA"); const L, D = 26, 2304
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:15, ",")), ","))
const OUT = joinpath(@__DIR__, "wind_T19_analysis.json")
const THETAS = collect(0.25:0.125:2.5)
BLAS.set_num_threads(parse(Int, get(ENV, "WIND_BLAS", "16")))
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L); open(joinpath(DIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])                     # J_1 .. J_25
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
chain(Js) = (P = Js[1]; for k in 2:length(Js); P = Js[k] * P; end; P)
out = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
for p in PIDX
    haskey(out, string(p)) && continue
    isfile(joinpath(DIR, "wind_J_p$(p)_AN.bin")) || (println("p$p : Jacobiennes absentes, sauté"); continue)
    t0 = time()
    meta = JSON.parsefile(joinpath(DIR, "wind_meta_p$(p).json"))
    X = [Float64.(v) for v in meta["X"]]; x = X[end]; γ = Float64.(meta["gamma_final"]); r = meta["rms_inv_final"]
    N = r .* (Diagonal(γ) * (I - (x * x') .* (r^2 / D)))
    JA = loadJ(p, "A"); JN = loadJ(p, "AN")
    xh(k) = X[k+1] ./ norm(X[k+1])
    lam(Js) = [dot(xh(k + 1), Js[k] * xh(k)) for k in 1:L-1]
    PA = chain(JA); PN = chain(JN)
    FA, FN, FNA, FNN = svd(PA), svd(PN), svd(N * PA), svd(N * PN)
    eA, eN = ent(FNA.S), ent(FNN.S)
    path = [ent(svdvals(N * chain([JA[k] + θ .* (JN[k] - JA[k]) for k in 1:L-1]))) for θ in THETAS]
    rec = Dict("ern_A" => eA, "ern_AN" => eN, "ratio" => eN / eA, "path_ratio" => path ./ eA,
               "g_raw" => FN.S[1] / FA.S[1], "g_N" => FNN.S[1] / FNA.S[1], "c_rad" => abs(dot(FN.U[:, 1], x ./ norm(x))),
               "c_dom" => abs(dot(FNN.U[:, 1], FNA.U[:, 1])), "p1_A" => FNA.S[1]^2 / sum(abs2, FNA.S),
               "lam_A" => lam(JA), "lam_AN" => lam(JN), "euler_d" => meta["euler_d"], "gates" => meta["gates"])
    out[string(p)] = rec
    open(OUT, "w") do f; JSON.print(f, out); end
    @printf("p%-3d ern A %6.2f AN %6.2f rapport %.3f | g_N %.2f c_dom %.3f p1_A %.3f | %.0f s\n", p, eA, eN, eN / eA, rec["g_N"], rec["c_dom"], rec["p1_A"], time() - t0)
    JA = nothing; JN = nothing; GC.gc()
end
