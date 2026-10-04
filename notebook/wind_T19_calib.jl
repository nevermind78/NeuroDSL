# WIND-T19 -- porte E1 : étalonnage de l'estimateur par esquisse (k = 256) sur Qwen, matrices EXACTES de WIND-T8.
# Compare ern(N·P_V) exact et ern(N·P_V·G/√k) pour V = A, AN (même G), et le rapport AN/A, sur les prompts 1..20.
using LinearAlgebra, Statistics, Printf, JSON, Random
const DIR = joinpath(@__DIR__, "wind_data_T8"); const L, D, K = 28, 1536, 256
BLAS.set_num_threads(16)
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L); open(joinpath(DIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
chain(Js) = (P = Js[1]; for k in 2:length(Js); P = Js[k] * P; end; P)
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
G = randn(MersenneTwister(20261004), D, K) ./ sqrt(K)
res = Dict{String,Any}()
for p in 1:20
    meta = JSON.parsefile(joinpath(DIR, "wind_meta_p$(p).json"))
    x = Float64.(meta["X"][end]); γ = Float64.(meta["gamma_final"]); r = meta["rms_inv_final"]
    N = r .* (Diagonal(γ) * (I - (x * x') .* (r^2 / D)))
    MA = N * chain(loadJ(p, "A")); MN = N * chain(loadJ(p, "AN")); GC.gc()
    eA, eN = ent(svdvals(MA)), ent(svdvals(MN)); sA, sN = ent(svdvals(MA * G)), ent(svdvals(MN * G))
    res[string(p)] = Dict("exact_A" => eA, "exact_AN" => eN, "sk_A" => sA, "sk_AN" => sN)
    @printf("p%-3d exact A %6.2f AN %6.2f rapport %.3f | esquisse A %6.2f AN %6.2f rapport %.3f\n", p, eA, eN, eN / eA, sA, sN, sN / sA)
end
re = [res[p]["exact_AN"] / res[p]["exact_A"] for p in keys(res)]; rs = [res[p]["sk_AN"] / res[p]["sk_A"] for p in keys(res)]
mae = median(abs.(rs .- re)); agree = mean((re .< 0.5) .== (rs .< 0.5))
@printf("\nE1 : erreur absolue médiane du rapport %.3f (seuil 0,05) ; accord de classe < 0,5 : %.0f %% (seuil 90 %%) -> %s\n",
        mae, 100agree, (mae <= 0.05 && agree >= 0.9) ? "OK" : "ÉCHEC")
open(joinpath(@__DIR__, "wind_T19_calib.json"), "w") do f; JSON.print(f, res); end
