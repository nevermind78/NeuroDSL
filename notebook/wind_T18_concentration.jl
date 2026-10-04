# WIND-T18 (EXPLORATOIRE) -- pourquoi GPT-2 ne s'effondre pas (pré-enregistrement : wind_T18_preregistration.md).
# Par prompt : g_raw, g_N, c_rad, c_dom, p1_A (définitions dans le pré-enregistrement).
# USAGE : WIND_T18_MODEL=gpt2|qwen WIND_PROMPTS=… julia --project=. notebook/wind_T18_concentration.jl
using LinearAlgebra, Statistics, Printf, JSON
const MODEL = get(ENV, "WIND_T18_MODEL", "gpt2")
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const DIR = joinpath(@__DIR__, MODEL == "gpt2" ? "wind_data_GPT2" : "wind_data_T8")
const L, D = MODEL == "gpt2" ? (12, 768) : (28, 1536)
const OUT = get(ENV, "WIND_T18_OUT", joinpath(@__DIR__, "wind_T18_$(MODEL).json"))
BLAS.set_num_threads(parse(Int, get(ENV, "WIND_BLAS", "16")))
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(DIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
chain(Js) = (P = Js[1]; for k in 2:length(Js); P = Js[k] * P; end; P)
function finalN_and_dir(meta)
    x = Float64.(meta["X"][end]); γ = Float64.(meta["gamma_final"]); r = meta["rms_inv_final"]
    if MODEL == "gpt2"
        xc = x .- mean(x); Pm = Matrix{Float64}(I, D, D) .- 1 / D
        return r .* (Diagonal(γ) * (Pm .- (xc * xc') .* (r^2 / D))), xc ./ norm(xc)
    else
        return r .* (Diagonal(γ) * (I - (x * x') .* (r^2 / D))), x ./ norm(x)
    end
end
out = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
for p in PIDX
    haskey(out, string(p)) && continue
    t0 = time()
    meta = JSON.parsefile(joinpath(DIR, "wind_meta_p$(p).json")); N, xd = finalN_and_dir(meta)
    PA = chain(loadJ(p, "A")); PN = chain(loadJ(p, "AN")); GC.gc()
    FA = svd(PA); FN = svd(PN); FNA = svd(N * PA); FNN = svd(N * PN)
    rec = Dict("g_raw" => FN.S[1] / FA.S[1], "g_N" => FNN.S[1] / FNA.S[1],
               "c_rad" => abs(dot(FN.U[:, 1], xd)), "c_rad_A" => abs(dot(FA.U[:, 1], xd)),
               "c_dom" => abs(dot(FNN.U[:, 1], FNA.U[:, 1])), "p1_A" => FNA.S[1]^2 / sum(abs2, FNA.S),
               "p1_AN" => FNN.S[1]^2 / sum(abs2, FNN.S))
    out[string(p)] = rec
    open(OUT, "w") do f; JSON.print(f, out); end
    @printf("%s p%-3d g_raw %6.2f g_N %6.2f c_rad %.3f c_dom %.3f p1_A %.3f | %.0f s\n", MODEL, p, rec["g_raw"], rec["g_N"],
            rec["c_rad"], rec["c_dom"], rec["p1_A"], time() - t0)
end
