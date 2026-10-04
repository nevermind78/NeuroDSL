# WIND-T15 (CPU) -- erank₂(N·Π(J^A + θ(J^AH − J^A))) sur θ ∈ {0} ∪ 0,25:0,125:2,5 ; θ_c^AH ; ern_AH(1).
# USAGE : WIND_PROMPTS=… julia --project=. notebook/wind_T15_analyze.jl
using LinearAlgebra, Statistics, Printf, Dates, JSON
const T8DIR = joinpath(@__DIR__, "wind_data_T8")
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const OUT = get(ENV, "WIND_T15_AN_OUT", joinpath(@__DIR__, "wind_T15_analysis.json"))
const RES = get(ENV, "WIND_T15_AN_RES", joinpath(@__DIR__, "wind_T15_analysis_results.txt"))
const L, D = 28, 1536
const THETAS = vcat(0.0, collect(0.25:0.125:2.5))
BLAS.set_num_threads(parse(Int, get(ENV, "WIND_BLAS", "16")))
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(T8DIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
θcross(θs, e) = (i = findfirst(<=(3.0), e); i === nothing ? Inf : i == 1 ? θs[1] :
                 θs[i-1] + (θs[i] - θs[i-1]) * (e[i-1] - 3) / (e[i-1] - e[i]))
out = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T15 (CPU) seuils sous la règle du demi -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PIDX
        haskey(out, string(p)) && continue
        t0 = time()
        meta = JSON.parsefile(joinpath(T8DIR, "wind_meta_p$(p).json"))
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x = Float64.(meta["X"][end])
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        JA = loadJ(p, "A"); JH = loadJ(p, "AH")
        e = Float64[]
        for θ in THETAS
            P = JA[1] + θ .* (JH[1] - JA[1])
            for k in 2:27; P = (JA[k] + θ .* (JH[k] - JA[k])) * P; end
            push!(e, ent(svdvals(N * P)))
        end
        JA = nothing; JH = nothing; GC.gc()
        thc = θcross(THETAS[2:end], e[2:end]); e1 = e[findfirst(==(1.0), THETAS)]
        out[string(p)] = Dict("thetas" => THETAS, "ern" => e, "theta_c_AH" => isfinite(thc) ? thc : 99.0, "ern1_AH" => e1)
        open(OUT, "w") do f; JSON.print(f, out); end
        emit(@sprintf("  p%-3d ern_AH(θ=1) %6.2f | θ_c^AH %s | %.0f s", p, e1, isfinite(thc) ? @sprintf("%.3f", thc) : "> 2,5", time() - t0))
    end
end
