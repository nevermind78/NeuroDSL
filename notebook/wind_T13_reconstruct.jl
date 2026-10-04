# ══════════════════════════════════════════════════════════════════════════════
# WIND-T13 (CPU) -- reconstruction exacte de J^AA = J^A + R¹ et J^AM = J^A + R² (pré-enregistrement T13),
# portes G1 (rang 1), G2 (JVP directes), G3 (gradients), puis balayage θ ↦ erank₂(N·Π(J^A + θR^V)).
#   R¹ = s₁ (a1 + b1) x̂ᵀ ; R − R¹ = s₂ mh ρᵀ (rang 1, ρ = (I + A₁)ᵀĥ) ; R² = s₂ mh (ρ − s₁(ĥ·a1) x̂)ᵀ.
# USAGE : WIND_PROMPTS=… julia --project=. notebook/wind_T13_reconstruct.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const T8DIR = joinpath(@__DIR__, "wind_data_T8")
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const FAC = get(ENV, "WIND_T13_OUT", joinpath(@__DIR__, "wind_T13_factors.json"))
const OUT = get(ENV, "WIND_T13_REC", joinpath(@__DIR__, "wind_T13_reconstruct.json"))
const RES = get(ENV, "WIND_T13_REC_RES", joinpath(@__DIR__, "wind_T13_reconstruct_results.txt"))
const L, D = 28, 1536
const THETAS = collect(0.25:0.125:2.5)
BLAS.set_num_threads(parse(Int, get(ENV, "WIND_BLAS", "16")))
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(T8DIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])                       # J_1 .. J_27
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
θcross(θs, e) = (i = findfirst(<=(3.0), e); i === nothing ? Inf : i == 1 ? θs[1] :
                 θs[i-1] + (θs[i] - θs[i-1]) * (e[i-1] - 3) / (e[i-1] - e[i]))

fac = JSON.parsefile(FAC)
out = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T13 (CPU) reconstruction J^AA / J^AM + seuils -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ") * "   prompts $(PIDX)")
    for p in PIDX
        haskey(out, string(p)) && continue
        t0 = time(); fp = fac[string(p)]
        meta = JSON.parsefile(joinpath(T8DIR, "wind_meta_p$(p).json"))
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; X = [Float64.(v) for v in meta["X"]]
        H = [Float64.(v) for v in meta["H"]]; x = X[end]
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        nchk = length(fp["check_blocks"])
        buf = Array{Float64}(undef, D * 27 * 3 + D * 29 * 2 + D * 3 * nchk)
        open(joinpath(T8DIR, "wind_T13_factors_p$(p).bin"), "r") do f; read!(f, buf); end
        o = 0
        A1 = reshape(buf[o+1:o+D*27], D, 27); o += D * 27
        B1 = reshape(buf[o+1:o+D*27], D, 27); o += D * 27
        MH = reshape(buf[o+1:o+D*27], D, 27); o += D * 27
        GAA = reshape(buf[o+1:o+D*29], D, 29); o += D * 29
        GAM = reshape(buf[o+1:o+D*29], D, 29); o += D * 29
        chk = [(fp["check_blocks"][i], buf[o+(3i-3)*D+1:o+(3i-2)*D], buf[o+(3i-2)*D+1:o+(3i-1)*D], buf[o+(3i-1)*D+1:o+3i*D]) for i in 1:nchk]
        s1 = Float64.(fp["s1"]); s2 = Float64.(fp["s2"])
        JA = loadJ(p, "A"); JN = loadJ(p, "AN")
        R1 = Vector{Matrix{Float64}}(undef, 27); R2 = Vector{Matrix{Float64}}(undef, 27); g1 = zeros(27)
        for k in 1:27
            xh = X[k+1] ./ norm(X[k+1]); hh = H[k+1] ./ norm(H[k+1])
            R = JN[k] - JA[k]
            R1[k] = s1[k] .* ((A1[:, k] .+ B1[:, k]) * xh')
            Z = R - R1[k]; mh = MH[:, k]
            ρ = (Z' * mh) ./ (s2[k] * sum(abs2, mh))
            g1[k] = norm(Z - s2[k] .* (mh * ρ')) / norm(R)
            R2[k] = s2[k] .* (mh * (ρ .- s1[k] * dot(hh, A1[:, k]) .* xh)')
        end
        JN = nothing; GC.gc()
        g2aa = [norm((JA[k] + R1[k]) * v - jaa) / norm(jaa) for (k, v, jaa, jam) in chk]
        g2am = [norm((JA[k] + R2[k]) * v - jam) / norm(jam) for (k, v, jaa, jam) in chk]
        g3(G, Rs) = maximum(norm((JA[k] + Rs[k])' * G[:, k+2] - G[:, k+1]) / norm(G[:, k+1]) for k in 1:27)
        g3aa = g3(GAA, R1); g3am = g3(GAM, R2)
        curves = Dict{String,Any}()
        for (nm, Rs) in (("AA", R1), ("AM", R2))
            e = Float64[]
            for θ in THETAS
                P = JA[1] + θ .* Rs[1]
                for k in 2:27; P = (JA[k] + θ .* Rs[k]) * P; end
                push!(e, ent(svdvals(N * P)))
            end
            curves[nm] = e
        end
        JA = nothing; GC.gc()
        i1 = findfirst(==(1.0), THETAS)
        ok = median(g1) < 5e-2 && median(vcat(g2aa, g2am)) < 1e-2 && g3aa < 1e-2 && g3am < 1e-2 && fp["gate_G4"] < 1e-4
        rec = Dict("G1_med" => median(g1), "G1_max" => maximum(g1), "G2_AA" => g2aa, "G2_AM" => g2am, "G3_AA" => g3aa,
                   "G3_AM" => g3am, "G4" => fp["gate_G4"], "ok" => ok, "thetas" => THETAS,
                   "ern_AA" => curves["AA"], "ern_AM" => curves["AM"],
                   "theta_c_AA" => θcross(THETAS, curves["AA"]), "theta_c_AM" => θcross(THETAS, curves["AM"]),
                   "ern1_AA" => curves["AA"][i1], "ern1_AM" => curves["AM"][i1])
        out[string(p)] = rec
        open(OUT, "w") do f; JSON.print(f, out); end
        f(v) = isfinite(v) ? @sprintf("%.3f", v) : "> 2,5"
        emit(@sprintf("  p%-3d ern(θ=1) AA %6.2f AM %6.2f | θ_c AA %s AM %s | G1 %.1e G2 %.1e/%.1e G3 %.1e/%.1e G4 %.1e | %s | %.0f s",
                      p, rec["ern1_AA"], rec["ern1_AM"], f(rec["theta_c_AA"]), f(rec["theta_c_AM"]), median(g1),
                      median(g2aa), median(g2am), g3aa, g3am, fp["gate_G4"], ok ? "retenu" : "EXCLU", time() - t0))
    end
end
println("Écrit : ", OUT)
