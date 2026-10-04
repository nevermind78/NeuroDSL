# ══════════════════════════════════════════════════════════════════════════════
# WIND-T7 -- tests pré-enregistrés P7.1–P7.3 (wind_T3_preregistration.md, section WIND-T7). EXPLORATOIRE.
# Vérité : ern_true(θ) = erank₂(N·Π_k (J_k^A + θ(J_k^AN − J_k^A))), θ = 0,25 : 0,125 : 2,5.
# Modèle d'ordre 2 : ern_mod(θ) = ern(M_A + θ𝒰(I − θW₂)⁻¹𝒱) ; porte : relèvement exact avec W.
# USAGE : julia --project=. notebook/wind_T7_sweep.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = joinpath(@__DIR__, "wind_data")
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "12,22,17,5,6"), ","))
const RES = joinpath(@__DIR__, "wind_T7_sweep_results.txt")
const L, D, Lb = 28, 1536, 27
const THETAS = collect(0.25:0.125:2.5)
const THC_MOD = Dict(22 => 0.851, 17 => 1.546, 5 => 1.621, 6 => 1.646)     # pré-enregistrés
BLAS.set_num_threads(16)
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
rows(i) = 2i+1:2Lb; cols(i) = 1:2i
function realizeW(W, r)
    O = Dict{Int,Matrix{Float64}}(); R = Dict{Int,Matrix{Float64}}()
    for i in 1:Lb-1
        F = svd(W[rows(i), cols(i)]); q = min(r, length(F.S))
        O[i] = F.U[:, 1:q] * Diagonal(sqrt.(F.S[1:q])); R[i] = Diagonal(sqrt.(F.S[1:q])) * F.Vt[1:q, :]
    end
    C = Dict(i + 1 => O[i][1:2, :] for i in 1:Lb-1); B = Dict(i => R[i][:, end-1:end] for i in 1:Lb-1)
    Φ = Dict(i + 1 => pinv(O[i+1]) * O[i][3:end, :] for i in 1:Lb-2)
    Wf = zeros(2Lb, 2Lb)
    for k in 1:Lb-1
        ζ = B[k]
        for j in k+1:Lb
            Wf[2j-1:2j, 2k-1:2k] = C[j] * ζ
            j <= Lb-1 && (ζ = Φ[j] * ζ)
        end
    end
    Wf
end
function θcross(θs, e)
    i = findfirst(<=(3.0), e)
    i === nothing && return Inf
    i == 1 && return θs[1]
    θs[i-1] + (θs[i] - θs[i-1]) * (e[i-1] - 3) / (e[i-1] - e[i])
end

res = Dict{Int,Any}()
open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T7 balayage du gain de rétroaction θ (tests P7.1–P7.3) -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PROMPT_IDXS
        t0 = time()
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x = Float64.(meta["X"][end])
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        buf = Array{Float64}(undef, D * 54 * 2 + D * D + 54 * 54)
        open(joinpath(OUTDIR, "wind_T7_lift_p$(p).bin"), "r") do f; read!(f, buf); end
        o = 0
        𝒰 = reshape(buf[o+1:o+D*54], D, 54); o += D * 54
        𝒱 = reshape(buf[o+1:o+54*D], 54, D); o += 54 * D
        MA = reshape(buf[o+1:o+D*D], D, D); o += D * D
        W = reshape(buf[o+1:o+54*54], 54, 54)
        W2 = realizeW(W, 2)
        JA = loadJ(p, "A"); JN = loadJ(p, "AN")
        et = Float64[]; em = Float64[]; gate = Float64[]
        for θ in THETAS
            P = JA[1] + θ .* (JN[1] - JA[1])
            for k in 2:27; P = (JA[k] + θ .* (JN[k] - JA[k])) * P; end
            MT = N * P
            push!(et, ent(svdvals(MT)))
            Mlift = MA + θ .* (𝒰 * inv(I - θ .* W) * 𝒱)
            push!(gate, norm(Mlift - MT) / norm(MT))
            push!(em, ent(svdvals(MA + θ .* (𝒰 * inv(I - θ .* W2) * 𝒱))))
        end
        JA = nothing; JN = nothing; GC.gc()
        thT = θcross(THETAS, et); thM = θcross(THETAS, em)
        G1 = inv(I - W); G2 = inv(I - 2 .* W); s1 = svdvals(G1); s2 = svdvals(G2)
        dev = maximum(abs.(log.(em) .- log.(et)))
        emit(@sprintf("\n==================== PROMPT %d  (%.0f s) ====================", p, time() - t0))
        emit("  θ        : " * join([@sprintf("%6.3f", t) for t in THETAS], " "))
        emit("  ern_true : " * join([@sprintf("%6.2f", v) for v in et], " "))
        emit("  ern_mod  : " * join([@sprintf("%6.2f", v) for v in em], " "))
        emit(@sprintf("  porte relèvement exact (max sur θ) : %.1e", maximum(gate)))
        emit(@sprintf("  θ_c^true = %s ; θ_c^mod (grille du test) = %s ; θ_c^mod pré-enregistré = %s",
                      isfinite(thT) ? @sprintf("%.3f", thT) : "> 2,5", isfinite(thM) ? @sprintf("%.3f", thM) : "> 2,5",
                      haskey(THC_MOD, p) ? @sprintf("%.3f", THC_MOD[p]) : "—"))
        emit(@sprintf("  max |ln ern_mod − ln ern_true| = %.3f (seuil ln 1,5 = %.3f)", dev, log(1.5)))
        emit(@sprintf("  G(1) : σ₁ = %.2f σ₂ = %.2f | G(2) : σ₁ = %.2f σ₂ = %.2f σ₂/σ₁ = %.4f", s1[1], s1[2], s2[1], s2[2], s2[2] / s2[1]))
        res[p] = (thT = thT, dev = dev, r21 = s2[2] / s2[1], s2ratio = s2[2] / s1[2], e05 = et[findfirst(==(0.5), THETAS)],
                  et = et, em = em)
    end
    emit("\nVERDICTS (pré-enregistrés) :")
    if all(haskey(res, p) for p in (12, 22, 17, 5, 6))
        ok1 = all(abs(res[p].thT - THC_MOD[p]) <= 0.15 * THC_MOD[p] for p in (22, 17, 5, 6)) &&
              res[22].thT < 1 < minimum(res[p].thT for p in (17, 5, 6))
        emit(@sprintf("P7.1 : %s  (θ_c^true : p22 %.3f [mod 0,851], p17 %.3f [1,546], p5 %.3f [1,621], p6 %.3f [1,646])",
                      ok1 ? "VRAI" : "FAUX", res[22].thT, res[17].thT, res[5].thT, res[6].thT))
        ok1b = 1.10 <= res[12].e05 <= 1.40
        emit(@sprintf("P7.1b (p12, ern_true(0,5) ∈ [1,10 ; 1,40]) : %s  (%.3f)", ok1b ? "VRAI" : "FAUX", res[12].e05))
        ok2 = all(res[p].dev <= log(1.5) for p in (12, 22, 17, 5, 6))
        emit("P7.2 : " * (ok2 ? "VRAI" : "FAUX") * "  (max |Δln ern| : " *
             join([@sprintf("p%d %.3f", p, res[p].dev) for p in (12, 22, 17, 5, 6)], ", ") * ")")
        ok3 = all(res[p].r21 <= 0.10 && res[p].s2ratio <= 3 for p in (12, 22, 17, 5, 6))
        emit("P7.3 : " * (ok3 ? "VRAI" : "FAUX") * "  (σ₂/σ₁ de G(2) ; σ₂(G(2))/σ₂(G(1)) : " *
             join([@sprintf("p%d %.3f ; %.2f", p, res[p].r21, res[p].s2ratio) for p in (12, 22, 17, 5, 6)], " | ") * ")")
    end
end
println("Écrit : ", RES)
