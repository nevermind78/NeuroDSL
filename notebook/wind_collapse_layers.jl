# ══════════════════════════════════════════════════════════════════════════════
# WIND-T5b (EXPLORATOIRE, post hoc) -- Quelle couche fabrique le sur-effondrement de AN (p12, p22) ?
# Test exact de F1 : J_AN − J_A = (I+J_m) r₁ + r₂ (I+J_a+r₁) est de rang ≤ 2 par couche (une norme figée =
# un terme de rang 1) -> σ₃/σ₁(J_AN − J_A) doit être au niveau du bruit de différences finies.
# Par couche : σ₁ de J_T, rapports ‖J_A − J_T‖₂/σ₁(J_T) (gel attention), ‖J_AN − J_A‖₂/σ₁(J_T) (gel normes),
# σ₂/σ₁ et σ₃/σ₁ de J_AN − J_A.
#
# USAGE : julia --project=. notebook/wind_collapse_layers.jl
# ══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra, Statistics, Printf, Dates

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const RES = joinpath(@__DIR__, "wind_collapse_layers_results.txt")
const L, D = 28, 1536
BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ 2))

loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 1:L])      # Js[k+1] = J_k

r3 = Dict(p => Float64[] for p in PROMPT_IDXS)
open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T5b (EXPLORATOIRE) -- effet du gel par couche ; test de rang ≤ 2 de J_AN − J_A")
    emit("Date : " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PROMPT_IDXS
        JT = loadJ(p, "T"); JA = loadJ(p, "A"); JN = loadJ(p, "AN")
        emit(@sprintf("\nPROMPT %d", p))
        emit("    k   σ₁(J_T)   gelQK/σ₁   gelNorm/σ₁   σ₂/σ₁(ΔN)   σ₃/σ₁(ΔN)   σ₁(J_AN)/σ₁(J_T)")
        for k in 0:27
            sT = svdvals(JT[k+1]); sN = svdvals(JN[k+1])
            dQ = opnorm(JA[k+1] - JT[k+1]); sD = svdvals(JN[k+1] - JA[k+1])
            k >= 1 && push!(r3[p], sD[3] / sD[1])
            emit(@sprintf("   %2d   %7.2f    %7.3f     %7.3f      %7.4f     %7.4f        %7.2f",
                          k, sT[1], dQ / sT[1], sD[1] / sT[1], sD[2] / sD[1], sD[3] / sD[1], sN[1] / sT[1]))
        end
        JT = nothing; JA = nothing; JN = nothing; GC.gc()
    end
    emit("\nTEST F1 (rang ≤ 2 de J_AN − J_A, couches 1..27) : σ₃/σ₁ médian / max par prompt")
    for p in PROMPT_IDXS
        emit(@sprintf("   p%-3d  %.4f / %.4f", p, median(r3[p]), maximum(r3[p])))
    end
end
println("\nÉcrit : ", RES)
