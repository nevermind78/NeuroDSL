# WIND-T6 -- contrôle du théorème 2 (partie QK) : rang(E_k) ≤ 12·(n_tok − 1), E_k = J_T − J_A.
# On rapporte σ_{r}/σ₁ et σ_{r+1}/σ₁ avec r = 12(n_tok−1), et σ_{r+1}/σ₁ doit être au niveau du bruit (~3e-4).
using LinearAlgebra, Statistics, Printf, JSON
const OUTDIR = joinpath(@__DIR__, "wind_data"); const L, D = 28, 1536
BLAS.set_num_threads(16)
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
open(joinpath(@__DIR__, "wind_T6_qkrank_results.txt"), "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T6 contrôle : rang de E_k = J_T − J_A (théorème 2 : ≤ 12(n_tok − 1))")
    for p in (12, 22)
        n = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))["n"]; r = 12 * (n - 1)
        JT = loadJ(p, "T"); JA = loadJ(p, "A")
        a = Float64[]; b = Float64[]; c = Int[]
        for k in 1:27
            s = svdvals(JT[k] - JA[k]); push!(a, s[r] / s[1]); push!(b, s[r+1] / s[1]); push!(c, count(>(1e-3), s ./ s[1]))
        end
        emit(@sprintf("p%d : n_tok = %d, r = %d ; σ_r/σ₁ médiane %.2e [min %.2e] ; σ_{r+1}/σ₁ médiane %.2e [max %.2e]",
                      p, n, r, median(a), minimum(a), median(b), maximum(b)))
        emit("   rang numérique (σ_i/σ₁ > 1e-3) par couche k=1..27 : " * join(string.(c), " "))
        JT = nothing; JA = nothing; GC.gc()
    end
end
