# WIND-T7 (EXPLORATOIRE) -- structure de W et de G = (I − W)⁻¹ à partir de wind_T7_lift.json
using LinearAlgebra, Statistics, Printf, JSON
d = JSON.parsefile(joinpath(@__DIR__, "wind_data", "wind_T7_lift.json"))
function loadW(v)
    M = reduce(hcat, [Float64.(c) for c in v])     # colonnes JSON -> matrice
    up = norm(triu(M, 1)); lo = norm(tril(M, -1))
    up > lo ? Matrix(M') : M                        # orientation : W strictement inférieure
end
blk(i) = 2i-1:2i
for p in (12, 22, 17, 5, 6)
    W = loadW(d["p$p"]["W"]); m = size(W, 1)
    G = inv(I - W)
    println(@sprintf("p%d : ‖triu(W)‖ = %.1e  det(I−W) = %.6f  σ(G) = %s", p, norm(triu(W, 0)), det(I - W),
                     join([@sprintf("%.2f", s) for s in svdvals(G)[1:5]], " ")))
    hw = [svdvals(W[2i+1:end, 1:2i]) for i in 1:26]; hg = [svdvals(G[2i+1:end, 1:2i]) for i in 1:26]
    println("   Hankel W σ₂/σ₁ : " * join([@sprintf("%.2f", h[2]/h[1]) for h in hw[2:25]], " "))
    println("   Hankel W σ₃/σ₁ : " * join([@sprintf("%.2f", h[3]/h[1]) for h in hw[2:25]], " "))
    println("   Hankel G σ₁    : " * join([@sprintf("%.1f", h[1]) for h in hg[2:25]], " "))
    println("   Hankel G σ₂/σ₁ : " * join([@sprintf("%.2f", h[2]/h[1]) for h in hg[2:25]], " "))
end
