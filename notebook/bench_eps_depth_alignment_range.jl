# =============================================================================
# PLAGE RÉELLE DE c_j SUR LE MODÈLE PRÉ-NORM SYNTHÉTIQUE -- referme un chiffre
# non traçable
#
# POURQUOI CE SCRIPT : une plage "c_j de -0.12 à 0.40" a été citée en discutant
# le nouveau résultat digamma (Remark~alignmentgap de branch_order_theorem.tex)
# sans qu'aucun fichier notebook/*_results.txt ne la contienne -- elle vient
# d'une exploration de tout début de session sur une AUTRE instance (B=10,
# facteurs aléatoires génériques), pas sur CE modèle pré-norm. Ce script mesure
# la vraie plage sur le MÊME modèle et la MÊME profondeur (L=32) déjà utilisés
# dans bench_eps_depth_synthetic.jl (fonctions Br/build/coeffs/two_scalar
# copiées verbatim, mêmes graines 200+s, s=1..6), pour que le nombre cité ait
# une source.
#
# Usage : julia --project=. notebook/bench_eps_depth_alignment_range.jl
# =============================================================================

using LinearAlgebra, Random, Statistics, Printf

const OUT = joinpath(@__DIR__, "bench_eps_depth_alignment_range_results.txt")

rmsn(x) = x ./ sqrt(mean(abs2, x) + 1e-6)
struct Br; W1::Matrix{Float64}; W2::Matrix{Float64}; end
mkbr(D, Dh, rng) = Br(randn(rng, Dh, D)/sqrt(D), randn(rng, D, Dh)/sqrt(Dh))
fwd(b::Br, x) = b.W2 * tanh.(b.W1 * rmsn(x))
function vjp(b::Br, x, v)
    s = sqrt(mean(abs2, x) + 1e-6); r = x ./ s
    a = tanh.(b.W1 * r)
    dr = b.W1' * ((b.W2' * v) .* (1 .- a.^2))
    (dr .- r .* (dot(r, dr)/length(x))) ./ s
end
struct Stack; brs::Vector{Br}; xs::Vector{Vector{Float64}}; d::Vector{Float64}; B::Int; end
function build(L; D=64, Dh=256, seed=7)
    rng = MersenneTwister(seed); B = 2L
    brs = [mkbr(D, Dh, rng) for _ in 1:B]
    x = randn(rng, D); xs = Vector{Vector{Float64}}(undef, B)
    for j in 1:B; xs[j] = x; x = x + fwd(brs[j], x); end
    Stack(brs, xs, randn(rng, D), B)
end
function readout(S::Stack, eps::Vector{Float64})
    v = copy(S.d)
    for j in S.B:-1:2
        v = v + eps[j] * vjp(S.brs[j], S.xs[j], v)
    end
    eps[1] * v
end
open_all(S) = ones(S.B)
close_one(S, j) = (e = ones(S.B); e[j] = 0.0; e)
function coeffs(S::Stack)
    g = readout(S, open_all(S))
    bet = [g - readout(S, close_one(S, j)) for j in 1:S.B]
    q  = [dot(bet[j], g)/dot(g, g) for j in 1:S.B]
    (g=g, bet=bet, q=q, nbar=sum(q), p=sum(q)/S.B)
end
function two_scalar(g, bet)
    B = length(bet)
    rho = zeros(B); c = zeros(B)
    for j in 1:B
        b = bet[j]; s = g - b
        ns = norm(s); nb = norm(b)
        if ns < 1e-300
            rho[j] = Inf; c[j] = NaN
        else
            rho[j] = nb/ns; c[j] = dot(b, s)/(nb*ns)
        end
    end
    (rho=rho, c=c)
end

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("PLAGE DE c_j -- modèle pré-norm synthétique, L=32, 6 graines (200+s)")
    emit("Mêmes fonctions et graines que bench_eps_depth_synthetic.jl.")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("")
    allc = Float64[]
    for s in 1:6
        S = build(32; seed=200+s)
        C = coeffs(S)
        T2 = two_scalar(C.g, C.bet)
        cs = filter(isfinite, T2.c[2:end])   # branche 1 = obligatoire, c_1 non défini
        append!(allc, cs)
        emit(@sprintf("  graine %d : c_j min=%.4f max=%.4f  (sur %d branches contournables)",
                      s, minimum(cs), maximum(cs), length(cs)))
    end
    emit("")
    emit(@sprintf("PLAGE GLOBALE (6 graines x 63 branches contournables) : min=%.4f  max=%.4f",
                  minimum(allc), maximum(allc)))
    emit(@sprintf("moyenne = %.4f   médiane = %.4f   %% positifs = %.1f",
                  mean(allc), median(allc), 100*count(>(0), allc)/length(allc)))
end
println("\nÉcrit : ", OUT)
