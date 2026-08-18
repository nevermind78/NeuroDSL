# =============================================================================
# sigma_c^2 EST-ELLE STABLE EN PROFONDEUR ?
#
# Suite de bench_eps_alignment_variance_check.jl : ce script-là a mesuré
# sigma_c^2 = 0.0895 à L=32 seulement (6 graines, seed=200+s), et la
# prédiction kappa_eff^2 = kappa_rho^2*(1-2*sigma_c^2) ne fermait pas
# proprement sur le kappa^2=0.864 FITTÉ GLOBALEMENT sur les 6 profondeurs du
# sweep (L=4..128, table tab:depth du papier). Comparer une quantité
# mesurée à UNE profondeur à une constante fittée sur toutes les profondeurs
# suppose implicitement que sigma_c^2 est stable en L -- non vérifié.
#
# Ce script mesure sigma_c^2 (et la moyenne, la fraction positive, et
# kappa_rho^2) aux 6 MÊMES profondeurs et avec les MÊMES graines
# (seed = 300+s, s=1..8) que la section "BALAYAGE EN PROFONDEUR" de
# bench_eps_depth_synthetic.jl, qui a produit le n_bar déjà publié dans
# Table tab:depth (kappa^2=0.864 fitté dessus). Donc ce sweep utilise
# exactement les mêmes tirages aléatoires que la mesure qu'il cherche à
# expliquer.
#
# Aucune nouvelle mesure sur un modèle entraîné ; aucun paquet ajouté.
# =============================================================================

using LinearAlgebra, Random, Printf, Statistics

const OUT = joinpath(@__DIR__, "bench_eps_alignment_variance_depth_sweep_results.txt")
const IO_ = open(OUT, "w")
say(s="") = (println(s); println(IO_, s))
sayf(fmt, args...) = say(Printf.format(Printf.Format(fmt), args...))

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
    rho = zeros(B); c = zeros(B); qf = zeros(B)
    for j in 1:B
        b = bet[j]; s = g - b
        ns = norm(s); nb = norm(b)
        if ns < 1e-300
            rho[j] = Inf; c[j] = NaN; qf[j] = 1.0
        else
            rho[j] = nb/ns; c[j] = dot(b, s)/(nb*ns)
            qf[j] = rho[j]*(rho[j] + c[j])/(1 + 2*rho[j]*c[j] + rho[j]^2)
        end
    end
    (rho=rho, c=c, qf=qf)
end

say(repeat("=", 78))
say("sigma_c^2 CONTRE L -- 6 profondeurs, 8 graines chacune (seed=300+s),")
say("mêmes tirages que le balayage en profondeur déjà publié (Table tab:depth)")
say(repeat("=", 78)); say()

Ls = [4, 8, 16, 32, 64, 128]
sayf("  %5s  %6s  %8s  %10s  %10s  %12s  %10s",
     "L", "n", "moy c", "sigma_c", "sigma_c^2", "%% pos", "kappa_rho^2(méd)")

rows = NamedTuple[]
for L in Ls
    c_all = Float64[]; kappa2_pointwise = Float64[]
    for s in 1:8
        S = build(L; seed=300+s)
        C = coeffs(S)
        T2 = two_scalar(C.g, C.bet)
        for j in 2:S.B
            push!(c_all, T2.c[j])
            push!(kappa2_pointwise, T2.rho[j]^2 * j)
        end
    end
    mc = mean(c_all); vc = var(c_all); sc = std(c_all)
    fp = 100*count(>(0), c_all)/length(c_all)
    kr2 = median(kappa2_pointwise)
    push!(rows, (L=L, n=length(c_all), mc=mc, sc=sc, vc=vc, fp=fp, kr2=kr2))
    sayf("  %5d  %6d  %+8.4f  %10.4f  %10.4f  %10.1f %%  %14.4f",
         L, length(c_all), mc, sc, vc, fp, kr2)
end
say()

vcs = [r.vc for r in rows]
sayf("sigma_c^2 sur les 6 profondeurs : min %.4f, médiane %.4f, max %.4f", minimum(vcs), median(vcs), maximum(vcs))
sayf("CV de sigma_c^2 sur les 6 profondeurs : %.1f %%", 100*std(vcs)/mean(vcs))
say()

say("--- Prédiction par profondeur : kappa_eff^2(L) = kappa_rho^2(L) * (1 - 2*sigma_c^2(L)) ---")
say("--- comparée à la kappa^2=0.864 fittée GLOBALEMENT sur les 6 profondeurs ---")
kappa_fit2 = 0.864
for r in rows
    pred = r.kr2 * (1 - 2*r.vc)
    sayf("  L=%-4d  kappa_eff^2 prédit = %.4f  (fitté global : %.4f, écart %+.1f %%)",
         r.L, pred, kappa_fit2, 100*(pred-kappa_fit2)/kappa_fit2)
end
say()

close(IO_)
println("\n-> $OUT")
