# =============================================================================
# VÉRIFICATION : le second moment de c_j explique-t-il l'écart entre le
# kappa^2 fitté (0.864, Prop. idealsum) et le kappa_rho^2 mesuré directement
# (~1, via rho_j*sqrt(j)) ?
#
# Piste testée (proposée après lecture de la Remarque rem:alignmentgap) :
# en développant q(rho,c) = rho*c + rho^2*(1-2c^2) + O(rho^3) et en sommant
# sur les branches avec E[c_j] ~ 0 mais Var(c_j) = sigma_c^2 != 0, le terme
# quadratique effectif devient kappa_rho^2 * (1 - 2*sigma_c^2), ce qui prédit
# kappa_eff^2 < kappa_rho^2. Prédiction numérique à tester : si
# kappa_rho^2 ~ 1 et kappa^2 fitté = 0.864, alors sigma_c^2 ~ 0.068.
#
# Ce script réutilise EXACTEMENT le même modèle et les mêmes graines
# (seed = 200+s, s=1..6, L=32) que bench_eps_depth_synthetic.jl, pour d'abord
# reproduire les chiffres déjà publiés (moyenne 0.037, plage [-0.672,0.784],
# 378 valeurs, 56% positifs) comme garde-fou, puis calculer la variance que
# le script original ne conservait pas.
#
# Aucune nouvelle mesure sur un modèle entraîné ; aucun paquet ajouté.
# =============================================================================

using LinearAlgebra, Random, Printf, Statistics

const OUT = joinpath(@__DIR__, "bench_eps_alignment_variance_check_results.txt")
const IO_ = open(OUT, "w")
say(s="") = (println(s); println(IO_, s))
sayf(fmt, args...) = say(Printf.format(Printf.Format(fmt), args...))

# --- copié tel quel depuis bench_eps_depth_synthetic.jl (mêmes fonctions) ---
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
# -----------------------------------------------------------------------------

say(repeat("=", 78))
say("SECOND MOMENT DE c_j, L=32, 6 GRAINES (mêmes graines que la Remarque")
say("rem:alignmentgap : seed = 200+s, s=1..6)")
say(repeat("=", 78)); say()

c_all = Float64[]
rho_all = Float64[]
kappa2_pointwise = Float64[]   # rho_j^2 * j, un par branche

for s in 1:6
    S = build(32; seed=200+s)
    C = coeffs(S)
    T2 = two_scalar(C.g, C.bet)
    for j in 2:S.B   # j=1 est la branche obligatoire, non "bypassable"
        push!(c_all, T2.c[j])
        push!(rho_all, T2.rho[j])
        push!(kappa2_pointwise, T2.rho[j]^2 * j)
    end
end

n = length(c_all)
mean_c = mean(c_all); std_c = std(c_all); var_c = var(c_all)
min_c, max_c = extrema(c_all)
frac_pos = 100*count(>(0), c_all)/n

say("--- Garde-fou : reproduction des chiffres déjà publiés ---")
sayf("n (branches contournables x graines) : %d  (attendu 378)", n)
sayf("moyenne de c_j                       : %.4f  (publié : 0.037)", mean_c)
sayf("plage de c_j                         : [%.4f, %.4f]  (publié : [-0.672, 0.784])", min_c, max_c)
sayf("%% de valeurs positives               : %.1f %%  (publié : 56%%)", frac_pos)
say()

say("--- Second moment, non publié jusqu'ici ---")
sayf("écart-type de c_j  (sigma_c)          : %.4f", std_c)
sayf("variance de c_j    (sigma_c^2)        : %.4f", var_c)
say()

say("--- kappa_rho^2 mesuré directement, sur les 378 points (pas seulement")
say("    les 6 points médians j in {2,4,8,16,32,64} du script original) ---")
kr2_mean = mean(kappa2_pointwise); kr2_med = median(kappa2_pointwise)
sayf("kappa_rho^2 = rho_j^2 * j : moyenne %.4f, médiane %.4f, plage [%.4f, %.4f]",
     kr2_mean, kr2_med, extrema(kappa2_pointwise)...)
say()

say("--- Prédiction : kappa_eff^2 = kappa_rho^2 * (1 - 2*sigma_c^2) ---")
say("(depuis q(rho,c) = rho*c + rho^2*(1-2c^2) + O(rho^3), E[c]~0)")
kappa_fit2 = 0.864
for (label, kr2) in [("kappa_rho^2 = 1 (approximation grossière)", 1.0),
                     ("kappa_rho^2 = moyenne mesurée sur 378 pts", kr2_mean),
                     ("kappa_rho^2 = médiane mesurée sur 378 pts", kr2_med)]
    pred = kr2 * (1 - 2*var_c)
    sayf("  %-42s -> kappa_eff^2 prédit = %.4f  (fitté : %.4f, écart %.1f %%)",
         label, pred, kappa_fit2, 100*abs(pred-kappa_fit2)/kappa_fit2)
end
say()

say("--- sigma_c^2 qu'il FAUDRAIT pour que la prédiction tombe exactement")
say("    sur le kappa^2 fitté (0.864), à kappa_rho^2 donné ---")
for (label, kr2) in [("kappa_rho^2 = 1", 1.0),
                     ("kappa_rho^2 = moyenne mesurée", kr2_mean),
                     ("kappa_rho^2 = médiane mesurée", kr2_med)]
    needed = (1 - kappa_fit2/kr2)/2
    sayf("  %-30s -> sigma_c^2 requis = %.4f  (mesuré : %.4f)", label, needed, var_c)
end
say()

say("--- Corrélation rho_j <-> c_j (l'hypothèse d'indépendance implicite) ---")
r_rho_c = cor(rho_all, c_all)
r_kappa2_c = cor(kappa2_pointwise, c_all)
sayf("corr(rho_j, c_j)          : %.4f", r_rho_c)
sayf("corr(rho_j^2*j, c_j)      : %.4f", r_kappa2_c)
say()

close(IO_)
println("\n-> $OUT")
