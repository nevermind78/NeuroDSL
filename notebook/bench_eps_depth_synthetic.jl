# =============================================================================
# PROFONDEUR ET IDENTITÉS EXACTES -- PILE PRÉ-NORM SYNTHÉTIQUE, INITIALISATION
# ALÉATOIRE.
#
# CE QUE CE SCRIPT EST, ET CE QU'IL N'EST PAS
#   Il mesure n_bar et p sur une pile résiduelle pré-norm SYNTHÉTIQUE à poids
#   ALÉATOIRES, de L = 4 à L = 128. Ce n'est PAS une mesure sur un modèle
#   entraîné et rien ici n'autorise à conclure sur un modèle entraîné : les
#   poids ne sont pas entraînés, les branches ne sont pas de l'attention, et
#   la seule profondeur entraînée dont on dispose est L = 28 (Qwen2.5-1.5B,
#   voir bench_eps_exact_ablation_qwen_results.txt). Les deux régimes ne
#   doivent jamais être mélangés.
#
# CE QU'IL VALIDE
#   1. les deux portes d'exactitude (q_j0 = 1 au site lu, q_j = 0 en amont) ;
#   2. l'identité d'ablation contre la différence finie, en double précision ;
#   3. la loi de signe sign(q_j) = sign(rho_j + c_j) ;
#   4. l'enveloppe exacte -rho/(1-rho) <= q <= rho/(1+rho) et le seuil q = 1/2 ;
#   5. l'identité de transport q^(s) = <beta, M g>/<g, M g>, M = T'T, et son
#      cas conforme (invariance exacte de site) ;
#   6. la famille de contre-exemples donnant n_bar < 1 et n_bar < 0 ;
#   7. la décroissance rho_j ~ j^{-1/2} imposée par la pré-norm, et la
#      croissance de n_bar en log L qui s'en déduit.
#
# Dépendances : bibliothèque standard seulement (LinearAlgebra, Random,
# Printf, Statistics). Aucun paquet ajouté, aucun Project.toml touché.
# =============================================================================

using LinearAlgebra, Random, Printf, Statistics

const OUT = joinpath(@__DIR__, "bench_eps_depth_synthetic_results.txt")
const IO_ = open(OUT, "w")
say(s="") = (println(s); println(IO_, s))
sayf(fmt, args...) = say(Printf.format(Printf.Format(fmt), args...))

say("PROFONDEUR ET IDENTITÉS EXACTES -- pile pré-norm SYNTHÉTIQUE, init ALÉATOIRE")
say("Poids NON entraînés. Ne pas lire comme une mesure sur un modèle entraîné.")
say("Double précision, dérivées écrites à la main, aucune différence finie")
say("dans les identités (seule la comparaison DF en utilise une).")
sayf("Date : %s", "2026-08-11T14:00:00Z")
say()

# ---------------------------------------------------------------------------
# Pile pré-norm : m = 2 branches séquentielles par couche.
#   branche(x) = W2 * tanh(W1 * rmsnorm(x)),  W ~ N(0, 1/fan_in)
# Le site lu est la SORTIE de la branche 1 (position 1), capturée APRÈS sa
# porte : la porte 1 est donc obligatoire et B = 2L.
# ---------------------------------------------------------------------------
rmsn(x) = x ./ sqrt(mean(abs2, x) + 1e-6)
struct Br; W1::Matrix{Float64}; W2::Matrix{Float64}; end
mkbr(D, Dh, rng) = Br(randn(rng, Dh, D)/sqrt(D), randn(rng, D, Dh)/sqrt(Dh))
fwd(b::Br, x) = b.W2 * tanh.(b.W1 * rmsn(x))

function vjp(b::Br, x, v)                       # J^T v, écrit à la main
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

# lecture au site : g = eps_1 * (cotangente arrivant au joint de la branche 1)
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

# rho_j et c_j viennent des deux mêmes passes : beta_j et sigma_j = g - beta_j
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

# ===========================================================================
say(repeat("=", 78))
say("PORTES D'EXACTITUDE ET IDENTITÉS (L = 16, 6 graines)")
say(repeat("=", 78)); say()

let
    dev_mand = 0.0; dev_up = 0.0; dev_two = 0.0; dev_fd = 0.0
    sign_bad = 0; env_bad = 0; half_bad = 0; ntot = 0; nneg = 0
    for s in 1:6
        S = build(16; seed=100+s); C = coeffs(S); T2 = two_scalar(C.g, C.bet)
        dev_mand = max(dev_mand, abs(C.q[1] - 1.0))
        # porte amont : fermer une porte AMONT du site (branche hors cône).
        # Ici le cône est le suffixe complet, donc on la teste sur un site
        # placé plus loin : site à la position k, portes 1..k-1 hors cône.
        k = 9
        gk(e) = begin
            v = copy(S.d)
            for j in S.B:-1:k+1; v = v + e[j]*vjp(S.brs[j], S.xs[j], v); end
            e[k]*v
        end
        g0 = gk(ones(S.B))
        for j in 1:k-1
            dev_up = max(dev_up, norm(gk(close_one(S, j)) - g0)/norm(g0))
        end
        for j in 2:S.B
            dev_two = max(dev_two, abs(C.q[j] - T2.qf[j])); ntot += 1
            r = T2.rho[j]; cc = T2.c[j]; qq = C.q[j]
            if sign(qq) != sign(r + cc); sign_bad += 1; end
            if qq < 0; nneg += 1; end
            lo = r < 1 ? -r/(1-r) : r/(1+r)
            hi = r < 1 ? r/(1+r)  : r/(r-1)
            if qq < lo - 1e-12 || qq > hi + 1e-12; env_bad += 1; end
            if (qq < 0.5) != (r < 1) && abs(r-1) > 1e-12; half_bad += 1; end
        end
        # ablation exacte contre différence finie centrée en log eps, h = 0.05
        h = 0.05
        gp = readout(S, fill(exp(h), S.B)); gm = readout(S, fill(exp(-h), S.B))
        nb_fd = (log(norm(gp)^2) - log(norm(gm)^2))/(4h)
        dev_fd = max(dev_fd, abs(nb_fd - C.nbar))
    end
    sayf("q_j0 = 1 au site lu           : écart max = %.3e", dev_mand)
    sayf("q_j = 0 en amont (relatif)    : écart max = %.3e", dev_up)
    sayf("forme à deux scalaires        : écart max = %.3e  sur %d coefficients", dev_two, ntot)
    sayf("ablation exacte vs DF (h=.05) : écart max sur n_bar = %.3e", dev_fd)
    say()
    sayf("loi de signe sign(q)=sign(rho+c)      : %d violation(s) sur %d", sign_bad, ntot)
    sayf("enveloppe exacte en rho seul          : %d violation(s) sur %d", env_bad, ntot)
    sayf("seuil q < 1/2 <=> rho < 1             : %d violation(s) sur %d", half_bad, ntot)
    sayf("coefficients négatifs                 : %d (%.1f %%)", nneg, 100*nneg/ntot)
end
say()

# ===========================================================================
say(repeat("=", 78))
say("VÉRITÉ TERRAIN : DÉVELOPPEMENT EXHAUSTIF SUR LES 2^B SOUS-ENSEMBLES")
say(repeat("=", 78)); say()
say("Instance synthétique, facteurs délibérément non normaux et non commutants,")
say("B = 12 donc 4096 sous-ensembles. Les coefficients a_k sont calculés")
say("exactement, ce qui donne n_bar sans passer par aucune identité.")
say()

let
    D = 48; B = 12
    rng = MersenneTwister(20260811)
    Ks = [randn(rng, D, D) .* (0.15 + 0.35*rand(rng)) ./ sqrt(D) .*
          (1 .+ 2*rand(rng, D, D).^3) for _ in 1:B]
    d0 = randn(rng, D)
    prod1(fs) = foldl(*, fs; init=Matrix{Float64}(I, D, D))
    G(e) = prod1([(I + e[j]*Ks[j]) for j in 1:B])*d0
    g1 = G(ones(B))
    # (a) vérité terrain : a_k par énumération des sous-ensembles
    a = [zeros(D) for _ in 0:B]
    for mask in 0:(2^B - 1)
        M = Matrix{Float64}(I, D, D); k = 0
        for j in 1:B
            if (mask >> (j-1)) & 1 == 1; M = M*Ks[j]; k += 1; end
        end
        a[k+1] .+= M*d0
    end
    w  = [dot(a[k+1], g1)/dot(g1, g1) for k in 0:B]
    nb_truth = sum(k*w[k+1] for k in 0:B)
    # (b) identité d'ablation
    bet = [g1 - G([k == j ? 0.0 : 1.0 for k in 1:B]) for j in 1:B]
    q_abl = [dot(bet[j], g1)/dot(g1, g1) for j in 1:B]
    # (c) forme à deux scalaires
    q_two = zeros(B)
    for j in 1:B
        b = bet[j]; s = g1 - b
        r = norm(b)/norm(s); c = dot(b, s)/(norm(b)*norm(s))
        q_two[j] = r*(r + c)/(1 + 2*r*c + r^2)
    end
    # (d) différence finie centrée en log eps
    h = 0.05
    nb_fd = (log(norm(G(fill(exp(h), B)))^2) - log(norm(G(fill(exp(-h), B)))^2))/(4h)

    sayf("n_bar par énumération exacte des 2^%d sous-ensembles : %.12f", B, nb_truth)
    sayf("n_bar par identité d'ablation (théorème)            : %.12f   écart %.3e",
         sum(q_abl), abs(sum(q_abl) - nb_truth))
    sayf("n_bar par forme à deux scalaires                    : %.12f   écart %.3e",
         sum(q_two), abs(sum(q_two) - nb_truth))
    sayf("n_bar par différence finie centrée (h = %.2f)        : %.12f   écart %.3e",
         h, nb_fd, abs(nb_fd - nb_truth))
    sayf("écart max par coefficient, ablation vs deux scalaires : %.3e",
         maximum(abs.(q_abl .- q_two)))
    sayf("somme des poids w_k (doit valoir 1)                  : %.12f", sum(w))
    sayf("w_k minimal : %+.4e à k = %d   -> des w_k NÉGATIFS existent : %s",
         minimum(w), argmin(w)-1, any(w .< 0) ? "OUI" : "non")
    say()
    say("La différence finie est correcte à ~1e-3 près, les deux identités à la")
    say("précision machine : elles ne corrigent pas l'estimateur, elles rendent")
    say("les coefficients par branche observables, ce qu'aucun schéma à deux")
    say("points ne peut faire.")
end
say()

# ===========================================================================
say(repeat("=", 78))
say("IDENTITÉ DE TRANSPORT ENTRE DEUX SITES")
say(repeat("=", 78)); say()

let
    D = 32; B1 = 10
    rng = MersenneTwister(11)
    prod1(fs) = foldl(*, fs; init=Matrix{Float64}(I, D, D))
    Ks = [randn(rng, D, D)*0.25/sqrt(D) for _ in 1:B1]
    Tf = [randn(rng, D, D)*0.25/sqrt(D) for _ in 1:3]
    d0 = randn(rng, D)
    gt(e) = prod1([(I + e[j]*Ks[j]) for j in 1:B1])*d0
    T  = prod1([(I + Tf[k]) for k in 1:3])
    on = ones(B1)
    gT = gt(on); gS = T*gT
    bet = [gT - gt([k == j ? 0.0 : 1.0 for k in 1:B1]) for j in 1:B1]
    qt = [dot(bet[j], gT)/dot(gT, gT) for j in 1:B1]
    qs = [1 - dot(T*gt([k == j ? 0.0 : 1.0 for k in 1:B1]), gS)/dot(gS, gS) for j in 1:B1]
    M  = T'*T
    qs_pred = [dot(bet[j], M*gT)/dot(gT, M*gT) for j in 1:B1]
    hs = T'*gS
    qs_h = [dot(bet[j], hs)/dot(gT, hs) for j in 1:B1]
    Q, _ = qr(randn(MersenneTwister(5), D, D)); Tc = 2.7*Matrix(Q)
    gSC = Tc*gT
    qsc = [1 - dot(Tc*gt([k == j ? 0.0 : 1.0 for k in 1:B1]), gSC)/dot(gSC, gSC) for j in 1:B1]
    sayf("q_j^(s) = <beta_j, M g^(t)>/<g^(t), M g^(t)>, M = T'T : écart max %.3e", maximum(abs.(qs .- qs_pred)))
    sayf("ligne engendrée par le seul vecteur h_s = T' g^(s)   : écart max %.3e", maximum(abs.(qs .- qs_h)))
    # défaut de métrique : q^(s)_j - q^(t)_j = <beta_j, u>, u indépendant de j
    u = (M*gT)/dot(gT, M*gT) - gT/dot(gT, gT)
    sayf("q^(s)_j - q^(t)_j = <beta_j, u>, u = Mg/<g,Mg> - g/||g||^2 : écart max %.3e",
         maximum(abs.((qs .- qt) .- [dot(bet[j], u) for j in 1:B1])))
    sayf("  ||u|| = %.5f ; u = 0 ssi Mg est colinéaire à g (transport conforme)", norm(u))
    sayf("dépendance au site réelle : max |q^(s) - q^(t)| = %.4f  (|q^(t)| moyen %.4f)",
         maximum(abs.(qs .- qt)), mean(abs.(qt)))
    sayf("T conforme (2.7 x orthogonale) : max |q^(s) - q^(t)| = %.3e  -> invariance EXACTE",
         maximum(abs.(qsc .- qt)))
end
say()

# ===========================================================================
say(repeat("=", 78))
say("CONTRE-EXEMPLE : n_bar < 1, ET NON BORNÉ INFÉRIEUREMENT")
say(repeat("=", 78)); say()
say("Site à porte obligatoire, une branche facultative, K v = -lambda v :")
say("theorie  n_bar = 1 - lambda/(1-lambda)  (soit rho = lambda, c = -1)")
say()
let
    D = 32; v = randn(MersenneTwister(3), D); h = 1e-7
    sayf("  %8s  %14s  %14s", "lambda", "n_bar mesuré", "n_bar théorie")
    for lam in [0.3, 0.5, 0.6, 0.9, 0.99]
        K = -lam*(v*v')/dot(v, v)
        g(e) = e*((I + e*K)*v)
        nb = dot((g(1+h) - g(1-h))/(2h), g(1.0))/dot(g(1.0), g(1.0))
        sayf("  %8.2f  %+14.5f  %+14.5f", lam, nb, 1 - lam/(1-lam))
    end
end
say()
say("=> n_bar >= 1 est FAUX en général : q_j0 = 1 ne contraint pas la somme")
say("   des autres coefficients, qui ne sont pas de signe constant.")
say()

# ===========================================================================
say(repeat("=", 78))
say("PRÉ-NORM : rho_j DÉCROÎT EN j^{-1/2}  (L = 32, 6 graines)")
say(repeat("=", 78)); say()

let
    sayf("  %4s  %10s  %14s  %10s", "j", "rho_j méd", "rho_j*sqrt(j)", "q_j méd")
    acc = Dict{Int,Vector{Float64}}(); accq = Dict{Int,Vector{Float64}}()
    for s in 1:6
        S = build(32; seed=200+s); C = coeffs(S); T2 = two_scalar(C.g, C.bet)
        for j in [2, 4, 8, 16, 32, 64]
            push!(get!(acc, j, Float64[]), T2.rho[j])
            push!(get!(accq, j, Float64[]), C.q[j])
        end
    end
    for j in [2, 4, 8, 16, 32, 64]
        r = median(acc[j])
        sayf("  %4d  %10.4f  %14.3f  %+10.4f", j, r, r*sqrt(j), median(accq[j]))
    end
end
say()
say("La pré-norm fixe l'échelle de l'ENTRÉE de branche ; le flux résiduel")
say("accumule les sorties de façon incohérente, donc ||x_j|| ~ sqrt(j) et le")
say("jacobien de branche porte un facteur 1/RMS(x_j) ~ j^{-1/2}. D'où")
say("q_j ~ rho_j^2 ~ 1/j et n_bar ~ somme 1/j ~ log L.")
say()

# ===========================================================================
say(repeat("=", 78))
say("BALAYAGE EN PROFONDEUR : n_bar ET p CONTRE L  (8 graines par L)")
say(repeat("=", 78)); say()

Ls = [4, 8, 16, 32, 64, 128]
nb_med = Float64[]; p_med = Float64[]
sayf("  %5s  %5s  %10s  %10s  %10s  %10s", "L", "B", "n_bar méd", "n_bar éc-t", "p méd", "n_bar/logL")
for L in Ls
    vals = [coeffs(build(L; seed=300+s)) for s in 1:8]
    nb = median(getfield.(vals, :nbar)); pp = median(getfield.(vals, :p))
    push!(nb_med, nb); push!(p_med, pp)
    sayf("  %5d  %5d  %10.4f  %10.4f  %10.5f  %10.4f",
         L, 2L, nb, std(getfield.(vals, :nbar)), pp, nb/log(L))
end
say()

# --- deux modèles ajustés sur la MÊME cible n_bar, R² comparables -----------
say("Deux formes ajustées à n_bar par moindres carrés, R² calculé sur n_bar :")
let
    x = log.(Ls); y = nb_med; ȳ = mean(y); sst = sum((y .- ȳ).^2)
    # (a) n_bar = a + b log L : linéaire
    Xa = hcat(ones(length(x)), x); βa = Xa \ y; ra = y - Xa*βa
    r2a = 1 - sum(ra.^2)/sst
    # (b) n_bar = a L^alpha : grille sur alpha, a par moindres carrés
    best = (-Inf, 0.0, 0.0)
    for al in 0.0:0.001:1.5
        z = Ls .^ al; a = dot(z, y)/dot(z, z); r = y - a*z
        r2 = 1 - sum(r.^2)/sst
        if r2 > best[1]; best = (r2, a, al); end
    end
    sayf("  n_bar = %+.4f %+.4f log L        R² = %.4f   résidu rel. max %.3f",
         βa[1], βa[2], r2a, maximum(abs.(ra ./ y)))
    sayf("  n_bar = %+.4f L^%.3f            R² = %.4f   résidu rel. max %.3f",
         best[2], best[3], best[1], maximum(abs.((y .- best[2]*(Ls .^ best[3])) ./ y)))
    say()
    sayf("  linéaire en L exigerait n_bar x%.0f de L=4 à L=128 ; mesuré x%.2f",
         Ls[end]/Ls[1], nb_med[end]/nb_med[1])
    sayf("  log L prédit x%.2f", log(Ls[end])/log(Ls[1]))
    say()
    if r2a > best[1]
        say("  => la forme logarithmique l'emporte sur la meilleure loi de")
        say("     puissance, sur la même cible et avec le même critère.")
    else
        say("  => la loi de puissance l'emporte ; la lecture log L ne tient pas.")
    end
    say()
    sayf("  p décroît de %.4f (L=%d) à %.5f (L=%d), soit un facteur %.1f,",
         p_med[1], Ls[1], p_med[end], Ls[end], p_med[1]/p_med[end])
    sayf("  tandis que L croît d'un facteur %d : p ~ log L / L.", Ls[end]÷Ls[1])
end
say()
say(repeat("=", 78))
say("PORTÉE")
say(repeat("=", 78))
say("Poids ALÉATOIRES, branches non attentionnelles, une seule largeur")
say("(D = 64, Dh = 256), un seul site par pile (position 1). La croissance")
say("en log L est donc établie À L'INITIALISATION et pour cette famille")
say("seulement. Sur le seul modèle ENTRAÎNÉ disponible (L = 28) on mesure")
say("n_bar = 8.9678 là où la tendance à l'initialisation prédirait bien")
say("moins : profondeur et entraînement ne sont pas séparables avec ces")
say("données. Lever le confondant demande n_bar à profondeur variable et à")
say("entraînement APPARIÉ, donc plusieurs modèles entraînés d'une même")
say("famille -- non disponibles ici.")
close(IO_)
println("\n-> $OUT")
