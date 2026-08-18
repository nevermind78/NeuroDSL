# =============================================================================
# VERIFICATION INDEPENDANTE ET FRAICHE -- INSTANCE DE VERITE TERRAIN DISTINCTE
#
# Objectif : verifier, sur une instance synthetique NEUVE (graine differente,
# B different, facteurs differents) et disjointe de celle de
# bench_eps_depth_synthetic.jl (section "VERITE TERRAIN", B=12, graine
# 20260811), les enonces suivants de branch_order_theorem.tex :
#
#   - Thm 3 (thm:ablation)   : q_j exact via B+1 evaluations (fermeture d'une
#                              porte a la fois) contre la vraie decomposition
#                              multilineaire obtenue par enumeration.
#   - Thm 6 (thm:twoscalar)  : q_j = q(rho_j, c_j), forme a deux scalaires.
#   - Cor 7 (cor:sign)       : sign(q_j) = sign(rho_j + c_j).
#   - Cor 8 (cor:envelope)   : enveloppe exacte -rho/(1-rho) <= q <= rho/(1+rho)
#                              (rho<1) et rho/(1+rho) <= q <= rho/(rho-1)
#                              (rho>1), et l'equivalence q>1/2 <=> rho>1.
#   - Cor 5 (cor:self)       : branche obligatoire, q_{j0} = 1 exact.
#   - Cor 4 (cor:upstream)   : branches hors du cone, q_j = 0 exact.
#
# CE QUI REND CETTE VERIFICATION INDEPENDANTE DE bench_eps_depth_synthetic.jl :
#   1. graine differente (20260815 contre 20260811) ;
#   2. B = 16 au lieu de B = 12 (2^16 = 65536 sous-ensembles enumeres, contre
#      4096) ;
#   3. la verite terrain n'est PAS seulement n_bar agrege : on accumule, pour
#      CHAQUE branche j, la somme des poids des sous-ensembles qui contiennent
#      j (bp[j] = somme_{S ni j} a_S), ce qui donne un q_j de verite terrain
#      PAR BRANCHE, par un chemin de calcul completement different de
#      l'identite d'ablation (combinatoire pure contre soustraction de deux
#      passes) ;
#   4. l'instance encode explicitement une branche obligatoire (site lu, cone
#      total) ET des branches hors cone (transparentes en amont), verifiees
#      toutes deux par enumeration exhaustive et pas seulement par un ecart
#      relatif sur une seule passe ;
#   5. les facteurs sont delibbrement heterogenes : quelques-uns a grande
#      norme pour forcer rho_j > 1 (regime le moins trivial de l'enveloppe et
#      du seuil q=1/2), et quelques-uns construits analytiquement au point
#      extreme c_j = -1 (reflexion -lambda le long du vecteur transporte) pour
#      garantir des q_j negatifs et un test serre de la loi de signe.
#
# Aucune dependance ajoutee : LinearAlgebra, Random, Printf, Statistics
# (bibliotheque standard). Aucun modele, aucun GPU. Cout : O(2^B * B * D^2).
# =============================================================================

using LinearAlgebra, Random, Printf, Statistics

const OUT = joinpath(@__DIR__, "bench_eps_fresh_ground_truth_check_results.txt")
const IO_ = open(OUT, "w")
say(s="") = (println(s); println(IO_, s))
sayf(fmt, args...) = say(Printf.format(Printf.Format(fmt), args...))

say("VERIFICATION INDEPENDANTE ET FRAICHE -- VERITE TERRAIN PAR ENUMERATION")
say("Instance synthetique NEUVE, graine et taille differentes de celle de")
say("bench_eps_depth_synthetic.jl. Verite terrain PAR BRANCHE (pas seulement")
say("agregee), obtenue par un chemin de calcul independant de l'identite")
say("d'ablation. Double precision, bibliotheque standard uniquement.")
sayf("Date : %s", "2026-08-15")
say()

let

t0 = time()

# ---------------------------------------------------------------------------
# Construction de l'instance
# ---------------------------------------------------------------------------
D     = 24
Mcone = 13          # branche 1 = obligatoire ; branches 2..13 = cone (12)
Bup   = 3           # branches 14..16 = en amont, hors cone
B     = Mcone + Bup # = 16

rng = MersenneTwister(20260815)   # graine FRAICHE, differente de 20260811

Q1raw, _ = qr(randn(rng, D, D))
C1 = 1.3 .* Matrix(Q1raw)          # facteur du site obligatoire : rotation*echelle,
                                    # preserve exactement angles et ratios de normes
d0 = randn(rng, D)

# Construction sequentielle des K_j (j = Mcone descendant a 2), avec quelques
# roles deliberement engineeres :
#   - 2 branches a grande norme          -> rho_j > 1 attendu
#   - 2 branches "reflexion" -lambda*v v'/(v'v) le long du vecteur transporte
#     courant -> point extreme EXACT c_j = -1, rho_j = lambda (q_j negatif,
#     valeur predite en forme close -lambda/(1-lambda))
#   - 8 branches generiques (echelle moderee, direction aleatoire)
K = Dict{Int,Matrix{Float64}}()
neg_lambda = Dict{Int,Float64}()
large_idx = Int[]
order = collect(Mcone:-1:2)   # ordre d'application dans la chaine
v = copy(d0)
for (pos, j) in enumerate(order)
    local Kj
    if pos <= 2
        Kj = 3.6 .* randn(rng, D, D) ./ sqrt(D)
        push!(large_idx, j)
    elseif pos <= 4
        lam = pos == 3 ? 0.40 : 0.85
        Kj = -lam .* (v * v') ./ dot(v, v)
        neg_lambda[j] = lam
    else
        Kj = 0.28 .* randn(rng, D, D) ./ sqrt(D)
    end
    K[j] = Kj
    v = v .+ Kj * v
end

Kfull = Dict{Int,Matrix{Float64}}()
for j in 2:Mcone
    Kfull[j] = K[j]
end
for j in (Mcone + 1):B
    Kfull[j] = zeros(D, D)   # branches en amont : transparentes en avant, K_j = 0
end

# ---------------------------------------------------------------------------
# G(eps) : fonction multilineaire reelle (evaluation directe, PAS l'enumeration)
#   branche 1 : F_1(eps_1) = eps_1 * C1               (obligatoire, sans terme I)
#   branches 2..Mcone : F_j(eps_j) = I + eps_j * K_j   (contournables)
#   branches Mcone+1..B : n'apparaissent pas dans la formule (hors cone)
# ---------------------------------------------------------------------------
function Gtrue(eps::Vector{Float64})
    v = copy(d0)
    for j in Mcone:-1:2
        v = v .+ eps[j] .* (K[j] * v)
    end
    eps[1] .* (C1 * v)
end

g = Gtrue(ones(B))

# ---------------------------------------------------------------------------
# (A) Identite d'ablation exacte, Thm. thm:ablation : B+1 evaluations
# ---------------------------------------------------------------------------
closej(j) = (e = ones(B); e[j] = 0.0; e)
bet = [g - Gtrue(closej(j)) for j in 1:B]
q_abl = [dot(bet[j], g) / dot(g, g) for j in 1:B]
nbar_abl = sum(q_abl)

# ---------------------------------------------------------------------------
# (B) Forme a deux scalaires, Thm. thm:twoscalar
# ---------------------------------------------------------------------------
rho = zeros(B); c = fill(NaN, B); q_two = zeros(B)
for j in 1:B
    b = bet[j]; s = g - b
    nb = norm(b); ns = norm(s)
    if nb < 1e-12
        rho[j] = 0.0; c[j] = NaN; q_two[j] = 0.0        # sigma = g, beta = 0 (amont)
    elseif ns < 1e-12
        rho[j] = Inf; c[j] = NaN; q_two[j] = 1.0          # sigma = 0 (obligatoire)
    else
        rho[j] = nb / ns
        c[j] = dot(b, s) / (nb * ns)
        q_two[j] = rho[j] * (rho[j] + c[j]) / (1 + 2 * rho[j] * c[j] + rho[j]^2)
    end
end

# ---------------------------------------------------------------------------
# (C) VERITE TERRAIN INDEPENDANTE : enumeration exhaustive des 2^B sous-ensembles
#     value_mask(S) = a_S . d0 = coefficient exact du monome eps^S dans le
#     developpement multilineaire de G (extraction combinatoire : bit=1
#     selectionne K_j, bit=0 selectionne I ; branche 1 speciale, sans I).
#     bp[j] = somme_{S ni j} a_S  =  d/deps_j G|_1   (definition meme de q_j)
#     -- chemin de calcul totalement independant de (A).
# ---------------------------------------------------------------------------
function value_mask(mask::Int)
    if (mask & 1) == 0
        return zeros(D)
    end
    vv = copy(d0)
    for j in B:-1:2
        if ((mask >> (j - 1)) & 1) == 1
            vv = Kfull[j] * vv
        end
    end
    C1 * vv
end

bp = [zeros(D) for _ in 1:B]
gsum = zeros(D)
nmasks = 2^B
for mask in 0:(nmasks - 1)
    val = value_mask(mask)
    gsum .+= val
    for j in 1:B
        if ((mask >> (j - 1)) & 1) == 1
            bp[j] .+= val
        end
    end
end

q_truth = [dot(bp[j], g) / dot(g, g) for j in 1:B]
nbar_truth = sum(q_truth)
dev_gsum = norm(gsum - g) / norm(g)   # identite multilineaire G(1)=somme_S a_S : recoupement interne

elapsed = time() - t0

# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
say(repeat("=", 78))
sayf("Instance : D=%d, B=%d (Mcone=%d dont 1 obligatoire + %d contournables, Bup=%d)",
     D, B, Mcone, Mcone - 1, Bup)
sayf("2^%d = %d sous-ensembles enumeres exhaustivement. Temps total : %.2f s", B, nmasks, elapsed)
say(repeat("=", 78))
say()

sayf("Recoupement interne  G(1) = somme_S a_S (sur les 2^%d sous-ensembles)  : ecart relatif %.3e",
     B, dev_gsum)
say()

say("--- (1) Thm. thm:ablation : q_j (B+1 evaluations) contre q_j de verite")
say("        terrain (enumeration combinatoire, chemin de calcul independant)")
dev_truth_vs_abl = maximum(abs.(q_truth .- q_abl))
sayf("    ecart max sur les %d coefficients : %.3e", B, dev_truth_vs_abl)
sayf("    n_bar par ablation      : %.12f", nbar_abl)
sayf("    n_bar par verite terrain: %.12f   ecart %.3e", nbar_truth, abs(nbar_abl - nbar_truth))
say()

say("--- (2) Thm. thm:twoscalar : q_j = q(rho_j,c_j) contre q_j de verite terrain")
dev_truth_vs_two = maximum(abs.(q_truth .- q_two))
dev_abl_vs_two = maximum(abs.(q_abl .- q_two))
sayf("    ecart max (verite terrain vs deux scalaires) : %.3e", dev_truth_vs_two)
sayf("    ecart max (ablation vs deux scalaires)        : %.3e", dev_abl_vs_two)
say()

say("--- (3) Cor. cor:self : branche obligatoire (j=1), q_{j0} = 1 exact")
sayf("    q_1 verite terrain = %.15f   ecart = %.3e", q_truth[1], abs(q_truth[1] - 1.0))
sayf("    q_1 ablation       = %.15f   ecart = %.3e", q_abl[1], abs(q_abl[1] - 1.0))
say()

say("--- (4) Cor. cor:upstream : branches hors cone (j = Mcone+1..B), q_j = 0 exact")
up_js = (Mcone + 1):B
dev_up_truth = maximum(abs.(q_truth[up_js]))
dev_up_abl = maximum(abs.(q_abl[up_js]))
sayf("    ecart max, verite terrain : %.3e  sur %d branche(s)", dev_up_truth, length(up_js))
sayf("    ecart max, ablation       : %.3e  sur %d branche(s)", dev_up_abl, length(up_js))
say()

say("--- (5) Cor. cor:sign : sign(q_j) = sign(rho_j + c_j)  --  Cor. cor:envelope :")
say("        enveloppe exacte en rho seul et seuil q_j > 1/2 <=> rho_j > 1")
say("        (teste sur les 12 branches du cone contournable, j=2..13,")
say("         hors la branche obligatoire et les branches hors cone)")
cone_js = 2:Mcone
sign_bad = 0; env_bad = 0; half_bad = 0
nrho_gt1 = 0; nneg = 0
max_env_slack = 0.0
for j in cone_js
    r = rho[j]; cc = c[j]; qq = q_truth[j]
    if qq < 0
        nneg += 1
    end
    if r > 1
        nrho_gt1 += 1
    end
    if sign(qq) != sign(r + cc) && abs(qq) > 1e-13
        sign_bad += 1
    end
    lo = r < 1 ? -r / (1 - r) : r / (1 + r)
    hi = r < 1 ? r / (1 + r) : r / (r - 1)
    if qq < lo - 1e-9 || qq > hi + 1e-9
        env_bad += 1
    end
    if (qq < 0.5) != (r < 1) && abs(r - 1) > 1e-9
        half_bad += 1
    end
end
sayf("    loi de signe   : %d violation(s) sur %d coefficients", sign_bad, length(cone_js))
sayf("    enveloppe      : %d violation(s) sur %d coefficients", env_bad, length(cone_js))
sayf("    seuil q > 1/2 <=> rho > 1 : %d violation(s) sur %d coefficients", half_bad, length(cone_js))
say()

sayf("    branches avec rho_j > 1 dans cette instance  : %d / %d", nrho_gt1, length(cone_js))
sayf("    branches avec q_j < 0 dans cette instance     : %d / %d", nneg, length(cone_js))
say()

say("--- Cas engineeres, verification en forme close ---")
sayf("    branches a grande norme (indices %s) : rho attendu > 1", string(large_idx))
for j in large_idx
    sayf("      j=%2d : rho_j = %.4f  c_j = %+.6f  q_j = %+.6f", j, rho[j], c[j], q_truth[j])
end
say()
say("    branches reflexion -lambda v v'/(v'v) : point extreme EXACT c_j=-1,")
say("    q_j predit = -lambda/(1-lambda) en forme close (Cor. cor:envelope)")
for (j, lam) in sort(collect(neg_lambda))
    qpred = -lam / (1 - lam)
    sayf("      j=%2d : lambda=%.2f  rho_j=%.6f (ecart %.2e)  c_j=%.6f (ecart a -1 : %.2e)  q_j=%+.8f  q_predit=%+.8f  ecart=%.3e",
         j, lam, rho[j], abs(rho[j] - lam), c[j], abs(c[j] - (-1.0)), q_truth[j], qpred, abs(q_truth[j] - qpred))
end
say()

say(repeat("=", 78))
say("RESUME DES ECARTS MAXIMAUX")
say(repeat("=", 78))
sayf("  recoupement interne G(1)=somme a_S                         : %.3e", dev_gsum)
sayf("  ablation vs verite terrain (par branche)                   : %.3e", dev_truth_vs_abl)
sayf("  n_bar ablation vs n_bar verite terrain                     : %.3e", abs(nbar_abl - nbar_truth))
sayf("  deux scalaires vs verite terrain                           : %.3e", dev_truth_vs_two)
sayf("  deux scalaires vs ablation                                 : %.3e", dev_abl_vs_two)
sayf("  branche obligatoire q_j0=1 (verite terrain / ablation)     : %.3e / %.3e", abs(q_truth[1]-1.0), abs(q_abl[1]-1.0))
sayf("  branches hors cone q_j=0 (verite terrain / ablation)       : %.3e / %.3e", dev_up_truth, dev_up_abl)
sayf("  loi de signe                                                : %d violation(s)", sign_bad)
sayf("  enveloppe exacte                                            : %d violation(s)", env_bad)
sayf("  seuil q>1/2 <=> rho>1                                       : %d violation(s)", half_bad)
say()
if dev_truth_vs_abl < 1e-9 && dev_truth_vs_two < 1e-9 && dev_up_truth < 1e-9 &&
   abs(q_truth[1] - 1.0) < 1e-9 && sign_bad == 0 && env_bad == 0 && half_bad == 0
    say("=> Toutes les identites verifiees tiennent a la precision machine sur")
    say("   cette instance fraiche et independante, y compris dans le regime")
    say("   rho_j > 1 et pour des coefficients q_j negatifs.")
else
    say("=> AU MOINS UN ECART DEPASSE LE SEUIL DE PRECISION MACHINE -- voir le")
    say("   detail ci-dessus.")
end

end # let

close(IO_)
println("\n-> $OUT")
