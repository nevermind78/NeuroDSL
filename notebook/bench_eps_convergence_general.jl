# =============================================================================
# THÉORÈMES GÉNÉRAUX DE CONVERGENCE POUR somme_j q_j -- VÉRIFICATIONS DE
# SANITÉ SEULEMENT, JAMAIS LA SOURCE DES ÉNONCÉS
#
# ORDRE DES OPÉRATIONS, IMPOSÉ ET RESPECTÉ
#   Les propositions de branch_order_theorem.tex (Section convergence) sont
#   prouvées ANALYTIQUEMENT, pour des suites {rho_j},{c_j} ARBITRAIRES avec
#   |c_j| <= 1, à partir de la seule forme exacte à deux scalaires
#       q(rho,c) = rho(rho+c)/(1+2 rho c + rho^2).
#   Aucun ansatz de décroissance (rho_j ~ kappa/sqrt(j) ou autre) n'intervient
#   dans les preuves. Ce script ne DÉMONTRE rien : il (1) re-teste les
#   inégalités des preuves sur des tirages aléatoires, ce qui attraperait une
#   faute d'algèbre, et (2) teste si les HYPOTHÈSES des théorèmes sont
#   satisfaites par le modèle pré-norm à l'initialisation -- un test des
#   théorèmes, pas leur origine.
#
# CE QUI NE PEUT PAS ÊTRE TESTÉ ICI, ET POURQUOI
#   Le régime ENTRAÎNÉ : bench_eps_depth_trained_results.txt ne conserve que
#   des agrégats par profondeur (nbar1, qbar, rho_lb, nneg, nins), PAS les
#   suites (rho_j, c_j) branche par branche. Les conditions nécessaires des
#   Propositions portent sur ces suites. Elles sont donc testables en principe
#   mais pas contre ce qui est sauvegardé : il faudrait une capture
#   supplémentaire (un tableau par branche) lors d'une nouvelle exécution. Ce
#   script ne touche ni ce fichier ni son script.
#
# Usage : julia --project=. notebook/bench_eps_convergence_general.jl
# =============================================================================

using LinearAlgebra, Random, Statistics, Printf

const OUT = joinpath(@__DIR__, "bench_eps_convergence_general_results.txt")

q(rho, c) = rho*(rho + c)/(1 + 2*rho*c + rho^2)

# --- modèle pré-norm synthétique, fonctions copiées verbatim de
#     bench_eps_depth_synthetic.jl (mêmes graines) ------------------------
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
function rho_c_q(S::Stack)
    g = readout(S, ones(S.B))
    rho = Float64[]; cc = Float64[]; qq = Float64[]
    for j in 2:S.B                        # branche 1 = obligatoire, exclue
        e = ones(S.B); e[j] = 0.0
        sig = readout(S, e)               # sigma_j
        bet = g - sig                     # beta_j
        ns = norm(sig); nb = norm(bet)
        (ns < 1e-300) && continue
        push!(rho, nb/ns)
        push!(cc, dot(bet, sig)/(nb*ns))
        push!(qq, dot(bet, g)/dot(g, g))
    end
    (rho=rho, c=cc, q=qq)
end

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("THÉORÈMES GÉNÉRAUX DE CONVERGENCE -- VÉRIFICATIONS DE SANITÉ")
    emit("Les preuves sont analytiques et pour suites ARBITRAIRES ; ce fichier")
    emit("ne fait que (1) re-tester les inégalités, (2) tester les hypothèses")
    emit("sur le modèle pré-norm à l'initialisation.")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("")

    # ---------------------------------------------------------------------
    emit("-"^78)
    emit("(1) INÉGALITÉS DES PREUVES, sur tirages aléatoires (rho,c)")
    emit("-"^78)
    rng = MersenneTwister(7)
    nA = nB = nF = nC = 0
    worstD = Inf; worst_rem = 0.0
    for _ in 1:400_000
        rho = 5*rand(rng); c = 2*rand(rng)-1
        # Prop A (c=0) : q <= rho^2 toujours ; rho^2 <= 2q dès rho <= 1
        q0 = q(rho, 0.0)
        (q0 <= rho^2 + 1e-12) || (nA += 1)
        (rho > 1.0 || rho^2 <= 2*q0 + 1e-9) || (nA += 1)
        # Prop B : D >= 1/2 dès rho <= 1/4, pour tout c
        if rho <= 0.25
            D = 1 + 2*rho*c + rho^2
            worstD = min(worstD, D)
            (D >= 0.5 - 1e-12) || (nB += 1)
            # Prop F : |q - rho*c - rho^2| <= (45/8) rho^2 dès rho <= 1/4
            rem = q(rho, c) - rho*c - rho^2
            worst_rem = max(worst_rem, rho^2 > 1e-14 ? abs(rem)/rho^2 : 0.0)
            (abs(rem) <= (45/8)*rho^2 + 1e-12) || (nF += 1)
        end
        # Prop C : q = 0 exactement ssi c = -rho  (rho > 0)
        if rho > 1e-9 && rho <= 1.0
            (abs(q(rho, -rho)) < 1e-10) || (nC += 1)
        end
    end
    emit(@sprintf("  Prop A  q<=rho^2 et rho^2<=2q (rho<=1)       : %d violation(s)", nA))
    emit(@sprintf("  Prop B  D>=1/2 dès rho<=1/4, tout c          : %d violation(s)  (min D vu %.4f)", nB, worstD))
    emit(@sprintf("  Prop F  |q-rho c-rho^2|<=(45/8)rho^2         : %d violation(s)  (pire ratio vu %.3f, borne 5.625)", nF, worst_rem))
    emit(@sprintf("  Prop C  q(rho,-rho)=0 exactement             : %d violation(s)", nC))
    emit("")
    emit("  Ces tirages ne prouvent rien : ils attraperaient une faute d'algèbre.")
    emit("")

    # ---------------------------------------------------------------------
    emit("-"^78)
    emit("(2) LES HYPOTHÈSES SONT-ELLES SATISFAITES ? modèle pré-norm, INIT")
    emit("-"^78)
    emit("  L=32, 6 graines (200+s), les 63 branches contournables de chaque.")
    emit("")
    emit(@sprintf("  %6s %12s %12s %12s %12s %10s", "graine", "rho monot.?",
                  "max|C_N|", "somme rho^2", "somme rho|c|", "q_j>=0 ?"))
    mono_ok = 0; allc_partial = Float64[]
    for s in 1:6
        S = build(32; seed=200+s)
        R = rho_c_q(S)
        # rho_j décroissante ? (hypothèse de Dirichlet dans Prop F)
        drops = count(j -> R.rho[j+1] <= R.rho[j] + 1e-12, 1:length(R.rho)-1)
        frac_mono = drops/(length(R.rho)-1)
        ismono = frac_mono > 0.95
        ismono && (mono_ok += 1)
        # sommes partielles de c_j bornées ? (hypothèse de Dirichlet)
        C = cumsum(R.c); push!(allc_partial, maximum(abs.(C)))
        s_rho2 = sum(R.rho.^2); s_rhoc = sum(R.rho .* abs.(R.c))
        nneg = count(<(0), R.q)
        emit(@sprintf("  %6d %12s %12.3f %12.4f %12.4f %10s", s,
                      ismono ? "oui($(round(100*frac_mono))%)" : "non($(round(100*frac_mono))%)",
                      maximum(abs.(C)), s_rho2, s_rhoc,
                      nneg == 0 ? "oui" : "non($nneg)"))
    end
    emit("")
    emit(@sprintf("  rho_j décroissante sur >95%% des pas : %d/6 graines", mono_ok))
    emit(@sprintf("  max |somme partielle de c_j| sur les graines : %.3f", maximum(allc_partial)))
    emit("")

    # Prop A relie somme q_j et somme rho_j^2 quand c=0. Les données ont c!=0 :
    # on mesure donc les DEUX directement et on regarde si la relation survit.
    emit("-"^78)
    emit("(3) PROP A CONTRE LES DONNÉES : somme q_j et somme rho_j^2 se suivent-elles ?")
    emit("-"^78)
    emit("  Prop A (c=0) : somme q_j converge SSI somme rho_j^2 converge. Les données")
    emit("  ont c_j != 0, donc l'hypothèse de Prop A ne tient PAS ; on teste si sa")
    emit("  conclusion survit quand même. 3 graines par profondeur, médianes.")
    emit("")
    emit(@sprintf("  %5s %6s %12s %12s %12s %12s", "L", "N",
                  "somme q_j", "somme rho^2", "ratio", "(s.rho^2)/logN"))
    for L in [8, 16, 32, 64]
        sq = Float64[]; s2 = Float64[]
        for s in 1:3
            R = rho_c_q(build(L; seed=200+s))
            push!(sq, sum(R.q)); push!(s2, sum(R.rho.^2))
        end
        N = 2L-1
        emit(@sprintf("  %5d %6d %12.4f %12.4f %12.4f %12.4f", L, N,
                      median(sq), median(s2), median(sq)/median(s2),
                      median(s2)/log(N)))
    end
    emit("")
    emit("  Lire le RATIO : Prop A (à c=0) le forcerait vers 1 quand rho_j -> 0.")
    emit("  Un ratio nettement inférieur à 1 mesure exactement ce que l'alignement")
    emit("  non nul retire à la somme -- l'écart entre le modèle idéalisé et les")
    emit("  données, quantifié sans ajuster quoi que ce soit.")

    emit("")
    emit("="^78)
    emit("CE QUI N'EST PAS TESTÉ ICI")
    emit("="^78)
    emit("  Le régime ENTRAÎNÉ. bench_eps_depth_trained_results.txt ne conserve")
    emit("  que des agrégats (nbar1, qbar, rho_lb, nneg, nins) ; les conditions")
    emit("  nécessaires portent sur les SUITES (rho_j, c_j) branche par branche,")
    emit("  qui n'y figurent pas. Les tester demanderait de sauvegarder ces")
    emit("  tableaux lors d'une nouvelle exécution -- non fait ici, et le fichier")
    emit("  existant n'est pas touché.")
end
println("\nÉcrit : ", OUT)
