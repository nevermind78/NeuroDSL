# =============================================================================
# LA SOMME D'EXCÈS SOUS LE MODÈLE IDÉALISÉ, FORME EXACTE VIA LA FONCTION
# DIGAMMA -- pas un ajustement, une évaluation exacte du modèle DÉJÀ ÉNONCÉ
#
# CE QUE CE SCRIPT FAIT ET NE FAIT PAS
# -------------------------------------
# Section 8 de l'article pose un modèle idéalisé À L'INITIALISATION :
# rho_j = kappa/sqrt(j) (mise à l'échelle pré-norm) et c_j = 0 (négligence
# d'alignement). Sous CE modèle, et uniquement sous ce modèle, la Corollaire
# enveloppe à c=0 donne q_j = rho_j^2/(1+rho_j^2) = kappa^2/(j+kappa^2)
# EXACTEMENT, donc la somme d'excès e(N) = somme_j q_j est une somme de
# fractions simples qui télescope EXACTEMENT via la fonction digamma :
#     e(N) = kappa^2 [psi(N+1+kappa^2) - psi(1+kappa^2)]
# Ceci n'est PAS un ajustement -- c'est une évaluation exacte d'une somme
# donnée le modèle. Le développement asymptotique standard de psi (série
# d'Euler-Maclaurin, Abramowitz & Stegun 1964, chap. 6) donne ensuite le
# terme suivant après log L, en forme close.
#
# CE QUI EST TESTÉ CONTRE LES DONNÉES : les six médianes n_bar DÉJÀ publiées
# dans Table~tab:depth (bench_eps_depth_synthetic_results.txt, L=4..128),
# recopiées ici en dur avec leur source citée -- aucune nouvelle mesure,
# aucun fichier existant modifié. Le modèle à UN SEUL paramètre libre
# (kappa^2) y est comparé, honnêtement, à l'ajustement log-linéaire à DEUX
# paramètres déjà publié.
#
# digamma() : implémentation pure Julia (récurrence + série asymptotique de
# Bernoulli), vérifiée ci-dessous contre trois valeurs exactes connues
# (psi(1)=-gamma, psi(1/2)=-gamma-2ln2, psi(5)=-gamma+1+1/2+1/3+1/4).
# AUCUN paquet ajouté, Project.toml INTACT.
#
# USAGE : julia --project=. notebook/bench_eps_depth_idealized_digamma.jl
# =============================================================================

using Printf, Statistics

function digamma(x::Float64)
    r = 0.0
    while x < 15.0
        r -= 1.0/x
        x += 1.0
    end
    ix2 = 1.0/(x*x)
    s = log(x) - 1.0/(2x) - ix2*(1.0/12 - ix2*(1.0/120 - ix2*(1.0/252 - ix2/240)))
    return r + s
end

const OUT = joinpath(@__DIR__, "bench_eps_depth_idealized_digamma_results.txt")

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("SOMME D'EXCÈS SOUS LE MODÈLE IDÉALISÉ -- FORME EXACTE PAR digamma")
    emit("Teste, sans nouvelle mesure, le modèle rho_j=kappa/sqrt(j), c_j=0 de la")
    emit("Section 8 contre les six médianes déjà publiées dans Table~tab:depth")
    emit("(source : bench_eps_depth_synthetic_results.txt).")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("")

    emit("-"^78)
    emit("PORTE : digamma() vérifiée contre trois valeurs exactes connues")
    emit("-"^78)
    EULER_GAMMA = 0.5772156649015329
    checks = [(1.0, -EULER_GAMMA, "psi(1) = -gamma"),
              (0.5, -EULER_GAMMA - 2*log(2), "psi(1/2) = -gamma-2ln2"),
              (5.0, -EULER_GAMMA + 1 + 1/2 + 1/3 + 1/4, "psi(5) = -gamma+1+1/2+1/3+1/4")]
    for (x, exact, label) in checks
        @printf(io, "  %-28s : digamma=%.13f  exact=%.13f  écart=%.2e\n",
                label, digamma(x), exact, abs(digamma(x)-exact))
        @printf("  %-28s : digamma=%.13f  exact=%.13f  écart=%.2e\n",
                label, digamma(x), exact, abs(digamma(x)-exact))
    end
    emit("")

    # Six médianes n_bar déjà publiées (Table tab:depth de l'article,
    # bench_eps_depth_synthetic_results.txt) -- recopiées, pas remesurées.
    Ls  = [4, 8, 16, 32, 64, 128]
    nb  = [2.1434, 2.8359, 3.3300, 4.6180, 5.2767, 5.4320]
    exc = nb .- 1.0
    Ns  = 2 .* Ls .- 1                       # N = B-1 = 2L-1 branches contournables
    ybar = mean(exc); sst = sum((exc .- ybar).^2)

    emit("-"^78)
    emit("MODÈLE EXACT (UN PARAMÈTRE) : e(N) = kappa^2 [psi(N+1+kappa^2)-psi(1+kappa^2)]")
    emit("-"^78)
    excess_model(N, a) = a*(digamma(N+1+a) - digamma(1+a))
    sse(a) = sum((excess_model(Ns[i], a) - exc[i])^2 for i in eachindex(Ns))
    function best_fit()
        ba, bs = 0.0, Inf
        for a in 0.001:0.001:5.0
            s = sse(a)
            if s < bs
                bs = s
                ba = a
            end
        end
        for a in (ba-0.001):0.00002:(ba+0.001)
            s = sse(a)
            if s < bs
                bs = s
                ba = a
            end
        end
        return ba, bs
    end
    a_hat, sse_hat = best_fit()
    r2_ideal = 1 - sse_hat/sst
    @printf(io, "\n  kappa^2 ajusté (1 paramètre, moindres carrés) : %.4f  (kappa=%.4f)\n", a_hat, sqrt(a_hat))
    @printf("\n  kappa^2 ajusté (1 paramètre, moindres carrés) : %.4f  (kappa=%.4f)\n", a_hat, sqrt(a_hat))
    @printf(io, "  R^2 (modèle exact digamma, 1 paramètre)       : %.4f\n", r2_ideal)
    @printf("  R^2 (modèle exact digamma, 1 paramètre)       : %.4f\n", r2_ideal)
    emit("")
    emit(@sprintf("%6s %6s %10s %10s %10s", "L", "N", "e mesuré", "e modèle", "résidu"))
    for i in eachindex(Ls)
        m = excess_model(Ns[i], a_hat)
        emit(@sprintf("%6d %6d %10.4f %10.4f %10.4f", Ls[i], Ns[i], exc[i], m, exc[i]-m))
    end

    emit("")
    emit("-"^78)
    emit("COMPARAISON AUX AJUSTEMENTS DÉJÀ PUBLIÉS")
    emit("-"^78)
    # log-linéaire, 2 paramètres (déjà publié : a=0.7182, b=1.0327, R²=0.9651)
    Xa = hcat(ones(length(Ls)), log.(Ls))
    beta = Xa \ exc
    pred_loglin = Xa*beta
    r2_loglin = 1 - sum((exc .- pred_loglin).^2)/sst
    @printf(io, "\n  log-linéaire (2 paramètres, déjà publié) : a=%.4f b=%.4f  R^2=%.4f\n",
            beta[1], beta[2], r2_loglin)
    @printf("\n  log-linéaire (2 paramètres, déjà publié) : a=%.4f b=%.4f  R^2=%.4f\n",
            beta[1], beta[2], r2_loglin)

    # sqrt(L), 1 paramètre -- pour quantifier l'exclusion d'une croissance en sqrt(L)
    z = sqrt.(Ls)
    a_sqrt = sum(z.*exc)/sum(z.*z)
    r2_sqrt = 1 - sum((exc .- a_sqrt.*z).^2)/sst
    @printf(io, "  sqrt(L)      (1 paramètre)               : a=%.4f          R^2=%.4f\n",
            a_sqrt, r2_sqrt)
    @printf("  sqrt(L)      (1 paramètre)               : a=%.4f          R^2=%.4f\n",
            a_sqrt, r2_sqrt)

    emit("")
    emit("-"^78)
    emit("DÉVELOPPEMENT ASYMPTOTIQUE (Euler-Maclaurin sur psi, ordre suivant après log L)")
    emit("-"^78)
    intercept_pred = a_hat*log(2) - a_hat*digamma(1+a_hat)
    coeff_1overL = a_hat*(2*a_hat-1)/4
    @printf(io, "\n  e(L) ~ kappa^2 log(L) + [kappa^2 log2 - kappa^2 psi(1+kappa^2)] + kappa^2(2kappa^2-1)/(4L)\n")
    @printf(io, "  ordonnée prédite (kappa^2=%.4f) : %.4f   (ordonnée ajustée publiée : 0.718)\n",
            a_hat, intercept_pred)
    @printf("\n  ordonnée prédite (kappa^2=%.4f) : %.4f   (ordonnée ajustée publiée : 0.718)\n",
            a_hat, intercept_pred)
    @printf(io, "  coefficient exact en 1/L : %.4f\n", coeff_1overL)
    @printf("  coefficient exact en 1/L : %.4f\n", coeff_1overL)

    emit("")
    emit("="^78)
    emit("CONCLUSION")
    emit("="^78)
    emit("  Le modèle idéalisé à un seul paramètre est une évaluation EXACTE (pas un")
    emit("  ajustement) sous ses hypothèses -- mais il n'atteint que R^2=" *
         string(round(r2_ideal,digits=2)) * ", sous l'ajustement log-linéaire à deux")
    emit("  paramètres déjà publié (R^2=" * string(round(r2_loglin,digits=3)) * "), et son ordonnée")
    emit("  prédite (" * string(round(intercept_pred,digits=2)) * ") est loin de l'ordonnée ajustée (0.718).")
    emit("  La croissance en sqrt(L), elle, est bien PLUS fortement exclue (R^2=" *
         string(round(r2_sqrt,digits=2)) * ")")
    emit("  que le modèle idéalisé lui-même -- ce qui suggère, sans le prouver, que")
    emit("  l'effet net de l'alignement (c_j, non nul individuellement) ne s'accumule")
    emit("  pas de façon cohérente sur la plage testée. Le modèle idéalisé n'explique")
    emit("  pas tout ; il donne un terme suivant")
    emit("  exact et une explication structurelle de l'écart, pas une clôture du problème.")
end
println("\nÉcrit : ", OUT)
