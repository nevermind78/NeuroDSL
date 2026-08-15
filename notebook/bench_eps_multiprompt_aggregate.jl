# =============================================================================
# AGRÉGAT SUR LES 22 PROMPTS, RECALCULÉ DEPUIS LE CSV
#
# POURQUOI CE SCRIPT EXISTE
# -------------------------
# bench_eps_exact_ablation_qwen_multiprompt.jl a tourné en QUATRE lots
# (reprise par MP_SKIP/MP_LIMIT, imposée par une contention VRAM externe :
# un lanceur de jeux occupait la carte par intermittence). Chaque processus
# ne connaît donc que SES prompts, et l'agrégat qu'il imprime ne porte que
# sur son lot -- celui du dernier lot annonce « 6 prompts » et serait pris
# pour un résumé global si on le lisait vite.
#
# Ce script relit le CSV complet (22 prompts x 1596 paires en cône) et
# recalcule TOUT sur les 22, une seule fois, sans rien remesurer. C'est la
# seule source à citer pour un chiffre agrégé.
#
# Usage : julia --project=. notebook/bench_eps_multiprompt_aggregate.jl
# =============================================================================

using Printf, Statistics

const CSV = joinpath(@__DIR__, "bench_eps_exact_ablation_qwen_multiprompt_matrix.csv")
const OUT = joinpath(@__DIR__, "bench_eps_multiprompt_aggregate_results.txt")

q2(rho, c) = rho*(rho + c)/(1 + 2*rho*c + rho^2)

# ─── lecture du CSV ──────────────────────────────────────────────────────────
# colonnes : prompt_idx,site,branch,site_pos,branch_pos,q,rho,cos
rows = Vector{NamedTuple}()
open(CSV) do fh
    readline(fh)                                    # en-tête
    for ln in eachline(fh)
        isempty(strip(ln)) && continue
        f = split(ln, ',')
        length(f) == 8 || continue
        f[1] == "prompt_idx" && continue            # en-tête répété éventuel
        push!(rows, (p = parse(Int, f[1]),
                     sp = parse(Int, f[4]), bp = parse(Int, f[5]),
                     q = parse(Float64, f[6]),
                     rho = parse(Float64, f[7]),
                     c = parse(Float64, f[8])))
    end
end

prompts = sort(unique(r.p for r in rows))

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("AGRÉGAT SUR LES $(length(prompts)) PROMPTS -- recalculé depuis le CSV complet")
    emit("Source : " * basename(CSV) * "  ($(length(rows)) lignes)")
    emit("Aucune nouvelle mesure : relecture seule.")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("")

    nbar = Float64[]; pL1 = Float64[]; pneg_c1 = Float64[]
    pneg_all = Float64[]; meanc = Float64[]; medc = Float64[]; fcneg = Float64[]
    two_max = 0.0; sgn_viol = 0; p1_max = 0.0

    for p in prompts
        R = filter(r -> r.p == p, rows)
        # site de couche 1 : site_pos == 1 ; son cône = branch_pos >= 1 (tout)
        c1 = filter(r -> r.sp == 1, R)
        push!(nbar, sum(r.q for r in c1))
        push!(pL1, sum(r.q for r in c1)/length(c1))
        byp = filter(r -> r.bp != r.sp, c1)          # hors branche obligatoire
        push!(pneg_c1, 100*count(r -> r.q < 0, byp)/length(byp))
        cs = [r.c for r in byp if isfinite(r.c)]
        push!(meanc, mean(cs)); push!(medc, median(cs))
        push!(fcneg, 100*count(r -> isfinite(r.c) && isfinite(r.rho) && r.c < -r.rho, byp)/length(byp))
        # matrice entière (paires en cône)
        push!(pneg_all, 100*count(r -> r.q < 0, R)/length(R))
        # portes recalculées
        for r in R
            r.bp == r.sp && (p1_max = max(p1_max, abs(r.q - 1.0)))
            if isfinite(r.rho) && isfinite(r.c)
                two_max = max(two_max, abs(r.q - q2(r.rho, r.c)))
                (sign(r.q) == sign(r.rho + r.c) || abs(r.rho + r.c) < 1e-6 ||
                 abs(r.q) < 1e-6) || (sgn_viol += 1)
            end
        end
    end

    f3(v) = (minimum(v), median(v), maximum(v))
    emit("-"^78); emit("PORTES, SUR LES 22 PROMPTS (recalculées ligne à ligne)"); emit("-"^78)
    emit(@sprintf("\n  P1  max |q_jj - 1|                    : %.3e", p1_max))
    emit(@sprintf("  2sc max |q - forme à deux scalaires|  : %.3e", two_max))
    emit(@sprintf("  loi de signe, violations cumulées     : %d  sur %d paires en cône",
                  sgn_viol, length(rows)))
    emit("")
    emit("-"^78); emit("QUANTITÉS PAR PROMPT, AGRÉGÉES"); emit("-"^78)
    emit(@sprintf("\n  n_bar (site couche 1)      : min %.4f  médian %.4f  max %.4f", f3(nbar)...))
    emit(@sprintf("  p_L1                       : min %.5f  médian %.5f  max %.5f", f3(pL1)...))
    emit(@sprintf("  %% négatifs, cône du site 1 : min %.1f  médian %.1f  max %.1f", f3(pneg_c1)...))
    emit(@sprintf("  %% négatifs, matrice entière: min %.1f  médian %.1f  max %.1f", f3(pneg_all)...))
    emit(@sprintf("  moyenne de c_j             : min %+.4f  médiane %+.4f  max %+.4f", f3(meanc)...))
    emit(@sprintf("  %% avec c_j < -rho_j        : min %.1f  médian %.1f  max %.1f", f3(fcneg)...))
    emit("")
    emit(@sprintf("  CV de p_L1 sur les prompts : %.1f %%", 100*std(pL1)/mean(pL1)))
    emit(@sprintf("  CV de la moyenne de c_j    : %.1f %%", 100*std(meanc)/abs(mean(meanc))))

    emit("")
    emit("-"^78); emit("DÉRIVE D'ALIGNEMENT : ENTRAÎNÉ CONTRE INITIALISATION"); emit("-"^78)
    nneg = count(<(0), meanc)
    emit(@sprintf("\n  moyenne de c_j NÉGATIVE sur %d des %d prompts", nneg, length(prompts)))
    emit(@sprintf("  moyenne des moyennes : %+.4f   (médiane des moyennes : %+.4f)",
                  mean(meanc), median(meanc)))
    if nneg < length(prompts)
        pos = [prompts[i] for i in eachindex(meanc) if meanc[i] >= 0]
        emit(@sprintf("  EXCEPTION(S) -- moyenne positive sur le(s) prompt(s) : %s",
                      join(pos, ", ")))
        emit("  Le signe n'est donc PAS universel sur les entrées testées ; il est")
        emit("  très majoritaire, ce qui n'est pas la même affirmation.")
    end
    emit("  Référence à l'INITIALISATION (pile pré-norm synthétique,")
    emit("  bench_eps_convergence_general_results.txt) : moyenne c_j = +0.037,")
    emit("  56 % de valeurs positives.")

    emit("")
    emit("-"^78); emit("LOI DE SIGNE, VÉRIFICATION CROISÉE"); emit("-"^78)
    emit("  Le Corollaire de la loi de signe prédit q_j < 0 SSI c_j < -rho_j, donc")
    emit("  les deux pourcentages doivent coïncider prompt par prompt.")
    dmax = maximum(abs.(pneg_c1 .- fcneg))
    emit(@sprintf("\n  écart max |%% négatifs - %% (c_j < -rho_j)| : %.3f point de %%", dmax))
    emit(dmax < 1e-9 ? "  ✓ coïncidence exacte sur les 22 prompts." :
                       "  ✗ écart non nul -- à expliquer.")

    # ─── PROFIL PAR COUCHE, SUR LES 22 PROMPTS ──────────────────────────────
    # Le site attn de la couche i est à la position 2i-1 ; son cône est le
    # suffixe qui commence à lui, donc B = 56 - pos + 1 = 57 - pos.
    emit("")
    emit("-"^78)
    emit("PROFIL PAR COUCHE (sites attn), MÉDIANE ET PLAGE SUR LES 22 PROMPTS")
    emit("-"^78)
    emit("  Remplace un profil mesuré sur UNE entrée. La dispersion est le point :")
    emit("  la donner en médiane + plage, jamais en valeur unique.")
    emit("")
    emit(@sprintf("\n%6s %5s %5s | %9s %9s %9s | %8s %8s %8s",
                  "couche", "L-i", "B", "n_bar min", "n_bar méd", "n_bar max",
                  "p min", "p méd", "p max"))
    for i in [1, 2, 5, 12, 16, 22, 27, 28]
        sp = 2i - 1
        B = 57 - sp
        nb_i = Float64[]
        for p in prompts
            s = sum(r.q for r in rows if r.p == p && r.sp == sp && r.bp >= sp)
            push!(nb_i, s)
        end
        pr_i = nb_i ./ B
        emit(@sprintf("%6d %5d %5d | %9.4f %9.4f %9.4f | %8.5f %8.5f %8.5f",
                      i, 28 - i, B, minimum(nb_i), median(nb_i), maximum(nb_i),
                      minimum(pr_i), median(pr_i), maximum(pr_i)))
    end
    emit("")
    emit("  p_deep = moyenne de p sur les sites attn des couches 1..14 :")
    pdeep = Float64[]
    for p in prompts
        acc = Float64[]
        for i in 1:14
            sp = 2i - 1; B = 57 - sp
            push!(acc, sum(r.q for r in rows if r.p == p && r.sp == sp && r.bp >= sp)/B)
        end
        push!(pdeep, mean(acc))
    end
    emit(@sprintf("    min %.5f   médian %.5f   max %.5f   (CV %.1f %%)",
                  minimum(pdeep), median(pdeep), maximum(pdeep),
                  100*std(pdeep)/mean(pdeep)))

    # ─── DÉPENDANCE AU SITE, SUR LES 22 PROMPTS ─────────────────────────────
    # Sites adjacents (positions i, i+1), même branche j, j dans les DEUX
    # cônes (bp >= i+1 suffit, le cône du site i+1 est le plus restrictif).
    # 1540 paires par prompt (= C(56,2)/... en fait sum_{i=1}^{55}(56-i) =
    # 1540, exactement le compte à n=1 d'origine) x 22 prompts.
    emit("")
    emit("-"^78)
    emit("DÉPENDANCE AU SITE (sites adjacents, même branche), SUR LES 22 PROMPTS")
    emit("-"^78)
    emit("  Referme le dernier holdout à n=1 signalé en Portée : gratuit, la")
    emit("  matrice complète par site existe déjà dans le CSV pour chaque prompt.")
    emit("")
    qat = Dict{Tuple{Int,Int,Int},Float64}()   # (prompt, site_pos, branch_pos) -> q
    for r in rows
        qat[(r.p, r.sp, r.bp)] = r.q
    end
    sym_all = Float64[]; abs_all = Float64[]; allq_all = Float64[]
    opp_count = 0; pair_count = 0
    for p in prompts
        for i in 1:55
            for bp in (i+1):56
                a = get(qat, (p, i, bp), nothing)
                b = get(qat, (p, i+1, bp), nothing)
                (a === nothing || b === nothing) && continue
                den = abs(a) + abs(b)
                push!(allq_all, abs(a)); push!(allq_all, abs(b))
                den < 1e-9 && continue
                push!(sym_all, abs(a-b)/den); push!(abs_all, abs(a-b))
                pair_count += 1
                abs(a-b)/den > 0.5 && (opp_count += 1)
            end
        end
    end
    emit(@sprintf("  %d paires (55 x 22 prompts, jusqu'à 1540 chacun)", pair_count))
    emit(@sprintf("  échelle de référence : |q| médian = %.4f", median(allq_all)))
    emit("")
    emit("  écart SYMÉTRIQUE |a-b|/(|a|+|b|) :")
    emit(@sprintf("    médian %.4f   p90 %.4f   p99 %.4f   max %.4f",
                  median(sym_all), quantile(sym_all,0.90), quantile(sym_all,0.99), maximum(sym_all)))
    emit(@sprintf("    paires de signes opposés (écart > 0.5) : %d (%.1f %%)",
                  opp_count, 100*opp_count/pair_count))
    emit("")
    emit("  écart ABSOLU |a-b| :")
    emit(@sprintf("    médian %.5f   p90 %.5f   max %.5f",
                  median(abs_all), quantile(abs_all,0.90), maximum(abs_all)))
    emit(@sprintf("    en fraction du |q| médian : médian %.1f %%, p90 %.1f %%",
                  100*median(abs_all)/median(allq_all), 100*quantile(abs_all,0.90)/median(allq_all)))
    emit("")
    emit("  Comparaison à la mesure à n=1 (un seul prompt) déjà publiée :")
    emit("  médian 0.0192, p90 0.2033, 76/1540 opposés (4.9%), médian absolu 0.0033.")

    emit("")
    emit("-"^78); emit("LONGUEUR DU PROMPT"); emit("-"^78)
    # longueurs relues depuis le fichier de prompts (pas dans le CSV)
    PF = joinpath(@__DIR__, "qwen_sweep_prompts.json")
    txt = read(PF, String)
    ns = [length(m.captures[1] === nothing ? "" : m.captures[1])
          for m in eachmatch(r"\"n\"\s*:\s*(\d+)", txt)]
    nvals = [parse(Int, m.captures[1]) for m in eachmatch(r"\"n\"\s*:\s*(\d+)", txt)]
    if length(nvals) == length(prompts)
        lens = [nvals[p] for p in prompts]
        cor(x, y) = begin
            mx, my = mean(x), mean(y)
            sum((x .- mx).*(y .- my))/sqrt(sum((x .- mx).^2)*sum((y .- my).^2))
        end
        r_len_c = cor(Float64.(lens), meanc)
        r_len_p = cor(Float64.(lens), pL1)
        emit(@sprintf("\n  n va de %d à %d sur les 22 prompts.", minimum(lens), maximum(lens)))
        emit(@sprintf("  corrélation n vs moyenne de c_j : r = %+.3f", r_len_c))
        emit(@sprintf("  corrélation n vs p_L1           : r = %+.3f", r_len_p))
        emit("  Sur 22 points et une plage de longueur étroite, une corrélation de")
        emit("  cet ordre est une indication, pas un résultat : elle ne sépare pas")
        emit("  la longueur du contenu, les prompts longs différant aussi par le sujet.")
    else
        emit("  (longueurs non appariables au CSV -- non calculé.)")
    end
end
println("\nÉcrit : ", OUT)
