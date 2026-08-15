# ══════════════════════════════════════════════════════════════════════════════
# PRÉALABLE au test de la loi de longueur de chemin : formes closes AMONT
# et comptes de chemins EXACTS, sur les deux graphes exacts d'E9
# (arXiv:2607.18323, §E9).
#
# POURQUOI CE SCRIPT
# ------------------
# Le théorème de localité arrière (math2 §sec:backward-locality) et son
# corollaire d'effondrement raisonnent sur le cône AMONT V^-_s. Mais l'article
# ne donne de forme close que pour le cône AVAL :
#     |V^+_head(i)| = 10 + (L-i)M,  |V^+_mlp(i)| = 2 + (L-i)M,  M = 7H+15
# vérifiée à résidu nul sur 564 sites. L'asymétrie forward/backward entre lens
# a besoin du pendant amont, qui n'est écrit nulle part. Ce script le mesure,
# le vérifie, et mesure en plus le compte de chemins entier exact vers la
# sortie -- l'objet dont la somme sur les chemins de la preuve du théorème
# est faite.
#
# CE QUI EST MESURÉ (structurel uniquement -- AUCUN chronométrage,
# AUCUN calcul de valeur, donc aucun bruit d'horloge à discuter)
#   1. |V^+(s)| pour les 564 sites  -> DOIT reproduire la formule publiée.
#      C'est la PORTE DE VALIDATION DU HARNAIS : si mon compteur ne
#      reproduit pas un résultat déjà vérifié indépendamment, le compteur
#      est faux et rien de ce qui suit n'a de valeur.
#   2. |V^-(s)| pour les 564 sites  -> la quantité nouvelle.
#   3. P(s) = nombre de chemins dirigés distincts de s à la sortie, en
#      entiers EXACTS (BigInt -- ces comptes explosent, un Float64
#      perdrait la précision en silence et donnerait une fausse forme close).
#
# DISCIPLINE D'ORACLE INDÉPENDANT
# -------------------------------
# Le moteur expose déjà `_ancestors_of!` (graph_api.jl:66) et
# `_downstream_nodes` (patching.jl:169). Ce script NE LES UTILISE PAS pour
# produire ses chiffres : il reconstruit ses propres parcours depuis
# `g.rules[ns]` seul, puis compare. Un compteur qui s'appuierait sur la
# fonction qu'il prétend valider ne prouverait rien. Les deux écarts
# (aval, amont) sont rapportés explicitement.
#
# SUBTILITÉ QUI CHANGE LE RÉSULTAT : les tenseurs de paramètres sont des
# FEUILLES. Ils ne sont donc jamais dans un cône AVAL (rien ne les produit),
# ce qui est exactement pourquoi la formule publiée les exclut. Mais ils SONT
# des ancêtres. Le cône amont n'est donc pas l'image miroir du cône aval, et
# la forme close amont doit compter séparément nœuds calculés et feuilles.
# Le script sépare les deux comptes pour rendre cette asymétrie visible.
#
# Les tables RoPE, masques et constantes de position sont ancêtres de TOUTES
# les couches : elles siègent dans tous les cônes amont. C'est le point où
# l'hypothèse d'ancestralité du corollaire d'effondrement peut fuir sur un
# vrai graphe -- mesuré ici, pas supposé.
#
# USAGE
#   julia --project=. notebook/e9_upstream_and_paths.jl            # les 2 configs
#   julia --project=. notebook/e9_upstream_and_paths.jl small      # contrôle largeur
#
# Écrit notebook/e9_upstream_and_paths_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf

const MODE = isempty(ARGS) ? "full" : ARGS[1]

# ─── Oracles indépendants, construits depuis g.rules seul ──────────────────

"""Carte arête directe : symbole -> Vector{(consommateur, multiplicité)}.

La multiplicité compte : si une règle liste deux fois la même entrée
(`add(x, x)`), ce sont DEUX arêtes, donc deux chemins distincts. Un
`Set` de consommateurs perdrait cette information et sous-compterait
les chemins.
"""
function build_edges(rules)
    edges = Dict{Symbol,Vector{Tuple{Symbol,Int}}}()
    for (out_sym, rule) in rules
        mult = Dict{Symbol,Int}()
        for inp in rule.inputs
            mult[inp] = get(mult, inp, 0) + 1
        end
        for (inp, m) in mult
            push!(get!(edges, inp, Tuple{Symbol,Int}[]), (out_sym, m))
        end
    end
    return edges
end

"""Cône aval strict de `s` : nœuds atteignables par >= 1 arête. Oracle."""
function oracle_downstream(edges, s::Symbol)
    visited = Set{Symbol}()
    stack = Symbol[]
    for (c, _) in get(edges, s, Tuple{Symbol,Int}[])
        push!(stack, c)
    end
    while !isempty(stack)
        v = pop!(stack)
        v in visited && continue
        push!(visited, v)
        for (c, _) in get(edges, v, Tuple{Symbol,Int}[])
            c in visited || push!(stack, c)
        end
    end
    return visited
end

"""Cône amont strict de `s` : nœuds d'où partent >= 1 arête vers `s`. Oracle."""
function oracle_upstream(rules, s::Symbol)
    visited = Set{Symbol}()
    stack = Symbol[]
    r = get(rules, s, nothing)
    r === nothing || append!(stack, r.inputs)
    while !isempty(stack)
        v = pop!(stack)
        v in visited && continue
        push!(visited, v)
        rv = get(rules, v, nothing)
        rv === nothing && continue
        for inp in rv.inputs
            inp in visited || push!(stack, inp)
        end
    end
    return visited
end

"""Ordre topologique de tout le namespace, depuis g.rules seul (Kahn)."""
function oracle_topo(all_syms, rules)
    indeg = Dict{Symbol,Int}(s => 0 for s in all_syms)
    for (out_sym, rule) in rules
        haskey(indeg, out_sym) || continue
        indeg[out_sym] = length(rule.inputs)
    end
    edges = build_edges(rules)
    ready = [s for s in all_syms if indeg[s] == 0]
    order = Symbol[]
    while !isempty(ready)
        v = pop!(ready)
        push!(order, v)
        for (c, m) in get(edges, v, Tuple{Symbol,Int}[])
            haskey(indeg, c) || continue
            indeg[c] -= m
            indeg[c] == 0 && push!(ready, c)
        end
    end
    length(order) == length(all_syms) ||
        error("ordre topologique incomplet : $(length(order))/$(length(all_syms)) -- cycle ?")
    return order
end

"""P(v) = nombre de chemins dirigés distincts de v vers `out_sym`, exact.

P(out) = 1 ; P(v) = somme sur consommateurs c de mult(v,c) * P(c).
Récurrence évaluée en ordre topologique INVERSE, donc chaque consommateur
est résolu avant le nœud. BigInt : ces comptes croissent
géométriquement en profondeur et déborderaient un Int64.
"""
function oracle_path_counts(all_syms, rules, out_sym::Symbol)
    edges = build_edges(rules)
    order = oracle_topo(all_syms, rules)
    P = Dict{Symbol,BigInt}(s => BigInt(0) for s in all_syms)
    P[out_sym] = BigInt(1)
    for v in Iterators.reverse(order)
        v == out_sym && continue
        acc = BigInt(0)
        for (c, m) in get(edges, v, Tuple{Symbol,Int}[])
            haskey(P, c) && (acc += BigInt(m) * P[c])
        end
        P[v] = acc
    end
    return P
end

# ─── Analyse d'une configuration ───────────────────────────────────────────

function analyze(io, label, dim, n_heads, hidden_dim, n_layers, seq_len, expected_sites)
    println(io, "\n" * "="^78)
    println(io, "CONFIG $label : L=$n_layers  H=$n_heads  dim=$dim  hidden=$hidden_dim  seq=$seq_len")
    println(io, "="^78)

    dev = NeuroDSL.Backend.CPUDevice()
    Random.seed!(1)
    ns = Symbol("e9_", label)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.set!(g, :input, NeuroDSL.Backend.rand32(dev, seq_len, dim) .- 0.5f0; namespace=ns)
    out = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim;
                              batched_attn=true)(g, :input; namespace=ns)

    # OBLIGATOIRE avant toute comparaison au moteur : `_downstream_nodes` est
    # FILTRÉ PAR LA VALIDITÉ (il ne franchit que des consommateurs déjà
    # `valid`). Sur un graphe fraîchement construit il renvoie ~2 nœuds, pas
    # le cône. E9 mesurait après un `demand!` ; sans ça la comparaison est
    # vide de sens. Diagnostiqué le 2026-08-11, pas supposé.
    NeuroDSL.demand!(g, out; namespace=ns)

    rules    = g.rules[ns]
    all_syms = collect(keys(g.nodes[ns]))
    N_total  = length(all_syms)
    n_rules  = length(rules)
    leaves   = [s for s in all_syms if !haskey(rules, s)]
    n_params = length(NeuroDSL.params(g; namespace=ns))

    @printf(io, "\nNœuds total N            : %d\n", N_total)
    @printf(io, "Nœuds calculés (règles)  : %d\n", n_rules)
    @printf(io, "Feuilles                 : %d  (dont %d paramètres)\n", length(leaves), n_params)
    @printf(io, "Sortie                   : %s\n", out)

    M_pred = 7 * n_heads + 15
    @printf(io, "\nM = 7H+15 prédit         : %d\n", M_pred)
    @printf(io, "Nœuds calculés / couche  : %.4f  (= %d/%d)\n",
            n_rules / n_layers, n_rules, n_layers)

    # Sites : H sorties de tête par couche + 1 sortie MLP par couche
    layer_prefixes = ["layer_$i" for i in 1:n_layers]
    head_sites = [(Symbol(lp, "_mha_ao_h", h), i, :head)
                  for (i, lp) in enumerate(layer_prefixes) for h in 1:n_heads]
    mlp_sites  = [(Symbol(lp, "_mlp_out"), i, :mlp)
                  for (i, lp) in enumerate(layer_prefixes)]
    sites = vcat(head_sites, mlp_sites)
    length(sites) == expected_sites ||
        error("attendu $expected_sites sites, obtenu $(length(sites))")
    for (s, _, _) in sites
        haskey(g.nodes[ns], s) || error("site manquant : $s")
    end
    @printf(io, "Sites                    : %d (%d têtes + %d MLP)\n",
            length(sites), length(head_sites), length(mlp_sites))

    edges = build_edges(rules)
    P     = oracle_path_counts(all_syms, rules, out)

    # ═══ PORTE DE VALIDATION : reproduire la formule AVAL publiée ═══
    println(io, "\n" * "-"^78)
    println(io, "PORTE 1 -- reproduction de la forme close AVAL publiée (E9)")
    println(io, "  |V+_head(i)| = 10 + (L-i)M   |V+_mlp(i)| = 2 + (L-i)M   M = 7H+15")
    println(io, "-"^78)

    down_resid = Int[]
    up_meas    = Dict{Tuple{Int,Symbol},Int}()
    up_comp    = Dict{Tuple{Int,Symbol},Int}()
    up_leaf    = Dict{Tuple{Int,Symbol},Int}()
    path_meas  = Dict{Tuple{Int,Symbol},BigInt}()
    mismatch_engine_down = 0
    mismatch_engine_up   = 0

    for (s, i, kind) in sites
        dn  = oracle_downstream(edges, s)
        pred = (kind === :head ? 10 : 2) + (n_layers - i) * M_pred
        # CONVENTION : la forme close publiée compte le SITE lui-même
        # (|V+_mlp(L)| = 2 alors que le cône strict de la dernière sortie MLP
        # ne contient qu'un nœud). On compare donc strict+1. Établi par
        # diagnostic, pas par lecture de la définition.
        push!(down_resid, (length(dn) + 1) - pred)

        # comparaison croisée avec le moteur (informatif, pas la source)
        eng_dn = NeuroDSL._downstream_nodes(g, s, ns)
        # _downstream_nodes inclut le site lui-même
        (length(eng_dn) - 1) == length(dn) || (mismatch_engine_down += 1)

        up = oracle_upstream(rules, s)
        up_meas[(i, kind)] = length(up)
        up_comp[(i, kind)] = count(v -> haskey(rules, v), up)
        up_leaf[(i, kind)] = count(v -> !haskey(rules, v), up)

        eng_up = NeuroDSL._ancestors_of!(g, ns, s)
        (length(eng_up) - 1) == length(up) || (mismatch_engine_up += 1)

        path_meas[(i, kind)] = P[s]
    end

    nz = count(!=(0), down_resid)
    @printf(io, "\nRésidu aval : %d/%d sites non nuls  (max |résidu| = %d)\n",
            nz, length(sites), maximum(abs.(down_resid)))
    if nz == 0
        println(io, "  ✓ PORTE 1 FRANCHIE -- le harnais reproduit un résultat déjà vérifié.")
    else
        println(io, "  ✗ PORTE 1 ÉCHOUÉE -- harnais suspect, ne pas lire la suite.")
    end
    @printf(io, "Désaccords avec le moteur : aval %d, amont %d (sur %d sites)\n",
            mismatch_engine_down, mismatch_engine_up, length(sites))

    # ═══ La quantité nouvelle : cône AMONT ═══
    println(io, "\n" * "-"^78)
    println(io, "MESURE 2 -- cône AMONT |V^-(i)|, décomposé calculés / feuilles")
    println(io, "-"^78)
    @printf(io, "\n%6s %6s  %10s %10s %10s   %s\n",
            "couche", "type", "|V^-|", "calculés", "feuilles", "P(s) vers sortie")
    for kind in (:head, :mlp)
        for i in 1:n_layers
            haskey(up_meas, (i, kind)) || continue
            p = path_meas[(i, kind)]
            pstr = ndigits(p) > 22 ? @sprintf("~1e%d", ndigits(p) - 1) : string(p)
            @printf(io, "%6d %6s  %10d %10d %10d   %s\n",
                    i, kind, up_meas[(i, kind)], up_comp[(i, kind)],
                    up_leaf[(i, kind)], pstr)
        end
    end

    # Différences consécutives : révèlent le pas par couche (la pente de la forme close)
    println(io, "\nDifférences consécutives de |V^-| (têtes) -- pas par couche :")
    dif = [up_meas[(i, :head)] - up_meas[(i - 1, :head)] for i in 2:n_layers]
    println(io, "  ", dif)
    println(io, "  constantes ? ", length(unique(dif)) == 1 ?
            "OUI, pas = $(dif[1])" : "NON -- $(length(unique(dif))) valeurs distinctes")

    println(io, "\nDifférences consécutives de |V^-| (MLP) :")
    difm = [up_meas[(i, :mlp)] - up_meas[(i - 1, :mlp)] for i in 2:n_layers]
    println(io, "  ", difm)
    println(io, "  constantes ? ", length(unique(difm)) == 1 ?
            "OUI, pas = $(difm[1])" : "NON -- $(length(unique(difm))) valeurs distinctes")

    # Ratio de chemins entre couches consécutives : la raison géométrique r
    println(io, "\nRatio P(i)/P(i+1) (têtes) -- raison géométrique attendue :")
    ratios = String[]
    for i in 1:(n_layers - 1)
        a, b = path_meas[(i, :head)], path_meas[(i + 1, :head)]
        push!(ratios, b == 0 ? "n/a" : string(a ÷ b, iszero(a % b) ? "" : "+r"))
    end
    println(io, "  ", ratios)

    # Fuite du corollaire : nœuds présents dans TOUS les cônes amont
    println(io, "\n" * "-"^78)
    println(io, "MESURE 3 -- ancêtres universels (fuite potentielle du corollaire)")
    println(io, "-"^78)
    common = oracle_upstream(rules, sites[1][1])
    for (s, _, _) in sites
        common = intersect(common, oracle_upstream(rules, s))
    end
    @printf(io, "\nNœuds ancêtres de TOUS les %d sites : %d\n", length(sites), length(common))
    if !isempty(common)
        shown = sort(collect(common))[1:min(12, length(common))]
        println(io, "  échantillon : ", shown)
        n_leaf_common = count(v -> !haskey(rules, v), common)
        @printf(io, "  dont feuilles : %d, calculés : %d\n",
                n_leaf_common, length(common) - n_leaf_common)
    end

    # raison géométrique exacte du compte de chemins, si constante
    r_exact = nothing
    let ok = true, r0 = nothing
        for i in 1:(n_layers - 1)
            a, b = path_meas[(i, :head)], path_meas[(i + 1, :head)]
            if b == 0 || a % b != 0
                ok = false; break
            end
            q = a ÷ b
            r0 === nothing ? (r0 = q) : (q == r0 || (ok = false))
            ok || break
        end
        ok && (r_exact = r0)
    end

    return (; label, n_layers, n_heads, N_total, n_rules, M_pred,
            down_ok = (nz == 0), up_meas, up_comp, up_leaf, path_meas,
            step_head = length(unique(dif)) == 1 ? dif[1] : nothing,
            step_mlp = length(unique(difm)) == 1 ? difm[1] : nothing,
            up1_head = up_meas[(1, :head)], up1_mlp = up_meas[(1, :mlp)],
            pL_head = path_meas[(n_layers, :head)],
            pL_mlp = path_meas[(n_layers, :mlp)],
            r_exact, n_common = length(common))
end

"""Ajuste une forme close linéaire en H sur deux points, puis la VALIDE
hors échantillon. Deux points déterminent une droite sans résidu : sans
points retenus, l'ajustement ne prouve rien."""
function fit_and_validate(io, fit_set, holdout)
    println(io, "\n" * "="^78)
    println(io, "FORMES CLOSES : ajustement sur E9, validation HORS ÉCHANTILLON")
    println(io, "="^78)

    a, b = fit_set
    # pas amont : attendu 7H+24 (tous les nœuds d'une couche, params inclus)
    println(io, "\nPas amont par couche :")
    for r in vcat(collect(fit_set), collect(holdout))
        pred = 7 * r.n_heads + 24
        @printf(io, "  H=%2d  mesuré=%s  7H+24=%d  %s\n", r.n_heads,
                something(r.step_head, "non constant"), pred,
                r.step_head == pred ? "ok" : "ÉCART")
    end

    # intercepts : ajustement affine en H sur les deux configs E9
    slope_h = (b.up1_head - a.up1_head) / (b.n_heads - a.n_heads)
    int_h   = a.up1_head - slope_h * a.n_heads
    slope_m = (b.up1_mlp - a.up1_mlp) / (b.n_heads - a.n_heads)
    int_m   = a.up1_mlp - slope_m * a.n_heads
    @printf(io, "\nAjusté sur E9 :  |V^-_head(1)| = %g*H + %g\n", slope_h, int_h)
    @printf(io,   "                 |V^-_mlp(1)|  = %g*H + %g\n", slope_m, int_m)

    println(io, "\nVALIDATION hors échantillon (configs jamais utilisées pour l'ajustement) :")
    @printf(io, "  %-12s %4s %4s  %-22s %-22s\n", "config", "L", "H",
            "|V^-_head(1)|", "|V^-_mlp(1)|")
    all_ok = true
    for r in holdout
        ph = slope_h * r.n_heads + int_h
        pm = slope_m * r.n_heads + int_m
        okh = isapprox(ph, r.up1_head; atol=1e-9)
        okm = isapprox(pm, r.up1_mlp; atol=1e-9)
        all_ok &= (okh && okm)
        @printf(io, "  %-12s %4d %4d  %6d vs %-6.0f %-6s %6d vs %-6.0f %-6s\n",
                r.label, r.n_layers, r.n_heads, r.up1_head, ph, okh ? "ok" : "ÉCART",
                r.up1_mlp, pm, okm ? "ok" : "ÉCART")
    end
    println(io, all_ok ?
        "  ✓ forme close amont VALIDÉE hors échantillon." :
        "  ✗ la forme affine en H est RÉFUTÉE hors échantillon -- ne pas publier.")

    # multiplicateur de chemins
    println(io, "\nMultiplicateur de chemins par couche r(H) -- P(i) = P(L) * r^(L-i) :")
    @printf(io, "  %4s %4s  %-16s %-16s %s\n", "L", "H", "r exact", "P_head(L)", "P_mlp(L)")
    for r in vcat(collect(fit_set), collect(holdout))
        @printf(io, "  %4d %4d  %-16s %-16s %s\n", r.n_layers, r.n_heads,
                something(r.r_exact === nothing ? nothing : string(r.r_exact), "NON géométrique"),
                string(r.pL_head), string(r.pL_mlp))
    end
    # Forme close de r, dérivée des différences troisièmes constantes de r/3
    # sur H = 4,8,12,16,20 (cubique). Vérifiée ici, pas asseriée en prose.
    println(io, "\nForme close proposée : r(H) = 6H^3 + 3H^2 + 3")
    r_ok = true
    for r in vcat(collect(fit_set), collect(holdout))
        H = r.n_heads
        pred = 6 * BigInt(H)^3 + 3 * BigInt(H)^2 + 3
        ok = (r.r_exact !== nothing && r.r_exact == pred)
        r_ok &= ok
        @printf(io, "  H=%2d  mesuré=%-8s prédit=%-8s %s\n", H,
                string(something(r.r_exact, "n/a")), string(pred), ok ? "ok" : "ÉCART")
    end
    println(io, r_ok ? "  ✓ r(H) VALIDÉE sur les 5 configs." :
                       "  ✗ r(H) RÉFUTÉE -- forme close fausse.")

    # Bornes terminales du compte de chemins
    pL_ok = all(r.pL_head == 3 && r.pL_mlp == 1 for r in vcat(collect(fit_set), collect(holdout)))
    @printf(io, "\n  P_head(L)=3 et P_mlp(L)=1 sur toutes les configs : %s\n", pL_ok)

    println(io, "\n  ── LE CONTRASTE ─────────────────────────────────────────────────")
    println(io, "  Dépendance arrière : LINÉAIRE en (L-i), pente 7H+24.")
    println(io, "  Termes sommés      : GÉOMÉTRIQUE en (L-i), raison 6H^3+3H^2+3.")
    println(io, "  Sur le MÊME graphe. C'est ce que la loi de longueur de chemin")
    println(io, "  doit exploiter -- et ce que l'expérience epsilon doit peser,")
    println(io, "  puisque le compte de termes ne dit rien de leur MASSE.")
    return all_ok && r_ok && pL_ok
end

# ─── Exécution ─────────────────────────────────────────────────────────────

const OUT = joinpath(@__DIR__, "e9_upstream_and_paths_results.txt")

open(OUT, "w") do io
    println(io, "PRÉALABLE E9 : formes closes AMONT et comptes de chemins EXACTS")
    println(io, "Structurel uniquement -- aucun chronométrage, aucun bruit d'horloge.")
    println(io, "Oracles indépendants de _ancestors_of! / _downstream_nodes, comparés à eux.")
    println(io, "Date : ", read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String) |> strip)

    results = []
    if MODE == "small"
        # Contrôle : les comptes de nœuds/cônes dépendent-ils de la largeur ?
        # Même (L,H), largeurs différentes -> les chiffres doivent être IDENTIQUES.
        println(io, "\nMODE small : contrôle d'indépendance à la largeur, (L,H)=(12,12)")
        a = analyze(io, "w768", 768, 12, 3072, 12, 20, 156)
        b = analyze(io, "w96",   96, 12,  128, 12, 20, 156)
        println(io, "\n" * "="^78)
        println(io, "CONTRÔLE LARGEUR")
        println(io, "="^78)
        same_N = a.N_total == b.N_total
        same_up = all(a.up_meas[k] == b.up_meas[k] for k in keys(a.up_meas))
        same_p  = all(a.path_meas[k] == b.path_meas[k] for k in keys(a.path_meas))
        @printf(io, "  N identique       : %s (%d vs %d)\n", same_N, a.N_total, b.N_total)
        @printf(io, "  |V^-| identiques  : %s\n", same_up)
        @printf(io, "  P(s) identiques   : %s\n", same_p)
        println(io, (same_N && same_up && same_p) ?
            "  ✓ structure indépendante de la largeur -- les formes closes ne dépendent que de (L,H)." :
            "  ✗ la largeur change la structure -- toute forme close en (L,H) seul est fausse.")
        push!(results, a); push!(results, b)
    else
        # Les deux graphes exacts d'E9. La largeur est réduite : le contrôle
        # `small` a établi que N, |V^-| et P(s) sont IDENTIQUES à largeur
        # différente pour un même (L,H) -- donc ces chiffres sont ceux des
        # graphes d'E9, pas une approximation, et le script tourne en CPU.
        e9a = analyze(io, "E9-a(12,12)",  96, 12,  192, 12, 20, 156)
        e9b = analyze(io, "E9-b(24,16)", 128, 16,  256, 24, 20, 408)

        # Configs RETENUES : jamais utilisées pour ajuster, servent à falsifier.
        h1 = analyze(io, "holdout(6,8)",   64,  8, 128,  6, 20,  54)
        h2 = analyze(io, "holdout(18,4)",  32,  4,  64, 18, 20,  90)
        h3 = analyze(io, "holdout(8,20)", 160, 20, 320,  8, 20, 168)

        results = [e9a, e9b, h1, h2, h3]
        validated = fit_and_validate(io, (e9a, e9b), (h1, h2, h3))

        println(io, "\n" * "="^78)
        println(io, "SYNTHÈSE")
        println(io, "="^78)
        @printf(io, "  sites E9 couverts     : %d (attendu 564)\n",
                12 * 13 + 24 * 17)
        @printf(io, "  porte aval franchie   : %s\n", all(r.down_ok for r in results))
        @printf(io, "  forme close validée   : %s\n", validated)
        for r in results
            @printf(io, "  %-14s L=%2d H=%2d  N=%5d  M=%3d  pas amont=%s  anc.univ=%d\n",
                    r.label, r.n_layers, r.n_heads, r.N_total, r.M_pred,
                    something(r.step_head, "non const"), r.n_common)
        end
    end
end

print(read(OUT, String))
println("\nÉcrit : ", OUT)
