# ══════════════════════════════════════════════════════════════════════════════
# EXPÉRIENCE EPSILON : la masse d'un readout arrière se concentre-t-elle
# en bas ordre de branche, ou suit-elle le compte de chemins ?
#
# LA QUESTION
# -----------
# Le préalable (notebook/e9_upstream_and_paths.jl) a établi, à résidu nul sur
# 564 sites, un contraste exact sur le MÊME graphe :
#   - dépendance arrière : LINÉAIRE en (L-i), pente 7H+24
#   - termes sommés      : GÉOMÉTRIQUE en (L-i), raison 6H^3+3H^2+3
# Un compte de termes ne dit RIEN de leur masse. Toute loi de dégradation
# pilotée par le comptage suppose implicitement des termes de magnitude
# comparable. Veit et al. 2016 prédit que cette supposition est fausse dans un
# réseau résiduel : le chemin de skip a un gain O(1) et zéro facteur non
# linéaire, et l'écrasante majorité des chemins exponentiellement nombreux ne
# porte presque rien. Ce script pèse les ordres, au lieu de les compter.
#
# LE MÉCANISME
# ------------
# Op `:bwd_eps` : identité en AVANT (epsilon n'intervient pas), epsilon*dy en
# ARRIÈRE. Interposé sur la BRANCHE de chaque jonction résiduelle -- le skip
# reste intact. La structure réelle d'un bloc (vérifiée, pas supposée) est
#     layer_i_res1 = add(skip,  layer_i_mha_output_out)
#     layer_i_out  = add(res1,  layer_i_mlp_out)
# soit DEUX jonctions par couche, la branche étant toujours le second argument.
# Le gradient en un site devient alors un polynôme en epsilon dont le
# coefficient de degré k est exactement la contribution des chemins traversant
# k branches.
#
# POURQUOI PAS LA DFT AUX RACINES DE L'UNITÉ (plan initial)
# ---------------------------------------------------------
# `NeuroGraph` code en dur `GraphNode{Float32}` et il n'existe AUCUN support
# complexe dans src/. La récupération de coefficients par DFT n'est pas
# disponible sans toucher au cœur du moteur. Un Vandermonde réel de degré
# 2(L-i) ~ 46 serait catastrophiquement mal conditionné, a fortiori sur des
# gradients Float32 (~1e-7 de précision relative).
#
# CE QUI EST MESURÉ À LA PLACE (exact, deux passes, bien conditionné)
# ------------------------------------------------------------------
# L'ORDRE DE BRANCHE MOYEN PONDÉRÉ PAR LA MASSE. Avec
# g(eps) = somme_k a_k eps^k, on a ||g(eps)||^2 = somme_{k,l} <a_k,a_l> eps^(k+l),
# donc la pente logarithmique en eps=1 vaut
#     d log||g||^2 / d log eps  =  2 <g'(1), g(1)> / ||g(1)||^2
# et sa moitié
#     n_bar = <g'(1), g(1)> / ||g(1)||^2
# est une moyenne des ordres pondérée par leur contribution au gradient réel.
# Si un SEUL ordre k domine, n_bar = k exactement -- c'est l'ancrage qui rend
# la quantité interprétable. Obtenue par différence centrée en log eps autour
# de 1, avec contrôle de Richardson à h/2 : aucune extraction de coefficient,
# aucun système mal conditionné.
#
# PORTES DE CORRECTION, AVANT TOUT CHIFFRE
#   G1 : forward BIT-IDENTIQUE au graphe non recâblé (l'op est l'identité en
#        avant -- si le forward bouge, le recâblage est faux).
#   G2 : à eps=1, gradient de CHAQUE site bit-identique au graphe non recâblé
#        (eps=1 doit être un no-op exact).
#   G3 : à eps=0, gradient de CHAQUE site EXACTEMENT nul. Un site de branche
#        n'atteint la perte qu'en traversant le noeud eps de sa propre
#        jonction, donc a_0 = 0 par construction. Si un gradient survit à
#        eps=0, une branche a été manquée au recâblage.
#
# LIMITE DE PORTÉE, À NE PAS TAIRE
# --------------------------------
# La distribution de masse dépend des POIDS. Ce script mesure des poids à
# l'initialisation aléatoire : il caractérise le régime architecture+init, PAS
# un modèle entraîné. Un modèle entraîné peut redistribuer la masse. C'est la
# suite évidente (Qwen2.5-1.5B est sur le disque), pas une conclusion d'ici.
#
# CRITÈRE PRÉENREGISTRÉ (avant d'avoir vu le moindre chiffre)
#   ÉCHEC de la voie « comptage » : n_bar reste borné (<= 3) alors que l'ordre
#     maximal possible 1+2(L-i) croît jusqu'à 47 -- la masse est découplée du
#     compte, et toute loi de dégradation pilotée par le nombre de chemins est
#     morte. Résultat publiable : confirmation EXACTE de Veit et al. dans un
#     cadre où la littérature n'a que de l'échantillonnage et des ablations.
#   SUCCÈS de la voie : n_bar croît avec (L-i), et le profil mesuré fournit la
#     distribution de masse par ordre dont une loi de dégradation a besoin.
#
# USAGE
#   julia --project=. notebook/bench_eps_branch_order.jl
# Écrit notebook/bench_eps_branch_order_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, LinearAlgebra

const EPS      = Ref{Float32}(1.0f0)
const CAPTURED = Dict{Symbol,Array{Float32}}()

# Les gradients des nœuds intermédiaires ne survivent PAS à la passe arrière :
# ils retournent au `GradPool` et `nd.gradient` est remis à `nothing` (seuls les
# nœuds `is_param` sont conservés). Vérifié dans src/backward.jl:869,897-899.
# On capture donc le gradient AU MOMENT où il traverse, depuis la règle de
# gradient elle-même. Pour que la règle sache DE QUEL nœud il s'agit, on
# enregistre un op distinct par branche, chacun fermé sur son propre symbole --
# plus robuste que de dépendre de la structure interne du `ctx`.
function ensure_eps_op!(branch::Symbol)
    op = Symbol("epsop_", branch)
    haskey(NeuroDSL.CUSTOM_OPS, op) && return op
    NeuroDSL.register_op!(op,
        (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> (out_buf .= inputs[1]))
    NeuroDSL.CUSTOM_SHAPE_RULES[op] = (inputs, attrs) -> size(inputs[1])
    # dy est d/dl par rapport à la SORTIE du nœud eps ; comme l'op est
    # l'identité en avant, eps*dy est exactement d/dl par rapport à la branche.
    NeuroDSL.GRAD_RULES[op] = (dev, dy, ctx, inputs) -> begin
        gr = EPS[] .* dy
        CAPTURED[branch] = copy(Array(gr))
        return (gr,)
    end
    return op
end

"""Construit le graphe, avec ou sans les nœuds `:bwd_eps` sur les branches."""
function build(ns::Symbol, L::Int, H::Int, dim::Int, hidden::Int, seq::Int;
               rewire::Bool, seed::Int=7)
    dev = NeuroDSL.Backend.CPUDevice()
    Random.seed!(seed)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.set!(g, :input, NeuroDSL.Backend.rand32(dev, seq, dim) .- 0.5f0; namespace=ns)
    out = NeuroDSL.LlamaModel(L, dim, H, hidden; batched_attn=true)(g, :input; namespace=ns)

    n_rewired = 0
    if rewire
        for i in 1:L
            # Les deux jonctions résiduelles, structure vérifiée par inspection.
            for (join_sym, branch_sym) in ((Symbol("layer_", i, "_res1"),
                                            Symbol("layer_", i, "_mha_output_out")),
                                           (Symbol("layer_", i, "_out"),
                                            Symbol("layer_", i, "_mlp_out")))
                rule = g.rules[ns][join_sym]
                length(rule.inputs) == 2 ||
                    error("$join_sym : arité $(length(rule.inputs)), attendu 2")
                rule.inputs[2] == branch_sym ||
                    error("$join_sym : branche attendue en 2e position, trouvé $(rule.inputs[2])")

                eps_sym = Symbol("eps_", branch_sym)
                NeuroDSL.addrule!(g, NeuroDSL.GraphRule(eps_sym, [branch_sym],
                                                        ensure_eps_op!(branch_sym);
                                                        namespace=ns))
                NeuroDSL.addrule!(g, NeuroDSL.GraphRule(join_sym,
                                                        [rule.inputs[1], eps_sym], rule.op;
                                                        attrs=rule.attrs, namespace=ns,
                                                        atom_type=rule.atom_type))
                # OBLIGATOIRE : addrule! ne réinvalide pas un nœud existant dont
                # on redéfinit la règle (bug latent documenté 2026-07). Sans ça
                # le join servirait une valeur périmée.
                NeuroDSL._invalidate_downstream!(g, join_sym, ns)
                n_rewired += 1
            end
        end
    end

    # Perte scalaire : mse contre une cible fixe. `backward_graph!` sème des
    # `ones` sur le nœud de perte, donc sur un scalaire c'est exactement dl/dl=1.
    Random.seed!(seed + 1000)
    tgt = Symbol(:target)
    NeuroDSL.set!(g, tgt, NeuroDSL.Backend.rand32(dev, seq, dim) .- 0.5f0; namespace=ns)
    loss = :loss
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss, [out, tgt], :mse_loss; namespace=ns))
    return g, out, loss, n_rewired
end

"""Une passe arrière à epsilon donné. Retourne (gradients de branche capturés,
gradients de paramètres conservés). Une seule passe donne TOUTES les branches."""
function backward_at(g, ns, loss, eps::Float32)
    EPS[] = eps
    empty!(CAPTURED)
    NeuroDSL.demand!(g, loss; namespace=ns)
    NeuroDSL.backward_graph!(g, loss; namespace=ns)
    pg = Dict{Symbol,Array{Float32}}()
    for (s, nd) in g.nodes[ns]
        nd.is_param && nd.gradient !== nothing && (pg[s] = copy(Array(nd.gradient)))
    end
    return copy(CAPTURED), pg
end

function run_config(io, label, L, H, dim, hidden, seq)
    println(io, "\n" * "="^78)
    println(io, "CONFIG $label : L=$L  H=$H  dim=$dim  hidden=$hidden  seq=$seq")
    println(io, "="^78)

    ns_r, ns_p = Symbol("eps_", label), Symbol("plain_", label)
    gr, outr, lossr, n_rw = build(ns_r, L, H, dim, hidden, seq; rewire=true)
    gp, outp, lossp, _    = build(ns_p, L, H, dim, hidden, seq; rewire=false)
    @printf(io, "\nJonctions recâblées : %d (attendu %d = 2L)\n", n_rw, 2 * L)

    # Sites mesurés : les 2L sorties de branche. Le gradient y est capturé au
    # vol par la règle de l'op eps (les intermédiaires ne survivent pas).
    sites = vcat([Symbol("layer_", i, "_mha_output_out") for i in 1:L],
                 [Symbol("layer_", i, "_mlp_out") for i in 1:L])
    @printf(io, "Sites de branche mesurés : %d (attendu %d = 2L)\n", length(sites), 2 * L)

    # ── G1 : forward bit-identique ────────────────────────────────────────
    EPS[] = 1.0f0
    vr = Array(NeuroDSL.demand!(gr, outr; namespace=ns_r))
    vp = Array(NeuroDSL.demand!(gp, outp; namespace=ns_p))
    g1 = (vr == vp)
    println(io, "\n" * "-"^78)
    @printf(io, "G1  forward bit-identique au graphe non recâblé : %s   (max|diff| = %.3e)\n",
            g1 ? "OUI" : "NON", maximum(abs.(vr .- vp)))

    # ── G2 : eps=1 est un no-op exact ─────────────────────────────────────
    # Comparé sur les gradients de PARAMÈTRES, les seuls conservés par le
    # moteur -- et un test fort : ils dépendent de tout le chemin arrière.
    br1, pr1 = backward_at(gr, ns_r, lossr, 1.0f0)
    _,   pp1 = backward_at(gp, ns_p, lossp, 1.0f0)
    common = intersect(keys(pr1), keys(pp1))
    n_bit = count(s -> pr1[s] == pp1[s], common)
    worst = isempty(common) ? 0.0 :
            maximum(maximum(abs.(pr1[s] .- pp1[s]); init=0.0) for s in common)
    g2 = (n_bit == length(common) && length(common) == length(pp1))
    @printf(io, "G2  gradients de paramètres à eps=1 bit-identiques : %d/%d   (max|diff| = %.3e)\n",
            n_bit, length(pp1), worst)
    @printf(io, "    branches capturées : %d/%d\n", length(br1), length(sites))

    # ── G3 : eps=0 annule EXACTEMENT tout gradient de paramètre de couche ──
    # Tout paramètre d'un bloc siège dans une branche (norm1->q/k/v->...->
    # mha_output_out, ou norm2->mlp->mlp_out), donc son seul accès à la perte
    # traverse un nœud eps. Non vide, contrairement à un test sur le nœud eps
    # lui-même où eps*dy est nul par construction.
    _, pr0 = backward_at(gr, ns_r, lossr, 0.0f0)
    nz = count(s -> any(!=(0.0f0), pr0[s]), keys(pr0))
    g3 = (nz == 0)
    @printf(io, "G3  gradients de paramètres à eps=0 exactement nuls : %d/%d non nuls\n",
            nz, length(pr0))

    if !(g1 && g2 && g3)
        println(io, "\n✗ PORTE DE CORRECTION ÉCHOUÉE -- aucun chiffre de masse n'est lisible.")
        return (; label, L, H, gates_ok=false)
    end
    println(io, "\n✓ G1, G2, G3 franchies -- les chiffres de masse sont lisibles.")

    # ── Mesure : ordre de branche moyen pondéré par la masse ──────────────
    # n_bar = (1/2) d log||g(eps)||^2 / d log eps  en eps=1, différence centrée.
    function nbar(h::Float64)
        bp, _ = backward_at(gr, ns_r, lossr, Float32(exp(h)))
        bm, _ = backward_at(gr, ns_r, lossr, Float32(exp(-h)))
        Dict(s => (log(sum(abs2, bp[s])) - log(sum(abs2, bm[s]))) / (4h) for s in sites)
    end
    nb_a = nbar(0.05)
    nb_b = nbar(0.025)   # contrôle de Richardson : les deux doivent concorder

    println(io, "\n" * "-"^78)
    println(io, "ORDRE DE BRANCHE MOYEN n_bar, par couche et par type de branche")
    println(io, "  ordre max possible = 1 + 2(L-i) ;  n_bar >= 1 toujours (a_0 = 0)")
    println(io, "-"^78)
    @printf(io, "\n%6s %6s %6s %10s %10s %10s %9s\n",
            "couche", "L-i", "type", "max ordre", "n_bar", "n_bar/max", "|h-h/2|")
    rows = Tuple{Int,Int,Symbol,Float64,Float64}[]
    # Ordre MAXIMAL par type de branche : une branche MLP traverse son propre
    # noeud eps puis 2 par couche aval (1+2(L-i)) ; une branche ATTN en traverse
    # un de plus, celui du MLP de SA PROPRE couche, via res1 -> norm2 -> mlp ->
    # eps -> out, soit 2+2(L-i). Erreur d'etiquetage revelee par l'ancrage.
    for i in 1:L, (tag, s) in ((:attn, Symbol("layer_", i, "_mha_output_out")),
                               (:mlp,  Symbol("layer_", i, "_mlp_out")))
        mx = (tag === :attn ? 2 : 1) + 2 * (L - i)
        @printf(io, "%6d %6d %6s %10d %10.3f %10.3f %9.2e\n",
                i, L - i, tag, mx, nb_a[s], nb_a[s] / mx, abs(nb_a[s] - nb_b[s]))
        push!(rows, (i, L - i, tag, nb_a[s], nb_a[s] / mx))
    end

    nb_max = maximum(r[4] for r in rows)
    richardson = maximum(abs(nb_a[s] - nb_b[s]) for s in sites)
    i1 = findfirst(r -> r[1] == 1 && r[3] === :attn, rows)
    deep = rows[i1]
    @printf(io, "\n  n_bar maximal sur toutes les couches : %.3f\n", nb_max)
    @printf(io, "  écart Richardson max (h vs h/2)       : %.3e\n", richardson)
    @printf(io, "  couche 1 (attn) : n_bar = %.3f pour un ordre max de %d (ratio %.4f)\n",
            deep[4], 2 + 2 * (L - 1), deep[5])

    # Profil de ||g(eps)|| : si l'ordre 1 domine, la norme est ~proportionnelle
    # à eps ; si un ordre élevé domine, elle s'effondre aux petits eps.
    println(io, "\nProfil ||g(eps)||/||g(1)|| au site attn de couche 1 :")
    s1 = Symbol("layer_1_mha_output_out")
    ref = sqrt(sum(abs2, br1[s1]))
    D1 = 2 + 2 * (L - 1)
    for e in (0.125f0, 0.25f0, 0.5f0, 1.0f0, 2.0f0)
        bb, _ = backward_at(gr, ns_r, lossr, e)
        @printf(io, "  eps=%-6.3f  mesuré=%.6e   eps^1=%.3e   eps^%d=%.3e\n",
                e, sqrt(sum(abs2, bb[s1])) / ref, Float64(e), D1, Float64(e)^D1)
    end

    return (; label, L, H, gates_ok=true, nb_max, richardson,
            nb_layer1=deep[4], max_order_layer1=2 + 2 * (L - 1),
            ratio_layer1=deep[5], rows)
end

const OUT = joinpath(@__DIR__, "bench_eps_branch_order_results.txt")

open(OUT, "w") do io
    println(io, "EXPÉRIENCE EPSILON : masse par ordre de branche d'un readout arrière")
    println(io, "Poids à l'INITIALISATION ALÉATOIRE -- régime architecture+init, pas un")
    println(io, "modèle entraîné. Limite de portée, pas un oubli.")
    println(io, "Critère préenregistré : n_bar borné (<=3) alors que l'ordre max croît")
    println(io, "  => masse découplée du compte => lois pilotées par le comptage mortes.")
    println(io, "Date : ", strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))

    res = []
    push!(res, run_config(io, "L6H4",   6, 4,  64, 128, 8))
    push!(res, run_config(io, "L12H12", 12, 12, 96, 192, 8))
    push!(res, run_config(io, "L24H16", 24, 16, 128, 256, 8))

    println(io, "\n" * "="^78)
    println(io, "VERDICT")
    println(io, "="^78)
    ok = all(r.gates_ok for r in res)
    @printf(io, "\n  portes de correction franchies partout : %s\n", ok)
    if ok
        @printf(io, "\n  %-8s %4s %4s %12s %12s %10s\n",
                "config", "L", "H", "max n_bar", "ordre max", "ratio")
        for r in res
            @printf(io, "  %-8s %4d %4d %12.3f %12.0f %10.4f\n",
                    r.label, r.L, r.H, r.nb_max, r.max_order_layer1, r.ratio_layer1)
        end
        grew = res[end].nb_max > 3.0
        println(io, "\n  n_bar dépasse-t-il 3 à L=24 (ordre max 47) ? ", grew ? "OUI" : "NON")
        println(io, grew ?
          "  => la masse suit la profondeur : voie « comptage » VIVANTE." :
          "  => la masse reste concentrée en bas ordre alors que le compte de chemins\n" *
          "     atteint ~1e101 : voie « comptage » MORTE. Veit et al. confirmé\n" *
          "     exactement, et toute loi de dégradation par nombre de chemins tombe.")
    end
end

print(read(OUT, String))
println("\nÉcrit : ", OUT)
