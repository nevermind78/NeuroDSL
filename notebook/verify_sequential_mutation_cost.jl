# ══════════════════════════════════════════════════════════════════════════════
# Piste 1.1 : vérification numérique du "coût de mutation séquentielle" ajouté à
# math.tex (nouvelle sous-section après §sec:aggregate). Trois vérifications à
# tolérance ZÉRO (identités entières, pas de convergence asymptotique) :
#   (A) régime "interleaved" (demand! complet entre chaque greffe) : le coût de
#       recalcul de la greffe k doit égaler exactement c0_k + h_k + sum_{j<k,
#       e_j en aval de e_k dans G0} h_j, pour 3 ordres (amont-d'abord,
#       aval-d'abord, aléatoire).
#   (B) régime "batché" (une seule demande finale) : coût total = |union des
#       cônes originaux| + somme des h_k, indépendant de l'ordre.
#   (C) dérive wall-clock (empirique, PAS un théorème) : 100 greffes minimales
#       successives sur le même graphe, régression du coût de câblage (hors
#       demand) contre la taille courante du graphe, bras témoin = 100
#       re-câblages sans ajout de nœud (taille constante).
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics

const N = 400   # longueur de la chaîne hôte

function build_chain(ns::Symbol)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=NeuroDSL.Backend.CPUDevice())
    NeuroDSL.set!(g, :input, zeros(Float32, 4); atom_type=NeuroDSL.Datom, namespace=ns)
    cur = :input
    chain = Symbol[cur]
    for i in 1:N
        nxt = Symbol(:c, i)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(nxt, [cur], :relu; namespace=ns))
        cur = nxt
        push!(chain, cur)
    end
    NeuroDSL.demand!(g, cur; namespace=ns)
    return g, chain
end

function insert_chain!(g, ns, after_sym::Symbol, h::Int, tag)
    old_consumers = collect(get(NeuroDSL._consumers_index!(g, ns), after_sym, Symbol[]))
    cur = after_sym
    for i in 1:h
        nxt = Symbol(:graft_, tag, :_, i)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(nxt, [cur], :relu; namespace=ns))
        cur = nxt
    end
    new_out = cur
    for c in old_consumers
        rule = g.rules[ns][c]
        new_inputs = [inp == after_sym ? new_out : inp for inp in rule.inputs]
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(c, new_inputs, rule.op;
                                                 namespace=ns, attrs=rule.attrs, atom_type=rule.atom_type))
    end
    return new_out
end

# Exclut les feuilles (Datom sans règle, ex. :input) : leur flag .valid n'est jamais mis
# à jour par demand! (elles n'ont rien à recalculer) -- les compter fausserait le total
# d'un décalage constant de 1 (diagnostiqué à la main sur un exemple à 10 nœuds).
n_invalid(g, ns) = count(!nd.valid for (sym, nd) in g.nodes[ns] if haskey(g.rules[ns], sym))

# ── Précalcul dans G0 : cônes c0 et relation "en aval" entre sites ────────────
const K = 6
const H = 20  # taille uniforme des blocs greffés

g0, chain0 = build_chain(:precompute)
site_positions = round.(Int, range(N ÷ (K + 1), stop = N - N ÷ (K + 1), length = K))
site_syms = [chain0[p + 1] for p in site_positions]   # after_sym = chain[position+1] (1-based, chain[1]=:input)

c0 = Dict{Symbol,Int}()
for s in site_syms
    c0[s] = length(NeuroDSL._downstream_nodes(g0, s, :precompute)) - 1  # -1 : exclut s lui-même (s ne change pas de valeur, seul son câblage aval change)
end
downstream_of = Dict{Symbol,Int}(s => p for (s, p) in zip(site_syms, site_positions))
# e_j ≻ e_k (e_j en aval de e_k) ssi position(e_j) > position(e_k)
succ(sj, sk) = downstream_of[sj] > downstream_of[sk]

println("="^70)
println("Sites (positions dans la chaîne) : ", site_positions)
println("Cônes c0 (nœuds originaux en aval, hors le site lui-même) : ", [c0[s] for s in site_syms])
println("="^70)

# ── (A) Régime interleaved, formule exacte ────────────────────────────────────
function run_interleaved(order::Vector{Int}; tagprefix)
    g, chain = build_chain(Symbol(:inter_, tagprefix))
    sites_in_order = [site_syms[i] for i in order]
    measured = Int[]
    predicted = Int[]
    applied = Symbol[]   # sites déjà greffés, dans l'ordre d'application
    for (k, s) in enumerate(sites_in_order)
        NeuroDSL.demand!(g, chain[end]; namespace=Symbol(:inter_, tagprefix))  # tout valide avant la greffe k
        insert_chain!(g, Symbol(:inter_, tagprefix), s, H, Symbol(tagprefix, :_, k))
        push!(measured, n_invalid(g, Symbol(:inter_, tagprefix)))
        extra = sum(H for j_s in applied if succ(j_s, s); init=0)
        push!(predicted, c0[s] + H + extra)
        push!(applied, s)
    end
    return measured, predicted
end

order_upstream_first = sortperm(site_positions)                       # 1..K croissant (amont d'abord)
order_downstream_first = sortperm(site_positions, rev=true)           # K..1 décroissant (aval d'abord)
Random.seed!(7)
order_random = shuffle(1:K)

meas_u, pred_u = run_interleaved(order_upstream_first; tagprefix=:up)
meas_d, pred_d = run_interleaved(order_downstream_first; tagprefix=:down)
meas_r, pred_r = run_interleaved(order_random; tagprefix=:rand)

println("\n--- (A) Régime interleaved : mesuré vs prédit (tolérance ZÉRO) ---")
for (name, meas, pred) in [("amont-d'abord", meas_u, pred_u), ("aval-d'abord", meas_d, pred_d), ("aléatoire", meas_r, pred_r)]
    ok = meas == pred
    println(name, " : mesuré=", meas, "  prédit=", pred, "  EXACT=", ok)
end

Delta(meas, pred_iso) = sum(meas) - pred_iso
C_iso = sum(c0[s] + H for s in site_syms)
println("\nC_iso (K greffes isolées, prédiction naïve) = ", C_iso)
println("Delta mesuré : amont-d'abord=", sum(meas_u) - C_iso,
        "  aval-d'abord=", sum(meas_d) - C_iso,
        "  aléatoire=", sum(meas_r) - C_iso)
println("Delta théorique : amont-d'abord=0, aval-d'abord=H*K*(K-1)/2=", H * K * (K - 1) ÷ 2)

# ── (B) Régime batché : une seule demande finale, indépendant de l'ordre ─────
# Compte les invalides juste après les K greffes, avant tout demand! -- c'est le total
# de nœuds qui devront être recalculés au prochain demand! (peu importe qu'on l'appelle
# ou non ensuite : la quantité d'intérêt est ce total, pas le résultat du calcul).
function run_batched_correct(order::Vector{Int}; tagprefix)
    ns = Symbol(:batch2_, tagprefix)
    g, chain = build_chain(ns)
    sites_in_order = [site_syms[i] for i in order]
    for (k, s) in enumerate(sites_in_order)
        insert_chain!(g, ns, s, H, Symbol(tagprefix, :_, k))
    end
    return n_invalid(g, ns)
end

# Prédiction : union des cônes originaux (chaque _downstream_nodes(s) inclut s lui-même).
# Correction importante (trouvée en diagnostiquant un écart constant de K-1) : un site s_i
# n'est exempté du compte que si sa PROPRE valeur ne change jamais -- ce qui n'est vrai que
# pour les sites qui ne sont eux-mêmes en aval d'AUCUN autre site retenu. Un site s_i en aval
# d'un autre site s_j EST légitimement recalculé (comme n'importe quel nœud ordinaire en aval
# de s_j) : sur une chaîne totalement ordonnée, un seul site (le plus en amont) qualifie pour
# l'exemption, pas les K.
exempt = [s for s in site_syms if !any(succ(s, other) for other in site_syms if other != s)]
predicted_batched = length(union((Set(NeuroDSL._downstream_nodes(g0, s, :precompute)) for s in site_syms)...)) - length(exempt) + K * H
# correction : chaque _downstream_nodes(s) inclut s lui-même (qui ne compte pas comme "recalculé" puisqu'il ne change pas de valeur) -> -K
b_u = run_batched_correct(order_upstream_first; tagprefix=:bu)
b_d = run_batched_correct(order_downstream_first; tagprefix=:bd)
b_r = run_batched_correct(order_random; tagprefix=:br)
println("\n--- (B) Régime batché : mesuré vs prédit (tolérance ZÉRO) ---")
println("Prédit (union des cônes originaux, sites exclus, + K*H) = ", predicted_batched)
println("Mesuré : amont-d'abord=", b_u, "  aval-d'abord=", b_d, "  aléatoire=", b_r)
println("Tous identiques et == prédit : ", (b_u == b_d == b_r == predicted_batched))
println("Batché <= C_iso (sous-additif) : ", b_u <= C_iso, "  (batché=", b_u, " vs C_iso=", C_iso, ")")

# ── (C) Dérive wall-clock (empirique, pas un théorème) ────────────────────────
println("\n--- (C) Dérive wall-clock : greffes successives vs re-câblages à taille constante ---")
function drift_grafts(n_grafts::Int)
    ns = :drift_grafts
    g, chain = build_chain(ns)
    cur = chain[end]
    sizes = Int[]; times = Float64[]
    rng = MersenneTwister(3)
    for i in 1:n_grafts
        sz_before = length(g.nodes[ns])
        t0 = time_ns()
        cur = insert_chain!(g, ns, cur, 1, Symbol(:d, i))
        dt = (time_ns() - t0) / 1e6
        push!(sizes, sz_before); push!(times, dt)
    end
    return sizes, times
end

function drift_rewire(n_rewire::Int)
    ns = :drift_rewire
    g, chain = build_chain(ns)
    rng = MersenneTwister(3)
    sizes = Int[]; times = Float64[]
    for i in 1:n_rewire
        pos = rand(rng, 2:N)
        sym = chain[pos]
        rule = g.rules[ns][sym]
        sz_before = length(g.nodes[ns])
        t0 = time_ns()
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(sym, rule.inputs, rule.op; namespace=ns, attrs=rule.attrs, atom_type=rule.atom_type))
        dt = (time_ns() - t0) / 1e6
        push!(sizes, sz_before); push!(times, dt)
    end
    return sizes, times
end

sizes_g, times_g = drift_grafts(100)
sizes_r, times_r = drift_rewire(100)

# OLS sur des temps aussi bruités (JIT/GC/Dict-resize) donne des pentes non fiables (intercept
# négatif, physiquement impossible) -- comparaison first-half vs second-half sur médiane tronquée
# à la place, plus robuste aux valeurs aberrantes et directement interprétable (le bras témoin a
# une taille CONSTANTE, donc toute dérive first→second y serait un pur effet d'historique/du
# nombre de mutations, indépendant de la taille du graphe).
trimmed_median(x) = (s = sort(x); n = length(s); q = max(1, round(Int, 0.1n)); median(s[q+1:end-q]))
half(x) = length(x) ÷ 2
tg1, tg2 = trimmed_median(times_g[1:half(times_g)]), trimmed_median(times_g[half(times_g)+1:end])
tr1, tr2 = trimmed_median(times_r[1:half(times_r)]), trimmed_median(times_r[half(times_r)+1:end])
println("Greffes successives (taille croît ", sizes_g[1], "→", sizes_g[end]+1, ") : médiane tronquée 1re moitié=",
        round(tg1, digits=4), " ms, 2e moitié=", round(tg2, digits=4), " ms")
println("Re-câblages (taille constante=", sizes_r[1], ") : médiane tronquée 1re moitié=",
        round(tr1, digits=4), " ms, 2e moitié=", round(tr2, digits=4), " ms")
println("Ordre de grandeur comparable entre les deux bras : ",
        round(trimmed_median(times_g), digits=4), " ms (greffes) vs ",
        round(trimmed_median(times_r), digits=4), " ms (re-câblages, taille constante)")
