# ══════════════════════════════════════════════════════════════════════════════
# patching.jl — Patching d'activation incrémental
#
# `set!` ne convient PAS pour patcher un nœud qui a déjà un GraphRule : il
# marque le nœud LUI-MÊME invalide (`_invalidate_downstream!` invalide `target`
# en plus de ses successeurs), donc le `demand!` suivant le recalculerait à
# partir de sa propre règle et écraserait la valeur qu'on vient d'imposer --
# vérifié empiriquement (`y.valid == false` après `set!(g,:y,...)` sur un nœud
# `:y` défini par une règle). `set!` est fait pour des feuilles (entrées,
# paramètres), pas pour piloter la valeur d'un nœud calculé.
#
# `patch_node!` mute donc directement `.value`/`.valid` (le nœud patché reste
# `valid=true`, `demand!` ne le recalculera pas) et invalide seulement ses
# VRAIS consommateurs via `_invalidate_downstream!` réutilisé tel quel sur
# chacun d'eux -- aucun nouveau mécanisme de graphe, juste une orchestration
# différente de l'invalidation existante.
# ══════════════════════════════════════════════════════════════════════════════

"""
    capture_activations(g, ns=g.active_ns) -> Dict{Symbol,Any}

Copie la valeur de tous les nœuds **calculés** (pas les paramètres) du
namespace `ns`. Utilisé pour mettre en cache un run "propre" ou "corrompu"
avant de patcher. La copie est indispensable : `demand!`/`node(g,...).value`
renvoient un alias vers le buffer interne, qui peut être écrasé en place lors
d'un appel ultérieur.

Les nœuds `is_param=true` sont exclus délibérément (correctif du 2026-07-12,
mesuré via `notebook/etape0_vram_breakdown.jl`) : les poids ne changent jamais
entre le run propre et le run corrompu (seule l'entrée diffère), et
`patch_node!`/`restore_from_cache!` ne ciblent jamais un nœud paramètre --
avant ce correctif, chaque cache dupliquait pourtant tout le modèle en plus
de ses vraies activations (mesuré : 452 Mo de poids dupliqués sur 454 Mo de
delta total pour un cache, soit 95,6% de gaspillage pur, sur un modèle à
l'échelle GPT-2 small). Aucun changement fonctionnel : les poids restent
accessibles directement via le graphe, jamais lus depuis ce cache.
"""
capture_activations(g::NeuroGraph, ns::Symbol=g.active_ns) =
    Dict{Symbol,Any}(sym => copy(nd.value) for (sym, nd) in g.nodes[ns]
                      if nd.value !== nothing && !nd.is_param)

"""
    patch_node!(g, sym, cache; namespace=g.active_ns)

Impose la valeur `cache[sym]` sur le nœud `sym` -- le nœud reste `valid=true`
(un `demand!` ultérieur ne le recalculera pas à partir de sa propre règle),
et seuls ses vrais consommateurs (successeurs dans le graphe de règles) sont
invalidés, en aval, via `_invalidate_downstream!` déjà existant.

`copy(cache[sym])` est INDISPENSABLE avant `to_device` -- correctif du
2026-07-11 (bug trouvé lors de la calibration d'une tâche d'induction à
plusieurs sauts, attention non batchée) : `Backend.to_device(::CUDADevice, x)`
retourne `x` TEL QUEL (sans copie) si `x` est déjà un `CuArray` sur le bon
device (`src/backend.jl:21-26`) -- sans le `copy()`, `nd.value` se retrouvait
littéralement le MÊME objet mémoire que `cache[sym]`. Un `demand!` ultérieur
recalculant ce même nœud depuis sa propre règle (ex. un `:matmul` d'attention
non batchée) écrit alors dans ce buffer -- et corrompt SILENCIEUSEMENT le
cache externe, sans jamais toucher au nœud patché lui-même entre-temps.
Symptôme observé : `greedy_patch_search!` trouvait une vraie trajectoire de
recovery (~0.98) en interne, mais un `patch_nodes!` ultérieur réutilisant le
même cache pour une vérification indépendante donnait exactement 0.0 -- un
cache RE-capturé au même point donnait la valeur correcte. Même patron déjà
utilisé par `restore_from_cache!` ci-dessous (`nd.value = copy(cache[sym])`).
"""
function patch_node!(g::NeuroGraph, sym::Symbol, cache; namespace::Symbol=g.active_ns)
    nd = node(g, sym; namespace=namespace)
    nd.value = Backend.to_device(g.device, copy(cache[sym]))
    nd.valid = true
    nd.backwarded = false
    for out_sym in get(_consumers_index!(g, namespace), sym, _EMPTY_SYMBOL_VEC)
        _invalidate_downstream!(g, out_sym, namespace)
    end
    return g
end

"""
    recovery_metric(patched, clean, corrupted) -> Float64

Métrique standard de causal tracing : 1.0 = récupération complète du
comportement propre, 0.0 = aucune récupération (sortie toujours corrompue).
"""
function recovery_metric(patched, clean, corrupted)
    patched_a, clean_a, corrupted_a = Array(patched), Array(clean), Array(corrupted)
    denom = norm(corrupted_a .- clean_a)
    denom == 0 && return 1.0
    return 1.0 - norm(patched_a .- clean_a) / denom
end

"""
    patch_and_measure!(g, output_sym, patch_sym, clean_cache, corrupted_cache,
                        clean_output, corrupted_output; namespace=g.active_ns)

Patch `patch_sym` avec sa valeur propre en cache, mesure le temps de
recalcul de `output_sym` et l'effet (`recovery_metric`) sur la sortie, puis
restaure `patch_sym` à sa valeur corrompue en cache et reconverge -- l'appel
suivant peut ainsi tester un autre site en repartant d'un état corrompu
cohérent, sans reconstruire le graphe.
"""
function patch_and_measure!(g::NeuroGraph, output_sym::Symbol, patch_sym::Symbol,
                             clean_cache, corrupted_cache,
                             clean_output, corrupted_output;
                             namespace::Symbol=g.active_ns)
    patch_node!(g, patch_sym, clean_cache; namespace=namespace)
    t0 = time_ns()
    patched_output = demand!(g, output_sym; namespace=namespace)
    dt_ms = (time_ns() - t0) / 1e6
    recovery = recovery_metric(patched_output, clean_output, corrupted_output)

    # Restauration : repart d'un état corrompu cohérent pour le site suivant.
    patch_node!(g, patch_sym, corrupted_cache; namespace=namespace)
    demand!(g, output_sym; namespace=namespace)

    return (; recovery, time_ms=dt_ms)
end

# ══════════════════════════════════════════════════════════════════════════════
# Balayage multi-sites amorti
#
# `patch_and_measure!` restaure en rappelant patch_node!+demand! sur le nœud
# corrompu -- ce qui RECALCULE le même cône qu'à l'étape de mesure. Sur un
# balayage complet (un site par couche), le coût cumulé de ces restaurations
# recalculées est du même ordre que la somme des patches eux-mêmes : la
# restauration n'apporte aucune information nouvelle (on connaît déjà la
# valeur corrompue, capturée une fois pour toutes par capture_activations),
# donc il n'y a aucune raison de la recalculer. `restore_from_cache!` la
# remplace par une copie directe des valeurs déjà connues suivie d'un
# marquage valid=true -- un coût mémoire, pas un coût de calcul (matmul,
# attention, normalisation). C'est une capacité propre à un graphe persistant
# où chaque nœud est individuellement adressable et son état de validité
# explicite ; un framework à bande n'a pas de notion de "nœud" à restaurer
# directement, seulement "refaire tourner le graphe".
# ══════════════════════════════════════════════════════════════════════════════

"""
    _downstream_nodes(g, sym, ns) -> Set{Symbol}

Calcule, en lecture seule, l'ensemble des nœuds que `patch_node!(g, sym, ...)`
toucherait -- `sym` lui-même plus tout nœud atteignable depuis ses
consommateurs directs par la même traversée (valid-gated) que
`_invalidate_downstream!`, sans muter `valid`. Appelé AVANT de patcher (donc
sur un graphe encore valide) pour capturer le cône affecté exactement comme
`patch_node!` le calculerait, avant qu'il ne soit invalidé puis recalculé.
"""
function _downstream_nodes(g::NeuroGraph, sym::Symbol, ns::Symbol)
    visited = Set{Symbol}([sym])
    queue = Symbol[]
    consumers_map = _consumers_index!(g, ns)
    for out_sym in get(consumers_map, sym, _EMPTY_SYMBOL_VEC)
        push!(queue, out_sym)
    end
    while !isempty(queue)
        cur = pop!(queue)
        cur ∈ visited && continue
        push!(visited, cur)
        nd = get(g.nodes[ns], cur, nothing)
        if nd !== nothing
            for w in nd.watchers
                w ∈ visited || push!(queue, w)
            end
        end
        for out_sym in get(consumers_map, cur, _EMPTY_SYMBOL_VEC)
            out_nd = get(g.nodes[ns], out_sym, nothing)
            if out_nd !== nothing && out_nd.valid
                push!(queue, out_sym)
            end
        end
    end
    return visited
end

"""
    restore_from_cache!(g, ns, cache, nodes)

Remplace directement `.value` par `copy(cache[sym])` et force `.valid = true`
pour chaque `sym` de `nodes` -- aucune règle exécutée, aucun appel à
`demand!`. Correct si et seulement si `cache` a été capturé sur ce même état
(déterminisme du calcul garanti par construction : mêmes poids, mêmes
entrées, mêmes opérations).
"""
function restore_from_cache!(g::NeuroGraph, ns::Symbol, cache, nodes)
    for sym in nodes
        haskey(cache, sym) || continue
        nd = g.nodes[ns][sym]
        nd.value = copy(cache[sym])
        nd.valid = true
        nd.backwarded = false
    end
    return g
end

# ══════════════════════════════════════════════════════════════════════════════
# Restauration GPU par lots : un seul lancement de kernel au lieu d'un par nœud
#
# `restore_from_cache!` ci-dessus boucle nœud par nœud et copie chacun
# séparément -- gratuit sur CPU (aucun coût de lancement), mais sur GPU chaque
# `copy()` est une soumission séparée au driver. Sous Windows/WDDM, ce coût
# fixe par appel (~0.01-0.02 ms, vérifié empiriquement : un lancement de
# kernel trivial coûte quasiment le même prix qu'une copie de tenseur
# complète) domine largement le volume de données réellement copié dès que le
# cône contient des centaines de nœuds (mesuré : 20.5 ms pour un cône de 1876
# nœuds, soit 0.0109 ms/nœud -- au plancher de lancement, pas au débit
# mémoire). `_multi_copy_kernel!` regroupe tous les nœuds du cône en un seul
# lancement -- un bloc CUDA par nœud, chaque bloc copiant son propre tableau
# en interne -- sans toucher à `restore_from_cache!` existant, toujours
# utilisé tel quel sur CPU et partout où aucun gain n'est en jeu.
#
# Chaque nœud reçoit systématiquement un tampon fraîchement alloué (jamais
# une réutilisation de `nd.value` existant) : `patch_node!` peut faire
# pointer `nd.value` directement vers `cache[sym]` sans copie
# (`Backend.to_device` ne copie pas un `CuArray` déjà sur le bon device, voir
# `src/backend.jl:21-26`), donc réutiliser le tampon existant risquerait
# d'écrire par-dessus le cache lui-même dans ce cas précis.
# ══════════════════════════════════════════════════════════════════════════════

if Backend.CUDA_AVAILABLE
    function _multi_copy_kernel!(dst_ptrs::CUDA.CuDeviceVector{CUDA.CuPtr{Float32}},
                                  src_ptrs::CUDA.CuDeviceVector{CUDA.CuPtr{Float32}},
                                  lens::CUDA.CuDeviceVector{Int32})
        bid = blockIdx().x
        n = Int(lens[bid])
        dst = CUDA.CuDeviceArray{Float32,1,CUDA.AS.Global}(
            reinterpret(Core.LLVMPtr{Float32,CUDA.AS.Global}, dst_ptrs[bid]), (n,))
        src = CUDA.CuDeviceArray{Float32,1,CUDA.AS.Global}(
            reinterpret(Core.LLVMPtr{Float32,CUDA.AS.Global}, src_ptrs[bid]), (n,))
        i = Int(threadIdx().x)
        while i <= n
            @inbounds dst[i] = src[i]
            i += Int(blockDim().x)
        end
        return
    end
end

"""
    restore_from_cache_batched!(g, ns, cache, nodes)

Équivalent fonctionnel exact de `restore_from_cache!` -- sur CPU, délègue
directement à `restore_from_cache!` (aucun gain à batcher sans coût de
lancement à amortir). Sur GPU, remplace la boucle de copies séparées par un
seul lancement de kernel couvrant tous les nœuds du cône.
"""
function restore_from_cache_batched!(g::NeuroGraph, ns::Symbol, cache, nodes)
    _restore_from_cache_batched_impl!(g.device, g, ns, cache, nodes)
end

_restore_from_cache_batched_impl!(::Backend.CPUDevice, g, ns, cache, nodes) =
    restore_from_cache!(g, ns, cache, nodes)

if Backend.CUDA_AVAILABLE
    function _restore_from_cache_batched_impl!(::Backend.CUDADevice, g::NeuroGraph, ns::Symbol, cache, nodes)
        syms = Symbol[]
        newvals = Any[]
        srcs = CUDA.CuPtr{Float32}[]
        dsts = CUDA.CuPtr{Float32}[]
        lens = Int32[]
        for sym in nodes
            haskey(cache, sym) || continue
            src = cache[sym]
            dst = similar(src)
            push!(syms, sym)
            push!(newvals, dst)
            push!(srcs, pointer(src))
            push!(dsts, pointer(dst))
            push!(lens, Int32(length(src)))
        end
        if !isempty(lens)
            src_gpu = CUDA.CuArray(srcs)
            dst_gpu = CUDA.CuArray(dsts)
            len_gpu = CUDA.CuArray(lens)
            threads = min(256, Int(maximum(lens)))
            @cuda threads=threads blocks=length(lens) _multi_copy_kernel!(dst_gpu, src_gpu, len_gpu)
        end
        for (sym, val) in zip(syms, newvals)
            nd = g.nodes[ns][sym]
            nd.value = val
            nd.valid = true
            nd.backwarded = false
        end
        return g
    end
end

"""
    sweep_patch_sites!(g, output_sym, sites, clean_cache, corrupted_cache,
                        clean_output, corrupted_output; namespace=g.active_ns,
                        restore_fn=restore_from_cache!, gc_every=0)

Balaie `sites` (patch + mesure comme `patch_and_measure!`), mais restaure
via `restore_fn` (par défaut `restore_from_cache!`) au lieu de
patch_node!+demand! -- élimine le recalcul côté restauration pour un
balayage complet. Passer `restore_fn=restore_from_cache_batched!` pour la
variante groupée (avantageuse sur GPU, identique à `restore_from_cache!` sur
CPU). Retourne un vecteur de `(; site, recovery, patch_ms, restore_ms)`.

`gc_every` (défaut `0` = désactivé, aucun changement de comportement) : si
`>0`, force un GC complet (`GC.gc(true)`) toutes les `gc_every` itérations.
Correctif mesuré le 2026-07-12 (`notebook/etape0b_sweep_residual.jl`) : sur
CUDA, chaque `demand!`/`restore_from_cache!` alloue de nouveaux buffers par
nœud recalculé ; les buffers des sites précédents deviennent orphelins mais
ne sont libérés côté pool CUDA que lorsque le GC Julia tourne réellement --
sans appel explicite, le pic mémoire pendant un long balayage croît avec le
nombre de sites (mesuré : résidu de 231 Mo à 10 sites contre 1759 Mo à 156
sites sur un modèle à l'échelle GPT-2 small), alors que le besoin réel de
mémoire vive par site ne croît pas. Un GC complet périodique (pas un GC
mineur, mesuré nettement moins efficace : les buffers morts survivent le
temps d'être promus vers l'ancienne génération) ramène le pic proche de la
résidence fixe (poids + caches), quel que soit le nombre de sites balayés.
"""
function sweep_patch_sites!(g::NeuroGraph, output_sym::Symbol, sites::AbstractVector{Symbol},
                             clean_cache, corrupted_cache,
                             clean_output, corrupted_output;
                             namespace::Symbol=g.active_ns,
                             restore_fn::Function=restore_from_cache!,
                             gc_every::Int=0)
    results = NamedTuple[]
    for (i, site) in enumerate(sites)
        affected = _downstream_nodes(g, site, namespace)

        patch_node!(g, site, clean_cache; namespace=namespace)
        t0 = time_ns()
        patched_output = demand!(g, output_sym; namespace=namespace)
        patch_ms = (time_ns() - t0) / 1e6
        recovery = recovery_metric(patched_output, clean_output, corrupted_output)

        t1 = time_ns()
        restore_fn(g, namespace, corrupted_cache, affected)
        restore_ms = (time_ns() - t1) / 1e6

        push!(results, (; site, recovery, patch_ms, restore_ms))

        if gc_every > 0 && i % gc_every == 0
            GC.gc(true)
        end
    end
    return results
end

# ══════════════════════════════════════════════════════════════════════════════
# Patching composable multi-nœuds
#
# L'invalidation n'a jamais supposé qu'un seul nœud change à la fois :
# `_invalidate_downstream!` marque `valid=false` sur tout nœud atteint, peu
# importe combien de sources l'ont invalidé, et `demand!` visite chaque nœud
# de l'ordre topologique une seule fois, ne recalculant que ceux marqués
# invalides. Patcher plusieurs nœuds puis appeler `demand!` une seule fois
# calcule donc déjà l'union de leurs cônes en aval correctement, chaque nœud
# partagé n'étant recalculé qu'une fois -- sans code nouveau pour la fusion
# des cônes elle-même. Ce qui suit expose cette propriété via une API
# multi-sites, en réutilisant patch_node!/_downstream_nodes/restore_from_cache!
# tels quels.
# ══════════════════════════════════════════════════════════════════════════════

"""
    _topo_sort_syms(g, ns, syms) -> Vector{Symbol}

Trie `syms` selon leur position dans `topo_order!(g; namespace=ns)`
(ancêtres avant descendants). Bug corrigé le 2026-07-10 (voir note ci-dessous
sur `patch_nodes!`) : appliquer des patches ancêtre-descendant dans un ordre
arbitraire laisse le dernier patch appliqué sujet à être invalidé et
RECALCULÉ (pas juste marqué invalide) par un patch ultérieur situé en amont
de lui -- ce tri garantit que le patch appliqué EN DERNIER pour un nœud donné
est toujours celui d'un site qui n'a plus aucun ancêtre restant à patcher
après lui dans la boucle.
"""
function _topo_sort_syms(g::NeuroGraph, ns::Symbol, syms)
    order = topo_order!(g; namespace=ns)
    pos = Dict{Symbol,Int}(s => i for (i, s) in enumerate(order))
    return sort(collect(syms); by = s -> get(pos, s, typemax(Int)))
end

"""
    patch_nodes!(g, syms, cache; namespace=g.active_ns)

Applique `patch_node!` à chaque symbole de `syms`, dans l'ORDRE TOPOLOGIQUE
(ancêtres avant descendants -- voir `_topo_sort_syms`), sans appeler `demand!`
entre les deux -- l'appelant choisit quand demander la sortie, une seule
fois pour l'ensemble des patches. L'union de leurs cônes en aval est gérée
par le mécanisme d'invalidation existant, pas par cette fonction.

Bug corrigé le 2026-07-10 (trouvé et vérifié par test avant correction,
voir `test/test_patching.jl`) : si `syms` contenait un site aval PUIS un
site amont (ordre non-topologique), patcher le site amont invalidait le
site aval déjà patché, et le `demand!` de l'appelant le recalculait depuis
sa propre règle -- effaçant silencieusement son patch, sans erreur. Ce
n'était pas qu'un risque théorique : `greedy_patch_search!` sélectionne
souvent un site aval AVANT de tester un site amont candidat (un site plus
proche de la sortie a généralement une meilleure recovery individuelle,
donc est choisi en premier) -- exactement le cas qui déclenche le bug. Sur
un graphe à structure résiduelle (un site en aval avec une entrée directe
non patchée en plus de l'entrée patchée en amont), l'écart mesuré était de
0.66 sur l'échelle de recovery -- pas du bruit.
"""
function patch_nodes!(g::NeuroGraph, syms, cache; namespace::Symbol=g.active_ns)
    for sym in _topo_sort_syms(g, namespace, syms)
        patch_node!(g, sym, cache; namespace=namespace)
    end
    return g
end

"""
    restore_nodes_from_cache!(g, ns, cache, syms)

Restaure l'union des cônes en aval de chaque symbole de `syms` (calculée
avant tout patch, sur le graphe encore valide) en un seul appel à
`restore_from_cache!` -- réutilise le mécanisme de restauration amortie déjà
construit, pas de nouvelle logique.
"""
function restore_nodes_from_cache!(g::NeuroGraph, ns::Symbol, cache, syms)
    affected = Set{Symbol}()
    for sym in syms
        union!(affected, _downstream_nodes(g, sym, ns))
    end
    restore_from_cache!(g, ns, cache, affected)
    return g
end

# ══════════════════════════════════════════════════════════════════════════════
# Recherche gloutonne automatisée sur des combinaisons de sites
#
# `restore_from_cache!` n'est prouvée correcte (Invariant 3) que pour annuler
# UN SEUL patch appliqué depuis l'état pleinement corrompu. Ici, on teste un
# candidat PAR-DESSUS des sites déjà retenus (`selected`) : si le cône du
# candidat recoupe le cône d'un site déjà retenu (deux têtes de la même
# couche convergent par exemple vers le même hcat_heads), le restaurer via le
# cache corrompu réinitialiserait aussi la partie partagée à sa valeur
# ENTIÈREMENT corrompue -- effaçant silencieusement l'effet du site déjà
# retenu sur cette même partie du graphe, qui reste pourtant actif. Ce n'est
# pas un bug de `restore_from_cache!` (son contrat reste correct pour ce
# qu'il promet), c'est une limite à respecter : on ne l'utilise que quand
# c'est prouvé sûr (cônes disjoints), et on retombe sur la restauration par
# recalcul -- toujours correcte, quel que soit le recoupement -- sinon.
# ══════════════════════════════════════════════════════════════════════════════

"""
    _adaptive_restore!(g, ns, cand, cand_cone, selected_cone_union, corrupted_cache, output_sym)

Restaure `cand` : par cache (bon marché) si `cand_cone` ne recoupe pas
`selected_cone_union` (sûr, Invariant 3), sinon par recalcul
(`patch_node!`+`demand!`, toujours correct, y compris en présence d'un
recoupement avec des sites déjà retenus et actifs).
"""
function _adaptive_restore!(g::NeuroGraph, ns::Symbol, cand::Symbol, cand_cone,
                             selected_cone_union, corrupted_cache, output_sym::Symbol)
    if isempty(intersect(cand_cone, selected_cone_union))
        restore_from_cache!(g, ns, corrupted_cache, cand_cone)
    else
        patch_node!(g, cand, corrupted_cache; namespace=ns)
        demand!(g, output_sym; namespace=ns)
    end
    return g
end

"""
    greedy_patch_search!(g, output_sym, candidates, clean_cache, corrupted_cache,
                          clean_output, corrupted_output; max_sites=length(candidates),
                          namespace=g.active_ns, metric=nothing)

À chaque étape, teste chaque candidat restant par-dessus les sites déjà
retenus (patch + mesure + `_adaptive_restore!`), retient celui qui maximise
la récupération jointe si elle améliore le meilleur score courant, sinon
s'arrête. Retourne `(selected, trajectory)` : la liste des sites retenus
dans l'ordre choisi, et la trajectoire de récupération cumulée.

`metric` (défaut `nothing`, comportement inchangé -- `recovery_metric(out,
clean_output, corrupted_output)` sur la sortie entière) : si fourni,
`metric(out)` remplace cet appel. Sert par exemple à restreindre la mesure à
une seule ligne/position (`out[j:j,:]` -- voir `position_patch_cache`) quand
la sortie entière noierait le signal d'un effet localisé à une seule position
dans une longue séquence.

Bug corrigé le 2026-07-10 (trouvé par analyse du code, confirmé par test
avant correction) : la mesure "par-dessus les sites déjà retenus" ne
réappliquait jamais leurs patches après le patch du candidat -- si le
candidat testé était en amont d'un site déjà retenu, `patch_node!(cand,...)`
invalidait ce site en aval, et le `demand!` suivant le RECALCULAIT depuis sa
propre règle au lieu de garder sa valeur clean imposée, effaçant
silencieusement sa contribution pendant la mesure de la "recovery jointe".
Comme un site aval a généralement une meilleure recovery individuelle qu'un
site amont (moins de recalcul corrompu entre lui et la sortie), il est
souvent sélectionné en PREMIER par ce même algorithme -- le bug se déclenche
donc précisément dans le cas d'usage nominal (composition ancêtre-descendant
d'un circuit réel), pas dans un cas limite. Correctif : `patch_nodes!`
(maintenant trié topologiquement, voir sa docstring) réapplique explicitement
`selected` à chaque point de mutation, garantissant que tout site déjà
retenu reste effectivement patché pendant toute mesure ultérieure.
"""
function greedy_patch_search!(g::NeuroGraph, output_sym::Symbol, candidates,
                               clean_cache, corrupted_cache, clean_output, corrupted_output;
                               max_sites::Int=length(candidates), namespace::Symbol=g.active_ns,
                               metric::Union{Nothing,Function}=nothing)
    selected = Symbol[]
    remaining = collect(candidates)
    trajectory = NamedTuple[]
    best_so_far = 0.0
    selected_cone_union = Set{Symbol}()

    for _ in 1:max_sites
        best_site = nothing
        best_recovery = best_so_far
        for cand in remaining
            cand_cone = _downstream_nodes(g, cand, namespace)
            # Épingle le candidat ET tous les sites déjà retenus, ensemble,
            # dans l'ordre topologique -- garantit que la mesure qui suit
            # reflète VRAIMENT "cand + selected" tous patchés à clean, même
            # si cand est en amont d'un site déjà retenu.
            patch_nodes!(g, vcat(selected, [cand]), clean_cache; namespace=namespace)
            out = demand!(g, output_sym; namespace=namespace)
            r = metric === nothing ? recovery_metric(out, clean_output, corrupted_output) : metric(out)

            _adaptive_restore!(g, namespace, cand, cand_cone, selected_cone_union,
                                corrupted_cache, output_sym)
            # La restauration du candidat peut avoir ré-invalidé un site déjà
            # retenu (même logique) -- le ré-épingler avant le candidat suivant.
            isempty(selected) || patch_nodes!(g, selected, clean_cache; namespace=namespace)

            if r > best_recovery
                best_recovery = r
                best_site = cand
            end
        end
        best_site === nothing && break

        best_cone = _downstream_nodes(g, best_site, namespace)
        push!(selected, best_site)
        # Réapplique TOUT `selected` (pas seulement `best_site`) dans l'ordre
        # topologique -- laisse le graphe dans l'état "tous les sites retenus
        # sont effectivement patchés" en précondition du round suivant.
        patch_nodes!(g, selected, clean_cache; namespace=namespace)
        demand!(g, output_sym; namespace=namespace)

        filter!(s -> s != best_site, remaining)
        union!(selected_cone_union, best_cone)
        best_so_far = best_recovery
        push!(trajectory, (; site=best_site, cumulative_recovery=best_recovery))
    end
    return selected, trajectory
end

# ══════════════════════════════════════════════════════════════════════════════
# Élagage arrière de la recherche gloutonne
#
# `greedy_patch_search!` n'ajoute jamais que des sites -- elle ne teste jamais
# si un site déjà retenu est devenu superflu une fois les autres en place.
# `backward_prune!` pose directement cette question a posteriori sur une
# trajectoire déjà obtenue.
#
# Piège évité par construction : les sites retenus par la recherche gloutonne
# peuvent être en relation ancêtre-descendant (ex. une tête de couche 1 est en
# amont d'une tête de couche 2 retenue plus tard, via le flux résiduel).
# Essayer de "défaire" un seul site en présence d'autres patches actifs
# (patch_node! sur ce site + demand!) invaliderait en aval tout site
# descendant déjà patché, et demand! le RECALCULERAIT depuis sa propre règle
# -- effaçant silencieusement son patch actif, sans erreur ni avertissement.
# La solution : ne jamais défaire un patch isolément. Pour tester un
# sous-ensemble S, repartir de zéro (restaurer TOUTE l'union des cônes des
# sites originaux à l'état corrompu, via restore_from_cache! déjà prouvé
# exact) puis ré-appliquer patch_nodes!(S, clean_cache) dans l'ordre
# topologique naturel de `selected` -- même principe de prudence que
# _adaptive_restore!, mais simplifié : reset complet plutôt qu'un cas par cas
# sur le recoupement.
# ══════════════════════════════════════════════════════════════════════════════

"""
    backward_prune!(g, output_sym, selected, clean_cache, corrupted_cache,
                     clean_output, corrupted_output; namespace=g.active_ns, tol=1f-6,
                     metric=nothing)

Teste, sur la trajectoire `selected` d'une recherche gloutonne déjà terminée,
si un site peut être retiré sans faire chuter la récupération cumulée de plus
de `tol`. Ne défait jamais un patch isolément (voir note ci-dessus) : chaque
sous-ensemble testé est obtenu en restaurant l'union complète des cônes des
sites de `selected` à l'état corrompu, puis en ré-appliquant seulement les
sites du sous-ensemble (`patch_nodes!`, maintenant trié topologiquement --
voir sa docstring -- donc sûr même si `selected` n'est pas dans un ordre
ancêtre-descendant). Retourne `(remaining, pruned)`.

`metric` (défaut `nothing`) : même sens que sur `greedy_patch_search!` --
si fourni, `metric(out)` remplace `recovery_metric(out, clean_output,
corrupted_output)`.
"""
function backward_prune!(g::NeuroGraph, output_sym::Symbol, selected,
                          clean_cache, corrupted_cache, clean_output, corrupted_output;
                          namespace::Symbol=g.active_ns, tol::Float32=1f-6,
                          metric::Union{Nothing,Function}=nothing)
    selected = collect(selected)
    full_cone = Set{Symbol}()
    for sym in selected
        union!(full_cone, _downstream_nodes(g, sym, namespace))
    end

    function recovery_of(subset)
        restore_from_cache!(g, namespace, corrupted_cache, full_cone)
        patch_nodes!(g, subset, clean_cache; namespace=namespace)
        out = demand!(g, output_sym; namespace=namespace)
        return metric === nothing ? recovery_metric(out, clean_output, corrupted_output) : metric(out)
    end

    full_recovery = recovery_of(selected)
    remaining = copy(selected)
    pruned = Symbol[]

    changed = true
    while changed
        changed = false
        for site in remaining
            trial = filter(s -> s != site, remaining)
            r = recovery_of(trial)
            if r >= full_recovery - tol
                remaining = trial
                push!(pruned, site)
                changed = true
                break
            end
        end
    end

    # Restaure l'état du graphe sur le sous-ensemble final retenu.
    recovery_of(remaining)

    return remaining, pruned
end

"""
    position_patch_cache(base_cache, patch_sym, clean_cache, row::Int) -> Dict{Symbol,Any}

Construit un cache hybride qui ne diffère de `base_cache[patch_sym]` qu'à la
ligne `row` (remplacée par `clean_cache[patch_sym][row,:]`) -- restreint un
patch à une seule position plutôt qu'au nœud entier (patcher un nœud entier
restaure trivialement 100% par déterminisme, sans information). Aucun
changement à `patch_node!` : la sélectivité de position est un artifice côté
appelant, `patch_node!` continue de recevoir un cache de forme normale.
Promu depuis `notebook/patching.ipynb`/`notebook/induction.ipynb`, où cette
fonction était dupliquée à l'identique dans les deux.
"""
function position_patch_cache(base_cache, patch_sym::Symbol, clean_cache, row::Int)
    hybrid = copy(base_cache[patch_sym])
    hybrid[row, :] .= clean_cache[patch_sym][row, :]
    return Dict(patch_sym => hybrid)
end

"""
    set_params!(g, ns, weights::Dict{Symbol,<:AbstractArray})

Injecte des poids déjà connus (circuit construit à la main, poids chargés) sur
des paramètres existants -- fine enveloppe autour de `set!`, qui gère déjà le
transfert CPU/CUDA (`Backend.to_device`). Convertit en `Float32` (les tenseurs
pré-entraînés réels sont souvent en Float16/BFloat16 ; `NeuroGraph.nodes` est
`Float32` uniquement, `src/types.jl`).
"""
function set_params!(g::NeuroGraph, ns::Symbol, weights::Dict{Symbol,<:AbstractArray})
    for (sym, arr) in weights
        set!(g, sym, Float32.(arr); is_param=true, namespace=ns)
    end
    return g
end
