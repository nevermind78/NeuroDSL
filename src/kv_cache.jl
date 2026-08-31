# ══════════════════════════════════════════════════════════════════════════════
# kv_cache.jl — primitive de cache KV pour la génération autorégressive
# incrémentale, 2026-07-28 (motivé par notebook/qwen2.ipynb).
#
# REWORK D'UNE PROPOSITION EXTÉRIEURE : la proposition d'origine supposait
# `struct KVCacheOp <: NeuroDSL.Operator` avec `NeuroDSL.evaluate!(op, ...)`
# -- CE MÉCANISME N'EXISTE PAS dans ce dépôt. Il n'y a AUCUN `abstract type
# Operator`, aucune hiérarchie de types dispatchée par méthode : chaque op
# est un `Symbol` (`GraphRule.op`), et un op personnalisé s'enregistre via
# `register_op!(name::Symbol, fn::Function)` dans un `Dict{Symbol,Function}`
# global (`CUSTOM_OPS`, `src/dispatch.jl:715`), exactement comme
# `src/flash_attention.jl`/`src/graph_surgery.jl` le font déjà. Ce fichier
# suit CE patron, pas celui de la proposition.
#
# CE QUI EST AUSSI REVU PAR RAPPORT À LA PROPOSITION : la proposition
# affirmait qu'une invalidation "ciblée" sur `:token_ids` (plutôt que
# `invalidate_all!`) évite de recalculer tout le graphe. Faux dans ce dépôt :
# `_invalidate_downstream!(g, :token_ids, ns)` (src/graph_api.jl) fait un
# parcours en aval qui, pour ce nœud racine, ATTEINT LITTÉRALEMENT tous les
# nœuds du modèle (token_ids alimente l'embedding, qui alimente la couche 1,
# qui alimente la couche 2, ..., jusqu'aux logits) -- ni plus ni moins que ce
# qu'`invalidate_all!` marque déjà. Le gain réel ne vient PAS du choix entre
# les deux fonctions d'invalidation (`invalidate_all!` ne touche d'ailleurs
# jamais `aux_data`, donc aucune des deux ne menace l'état du cache) : il
# vient (a) de nourrir le graphe avec un `:token_ids` de taille 1 (les
# projections Q/K/V et le MLP du nouveau token redeviennent alors des
# multiplications bon marché, par construction, sans rien de spécial à
# coder), et (b) de CE fichier, qui réinjecte l'historique K/V dans
# l'attention pour ce seul nouveau token.
#
# CE QUI EST SIMPLIFIÉ PAR RAPPORT À LA PROPOSITION (delta explicite, pas
# silencieux) : la proposition veut un tampon PRÉ-ALLOUÉ à `max_seq_len`,
# jamais réalloué, avec une VUE (`view`) zero-copy retournée à chaque pas.
# Ce mécanisme existe dans ce dépôt (`_VIEW_OPS`, src/dispatch.jl:42) mais
# est actuellement câblé en dur pour exactement deux ops (`:view_cols`,
# `:head_view`) et EXPLICITEMENT incompatible avec `execute_rule_pooled!`
# (src/dispatch.jl:275-278). L'étendre à un troisième op aurait touché le
# cœur du dispatcher (utilisé par CHAQUE op de ce dépôt) pour un gain de
# performance secondaire (éviter une copie de l'historique déjà en cache,
# elle-même bien moins coûteuse que ce que le cache économise déjà -- le
# recalcul des projections/MLP des anciens tokens à travers 28 couches).
# Choix fait ici : rester sur le chemin `CUSTOM_OPS` standard (allocation
# fraîche à chaque pas via `execute_rule!`, copie de l'historique + nouvelle
# ligne dans ce tampon frais), qui ne touche AUCUN mécanisme central du
# dispatcher. Correct d'abord, zero-copy plus tard si mesuré nécessaire.
#
# GARDE-FOU (voir aussi patching.jl) : chaque nœud produit par
# `:kv_cache_append` porte `aux_data[:kv_cache_active] = true`.
# `capture_activations`/`patch_node!` échouent bruyamment sur un tel nœud --
# leur état réel (l'historique) vit dans `aux_data`, invisible à ces deux
# fonctions, exactement la classe de bug qui a déjà mordu ce dépôt deux fois
# cette session (poids dupliqués dans `capture_activations`, alias sans
# `copy()` dans `patch_node!`).
# ══════════════════════════════════════════════════════════════════════════════

"""
    kv_cache_append!(dev, output_buffer, inputs, attrs, out_sym, out_node, ctx_store)

Op `:kv_cache_append`. Entrées `[new_row, cur_step]` :
  - `new_row`  :: `(1, d)` -- la ligne K (ou V) du nouveau token pour CETTE tête.
  - `cur_step` :: `(1,)`   -- position 1-indexée du nouveau token (= nombre de
    lignes valides dans l'historique APRÈS cet appel).

`output_buffer` est déjà alloué à la forme `(cur_step, d)` par le chemin
standard (`CUSTOM_SHAPE_RULES[:kv_cache_append]`, ci-dessous) -- ce n'est PAS
un tampon pré-alloué à `max_seq_len` réutilisé d'un pas à l'autre (voir la
note de conception plus haut) : une nouvelle allocation, de la taille
EXACTE requise, à chaque pas. `out_node.aux_data[:history]` porte la copie
qui survit à l'appel suivant (où `output_buffer` sera un objet Julia
différent -- l'ancien peut être libéré par `execute_rule!` entre deux appels,
donc TOUJOURS copier avant de stocker dans `aux_data`, jamais aliaser
`output_buffer` lui-même : c'est exactement le bug que `patch_node!` a dû
corriger le 2026-07-11 pour une raison différente, même leçon ici).
"""
function kv_cache_append!(dev, output_buffer, inputs, attrs, out_sym, out_node, ctx_store)
    new_row = inputs[1]::AbstractArray{Float32}
    cur_step = Int(Array(inputs[2])[1])
    d = size(new_row, 2)
    size(new_row, 1) == 1 || error("kv_cache_append! : new_row doit avoir exactement 1 ligne (un seul nouveau token), a $(size(new_row,1))")

    hist = get(out_node.aux_data, :history, nothing)
    if hist !== nothing && size(hist, 1) == cur_step - 1
        output_buffer[1:cur_step-1, :] .= hist
    elseif cur_step > 1
        error("kv_cache_append! (:$out_sym) : historique absent ou de taille incohérente " *
              "(attendu $(cur_step-1) lignes, trouvé $(hist === nothing ? "aucun" : size(hist,1))) -- " *
              "cur_step a-t-il été incrémenté sans appeler ce nœud, ou le cache a-t-il été " *
              "réinitialisé sans repartir de cur_step=1 ?")
    end
    output_buffer[cur_step, :] .= view(new_row, 1, :)

    # `copy(output_buffer)`, PAS `copy(Array(output_buffer))` : sur CUDA,
    # `output_buffer` est un `CuArray` -- un aller-retour host (`Array(...)`)
    # forcerait un transfert GPU->CPU à CHAQUE token généré, puis l'écriture
    # `output_buffer[1:cur_step-1,:] .= hist` de l'appel SUIVANT ferait un
    # broadcast CPU->GPU implicite (au mieux lent, au pire une erreur selon
    # la version de CUDA.jl). `copy()` reste sur le même device que
    # `output_buffer` dans les deux cas (CPU comme CUDA) -- toujours copie
    # réelle, jamais un alias, même garde-fou qu'avant.
    #
    # CORRECTIF MÉMOIRE 2026-08-31 (fuite multi-tours, trouvée en instrumentant
    # une VRAIE conversation à 4 tours dans notebook/qwen2.ipynb -- voir
    # notebook/diag_multiturn_vram_growth.jl) : `hist` (l'ANCIEN
    # `aux_data[:history]`, déjà copié dans `output_buffer` ci-dessus, donc
    # plus référencé par personne d'autre) était jusqu'ici simplement remplacé
    # par la ligne suivante, sans `Backend.free!` -- contrairement à `.value`
    # (`execute_rule!`, src/dispatch.jl:243-258), qui LUI est explicitement
    # libéré de façon SYNCHRONE avant réallocation, précisément pour éviter
    # d'attendre un passage du GC Julia qui "peut ne jamais arriver au milieu
    # d'une boucle chaude" (citation du commentaire de `Backend.free!`,
    # src/backend.jl -- exactement le cas ici : une réponse de ~200-300
    # tokens appelle ce nœud, pour CHAQUE tête K/V de CHAQUE couche
    # (`2 * n_layers * n_kv_heads` nœuds), une fois PAR TOKEN GÉNÉRÉ, sans
    # qu'aucun `GC.gc()` n'ait lieu entre deux tokens ni entre deux tours de
    # `chat(...)`). Mesuré : cette fuite (`aux_data[:history]` abandonné à
    # chaque pas, jamais rendu au pool CUDA avant le prochain passage
    # -- éventuel -- du GC) explique la quasi-totalité du saut de VRAM observé
    # par l'utilisateur (~8.5 Go -> ~13.3 Go) sur une conversation réelle à
    # 4 tours dont les tours 3-4 génèrent ~200-300 tokens chacun. Le même
    # patron existe dans `prime_kv_cache_from_prefix!` (src/layers.jl),
    # appelé une fois par tour plutôt qu'une fois par token -- corrigé de la
    # même façon, effet bien plus petit (un seul remplacement par tour et par
    # tête, pas par token).
    old_hist = hist
    out_node.aux_data[:history] = copy(output_buffer)
    old_hist !== nothing && Backend.free!(dev, old_hist)
    out_node.aux_data[:kv_cache_active] = true
    return output_buffer
end

"""
    kv_cache_shape(inputs, attrs) -> (cur_step, d)

`CUSTOM_SHAPE_RULES[:kv_cache_append]`. Ne reçoit PAS `out_node` (signature
imposée par `_infer_output_shape`, `src/dispatch.jl:70`) -- c'est précisément
pourquoi `cur_step` est un vrai nœud d'entrée du graphe (`inputs[2]`, réglé
via `set!` comme `:token_ids`/`:pos_ids`) plutôt qu'un compteur caché dans
`aux_data` : la règle de forme n'a accès qu'aux valeurs d'entrée courantes,
jamais à l'état du nœud de sortie.
"""
kv_cache_shape(inputs, attrs) = (Int(Array(inputs[2])[1]), size(inputs[1], 2))

"""
    scale_no_mask!(dev, output_buffer, inputs, attrs, out_sym, out_node, ctx_store)

Op `:scale_no_mask` -- mise à l'échelle `1/sqrt(d_head)` SANS masque causal.
Nécessaire pour le décodage incrémental : la requête du nouveau token peut
voir la totalité de l'historique du cache sans exception (le cache, par
construction, ne contient que des positions PASSÉES) -- `:scale_mask`
(existant) construit un masque causal CARRÉ via `causal_mask(dev,seqlen)`
avec `seqlen=size(scores,1)`, qui suppose implicitement que la requête et les
clés ont la même longueur -- faux ici (1 requête, `cur_step` clés). Utiliser
`:scale_mask` tel quel produirait soit une erreur de forme, soit (pire) un
masquage silencieusement incorrect selon comment `causal_mask` traite un
`scores` non carré. Ce nouvel op est un multiply élément par élément par une
CONSTANTE -- aucune non-linéarité, aucun état, rien à raisonner de plus.
"""
function scale_no_mask!(dev, output_buffer, inputs, attrs, out_sym, out_node, ctx_store)
    scores = inputs[1]::AbstractArray{Float32}
    d_head = attrs[:d_head]::Int
    output_buffer .= scores .* (1f0 / sqrt(Float32(d_head)))
    return output_buffer
end

"""
    _register_kv_cache_ops!()

Enregistrement paresseux (même patron que `_register_scalar_gate_op!` dans
`graph_surgery.jl` et `_register_flash_attn_in_dispatch!` dans
`flash_attention.jl` — un `Ref{Bool}` pour rendre l'appel idempotent, appelé
depuis le premier constructeur de graphe qui a réellement besoin de ces ops
plutôt qu'inconditionnellement au chargement du module).
"""
const _KV_CACHE_OPS_REGISTERED = Ref(false)
"""
    rope_at_pos!(dev, output_buffer, inputs, attrs, out_sym, out_node, ctx_store)

Op `:rope_at_pos` -- RoPE appliqué à UNE seule ligne (`(1, d)`) à une
position ABSOLUE explicite plutôt qu'à toute une séquence commençant à la
position 0. Nécessaire car l'op `:rope` existant (`src/dispatch.jl:583`)
calcule TOUJOURS `pos = 0:seqlen-1` -- correct pour un forward pass complet
(la ligne `i` d'un batch de longueur `seqlen` EST la position `i-1`), mais
FAUX en décodage incrémental : la nouvelle ligne a `seqlen=1`, donc `:rope`
lui assignerait systématiquement `pos=0`, quel que soit le nombre réel de
tokens déjà générés -- silencieusement incorrect (pas une erreur, un angle
de rotation faux) si réutilisé tel quel ici. Entrées `[x, pos_id]` :
  - `x`      :: `(1, d)`  -- une tête Q ou K du nouveau token.
  - `pos_id` :: `(1,)`    -- position 0-indexée absolue de ce token
    (= `cur_step - 1`, le même `cur_step` que celui donné à
    `:kv_cache_append`, transmis comme un vrai nœud d'entrée du graphe pour
    la même raison -- voir `kv_cache_shape` ci-dessus).

Ne met RIEN en cache dans `aux_data` (contrairement à `:rope`, qui met en
cache `cos_a`/`sin_a` par longueur de séquence) : la position change à
CHAQUE appel ici, donc rien à réutiliser ; le coût (un `cos`/`sin` sur un
vecteur de longueur `d/2`) est négligeable face à ce que le cache KV évite
déjà (le recalcul des projections/MLP des tokens précédents à travers
28 couches).
"""
function rope_at_pos!(dev, output_buffer, inputs, attrs, out_sym, out_node, ctx_store)
    x = inputs[1]::AbstractArray{Float32}
    pos = Float32(Array(inputs[2])[1])
    d = size(x, 2)
    half = d ÷ 2
    theta_base = Float32(get(attrs, :theta, 10000f0))
    theta = Backend.to_device(dev, Float32.(1f0 ./ (theta_base .^ ((0:half-1) ./ half))))
    angles = pos .* theta                          # (half,)
    cos_a = reshape(cos.(angles), 1, half)
    sin_a = reshape(sin.(angles), 1, half)
    output_buffer[:, 1:half]     .= x[:, 1:half] .* cos_a .- x[:, half+1:end] .* sin_a
    output_buffer[:, half+1:end] .= x[:, 1:half] .* sin_a .+ x[:, half+1:end] .* cos_a
    return output_buffer
end

function _register_kv_cache_ops!()
    _KV_CACHE_OPS_REGISTERED[] && return
    CUSTOM_SHAPE_RULES[:kv_cache_append] = kv_cache_shape
    CUSTOM_OPS[:kv_cache_append] = kv_cache_append!
    CUSTOM_SHAPE_RULES[:scale_no_mask] = (inputs, attrs) -> size(inputs[1])
    CUSTOM_OPS[:scale_no_mask] = scale_no_mask!
    CUSTOM_SHAPE_RULES[:rope_at_pos] = (inputs, attrs) -> size(inputs[1])
    CUSTOM_OPS[:rope_at_pos] = rope_at_pos!
    _KV_CACHE_OPS_REGISTERED[] = true
end
