# ══════════════════════════════════════════════════════════════════════════════
# graph_surgery.jl — Chirurgie à chaud : mutation d'architecture en cours
# d'entraînement, avec préservation exacte de la fonction calculée.
#
# Nom distinct de `checkpoint.jl` (recalcul d'activations en RAM) et
# `serialization.jl` (persistance disque) -- sans rapport avec ces deux fichiers.
#
# Insérer une couche dans un réseau PyTorch/JAX déjà entraîné exige de
# reconstruire le module, migrer l'état de l'optimiseur à la main, retracer/
# recompiler. Ici, ça se réduit à re-clé quelques `GraphRule` (`addrule!`) et à
# un `_invalidate_downstream!` ciblé -- le coût est exactement le cône aval du
# point d'insertion, pas un rebuild complet. Technique reconnue dans la
# littérature (progressive stacking, Net2Net, LiGO).
# ══════════════════════════════════════════════════════════════════════════════

"""
    insert_block!(g, ns, after_sym, dim, n_heads, hidden_dim; prefix) -> Symbol

Insère un nouveau `LlamaBlock` juste après `after_sym` dans le flux résiduel,
initialisé pour calculer l'IDENTITÉ EXACTE (voir ci-dessous), et rebranche tous
les consommateurs PRÉEXISTANTS de `after_sym` vers la sortie du nouveau bloc.
Retourne le symbole de sortie du nouveau bloc.

Mécanisme d'identité exacte : `LlamaBlock` calcule `r1 = x + attn_out(x)` puis
`out = r1 + mlp_out(r1)` (`src/layers.jl:96-125`). En mettant à zéro la
projection de sortie de l'attention (`<prefix>_mha_output_W`) ET la dernière
matrice du MLP (`<prefix>_mlp_w2`), `attn_out ≡ 0` et `mlp_out ≡ 0` quelle que
soit l'entrée -- le bloc entier calcule alors `out = x` exactement, quels que
soient les autres poids (Q/K/V/mlp_w1/mlp_w3, laissés à leur initialisation
aléatoire habituelle) puisque leurs contributions sont multipliées par zéro en
sortie.

**Historique de correction** : la découverte de ce mécanisme a révélé un vrai trou du filet de
sécurité du moteur réactif lui-même -- `addrule!` (`src/graph_api.jl`) ne réinvalidait PAS un
nœud déjà existant quand on redéfinissait sa règle avec de nouveaux `inputs` (seuls les caches
de topologie/consommateurs étaient réinitialisés). Rebrancher un consommateur sans appeler
explicitement `_invalidate_downstream!` dessus servait alors silencieusement une valeur
périmée. **Corrigé à la source dans `addrule!`** (qui réinvalide maintenant automatiquement
tout nœud existant dont la règle est redéfinie) plutôt que de compter sur chaque appelant pour
s'en souvenir -- `insert_block!` n'a donc plus besoin d'appeler `_invalidate_downstream!`
lui-même, `addrule!` seul suffit désormais. Voir `test/test_surgery.jl`, section F1, qui vérifie
directement cette garantie.
"""
function insert_block!(g::NeuroGraph, ns::Symbol, after_sym::Symbol,
                        dim::Int, n_heads::Int, hidden_dim::Int;
                        prefix::Symbol=Symbol(:surgery_, after_sym))
    # 1. Capturer les consommateurs AVANT de construire le nouveau bloc -- sinon
    #    on capture aussi la propre consommation de after_sym par le nouveau
    #    bloc lui-même (son _norm1).
    old_consumers = collect(get(_consumers_index!(g, ns), after_sym, Symbol[]))

    # 2. Construire le nouveau bloc à partir de after_sym.
    new_out = LlamaBlock(dim, n_heads, hidden_dim)(g, after_sym, prefix; namespace=ns)

    # 3. Initialisation à zéro EXACTE (pas approximative) des deux poids qui
    #    annulent les deux branches résiduelles du bloc.
    mha_prefix = Symbol(prefix, :_mha)
    output_W_sym = Symbol(mha_prefix, :_output_W)
    mlp_w2_sym   = Symbol(prefix, :_mlp_w2)
    set_params!(g, ns, Dict(
        output_W_sym => zeros(Float32, dim, dim),
        mlp_w2_sym   => zeros(Float32, dim, hidden_dim),
    ))

    # 4. Rebrancher les consommateurs préexistants vers new_out (un LlamaBlock
    #    inséré au milieu d'un LlamaModel a typiquement DEUX consommateurs
    #    directs -- le _norm1 et le _res1 du bloc suivant -- capturés tous les
    #    deux automatiquement via _consumers_index!, aucun cas particulier).
    # addrule! réinvalide maintenant lui-même le nœud redéfini -- aucun appel
    # supplémentaire à _invalidate_downstream! n'est nécessaire ici.
    for c in old_consumers
        rule = g.rules[ns][c]
        new_inputs = [inp == after_sym ? new_out : inp for inp in rule.inputs]
        addrule!(g, GraphRule(c, new_inputs, rule.op; attrs=rule.attrs,
                               namespace=ns, atom_type=rule.atom_type))
    end
    return new_out
end

# ══════════════════════════════════════════════════════════════════════════════
# graft_shadow_block! — greffe gatée : F(x) = x + alpha * R(x; theta)
#
# `insert_block!` (ci-dessus) force l'identité en mettant à zéro les poids de
# SORTIE du bloc (Net2Net/Fixup) -- aucun gate, `theta` normal partout sauf ces
# deux matrices. `graft_shadow_block!` isole une deuxième façon d'obtenir
# F(x)=x à l'initialisation, structurellement différente : un scalaire APPRENABLE
# `alpha` multiplie la sortie de toute la branche, `theta` restant ENTIÈREMENT
# aléatoire (ReZero / Gradient Shadowing). Les deux mécanismes sont comparables
# côte à côte via les mots-clés `alpha0`/`zero_out_proj` :
#   - alpha0=0, zero_out_proj=false : « rezero »     -- gate à zéro, theta random.
#   - alpha0=1, zero_out_proj=true  : « net2net »     -- gate à un, sorties à zéro.
#   - alpha0=0, zero_out_proj=true  : « degenerate »  -- point de selle vrai
#     (voir plus bas -- ∂L/∂alpha ET ∂L/∂theta restent EXACTEMENT nuls pour
#     toujours, pas seulement petits).
# ══════════════════════════════════════════════════════════════════════════════

const _SCALAR_GATE_REGISTERED = Ref(false)
function _register_scalar_gate_op!()
    _SCALAR_GATE_REGISTERED[] && return
    # Forward identique à :mul (out .= a .* b, avec b un scalaire (1,) qui
    # broadcast naturellement) -- seule la règle backward diffère : :mul
    # suppose que les deux entrées ont la même forme que dy (correct pour un
    # multiply élément-par-élément classique), ce qui donnerait un gradient de
    # forme pleine pour `alpha` au lieu de (1,), et ferait échouer accum_grad!.
    register_op!(:scalar_gate, (dev, out, inputs, attrs, out_sym, out_node, ctx_store) ->
        (out .= inputs[1] .* inputs[2]; out))
    GRAD_RULES[:scalar_gate] = (dev, dy, ctx, inputs) -> begin
        R, alpha = inputs[1], inputs[2]
        dR     = dy .* alpha                                   # forme de dy/R (broadcast)
        dalpha = Backend.to_device(dev, Float32[sum(dy .* R)])  # réduit à la forme (1,) d'alpha
        (dR, dalpha)
    end
    _SCALAR_GATE_REGISTERED[] = true
end

"""
    graft_shadow_block!(g, ns, after_sym, dim, n_heads, hidden_dim;
                         alpha0, zero_out_proj, prefix) -> (Symbol, NamedTuple)

Greffe une branche résiduelle gatée `R(x;theta)` juste après `after_sym` :
`new_out = after_sym + alpha * R(x)`, avec `R = ao + mo` où `ao` (attention) et
`mo` (MLP) lisent TOUTES LES DEUX `after_sym` directement et indépendamment
(bloc "parallèle", PAS la chaîne séquentielle de `LlamaBlock` où le MLP lit la
sortie de l'attention) -- indispensable pour que `zero_out_proj` désactive
chaque branche séparément sans affamer l'autre (si le MLP lisait `norm(ao)`,
zéroter `W_O` donnerait `ao≡0` donc une entrée MLP ≡0, tuant aussi son
gradient). SANS jamais additionner `x` à l'intérieur -- l'addition finale avec
`after_sym` est la SEULE, portée par le gate `alpha`. Contrairement à
`insert_block!`, `theta` (Q/K/V/W_O/mlp_w1/w2/w3) est laissé à son initialisation
aléatoire habituelle par défaut ; `zero_out_proj=true` reproduit en plus la
recette Net2Net/Fixup (mise à zéro de `W_O` et `mlp_w2`) pour comparaison directe.

Retourne `(new_out, handle)` où `handle = (; alpha_sym, R_sym, out_sym, prefix)`
donne les symboles nécessaires pour lire/patcher `alpha` et `R(x)` séparément.

Rebranche tous les consommateurs préexistants de `after_sym` vers `new_out`,
exactement comme `insert_block!` (même bug historique déjà corrigé dans
`addrule!` -- aucun appel explicite à `_invalidate_downstream!` nécessaire ici).
"""
function graft_shadow_block!(g::NeuroGraph, ns::Symbol, after_sym::Symbol,
                              dim::Int, n_heads::Int, hidden_dim::Int;
                              alpha0::Float32, zero_out_proj::Bool,
                              prefix::Symbol=Symbol(:shadow_, after_sym))
    _register_scalar_gate_op!()

    old_consumers = collect(get(_consumers_index!(g, ns), after_sym, Symbol[]))

    # R(x;theta) : bloc "parallèle" -- attention et MLP lisent TOUTES LES DEUX
    # directement after_sym (pas l'une la sortie de l'autre, contrairement à
    # LlamaBlock qui les chaîne en séquence via r1). Nécessaire pour que
    # zero_out_proj contrôle chaque branche INDÉPENDAMMENT : si le MLP lisait
    # la sortie de l'attention (norm(ao)) et que W_O est mis à zéro, ao≡0 donc
    # xn2≡0 donc swiglu(0,0)≡0 -- la branche MLP se retrouverait AFFAMÉE
    # (aucun signal, ni forward ni backward) par un zéro qui ne la concerne
    # pas. Découvert empiriquement : `d(mlp_w2)` sortait exactement nul avec
    # zero_out_proj=true même si mlp_w2 lui-même n'est jamais zéro-initialisé
    # comme facteur direct -- tracé jusqu'à `sg` (sortie swiglu) elle-même
    # exactement nulle à cause de ce couplage, pas un bug de :matmul/:add.
    xn1 = LayerNorm(dim)(g, after_sym, Symbol(prefix, :_norm1); namespace=ns)
    ao  = MultiHeadAttention(dim, n_heads)(g, xn1, Symbol(prefix, :_mha); namespace=ns)
    xn2 = LayerNorm(dim)(g, after_sym, Symbol(prefix, :_norm2); namespace=ns)

    k = 1f0/sqrt(Float32(dim))
    for (wname, sh) in [(:_mlp_w1,(hidden_dim,dim)), (:_mlp_w2,(dim,hidden_dim)), (:_mlp_w3,(hidden_dim,dim))]
        W = (Backend.rand32(g.device, sh...) .- 0.5f0) .* (2k)
        set!(g, Symbol(prefix, wname), W; is_param=true, namespace=ns)
    end
    gt = Symbol(prefix, :_gate); up = Symbol(prefix, :_up); sg = Symbol(prefix, :_swiglu)
    mo = Symbol(prefix, :_mlp_out)
    addrule!(g, GraphRule(gt, [xn2, Symbol(prefix,:_mlp_w1)], :matmul;
             attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=ns))
    addrule!(g, GraphRule(up, [xn2, Symbol(prefix,:_mlp_w3)], :matmul;
             attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=ns))
    addrule!(g, GraphRule(sg, [gt, up], :swiglu; namespace=ns))
    addrule!(g, GraphRule(mo, [sg, Symbol(prefix,:_mlp_w2)], :matmul;
             attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=ns))

    R_sym = Symbol(prefix, :_R)
    addrule!(g, GraphRule(R_sym, [ao, mo], :add; namespace=ns))

    alpha_sym = Symbol(prefix, :_alpha)
    set!(g, alpha_sym, Float32[alpha0]; is_param=true, namespace=ns)

    gated_sym = Symbol(prefix, :_gated)
    addrule!(g, GraphRule(gated_sym, [R_sym, alpha_sym], :scalar_gate; namespace=ns))

    out_sym = Symbol(prefix, :_out)
    addrule!(g, GraphRule(out_sym, [after_sym, gated_sym], :add; namespace=ns))

    if zero_out_proj
        mha_prefix   = Symbol(prefix, :_mha)
        output_W_sym = Symbol(mha_prefix, :_output_W)
        mlp_w2_sym   = Symbol(prefix, :_mlp_w2)
        set_params!(g, ns, Dict(
            output_W_sym => zeros(Float32, dim, dim),
            mlp_w2_sym   => zeros(Float32, dim, hidden_dim),
        ))
    end

    for c in old_consumers
        rule = g.rules[ns][c]
        new_inputs = [inp == after_sym ? out_sym : inp for inp in rule.inputs]
        addrule!(g, GraphRule(c, new_inputs, rule.op; attrs=rule.attrs,
                               namespace=ns, atom_type=rule.atom_type))
    end

    return out_sym, (; alpha_sym, R_sym, out_sym, prefix)
end
