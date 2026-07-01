# ════════════════════════════════════════════════════════════════════════════════
# NeuroDSL — compiler_config.jl
# Types de configuration, règles de réécriture et résultat du compilateur
#
# Design philosophy :
#   • Additif : zéro modification des fichiers existants (neuro_graph.jl,
#     dsl_macros.jl, liveness.jl, checkpoint.jl)
#   • Zéro dépendance externe : Metatheory.jl est optionnel (injecté par
#     compiler.jl si disponible), ce fichier ne le référence pas
#   • Non-destructif : RewriteRule enregistre des équivalences, elle ne supprime
#     jamais de nœuds — contrairement à _fuse! + FUSION_TABLE
#   • Réactif : CompiledPlan écoute le NeuroGraph via on_change et accumule
#     les dirty_nodes pour déclencher une re-saturation incrémentale ciblée
#
# Ordre d'inclusion recommandé dans NeuroDSL.jl :
#   include("liveness.jl")           # MemoryPlan, BufferPool
#   include("compiler_config.jl")    # ce fichier
#   include("compiler_rules.jl")     # règles métier avancées
#   include("compiler.jl")           # compile(), _saturate!, _extract_plan
# ════════════════════════════════════════════════════════════════════════════════


# ── 1. RewriteRule ────────────────────────────────────────────────────────────

"""
    RewriteRule

Règle de réécriture sémantique NON-DESTRUCTIVE pour l'e-graph NeuroDSL.

Contrairement à `FUSION_TABLE + _fuse!` (qui supprime physiquement les nœuds
intermédiaires), une `RewriteRule` déclare simplement que deux sous-graphes
sont *sémantiquement équivalents*. L'e-graph conserve les deux formes ; c'est
la fonction de coût qui choisit laquelle exécuter à l'extraction.

Champs
──────
- `name`       : identifiant unique de la règle  (ex. `:matmul_relu_fusion`)
- `pattern`    : séquence ordonnée d'opérateurs à reconnaître dans le graphe
                 (ex. `(:matmul, :relu)`)
- `result`     : opérateur fusionné qui remplace la chaîne dans le plan extrait
                 (ex. `:fused_matmul_relu`)
- `condition`  : garde optionnelle `(g, chain, ns) -> Bool` — évaluée avant
                 d'ajouter l'équivalence à l'e-graph ; `nothing` = toujours vraie
- `cost_delta` : réduction de coût estimée (unités normalisées, valeur > 0 = gain)
                 utilisée par `default_cost` pour comparer les classes d'équivalence
- `verified`   : `true` si la règle est accompagnée d'une preuve symbolique
                 (IntervalArithmetic.jl ou Symbolics.jl) de préservation numérique

Exemple
───────
```julia
# Fusion standard linear + activation
linear_gelu = RewriteRule(
    :linear_gelu_fusion,
    (:matmul, :gelu),
    :fused_linear_gelu;
    cost_delta = 0.35f0,
    verified   = false
)

# Règle conditionnelle : flash attention seulement si séquence > 512
flash_rule = RewriteRule(
    :flash_attention,
    (:softmax_attn,),
    :flash_attention;
    condition  = (g, chain, ns) -> begin
        nd = get(g.nodes[ns], chain[1], nothing)
        nd !== nothing && nd.value !== nothing && size(nd.value, 1) > 512
    end,
    cost_delta = 0.6f0,
    verified   = true
)
```
"""
struct RewriteRule
    name       :: Symbol
    pattern    :: Tuple{Vararg{Symbol}}
    result     :: Symbol
    condition  :: Union{Nothing, Function}   # (g, chain::Vector{Symbol}, ns) -> Bool
    cost_delta :: Float32
    verified   :: Bool
end

"""
    RewriteRule(name, pattern, result; condition, cost_delta, verified)

Constructeur avec valeurs par défaut — `condition = nothing`, `cost_delta = 1f0`,
`verified = false`.
"""
function RewriteRule(name::Symbol,
                     pattern::Tuple{Vararg{Symbol}},
                     result::Symbol;
                     condition  :: Union{Nothing, Function} = nothing,
                     cost_delta :: Real                     = 1.0f0,
                     verified   :: Bool                     = false)
    return RewriteRule(name, pattern, result, condition, Float32(cost_delta), verified)
end

function Base.show(io::IO, r::RewriteRule)
    verified_tag = r.verified ? " ✓" : ""
    cond_tag     = r.condition !== nothing ? " [conditionnel]" : ""
    pattern_str  = join(r.pattern, ", ")
    print(io, "RewriteRule(:$(r.name), ($(pattern_str)) → :$(r.result), ",
          "Δcost=$(r.cost_delta)$(verified_tag)$(cond_tag))")
end

function _flash_condition(g, chain, ns)
    nd = get(g.nodes[ns], chain[1], nothing)
    return nd !== nothing &&
           nd.value !== nothing &&
           ndims(nd.value) >= 2 &&
           size(nd.value, 1) > 512
end


# ── 2. Règles de réécriture par défaut ────────────────────────────────────────

"""
    DEFAULT_RULES :: Vector{RewriteRule}

Ensemble de règles sémantiques incluses dans toute compilation NeuroDSL.
Ces règles migrent `FUSION_TABLE` vers le paradigme non-destructif de l'e-graph.

`FUSION_TABLE` reste intacte pour la compatibilité ascendante avec `_fuse!` ;
le compilateur opère en parallèle et de façon non-destructive.

Règles incluses
───────────────
- Fusions classiques      : matmul+relu, relu+matmul
- Fusions Transformer     : matmul+gelu, layernorm+matmul, matmul+softmax
- Fusions mémoire GPU     : attention naïve → flash attention (conditionnel)
- Identités algébriques   : élimination de double-transposition,
                             absorption de scaling dans matmul

Chaque règle est accompagnée d'un `cost_delta` calibré sur des benchmarks
NeuroDSL vs Flux.jl (voir article de recherche, section 4.2).
"""
const DEFAULT_RULES = RewriteRule[

    # ── Fusions activations ────────────────────────────────────────────────

    RewriteRule(:matmul_relu_fusion,
                (:matmul, :relu),
                :fused_matmul_relu;
                cost_delta = 0.30f0,
                verified   = false),

    RewriteRule(:relu_matmul_fusion,
                (:relu, :matmul),
                :fused_relu_matmul;
                cost_delta = 0.20f0,
                verified   = false),

    RewriteRule(:linear_gelu_fusion,
                (:matmul, :gelu),
                :fused_linear_gelu;
                cost_delta = 0.35f0,
                verified   = false),

    RewriteRule(:matmul_sigmoid_fusion,
                (:matmul, :sigmoid),
                :fused_matmul_sigmoid;
                cost_delta = 0.25f0,
                verified   = false),

    # ── Fusions normalisation + projection ────────────────────────────────

    RewriteRule(:layernorm_linear_fusion,
                (:layernorm, :matmul),
                :fused_layernorm_linear;
                cost_delta = 0.40f0,
                verified   = false),

    RewriteRule(:rmsnorm_linear_fusion,
                (:rmsnorm, :matmul),
                :fused_rmsnorm_linear;
                cost_delta = 0.45f0,       # RMSNorm déjà kernel custom → gain élevé
                verified   = false),

    # ── Flash Attention (conditionnel sur la taille de séquence) ──────────

    RewriteRule(:flash_attention,
            (:softmax_attn,),
            :flash_attention;
            condition  = _flash_condition,
            cost_delta = 0.60f0,
            verified   = true),

    # ── Identités algébriques ──────────────────────────────────────────────

    RewriteRule(:double_transpose_elim,
                (:transpose, :transpose),
                :identity;
                cost_delta = 1.00f0,       # coût nul → gain maximal
                verified   = true),

    RewriteRule(:scale_absorb_matmul,
                (:scale, :matmul),
                :scaled_matmul;
                cost_delta = 0.15f0,
                verified   = true),

    RewriteRule(:matmul_scale_absorb,
                (:matmul, :scale),
                :scaled_matmul;
                cost_delta = 0.15f0,
                verified   = true),
]


# ── 3. Fonction de coût par défaut ────────────────────────────────────────────

"""
    symbolic_cost(op, input_shapes) → Float32

Estime le coût d'exécution d'une opération à partir de la forme de ses tenseurs
d'entrée. Mode **symbolique** : aucune mesure réelle, pas d'exécution GPU.

Convention de coût
──────────────────
- matmul (M, K) × (K, N)  → M × K × N / 1e9  (en GFLOP)
- ops élémentaires         → prod(shape) / 1e8
- ops de structure légères → 0.0  (transpose, identity, scale seul)
- ops custom NeuroDSL      → heuristique conservative (prod(shape) / 5e7)

Ce calcul est intentionnellement conservateur. Le mode `:profiled`
(activable via `CompilerConfig(cost_mode=:profiled)`) remplace ces estimations
par des mesures CUDA.jl réelles pour les nœuds ambigus.
"""
function symbolic_cost(op::Symbol, input_shapes::Vector{<:Tuple})::Float32
    isempty(input_shapes) && return 0.0f0

    if op ∈ (:matmul, :fused_matmul_relu, :fused_linear_gelu,
             :fused_matmul_sigmoid, :scaled_matmul,
             :fused_rmsnorm_linear, :fused_layernorm_linear)
        # Coût quadratique-cubique selon les dimensions
        length(input_shapes) >= 2 || return 0.0f0
        A_shape, B_shape = input_shapes[1], input_shapes[2]
        M = length(A_shape) >= 1 ? A_shape[1] : 1
        K = length(A_shape) >= 2 ? A_shape[2] : 1
        N = length(B_shape) >= 2 ? B_shape[2] : (length(B_shape) >= 1 ? B_shape[1] : 1)
        return Float32(M * K * N) / 1f9

    elseif op ∈ (:relu, :gelu, :sigmoid, :tanh, :softmax, :scale,
                 :add, :wsum, :nsum, :scale_add)
        # Ops élémentaires : proportionnel au volume
        return Float32(prod(input_shapes[1])) / 1f8

    elseif op ∈ (:layernorm, :rmsnorm)
        # Normalisation : légèrement plus coûteuse qu'élémentaire
        return Float32(prod(input_shapes[1])) / 5f7

    elseif op ∈ (:softmax_attn, :flash_attention)
        # Attention : coût quadratique en séquence
        s = input_shapes[1]
        seq = length(s) >= 1 ? s[1] : 1
        d   = length(s) >= 2 ? s[2] : 1
        return Float32(seq * seq * d) / 1f9

    elseif op ∈ (:transpose, :identity, :double_transpose_elim)
        return 0.0f0                             # restructuration pure, pas de calcul

    else
        # Heuristique conservative pour ops custom
        return Float32(prod(input_shapes[1])) / 5f7
    end
end

"""
    default_cost(op, input_shapes; rule_cost_delta) → Float32

Fonction de coût finale utilisée par l'extracteur de l'e-graph.
Combine le coût symbolique de base et le delta apporté par la règle appliquée.

Un coût plus bas = préféré par l'extracteur.
`rule_cost_delta` représente une réduction : cost_final = base - delta.
"""
function default_cost(op::Symbol,
                      input_shapes::Vector{<:Tuple};
                      rule_cost_delta::Float32 = 0.0f0)::Float32
    base = symbolic_cost(op, input_shapes)
    return max(0.0f0, base - rule_cost_delta)
end


# ── 4. CompilerConfig ─────────────────────────────────────────────────────────

"""
    CompilerConfig(; ...)

Configuration complète du compilateur NeuroDSL.
Passer à `compile(g, config)` pour contrôler le comportement de la compilation.

Champs
──────
- `rules`       : règles de réécriture appliquées pendant la saturation e-graph.
                  Défaut : `DEFAULT_RULES` (fusions standards + flash attention).
                  Extensible : `CompilerConfig(rules = [DEFAULT_RULES..., ma_regle])`

- `cost_fn`     : fonction de coût symbolique `(op, shapes) -> Float32`.
                  Permet de brancher un modèle de coût matériel spécifique (A5500,
                  3060 Ti) en remplaçant `default_cost`.

- `budget`      : nombre maximum d'itérations de saturation e-graph.
                  Borne le coût de compilation sur les grands graphes.
                  1 000 suffit pour les architectures Transformer standard.

- `target`      : backend d'exécution visé — influe sur le choix des règles
                  conditionnelles (flash attention requiert `:gpu`).

- `incremental` : si `true`, une modification partielle du NeuroGraph ne
                  recompile que la région invalidée (via `dirty_nodes`).
                  Désactiver pour forcer une recompilation complète.

- `inspect`     : si `true`, `CompiledPlan.egraph_cache` conserve l'e-graph
                  final pour analyse. Coûteux en mémoire sur les grands modèles.

- `cost_mode`   : `:symbolic` (rapide, par défaut) ou `:profiled` (mesure
                  CUDA.jl réelle pour les nœuds ambigus — à utiliser avec
                  `CompilerConfig(cost_mode=:profiled)` post-convergence).

- `verify_rules`: si `true`, les règles avec `verified=false` émettent un warning.
                  Activer pour un pipeline de recherche rigoureux.

- `training`    : si `true` (défaut), `compile()` refuse d'appliquer toute `RewriteRule` dont
                  le `result` n'a pas d'entrée dans `GRAD_RULES` — filet de sécurité qui empêche
                  `backward_graph!` de planter sur un op fusionné sans règle de gradient. Passer
                  `training=false` pour autoriser aussi les fusions inference-only (aucune passe
                  backward ne sera possible sur le graphe fusionné après coup).

Exemples
────────
```julia
# Configuration minimale (GPU, règles par défaut)
cfg = CompilerConfig()

# Recherche : inspection de l'e-graph + vérification des règles
cfg_research = CompilerConfig(inspect=true, verify_rules=true, budget=5000)

# Production CPU : pas d'incremental, règles légères seulement
cpu_rules = filter(r -> r.result != :flash_attention, DEFAULT_RULES)
cfg_cpu = CompilerConfig(rules=cpu_rules, target=:cpu, incremental=false)

# Règle custom ajoutée au-dessus des defaults
my_rule = RewriteRule(:my_fusion, (:conv, :bn), :fused_conv_bn; cost_delta=0.5f0)
cfg_custom = CompilerConfig(rules=[DEFAULT_RULES..., my_rule])
```
"""
@kwdef struct CompilerConfig
    rules        :: Vector{RewriteRule} = DEFAULT_RULES
    cost_fn      :: Function            = default_cost
    budget       :: Int                 = 1_000
    target       :: Symbol              = :gpu
    incremental  :: Bool                = true
    inspect      :: Bool                = false
    cost_mode    :: Symbol              = :symbolic
    verify_rules :: Bool                = false
    training     :: Bool                = true
end

function Base.show(io::IO, cfg::CompilerConfig)
    println(io, "CompilerConfig:")
    println(io, "  rules       : $(length(cfg.rules)) règles",
                cfg.inspect ? " [inspect=on]" : "")
    println(io, "  budget      : $(cfg.budget) itérations")
    println(io, "  target      : :$(cfg.target)")
    println(io, "  cost_mode   : :$(cfg.cost_mode)")
    print(io,   "  incremental : $(cfg.incremental)")
    cfg.verify_rules && print(io, "  [verify_rules=on]")
end


# ── 5. CompiledPlan ───────────────────────────────────────────────────────────

"""
    CompiledPlan

Résultat d'un appel à `compile(g, config)`. Objet callable.

Usage
─────
```julia
plan = compile(g)                            # compile le graphe
loss_val = plan(g, :loss)                    # exécution optimisée
loss_val = plan(g, :loss; ctx=mon_ctxstore)  # avec contexte externe
```

Réactivité
──────────
`CompiledPlan` s'enregistre automatiquement sur les callbacks `on_change` du
NeuroGraph. Quand un nœud est modifié (via `set!`), son symbole est ajouté à
`dirty_nodes`. Au prochain appel, seule la région affectée est re-saturée —
pas le graphe entier.

Ce comportement est transparent pour l'utilisateur : `plan(g, :loss)` reste
la même API avant et après une modification du graphe.

Champs (lecture seule après construction)
─────────────────────────────────────────
- `namespace`     : espace de noms ciblé dans le NeuroGraph
- `exec_order`    : ordre topologique post-fusion (optimisé)
- `fused_ops`     : nœud original → opérateur fusionné choisi par l'extracteur
- `memory_plan`   : MemoryPlan de liveness.jl (slots de buffers)
- `pool`          : BufferPool associé au plan
- `dirty_nodes`   : nœuds invalidés depuis la dernière (re-)compilation
- `egraph_cache`  : e-graph persistant (peuplé si `config.inspect=true` ou
                    si `config.incremental=true`)
- `config`        : CompilerConfig utilisée
- `n_recompiles`  : compteur de re-saturations incrémentales
- `compiled_at`   : timestamp `time()` de la dernière compilation complète
"""
mutable struct CompiledPlan
    namespace    :: Symbol
    exec_order   :: Vector{Symbol}
    fused_ops    :: Dict{Symbol, Symbol}
    memory_plan  :: MemoryPlan
    pool         :: BufferPool
    dirty_nodes  :: Set{Symbol}
    egraph_cache :: Any                      # ::EGraph dans compiler.jl (optionnel)
    config       :: CompilerConfig
    n_recompiles :: Int
    compiled_at  :: Float64                  # time()
end

"""
    CompiledPlan(namespace, exec_order, fused_ops, memory_plan, pool, config)

Constructeur interne utilisé par `compile()`. Les champs mutables (`dirty_nodes`,
`egraph_cache`, `n_recompiles`, `compiled_at`) sont initialisés à des valeurs
vides/par défaut.
"""
function CompiledPlan(namespace    :: Symbol,
                      exec_order   :: Vector{Symbol},
                      fused_ops    :: Dict{Symbol, Symbol},
                      memory_plan  :: MemoryPlan,
                      pool         :: BufferPool,
                      config       :: CompilerConfig)
    return CompiledPlan(
        namespace,
        exec_order,
        fused_ops,
        memory_plan,
        pool,
        Set{Symbol}(),   # dirty_nodes vide à la construction
        nothing,          # egraph_cache — peuplé par compiler.jl si inspect=true
        config,
        0,                # n_recompiles
        time()            # compiled_at
    )
end

function Base.show(io::IO, plan::CompiledPlan)
    age_s    = round(time() - plan.compiled_at; digits=1)
    n_fused  = count(v -> v != :identity, values(plan.fused_ops))
    n_dirty  = length(plan.dirty_nodes)

    println(io, "╔══════════════════════════════════════╗")
    println(io, "║     NeuroDSL — CompiledPlan          ║")
    println(io, "╚══════════════════════════════════════╝")
    println(io, "  Namespace   : :$(plan.namespace)")
    println(io, "  Nœuds       : $(length(plan.exec_order))")
    println(io, "  Fusions     : $n_fused  ($(length(plan.exec_order)) → opérateurs fusionnés)")
    println(io, "  Mémoire     : $(plan.memory_plan.n_slots) slots  ",
                "(pic réduit vs naïf)")
    println(io, "  Recompiles  : $(plan.n_recompiles) incrémentales")
    println(io, "  Compilé il y a : $(age_s)s")
    n_dirty > 0 && println(io, "  ⚠ Dirty nodes : $n_dirty  (re-saturation en attente)")
    plan.egraph_cache !== nothing && println(io, "  E-graph     : conservé (inspect=true)")
end


# ── 6. Utilitaires ────────────────────────────────────────────────────────────

"""
    is_dirty(plan) → Bool
Renvoie `true` si le plan a des nœuds invalidés en attente de re-saturation.
"""
is_dirty(plan::CompiledPlan) = !isempty(plan.dirty_nodes)

"""
    n_rules_verified(config) → Int
Nombre de règles vérifiées symboliquement dans la configuration.
"""
n_rules_verified(cfg::CompilerConfig) = count(r -> r.verified, cfg.rules)

"""
    rules_for_target(config) → Vector{RewriteRule}
Filtre les règles applicables au backend cible (ex. flash_attention exclue sur CPU).
"""
function rules_for_target(cfg::CompilerConfig)::Vector{RewriteRule}
    cfg.target == :gpu && return cfg.rules
    # CPU : exclure les règles GPU-spécifiques
    return filter(r -> r.result ∉ (:flash_attention, :fused_rmsnorm_linear), cfg.rules)
end

"""
    _has_backward_support(op) → Bool
Renvoie `true` si `op` a une entrée dans `GRAD_RULES` — utilisé comme garde de sécurité par
`compile()` : en mode `training=true` (défaut), une fusion vers un op sans règle de gradient
est refusée plutôt que de laisser `backward_graph!` planter plus tard.
"""
_has_backward_support(op::Symbol)::Bool = haskey(GRAD_RULES, op)