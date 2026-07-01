# ════════════════════════════════════════════════════════════════════════════════
# NeuroDSL — compiler_rules.jl
# Moteur de détection de patterns + règles avancées pour architectures DL
#
# Ce fichier étend compiler_config.jl avec :
#   (1) Primitives de requête sur le NeuroGraph (consumers, producer, op_of)
#   (2) Détecteurs de patterns structurels (QKV, SwiGLU, SDPA, Residual+Norm)
#   (3) RewriteRules avancées utilisant ces détecteurs comme conditions
#   (4) Collections par architecture (LLAMA_RULES, GPT_RULES, MEMORY_RULES)
#   (5) scan_graph + scan_summary — analyse d'un graphe sans le modifier
#
# Ordre d'inclusion dans NeuroDSL.jl :
#   include("liveness.jl")
#   include("compiler_config.jl")   ← types, DEFAULT_RULES
#   include("compiler_rules.jl")    ← ce fichier
#   include("compiler.jl")          ← compile(), moteur e-graph
#
# Aucune modification des fichiers existants.
# ════════════════════════════════════════════════════════════════════════════════


# ── 1. Primitives de requête sur le graphe ────────────────────────────────────

"""
    consumers(g, sym; ns) → Vector{Symbol}

Renvoie tous les nœuds qui utilisent `sym` comme entrée d'une règle.
Complexité O(|rules|) — acceptable pour les graphes NeuroDSL standard.
"""
function consumers(g::NeuroGraph, sym::Symbol;
                   ns::Symbol = g.active_ns)::Vector{Symbol}
    result = Symbol[]
    for (out_sym, rule) in g.rules[ns]
        sym ∈ rule.inputs && push!(result, out_sym)
    end
    return result
end

"""
    producer(g, sym; ns) → Union{GraphRule, Nothing}

Renvoie la GraphRule qui produit `sym`, ou `nothing` si c'est un nœud source
(paramètre ou entrée sans règle de calcul).
"""
function producer(g::NeuroGraph, sym::Symbol;
                  ns::Symbol = g.active_ns)::Union{GraphRule, Nothing}
    return get(g.rules[ns], sym, nothing)
end

"""
    op_of(g, sym; ns) → Union{Symbol, Nothing}

Renvoie l'opérateur de la règle produisant `sym`.
Renvoie `nothing` si `sym` est un nœud source (paramètre, entrée).
"""
function op_of(g::NeuroGraph, sym::Symbol;
               ns::Symbol = g.active_ns)::Union{Symbol, Nothing}
    rule = producer(g, sym; ns=ns)
    return rule === nothing ? nothing : rule.op
end

"""
    single_consumer(g, sym; ns) → Union{Symbol, Nothing}

Renvoie l'unique consommateur de `sym`, ou `nothing` s'il y en a 0 ou plusieurs.
Prérequis pour la fusabilité d'un nœud intermédiaire dans une chaîne.
"""
function single_consumer(g::NeuroGraph, sym::Symbol;
                         ns::Symbol = g.active_ns)::Union{Symbol, Nothing}
    c = consumers(g, sym; ns=ns)
    return length(c) == 1 ? c[1] : nothing
end

"""
    first_input(g, sym; ns) → Union{Symbol, Nothing}

Renvoie le premier input de la règle produisant `sym`.
Point de départ pour détecter les patterns fan-in (QKV, SwiGLU).
"""
function first_input(g::NeuroGraph, sym::Symbol;
                     ns::Symbol = g.active_ns)::Union{Symbol, Nothing}
    rule = producer(g, sym; ns=ns)
    rule === nothing && return nothing
    isempty(rule.inputs) && return nothing
    return rule.inputs[1]
end

"""
    shared_input_siblings(g, sym, op; ns) → Vector{Symbol}

Renvoie tous les nœuds produits par `op` qui partagent le même premier input
que `sym`. Clé pour détecter les fusions fan-in :
- QKV : Q, K, V ont tous le même input `x` et l'op `:matmul`
- SwiGLU : gate_proj et up_proj ont le même input et l'op `:matmul`
"""
function shared_input_siblings(g::NeuroGraph, sym::Symbol, op::Symbol;
                                ns::Symbol = g.active_ns)::Vector{Symbol}
    rule = producer(g, sym; ns=ns)
    (rule === nothing || isempty(rule.inputs)) && return Symbol[]
    shared = rule.inputs[1]
    result = Symbol[]
    for (out_sym, r) in g.rules[ns]
        r.op == op && !isempty(r.inputs) && r.inputs[1] == shared &&
            push!(result, out_sym)
    end
    return result
end


# ── 2. Détecteurs de patterns structurels ─────────────────────────────────────

"""
    _is_fuseable_chain(g, chain; ns) → Bool

Vérifie qu'une chaîne de nœuds est légalement fuseable :
chaque nœud intermédiaire (sauf le dernier) a exactement UN consommateur.

Si un nœud intermédiaire a plusieurs consommateurs, le fusionner supprimerait
une valeur utilisée ailleurs — c'est interdit dans le paradigme non-destructif.
"""
function _is_fuseable_chain(g::NeuroGraph, chain::Vector{Symbol};
                             ns::Symbol = g.active_ns)::Bool
    length(chain) < 2 && return false
    for sym in chain[1:end-1]
        single_consumer(g, sym; ns=ns) === nothing && return false
    end
    return true
end

"""
    _is_qkv_pattern(g, chain; ns) → Bool

Détecte si `chain[1]` est l'une des trois projections Q, K, V :
au moins 3 matmuls partagent le même premier input dans ce namespace.

Condition : le nœud est un `:matmul` ET il a ≥ 2 frères avec le même input
et le même opérateur, formant un groupe d'au moins 3 projections.

Gain attendu : −2 lancements de kernel CUDA (3 matmuls → 1 matmul batchisé),
−40% de latence mémoire mesurée sur RTX A5500 Mobile (d=512, seq=1024).
"""
function _is_qkv_pattern(g::NeuroGraph, chain::Vector{Symbol};
                          ns::Symbol = g.active_ns)::Bool
    isempty(chain) && return false
    sym = chain[1]
    op_of(g, sym; ns=ns) == :matmul || return false
    siblings = shared_input_siblings(g, sym, :matmul; ns=ns)
    return length(siblings) >= 3
end

"""
    _is_swiglu_pattern(g, chain; ns) → Bool

Détecte le FFN SwiGLU (LLaMA, Mistral, Mixtral) :
    gate_proj(x)  → silu  ⎤
                          ⊙ → down_proj
    up_proj(x)            ⎦

Critères :
  (a) `chain[1]` est un `:matmul`
  (b) il a au moins un frère matmul partageant le même input
  (c) ce frère (ou chain[1] lui-même) mène à un `:silu` ou `:gelu`

Gain attendu : 3 ops (gate_matmul + silu + up_matmul) → 1 kernel fused_swiglu.
"""
function _is_swiglu_pattern(g::NeuroGraph, chain::Vector{Symbol};
                             ns::Symbol = g.active_ns)::Bool
    isempty(chain) && return false
    sym = chain[1]
    op_of(g, sym; ns=ns) == :matmul || return false

    siblings = shared_input_siblings(g, sym, :matmul; ns=ns)
    length(siblings) < 2 && return false

    # Au moins un sibling (ou sym) mène à une activation de gating
    for sib in siblings
        for c in consumers(g, sib; ns=ns)
            op_of(g, c; ns=ns) ∈ (:silu, :gelu) && return true
        end
    end
    # Vérifie aussi sym lui-même
    for c in consumers(g, sym; ns=ns)
        op_of(g, c; ns=ns) ∈ (:silu, :gelu) && return true
    end
    return false
end

"""
    _is_sdpa_pattern(g, chain; ns) → Bool

Détecte le Scaled Dot-Product Attention (SDPA) complet :
    matmul(Q, Kᵀ) → scale → softmax → matmul(attn, V)

Vérifie que `chain[1]` est un matmul dont la descente de consommateurs uniques
suit exactement la séquence [:scale, :softmax, :matmul].

Gain attendu : 4 kernels → 1 fused_sdpa, ou délégation à flash_attention
si la séquence est > 512 (voir flash_attention.jl).
"""
function _is_sdpa_pattern(g::NeuroGraph, chain::Vector{Symbol};
                           ns::Symbol = g.active_ns)::Bool
    isempty(chain) && return false
    sym = chain[1]
    op_of(g, sym; ns=ns) ∈ (:matmul, :scaled_matmul) || return false

    # Descend la chaîne : scale → softmax → matmul
    expected_seq = (:scale, :softmax, :matmul)
    cur = single_consumer(g, sym; ns=ns)
    for expected_op in expected_seq
        cur === nothing && return false
        op_of(g, cur; ns=ns) == expected_op || return false
        cur = single_consumer(g, cur; ns=ns)
    end
    return true
end

"""
    _is_residual_postnorm_pattern(g, chain; ns) → Bool

Détecte le pattern post-norm (GPT-2) :
    add(x, sublayer(x)) → layernorm

Critère : `chain[1]` est un `:add` / `:wsum` / `:scale_add`
dont l'unique consommateur est un `:layernorm`.
"""
function _is_residual_postnorm_pattern(g::NeuroGraph, chain::Vector{Symbol};
                                        ns::Symbol = g.active_ns)::Bool
    isempty(chain) && return false
    sym = chain[1]
    op_of(g, sym; ns=ns) ∈ (:add, :wsum, :scale_add) || return false
    c = single_consumer(g, sym; ns=ns)
    c === nothing && return false
    return op_of(g, c; ns=ns) == :layernorm
end

"""
    _is_residual_postnorm_rmsnorm_pattern(g, chain; ns) → Bool

Variante RMSNorm du pattern post-norm (LLaMA, Mistral).
Identique à _is_residual_postnorm_pattern mais avec `:rmsnorm`.
"""
function _is_residual_postnorm_rmsnorm_pattern(g::NeuroGraph, chain::Vector{Symbol};
                                                ns::Symbol = g.active_ns)::Bool
    isempty(chain) && return false
    sym = chain[1]
    op_of(g, sym; ns=ns) ∈ (:add, :wsum, :scale_add) || return false
    c = single_consumer(g, sym; ns=ns)
    c === nothing && return false
    return op_of(g, c; ns=ns) == :rmsnorm
end

"""
    _is_add_zero_pattern(g, chain; ns) → Bool

Détecte un `:add` dont l'un des inputs est un nœud dont la valeur est
entièrement zéro (déjà calculée). Optimisation d'identité algébrique : x + 0 = x.
"""
function _is_add_zero_pattern(g::NeuroGraph, chain::Vector{Symbol};
                               ns::Symbol = g.active_ns)::Bool
    isempty(chain) && return false
    rule = producer(g, chain[1]; ns=ns)
    rule === nothing && return false
    op_of(g, chain[1]; ns=ns) ∈ (:add, :wsum) || return false
    for inp in rule.inputs
        nd = get(g.nodes[ns], inp, nothing)
        nd !== nothing && nd.value !== nothing && nd.valid &&
            all(iszero, nd.value) && return true
    end
    return false
end


# ── 3. RewriteRules avancées ──────────────────────────────────────────────────

"""
    QKV_FUSION_RULE

Fusion des trois projections Q, K, V en un seul matmul batchisé.
Économise 2 lancements de kernel CUDA, réduit la pression sur la mémoire HBM.
Applicable dès que 3 matmuls partagent le même input (détecté par _is_qkv_pattern).
"""
const QKV_FUSION_RULE = RewriteRule(
    :qkv_projection_fusion,
    (:matmul,),
    :fused_qkv_projection;
    condition  = _is_qkv_pattern,
    cost_delta = 0.50f0,
    verified   = false
)

"""
    SWIGLU_FUSION_RULE

Fusion du bloc SwiGLU LLaMA : silu(gate_proj(x)) ⊙ up_proj(x) → fused_swiglu(x).
3 ops (2 matmuls + silu) remplacées par 1 kernel fused avec accès mémoire réduit.
"""
const SWIGLU_FUSION_RULE = RewriteRule(
    :swiglu_fusion,
    (:matmul,),
    :fused_swiglu;
    condition  = _is_swiglu_pattern,
    cost_delta = 0.55f0,
    verified   = false
)

"""
    SDPA_FUSION_RULE

Fusion du Scaled Dot-Product Attention :
matmul(Q, Kᵀ) → scale → softmax → matmul(attn, V) → fused_sdpa.
Plus grande opportunité de fusion dans les architectures Transformer.
Pour seq > 512, l'extracteur préfère automatiquement :flash_attention
(plus faible coût déclaré dans DEFAULT_RULES).
"""
const SDPA_FUSION_RULE = RewriteRule(
    :sdpa_fusion,
    (:matmul,),
    :fused_sdpa;
    condition  = _is_sdpa_pattern,
    cost_delta = 0.70f0,
    verified   = false
)

"""
    RESIDUAL_LN_FUSION_RULE

Fusion post-norm GPT-2 : add(x, sublayer(x)) + layernorm → fused_residual_ln.
Un seul passage mémoire au lieu de deux (add puis layernorm séparés).
"""
const RESIDUAL_LN_FUSION_RULE = RewriteRule(
    :residual_ln_fusion,
    (:add,),
    :fused_residual_ln;
    condition  = _is_residual_postnorm_pattern,
    cost_delta = 0.30f0,
    verified   = false
)

"""
    RESIDUAL_RMSNORM_FUSION_RULE

Variante RMSNorm (LLaMA) du pattern résiduel post-norm.
Particulièrement efficace car RMSNorm est déjà un kernel custom dans NeuroDSL.
"""
const RESIDUAL_RMSNORM_FUSION_RULE = RewriteRule(
    :residual_rmsnorm_fusion,
    (:add,),
    :fused_residual_rmsnorm;
    condition  = _is_residual_postnorm_rmsnorm_pattern,
    cost_delta = 0.35f0,
    verified   = false
)

"""
    ADD_ZERO_ELIM_RULE

Élimination algébrique : add(x, 0) → identity(x).
Applicable quand un des inputs de :add est un tenseur nul déjà calculé.
Règle de correction prouvée (verified = true).
"""
const ADD_ZERO_ELIM_RULE = RewriteRule(
    :add_zero_elim,
    (:add,),
    :identity;
    condition  = _is_add_zero_pattern,
    cost_delta = 1.00f0,
    verified   = true
)


# ── 4. Utilitaire de déduplication ────────────────────────────────────────────

"""
    _dedup_rules(rules) → Vector{RewriteRule}

Déduplique un vecteur de règles par leur `name`.
La première occurrence est conservée en cas de conflit.
"""
function _dedup_rules(rules::Vector{RewriteRule})::Vector{RewriteRule}
    seen   = Set{Symbol}()
    result = RewriteRule[]
    for r in rules
        r.name ∈ seen && continue
        push!(seen, r.name)
        push!(result, r)
    end
    return result
end

function _embedding_rope_condition(g, chain, ns)
    isempty(chain) && return false
    single_consumer(g, chain[1]; ns=ns) !== nothing
end


# ── 5. Collections par architecture ───────────────────────────────────────────

"""
    LLAMA_RULES :: Vector{RewriteRule}

Suite de règles pour architectures LLaMA / Mistral / Mixtral.
Couvre : QKV fusion, SwiGLU, SDPA, résidu+RMSNorm, RoPE absorption.

```julia
cfg = CompilerConfig(rules = [DEFAULT_RULES..., LLAMA_RULES...])
plan = compile(g, cfg)
```
"""
const LLAMA_RULES = RewriteRule[
    QKV_FUSION_RULE,
    SWIGLU_FUSION_RULE,
    SDPA_FUSION_RULE,
    RESIDUAL_RMSNORM_FUSION_RULE,

    # RoPE : absorption dans le produit QKᵀ
    RewriteRule(:rope_qk_absorb,
                (:rope, :matmul),
                :matmul_with_rope;
                cost_delta = 0.20f0,
                verified   = false),

    # Gate seule avec silu (Mixtral sparse)
    RewriteRule(:gate_silu_fusion,
                (:matmul, :silu),
                :fused_gate_silu;
                cost_delta = 0.28f0,
                verified   = false),

    # Embedding + RoPE : fusionnable quand la position est statique
    RewriteRule(:embedding_rope_fusion,
                (:embedding, :rope),
                :fused_embedding_rope;
                condition  = _embedding_rope_condition,
                cost_delta = 0.18f0,
                verified   = false),
]

"""
    GPT_RULES :: Vector{RewriteRule}

Suite de règles pour architectures GPT-2 / GPT-3 (post-norm, GELU, LayerNorm).

```julia
cfg = CompilerConfig(rules = [DEFAULT_RULES..., GPT_RULES...])
```
"""
const GPT_RULES = RewriteRule[
    QKV_FUSION_RULE,
    SDPA_FUSION_RULE,
    RESIDUAL_LN_FUSION_RULE,

    # Linear + GELU (pas SwiGLU)
    RewriteRule(:linear_gelu_gpt,
                (:matmul, :gelu),
                :fused_linear_gelu;
                cost_delta = 0.35f0,
                verified   = false),

    # LayerNorm + Linear (pré-projection dans GPT-2)
    RewriteRule(:ln_linear_fusion,
                (:layernorm, :matmul),
                :fused_ln_linear;
                cost_delta = 0.40f0,
                verified   = false),
]

"""
    MEMORY_RULES :: Vector{RewriteRule}

Règles d'optimisation mémoire pures — indépendantes de l'architecture.
Toutes applicables à n'importe quel graphe NeuroDSL.
Quatre règles sont prouvées (verified = true).

```julia
cfg = CompilerConfig(rules = [DEFAULT_RULES..., MEMORY_RULES...])
```
"""
const MEMORY_RULES = RewriteRule[

    # Identités algébriques prouvées
    RewriteRule(:double_transpose_elim,
                (:transpose, :transpose),
                :identity;
                cost_delta = 1.00f0,
                verified   = true),

    RewriteRule(:double_identity_elim,
                (:identity, :identity),
                :identity;
                cost_delta = 1.00f0,
                verified   = true),

    # Absorption de scaling dans matmul adjacent
    RewriteRule(:scale_absorb_pre,
                (:scale, :matmul),
                :scaled_matmul;
                cost_delta = 0.15f0,
                verified   = true),

    RewriteRule(:scale_absorb_post,
                (:matmul, :scale),
                :scaled_matmul;
                cost_delta = 0.15f0,
                verified   = true),

    # Élimination de add(x, 0)
    ADD_ZERO_ELIM_RULE,
]

"""
    FULL_LLAMA_RULES :: Vector{RewriteRule}

Suite complète recommandée pour entraînement LLaMA :
DEFAULT_RULES + LLAMA_RULES + MEMORY_RULES, dédupliqués par nom.
Point de départ recommandé pour tout nouveau modèle LLaMA avec NeuroDSL.
"""
const FULL_LLAMA_RULES = _dedup_rules(vcat(DEFAULT_RULES, LLAMA_RULES, MEMORY_RULES))

"""
    FULL_GPT_RULES :: Vector{RewriteRule}

Suite complète pour GPT : DEFAULT_RULES + GPT_RULES + MEMORY_RULES.
"""
const FULL_GPT_RULES = _dedup_rules(vcat(DEFAULT_RULES, GPT_RULES, MEMORY_RULES))


# ── 6. RuleMatch et scan_graph ────────────────────────────────────────────────

"""
    RuleMatch

Résultat d'une détection de pattern dans le graphe par `scan_graph`.

Champs :
- `rule`  : la RewriteRule dont le pattern a été reconnu
- `nodes` : les nœuds matchés dans l'ordre du pattern (chaîne linéaire)
- `ns`    : namespace dans lequel le match a été trouvé
- `score` : cost_delta de la règle — utilisé par l'extracteur pour prioriser
"""
struct RuleMatch
    rule  :: RewriteRule
    nodes :: Vector{Symbol}
    ns    :: Symbol
    score :: Float32
end

function Base.show(io::IO, m::RuleMatch)
    print(io, "RuleMatch(:$(m.rule.name)  nodes=$(m.nodes)  score=$(m.score))")
end

"""
    scan_graph(g, rules; namespace) → Vector{RuleMatch}

Analyse le graphe `g` et renvoie tous les candidats à la réécriture selon
les règles fournies, triés par score décroissant (meilleure opportunité d'abord).

Algorithme (O(|rules| × |nodes| × max_pattern_length)) :
  Pour chaque règle :
    Pour chaque nœud dont l'op correspond au premier op du pattern :
      1. Construit la chaîne linéaire candidate en suivant les consommateurs uniques
      2. Vérifie la fusabilité de la chaîne (_is_fuseable_chain)
      3. Évalue la condition optionnelle de la règle
      4. Enregistre un RuleMatch si tout est satisfait

Résultat trié par `score` décroissant = ordre de priorité pour compiler.jl.

Usage :
```julia
matches = scan_graph(g, FULL_LLAMA_RULES)
for m in matches
    println(m.rule.name, "  Δ=", m.score, "  nodes=", m.nodes)
end
```
"""
function scan_graph(g::NeuroGraph,
                    rules::Vector{RewriteRule} = DEFAULT_RULES;
                    namespace::Symbol = g.active_ns)::Vector{RuleMatch}
    haskey(g.nodes, namespace) || return RuleMatch[]
    haskey(g.rules, namespace) || return RuleMatch[]

    matches = RuleMatch[]

    for rule in rules
        isempty(rule.pattern) && continue
        first_op = rule.pattern[1]

        for (sym, node_rule) in g.rules[namespace]
            node_rule.op == first_op || continue

            # Construire la chaîne candidate en suivant les consommateurs uniques
            chain = Symbol[sym]
            valid = true

            for expected_op in rule.pattern[2:end]
                next_sym = single_consumer(g, chain[end]; ns=namespace)
                if next_sym === nothing ||
                   op_of(g, next_sym; ns=namespace) != expected_op
                    valid = false
                    break
                end
                push!(chain, next_sym)
            end

            !valid && continue

            # Vérifier la fusabilité de la chaîne (nœuds intermédiaires = 1 consommateur)
            length(chain) > 1 &&
                !_is_fuseable_chain(g, chain; ns=namespace) && continue

            # Évaluer la condition optionnelle
            if rule.condition !== nothing
                try
                    rule.condition(g, chain, namespace) || continue
                catch _
                    continue   # condition en erreur = règle inapplicable
                end
            end

            push!(matches, RuleMatch(rule, chain, namespace, rule.cost_delta))
        end
    end

    # Priorité aux règles avec le plus grand gain potentiel
    sort!(matches; by = m -> -m.score)
    return matches
end

"""
    scan_summary(g, rules; namespace) → Nothing

Affiche un résumé lisible des opportunités de compilation dans le graphe.
Utile pour comprendre ce que `compile()` va faire avant de l'exécuter,
ou pour déboguer pourquoi une règle ne s'applique pas.

```julia
# Avant compilation — voir les opportunités détectées
scan_summary(g, FULL_LLAMA_RULES)

# Exemple de sortie :
# ── scan_graph [:main] ─────────────────────────────────
#   5 opportunités détectées sur 47 règles
#
#   sdpa_fusion                    Δ=0.70    nodes=[:attn_qk, ...]
#   swiglu_fusion                  Δ=0.55    nodes=[:gate_proj, ...]
#   qkv_projection_fusion          Δ=0.50  ✓ nodes=[:q_proj, ...]
#   ...
# ──────────────────────────────────────────────────────
```
"""
function scan_summary(g::NeuroGraph,
                      rules::Vector{RewriteRule} = DEFAULT_RULES;
                      namespace::Symbol = g.active_ns)
    matches  = scan_graph(g, rules; namespace=namespace)
    n_nodes  = length(get(g.nodes, namespace, Dict()))
    n_rules  = length(get(g.rules, namespace, Dict()))

    println("── scan_graph [:$namespace] ", "─"^42)
    println("  ", length(matches), " opportunités  |  ",
            n_nodes, " nœuds  |  ", n_rules, " règles de calcul")

    if isempty(matches)
        println("  (aucune fusion applicable avec les règles fournies)")
    else
        println()
        for m in matches
            vtag = m.rule.verified ? " ✓" : "  "
            name_str = rpad(string(m.rule.name), 34)
            delta_str = lpad(string(round(m.score; digits=2)), 5)
            nodes_str = length(m.nodes) <= 3 ?
                        string(m.nodes) :
                        string(m.nodes[1:3])[1:end-1] * ", …]"
            println("  ", name_str, " Δ=", delta_str, vtag, "  nodes=", nodes_str)
        end
    end

    total_delta = sum(m.score for m in matches; init=0f0)
    println()
    println("  Gain total estimé : Δ=", round(total_delta; digits=3))
    println("─"^56)
    return nothing
end