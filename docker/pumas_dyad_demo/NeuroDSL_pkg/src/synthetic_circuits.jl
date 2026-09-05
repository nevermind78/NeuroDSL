# ══════════════════════════════════════════════════════════════════════════════
# synthetic_circuits.jl — Circuits synthétiques pour valider le patching à plus
# grande échelle, sans checkpoint externe (décision utilisateur confirmée).
#
# Nom distinct de `checkpoint.jl` (recalcul d'activations en RAM, sans rapport).
#
# Deux constructions, complémentaires :
#   1. `selection_circuit_weights`/`build_selection_circuit` -- circuit construit
#      À LA MAIN dans une seule tête d'attention, dont l'effet a une prédiction
#      EXACTE en forme close (pas d'apprentissage, pas de recherche). Sert de
#      vérité terrain pour tester le patching, indépendamment de tout
#      entraînement.
#   2. `build_induction_graph`/`train_induction!`/`evaluate_induction` --
#      adapté de `notebook/induction.ipynb`, paramétré pour une échelle plus
#      grande, pour que `greedy_patch_search!`/`backward_prune!` (inchangés)
#      retrouvent un VRAI circuit d'induction appris, pas un circuit imposé.
# ══════════════════════════════════════════════════════════════════════════════

"""
    selection_circuit_weights(dim, n_heads, target_head, seq_len, query_pos, key_pos;
                               qk_scale=8f0, v_target=nothing)
    -> (weights::Dict{Symbol,Array{Float32}}, v_target::Vector{Float32})

Construit des poids Q/K/V/sortie pour `MultiHeadAttention(dim, n_heads)` tels que,
à la position de requête `query_pos`, la tête `target_head` porte quasiment toute
son attention (à la précision Float32) sur la position clé `key_pos` (causal :
`key_pos < query_pos`), et copie un vecteur `v_target` connu vers cette position
de sortie -- une prédiction en forme close, pas un résultat mesuré après coup.

Construction (`:input` doit être une matrice one-hot par position, ligne `i` =
`e_i`, ce que `build_selection_circuit` fournit) :
  - `q_W[s:e, query_pos] = q0`, `k_W[s:e, key_pos] = q0` (mêmes colonnes,
    colonnes 0 ailleurs) où `s:e` = plage de lignes de la tête cible.
    Donne `score[query_pos, key_pos] = |q0|² = qk_scale²`,
    `score[query_pos, i] = 0` pour tout autre `i` visible (masque causal).
  - `v_W[s:e, key_pos] = v_target`, colonnes 0 ailleurs -- la seule position
    dont la valeur-V est non nulle est `key_pos`, donc la sortie de la tête en
    `query_pos` vaut EXACTEMENT `pr[query_pos,key_pos] * v_target` (les autres
    positions contribuent un poids quelconque × 0 = 0, littéralement, pas
    approximativement).
  - `output_W = I` (identité) : la projection de sortie ne mélange jamais les
    colonnes entre têtes, donc la tranche de colonnes de la tête cible traverse
    la couche de sortie inchangée -- la fermeture algébrique reste exacte même
    si les autres têtes gardent leurs poids aléatoires d'origine.

`pr[query_pos,key_pos]` (calculable en forme close par la formule softmax
standard, voir `test/test_pretrained_scale.jl`) tend vers 1.0 mais n'est jamais
exactement 1.0 -- c'est la valeur exacte comparée dans le test, pas "recovery
élevée".
"""
function selection_circuit_weights(dim::Int, n_heads::Int, target_head::Int,
                                    seq_len::Int, query_pos::Int, key_pos::Int;
                                    qk_scale::Float32=8f0,
                                    v_target::Union{Nothing,Vector{Float32}}=nothing)
    dim % n_heads == 0 || error("❌ selection_circuit_weights : dim doit être divisible par n_heads")
    d_head = dim ÷ n_heads
    1 <= target_head <= n_heads || error("❌ selection_circuit_weights : target_head hors bornes")
    seq_len <= dim || error("❌ selection_circuit_weights : nécessite dim >= seq_len (encodage one-hot)")
    (1 <= key_pos < query_pos <= seq_len) ||
        error("❌ selection_circuit_weights : il faut 1 <= key_pos < query_pos <= seq_len (causal)")

    vt = v_target === nothing ? Float32.(collect(1:d_head)) : v_target
    length(vt) == d_head || error("❌ selection_circuit_weights : v_target doit être de longueur d_head=$d_head")

    q_W = zeros(Float32, dim, dim)
    k_W = zeros(Float32, dim, dim)
    v_W = zeros(Float32, dim, dim)
    o_W = Matrix{Float32}(LinearAlgebra.I, dim, dim)

    s = (target_head - 1) * d_head + 1
    e = target_head * d_head
    q0 = zeros(Float32, d_head); q0[1] = qk_scale

    q_W[s:e, query_pos] .= q0
    k_W[s:e, key_pos]   .= q0
    v_W[s:e, key_pos]   .= vt

    weights = Dict{Symbol,Array{Float32}}(:q_W => q_W, :k_W => k_W, :v_W => v_W, :output_W => o_W)
    return weights, vt
end

"""
    build_selection_circuit(dev, dim, n_heads, seq_len, target_head, query_pos, key_pos;
                             qk_scale=8f0, v_target=nothing, namespace=:selection_circuit)
    -> (g, out_sym, mha_prefix, v_target)

Construit un `MultiHeadAttention(dim, n_heads)` sur une entrée one-hot par
position (`:input[i,:] = e_i`), puis injecte les poids de
`selection_circuit_weights` via `set_params!`.
"""
function build_selection_circuit(dev, dim::Int, n_heads::Int, seq_len::Int,
                                  target_head::Int, query_pos::Int, key_pos::Int;
                                  qk_scale::Float32=8f0,
                                  v_target::Union{Nothing,Vector{Float32}}=nothing,
                                  namespace::Symbol=:selection_circuit)
    ns = namespace
    g = NeuroGraph(namespace=ns, device=dev)
    X = zeros(Float32, seq_len, dim)
    for i in 1:seq_len
        X[i, i] = 1f0
    end
    set!(g, :input, X; namespace=ns)
    prefix = :circuit_mha
    out_sym = MultiHeadAttention(dim, n_heads)(g, :input, prefix; namespace=ns)

    weights, vt = selection_circuit_weights(dim, n_heads, target_head, seq_len, query_pos, key_pos;
                                             qk_scale=qk_scale, v_target=v_target)
    set_params!(g, ns, Dict(
        Symbol(prefix, :_q_W)      => weights[:q_W],
        Symbol(prefix, :_k_W)      => weights[:k_W],
        Symbol(prefix, :_v_W)      => weights[:v_W],
        Symbol(prefix, :_output_W) => weights[:output_W],
    ))
    return g, out_sym, prefix, vt
end

# ══════════════════════════════════════════════════════════════════════════════
# Tâche d'induction (Olsson et al. 2022) -- adapté de notebook/induction.ipynb,
# paramétré pour une échelle plus grande que le notebook (dim=64/3 couches).
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_induction_graph(dev, ns; vocab_size, dim, n_heads, hidden_dim, n_layers, prefix_len)
    -> (g, logits_sym)

Embedding(token)+Embedding(position) -> LlamaModel -> Linear -> :cross_entropy.
`seq_len = 2*prefix_len` (préfixe répété deux fois, protocole d'induction).
"""
function build_induction_graph(dev, ns::Symbol; vocab_size::Int, dim::Int, n_heads::Int,
                                hidden_dim::Int, n_layers::Int, prefix_len::Int)
    seq_len = 2 * prefix_len
    g = NeuroGraph(namespace=ns, device=dev)
    set!(g, :token_ids, ones(Int, seq_len); atom_type=Datom, namespace=ns)
    set!(g, :pos_ids, collect(1:seq_len); atom_type=Datom, namespace=ns)
    tok_emb = Embedding(vocab_size, dim)(g, :token_ids, :tok; namespace=ns)
    pos_emb = Embedding(seq_len, dim)(g, :pos_ids, :pos; namespace=ns)
    x = :embed_sum
    addrule!(g, GraphRule(x, [tok_emb, pos_emb], :add; namespace=ns))
    out = LlamaModel(n_layers, dim, n_heads, hidden_dim)(g, x; namespace=ns)
    logits = Linear(dim, vocab_size)(g, out, :lm_head; namespace=ns)
    set!(g, :labels, ones(Int, seq_len); atom_type=Datom, namespace=ns)
    addrule!(g, GraphRule(:loss, [logits, :labels], :cross_entropy; namespace=ns))
    return g, logits
end

"""
    sample_induction_sequence(rng, vocab_size, prefix_len) -> (tokens, labels)

Préfixe aléatoire répété deux fois ; seule la seconde moitié est résoluble par
induction (retrouver la dernière occurrence du token courant, copier ce qui le
suivait). Nouveau préfixe à chaque appel -- l'entraînement doit apprendre
l'algorithme, pas mémoriser un jeu fixe.
"""
function sample_induction_sequence(rng, vocab_size::Int, prefix_len::Int)
    prefix = rand(rng, 1:vocab_size, prefix_len)
    tokens = vcat(prefix, prefix)
    labels = vcat(tokens[2:end], tokens[1])
    return tokens, labels
end

"""
    train_induction!(g, ns; vocab_size, prefix_len, n_steps, seed=123, lr=3f-3, ...) -> losses

Boucle AdamW pas-à-pas (patron déjà établi ailleurs dans le dépôt : `demand!` ->
`backward_graph!` -> `adamw_step!` par paramètre -> `invalidate_all!`), un
nouveau préfixe aléatoire à chaque pas.
"""
function train_induction!(g::NeuroGraph, ns::Symbol; vocab_size::Int, prefix_len::Int, n_steps::Int,
                           seed::Int=123, lr::Float32=3f-3, b1::Float32=0.9f0, b2::Float32=0.999f0,
                           eps_v::Float32=1f-8, clip::Float32=1f0, wd::Float32=0f0)
    dev = g.device
    ps = params(g; namespace=ns)
    m1s = [Backend.zeros32(dev, size(p.value)...) for p in ps]
    m2s = [Backend.zeros32(dev, size(p.value)...) for p in ps]
    rng = MersenneTwister(seed)
    losses = Float64[]
    for t in 1:n_steps
        tokens, labels = sample_induction_sequence(rng, vocab_size, prefix_len)
        set!(g, :token_ids, tokens; atom_type=Datom, namespace=ns)
        set!(g, :labels, labels; atom_type=Datom, namespace=ns)
        invalidate_all!(g; namespace=ns)
        loss_val = demand!(g, :loss; namespace=ns)
        push!(losses, Float64(sum(Array(loss_val))))
        backward_graph!(g, :loss; namespace=ns)
        for (i, p) in enumerate(ps)
            adamw_step!(dev, p.value, p.gradient, m1s[i], m2s[i], lr, b1, b2, eps_v, t, clip, wd)
        end
        invalidate_all!(g; namespace=ns)
    end
    return losses
end

"""
    evaluate_induction(g, logits, ns; vocab_size, prefix_len, n_eval=100, seed=999)
    -> (; first_half_acc, second_half_acc)

Précision sur des séquences FRAÎCHES (jamais vues à l'entraînement) : la
seconde moitié doit être quasi parfaite (résoluble par induction), la première
proche du hasard (imprédictible par construction) -- sinon le modèle a
mémorisé plutôt qu'appris l'algorithme.
"""
function evaluate_induction(g::NeuroGraph, logits::Symbol, ns::Symbol; vocab_size::Int, prefix_len::Int,
                             n_eval::Int=100, seed::Int=999)
    seq_len = 2 * prefix_len
    eval_rng = MersenneTwister(seed)
    sh_ok = sh_tot = fh_ok = fh_tot = 0
    for _ in 1:n_eval
        tokens, labels = sample_induction_sequence(eval_rng, vocab_size, prefix_len)
        set!(g, :token_ids, tokens; atom_type=Datom, namespace=ns)
        invalidate_all!(g; namespace=ns)
        lg = Array(demand!(g, logits; namespace=ns))
        preds = [argmax(lg[i, :]) for i in 1:seq_len]
        for i in 1:(seq_len - 1)
            correct = preds[i] == labels[i]
            if i <= prefix_len
                fh_tot += 1; fh_ok += correct ? 1 : 0
            else
                sh_tot += 1; sh_ok += correct ? 1 : 0
            end
        end
    end
    return (; first_half_acc = fh_ok / fh_tot, second_half_acc = sh_ok / sh_tot)
end

# ══════════════════════════════════════════════════════════════════════════════
# Induction à plusieurs sauts -- extension additive, aucune modification des
# fonctions à 1 saut ci-dessus. Conçue pour donner un levier de difficulté
# CONTRÔLABLE (par `n_hops`) plutôt que d'espérer trouver un vrai trou causal
# dans du texte réel (voir le screening du 2026-07-11 sur `real_llm_surgery_v2` :
# 0/8-0/55 fenêtres d'induction sur en-têtes de locuteur sont sous 0.85 de
# recovery -- la tâche à 1 saut sature un char-LM à 4 couches partout).
#
# Construction (proche des bancs "associative recall multi-hop" de la
# littérature, ex. Zoology/MQAR) : une chaîne de `n_hops+1` symboles distincts
# c_1..c_{n_hops+1} est présentée comme `n_hops` paires (clé,valeur) où
# valeur_i = clé_{i+1} (identité, pas juste égalité de valeur) --
#   tokens = c_1 c_2  c_2 c_3  c_3 c_4  ...  c_n c_{n+1}  [requête = c_1]
# et la BONNE réponse à la requête est c_{n_hops+1} -- PAS c_2, que donnerait
# une tête d'induction à 1 saut classique (elle retrouverait la première
# occurrence de c_1 et copierait ce qui suivait, c'est-à-dire c_2). Avec
# `n_hops=1` la tâche dégénère exactement en induction à 1 saut existante.
# Un seul token requiert `n_hops` sauts de composition séquentielle -- la
# difficulté est donc pilotée par `n_hops` (et par `n_layers`/pas
# d'entraînement), pas par la chance d'un tirage de texte réel.
# ══════════════════════════════════════════════════════════════════════════════

"""
    sample_multihop_sequence(rng, vocab_size, n_hops) -> (tokens, labels, q_pos)

Chaîne de `n_hops+1` symboles distincts, présentée en `n_hops` paires
(clé,valeur) où `valeur_i == clé_{i+1}`, suivie d'une requête répétant `clé_1`.
**Les `n_hops` paires sont présentées dans un ORDRE MÉLANGÉ** (pas dans l'ordre
de la chaîne) -- indispensable : sans mélange, la valeur finale `c_{n_hops+1}`
se trouve TOUJOURS au token juste avant la requête, ce qui rend la tâche
soluble par un simple raccourci positionnel ("copier le token précédent"),
sans aucune composition multi-sauts (bug trouvé empiriquement lors de la
calibration du 2026-07-11 : précision ≈1.0 dès 500 pas et recovery causale
EXACTEMENT nulle -- signe d'un raccourci, pas d'un circuit). Le mélange force
une recherche par CONTENU (quelle paire contient telle clé), pas par position,
pour chacun des `n_hops` sauts.

`labels[i] = tokens[i+1]` pour `i < seq_len` (contenu de la chaîne, non
prédictible en soi -- pur remplissage, comme la première moitié de
`sample_induction_sequence`) ; `labels[seq_len] = c_{n_hops+1}` est la SEULE
position dont la réponse exige la traversée complète de la chaîne -- c'est la
position `q_pos = seq_len` à cibler pour toute analyse causale (patching,
`greedy_patch_search!`), exactement comme la ligne `j` du protocole texte réel.
"""
function sample_multihop_sequence(rng, vocab_size::Int, n_hops::Int)
    n_hops >= 1 || error("❌ sample_multihop_sequence : n_hops doit être >= 1")
    n_hops + 1 <= vocab_size || error("❌ sample_multihop_sequence : vocab_size trop petit pour $(n_hops+1) symboles distincts")

    chain = Int[]
    while length(chain) < n_hops + 1
        cand = rand(rng, 1:vocab_size)
        cand in chain || push!(chain, cand)
    end

    pair_order = shuffle(rng, 1:n_hops)
    tokens = Int[]
    for i in pair_order
        push!(tokens, chain[i])       # clé_i   = c_i
        push!(tokens, chain[i+1])     # valeur_i = c_{i+1}
    end
    push!(tokens, chain[1])           # requête : clé_1 répétée
    seq_len = length(tokens)          # = 2*n_hops + 1
    answer = chain[end]               # c_{n_hops+1}, réponse correcte à la requête

    labels = vcat(tokens[2:end], [answer])
    return tokens, labels, seq_len
end

"""
    build_multihop_graph(dev, ns; vocab_size, dim, n_heads, hidden_dim, n_layers, n_hops)
    -> (g, logits_sym)

Même câblage que `build_induction_graph` (Embedding(token)+Embedding(position)
-> LlamaModel -> Linear -> `:cross_entropy`), `seq_len = 2*n_hops + 1`.

`batched` (défaut `true`, transmis à `LlamaModel(...; batched_attn=batched)`) --
défaut délibérément différent de `LlamaModel`/`build_induction_graph`
(`batched_attn=false`) : diagnostiqué le 2026-07-11 lors de la calibration de
cette tâche qu'une greffe `ao_h` NON batchée (`:matmul` direct, sortie
possiblement réécrite en place par le prochain `demand!`) peut faire dériver
silencieusement un `clean_cache` externe capturé par `capture_activations` une
fois que `patch_node!` l'a aliasé sur `.value` d'un nœud CUDA (`to_device` ne
copie jamais un `CuArray` déjà sur le bon device) -- observé concrètement :
`greedy_patch_search!`/`backward_prune!` trouvent une vraie trajectoire de
recovery (~0.96), mais un `patch_nodes!` ultérieur réutilisant le MÊME cache
externe donne 0.0 exactement, alors qu'un cache RE-capturé au même point donne
~0.97. En mode batché, `ao_h` est une vue (`:head_view` sur `ao3`), jamais
réécrite en place de la même façon -- le bug ne s'est jamais manifesté dans
aucune expérience de cette session utilisant `batched_attn=true`
(`real_llm_surgery_v2`, P1-bis, P4). Correctif de fond dans
`src/patching.jl`/`dispatch.jl` non fait ici (hors scope de cette tâche de
calibration) -- `batched=true` par défaut est le contournement adopté,
cohérent avec le reste de la session.
"""
function build_multihop_graph(dev, ns::Symbol; vocab_size::Int, dim::Int, n_heads::Int,
                               hidden_dim::Int, n_layers::Int, n_hops::Int, batched::Bool=true)
    seq_len = 2 * n_hops + 1
    g = NeuroGraph(namespace=ns, device=dev)
    set!(g, :token_ids, ones(Int, seq_len); atom_type=Datom, namespace=ns)
    set!(g, :pos_ids, collect(1:seq_len); atom_type=Datom, namespace=ns)
    tok_emb = Embedding(vocab_size, dim)(g, :token_ids, :tok; namespace=ns)
    pos_emb = Embedding(seq_len, dim)(g, :pos_ids, :pos; namespace=ns)
    x = :embed_sum
    addrule!(g, GraphRule(x, [tok_emb, pos_emb], :add; namespace=ns))
    out = LlamaModel(n_layers, dim, n_heads, hidden_dim; batched_attn=batched)(g, x; namespace=ns)
    logits = Linear(dim, vocab_size)(g, out, :lm_head; namespace=ns)
    set!(g, :labels, ones(Int, seq_len); atom_type=Datom, namespace=ns)
    addrule!(g, GraphRule(:loss, [logits, :labels], :cross_entropy; namespace=ns))
    return g, logits
end

"""
    train_multihop!(g, ns; vocab_size, n_hops, n_steps, seed=123, lr=3f-3, ...) -> losses

Même boucle AdamW pas-à-pas que `train_induction!`, un nouveau tirage de
chaîne à chaque pas (`sample_multihop_sequence`).
"""
function train_multihop!(g::NeuroGraph, ns::Symbol; vocab_size::Int, n_hops::Int, n_steps::Int,
                          seed::Int=123, lr::Float32=3f-3, b1::Float32=0.9f0, b2::Float32=0.999f0,
                          eps_v::Float32=1f-8, clip::Float32=1f0, wd::Float32=0f0)
    dev = g.device
    ps = params(g; namespace=ns)
    m1s = [Backend.zeros32(dev, size(p.value)...) for p in ps]
    m2s = [Backend.zeros32(dev, size(p.value)...) for p in ps]
    rng = MersenneTwister(seed)
    losses = Float64[]
    for t in 1:n_steps
        tokens, labels, _ = sample_multihop_sequence(rng, vocab_size, n_hops)
        set!(g, :token_ids, tokens; atom_type=Datom, namespace=ns)
        set!(g, :pos_ids, collect(1:length(tokens)); atom_type=Datom, namespace=ns)
        set!(g, :labels, labels; atom_type=Datom, namespace=ns)
        invalidate_all!(g; namespace=ns)
        loss_val = demand!(g, :loss; namespace=ns)
        push!(losses, Float64(sum(Array(loss_val))))
        backward_graph!(g, :loss; namespace=ns)
        for (i, p) in enumerate(ps)
            adamw_step!(dev, p.value, p.gradient, m1s[i], m2s[i], lr, b1, b2, eps_v, t, clip, wd)
        end
        invalidate_all!(g; namespace=ns)
    end
    return losses
end

"""
    evaluate_multihop(g, logits, ns; vocab_size, n_hops, n_eval=100, seed=999)
    -> (; query_acc, body_acc)

`query_acc` : précision à la SEULE position qui exige la traversée complète de
la chaîne (`q_pos = seq_len`) -- la métrique qui compte. `body_acc` : précision
sur le reste de la séquence (remplissage non prédictible par construction,
sert de témoin de non-mémorisation, comme `first_half_acc` pour l'induction).
"""
function evaluate_multihop(g::NeuroGraph, logits::Symbol, ns::Symbol; vocab_size::Int, n_hops::Int,
                            n_eval::Int=100, seed::Int=999)
    eval_rng = MersenneTwister(seed)
    q_ok = q_tot = b_ok = b_tot = 0
    for _ in 1:n_eval
        tokens, labels, q_pos = sample_multihop_sequence(eval_rng, vocab_size, n_hops)
        set!(g, :token_ids, tokens; atom_type=Datom, namespace=ns)
        set!(g, :pos_ids, collect(1:length(tokens)); atom_type=Datom, namespace=ns)
        invalidate_all!(g; namespace=ns)
        lg = Array(demand!(g, logits; namespace=ns))
        for i in 1:q_pos
            pred = argmax(lg[i, :])
            correct = pred == labels[i]
            if i == q_pos
                q_tot += 1; q_ok += correct ? 1 : 0
            else
                b_tot += 1; b_ok += correct ? 1 : 0
            end
        end
    end
    return (; query_acc = q_ok / q_tot, body_acc = b_ok / b_tot)
end
