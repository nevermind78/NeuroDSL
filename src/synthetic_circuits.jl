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
