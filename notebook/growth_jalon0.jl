# ══════════════════════════════════════════════════════════════════════════════
# Jalon 0 (piste "économie de la croissance", proposée par Fable le 2026-07-13,
# mémoire project_neurodsl_growth_schedules_pitch_2026-07-13.md) -- kill-switch :
# est-ce que faire GRANDIR un char-LM pendant l'entraînement (2→3→4 couches, ou
# 1→2→4) bat un modèle à taille fixe, à budget de calcul RIGOUREUSEMENT égal ?
#
# Normalisation du budget : proxy "layer-steps" = Σ_t n_layers(t) cumulé sur
# l'entraînement. Le coût d'un pas est dominé par les blocs LlamaBlock (le coût
# de l'embedding/tête LM/cross-entropy est indépendant de n_layers et identique
# entre bras) -- ce proxy est linéaire dans le nombre de couches, cohérent avec
# la pratique de la littérature (Net2Net/LiGO/G_stack normalisent aussi par un
# proxy FLOPs analytique, pas par un profilage matériel exact).
#
# Bras :
#   A (référence) : n_layers fixe tout du long.
#   B : schedule [2,3,4] -- budget total réparti en parts ÉGALES de layer-steps
#       entre les 3 paliers (donc plus de PAS aux paliers peu profonds, moins
#       aux paliers profonds -- même budget de calcul, plus de mises à jour de
#       gradient au total si le schedule croît).
#   C : schedule [1,2,4].
#
# Croissance : insert_block!(g, ns, current_out, dim, n_heads, hidden_dim;
# batched_attn=true) -- identité EXACTE au moment de l'insertion (poids de
# sortie mis à zéro), donc AUCUN saut de perte au pas de croissance lui-même,
# seulement une divergence progressive ensuite due à l'entraînement de la
# nouvelle branche -- signal à vérifier explicitement avant tout chiffre.
#
# AdamW : moments keyés par SYMBOLE (Dict{Symbol,Array}, pas par position) --
# extend_adamw_state!-style (src/serialization.jl) pour ne jamais désaligner
# une nouvelle tranche de paramètres après une croissance.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics, Printf, JSON, Downloads

dev = NeuroDSL.Backend.CUDADevice()

# ── Corpus (identique à real_llm.ipynb) ──────────────────────────────────────
const CORPUS_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
const CORPUS_PATH = joinpath(@__DIR__, "data", "tinyshakespeare", "input.txt")
function load_corpus(path::String, url::String)
    isfile(path) || (mkpath(dirname(path)); Downloads.download(url, path))
    return read(path, String)
end
text = load_corpus(CORPUS_PATH, CORPUS_URL)
chars = sort(unique(collect(text)))
stoi = Dict(c => i for (i, c) in enumerate(chars))
encode(s, stoi) = [stoi[c] for c in s]
data = encode(text, stoi)
n_train = floor(Int, 0.9 * length(data))
train_ids, val_ids = data[1:n_train], data[n_train+1:end]
vocab_size = length(chars)

const BLOCK_SIZE = 256
const DIM, N_HEADS, HIDDEN_DIM = 256, 4, 512

function sample_window(rng, ids, block_size)
    i = rand(rng, 1:(length(ids) - block_size))
    return ids[i:i+block_size-1], ids[i+1:i+block_size]
end

function val_loss(g, ns, block_size; n_windows=64)
    max_start = length(val_ids) - block_size
    starts = round.(Int, range(1, max_start, length=n_windows))
    total = 0.0
    for i in starts
        tokens, labels = val_ids[i:i+block_size-1], val_ids[i+1:i+block_size]
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        loss_val = NeuroDSL.demand_release!(g, :loss; namespace=ns)
        total += Float64(sum(Array(loss_val)))
    end
    return total / n_windows
end

function build_char_lm_graph(dev, ns; n_layers, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.set!(g, :token_ids, ones(Int, BLOCK_SIZE); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:BLOCK_SIZE); atom_type=NeuroDSL.Datom, namespace=ns)
    tok_emb = NeuroDSL.Embedding(vocab_size, dim)(g, :token_ids, :tok; namespace=ns)
    pos_emb = NeuroDSL.Embedding(BLOCK_SIZE, dim)(g, :pos_ids, :pos; namespace=ns)
    x = :embed_sum
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(x, [tok_emb, pos_emb], :add; namespace=ns))
    out = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim; batched_attn=true)(g, x; namespace=ns)
    logits = NeuroDSL.Linear(dim, vocab_size)(g, out, :lm_head; namespace=ns)
    NeuroDSL.set!(g, :labels, ones(Int, BLOCK_SIZE); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [logits, :labels], :cross_entropy; namespace=ns))
    return g, logits, out
end

# ── Bras de croissance générique ─────────────────────────────────────────────
"""
    train_growth_arm!(schedule, layer_step_budget; seed, val_every, weight_fn) -> NamedTuple

`schedule` ex. [4] (fixe) ou [2,3,4] -- le budget total (en layer-steps) est
réparti entre les `length(schedule)` paliers selon `weight_fn(L)` (défaut :
`L -> 1.0`, poids égal quel que soit `L` -- comportement de jalon 0, INCHANGÉ).

Correctif 2026-07-13 (jalon 1) : le partage à POIDS ÉGAUX confond "nombre de
paliers" et "part du budget consacrée aux profondeurs bon marché" -- plus il y
a de paliers, moins celui à profondeur 1 reçoit de budget (1/n_stages au lieu
d'une part fixe), donc moins de pas totaux, ce qui pilotait presque tout
l'écart de perte observé dans jalon 1 (non conçu pour répondre à la question
de granularité). Passer `weight_fn = L -> 1.0/L` (ou `1/L^2`) donne à chaque
palier un budget ∝ 1/L : ajouter des paliers intermédiaires ne vole alors
qu'une petite part au palier le moins cher, au lieu de la diluer
proportionnellement à `length(schedule)`.

`lr_schedule` (optionnel, `nothing` par défaut -- comportement INCHANGÉ, `lr`
constant comme avant) : fonction `(t_global, total_steps_planned) -> Float32`
appelée à chaque pas pour obtenir le taux d'apprentissage courant, ex. pour une
décroissance cosinus `(t, T) -> lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*t/T))`.
"""
function train_growth_arm!(schedule::Vector{Int}, layer_step_budget::Int;
                            seed::Int=123, val_every::Int=100, lr=1f-3,
                            weight_fn::Function=(L -> 1.0),
                            lr_schedule::Union{Nothing,Function}=nothing,
                            dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM)
    ns = Symbol(:growth_, join(schedule, "_"), :_, seed)
    n_layers = schedule[1]
    # Correctif 2026-07-13 : `seed` doit contrôler aussi les poids initiaux
    # (tirés via Backend.rand32 sur le RNG global), pas seulement l'ordre des
    # minibatchs -- sinon "même seed" entre bras A/B/C ne veut rien dire.
    # Couvre aussi les poids du bloc greffé par insert_block! plus loin (même
    # flux de RNG global, jamais touché par le `rng` local ci-dessous, qui est
    # un objet MersenneTwister séparé dédié aux minibatchs).
    Random.seed!(seed)
    g, logits_sym, cur_out = build_char_lm_graph(dev, ns; n_layers=n_layers, dim=dim,
                                                  n_heads=n_heads, hidden_dim=hidden_dim)

    m1 = Dict{Symbol,Any}(); m2 = Dict{Symbol,Any}()
    function sync_moments!()
        for nd in NeuroDSL.params(g; namespace=ns)
            haskey(m1, nd.name) && continue
            m1[nd.name] = NeuroDSL.Backend.zeros32(dev, size(nd.value)...)
            m2[nd.name] = NeuroDSL.Backend.zeros32(dev, size(nd.value)...)
        end
    end
    sync_moments!()

    raw_weights = [weight_fn(L) for L in schedule]
    norm_weights = raw_weights ./ sum(raw_weights)
    stage_budgets = [max(1, round(Int, layer_step_budget * w)) for w in norm_weights]
    total_steps_planned = sum(max(1, stage_budgets[i] ÷ schedule[i]) for i in eachindex(schedule))
    rng = MersenneTwister(seed)
    train_losses = Float64[]
    val_history = Tuple{Int,Float64,Int}[]   # (global_step, val_loss, n_layers_at_eval)
    growth_events = Tuple{Int,Int,Int}[]     # (global_step, layer_steps_consumed, new_n_layers)
    t_global = 0
    layer_steps_consumed = 0
    t_start = time()

    for (stage_idx, L) in enumerate(schedule)
        if stage_idx > 1
            new_out = NeuroDSL.insert_block!(g, ns, cur_out, dim, n_heads, hidden_dim;
                                              prefix=Symbol(:layer_, L), batched_attn=true)
            cur_out = new_out
            sync_moments!()
            push!(growth_events, (t_global, layer_steps_consumed, L))
        end
        n_steps_stage = max(1, stage_budgets[stage_idx] ÷ L)
        ps = NeuroDSL.params(g; namespace=ns)
        for _ in 1:n_steps_stage
            t_global += 1
            layer_steps_consumed += L
            tokens, labels = sample_window(rng, train_ids, BLOCK_SIZE)
            NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.set!(g, :pos_ids, collect(1:BLOCK_SIZE); atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.invalidate_all!(g; namespace=ns)
            loss_val = NeuroDSL.demand!(g, :loss; namespace=ns)
            push!(train_losses, Float64(sum(Array(loss_val))))
            NeuroDSL.backward_graph!(g, :loss; namespace=ns)
            m1_flat = [m1[nd.name] for nd in ps]; m2_flat = [m2[nd.name] for nd in ps]
            cur_lr = lr_schedule === nothing ? lr : Float32(lr_schedule(t_global, total_steps_planned))
            NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps], [p.gradient for p in ps],
                                         m1_flat, m2_flat, cur_lr, 0.9f0, 0.999f0, 1f-8, t_global, 1f0, 0f0)
            NeuroDSL.invalidate_all!(g; namespace=ns)
            if t_global % val_every == 0
                vl = val_loss(g, ns, BLOCK_SIZE)
                push!(val_history, (t_global, vl, L))
            end
        end
    end
    elapsed = time() - t_start
    return (; schedule, layer_step_budget, n_steps_total=t_global, elapsed,
            train_losses, val_history, growth_events,
            final_val=val_history[end][2])
end

println("Module de croissance chargé. Prêt pour le pilote (voir growth_jalon0_pilot.jl).")
