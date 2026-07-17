# ══════════════════════════════════════════════════════════════════════════════
# Piste 3.3 (transitoire des moments d'optimiseur), motivée par une question de
# l'utilisateur sur E6 : le rejet initial (lambda > 0, alpha decroit depuis 1)
# est-il lié à une DÉSYNCHRONISATION d'AdamW entre la branche fraîchement
# greffée (second moment v initialisé à zéro, "standard and correct choice" per
# math.tex §sec:locality remark) et le seul consommateur aval direct de la
# greffe (:lm_head_W, qui reçoit soudain un gradient perturbé par une branche
# dont le v est encore froid) ?
#
# Réutilise EXACTEMENT le protocole de verify_attribution_curve.jl (même tâche,
# mêmes deux bras early/late, même construction Net2Net-style alpha0=1,
# zero_out_proj=true -- celle qui a produit le vrai signal de E6), en ajoutant
# le suivi de v (second moment AdamW, déjà tenu dans le Dict m2 de la boucle
# d'entraînement -- aucune nouvelle infrastructure requise) pour 3 groupes :
#   - "branche"  : les paramètres propres du greffon (préfixe handle.prefix)
#   - "lm_head"  : :lm_head_W, l'UNIQUE consommateur paramétré directement en
#                  aval de la greffe (le graphe se termine par out -> lm_head)
#   - "reste"    : tout le reste de l'hôte (embeddings + couches LlamaModel),
#                  comme référence de ce qu'est un v "non perturbé"
# Résolution temporelle augmentée (tous les 25 pas, pas 100) pour bien résoudre
# la fenêtre critique 0-700 pas où E6 a mesuré le croisement de signe de A(t).
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics, Printf, JSON, Downloads

dev = NeuroDSL.Backend.CUDADevice()

const CORPUS_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
const CORPUS_PATH = joinpath(@__DIR__, "data", "tinyshakespeare", "input.txt")
isfile(CORPUS_PATH) || (mkpath(dirname(CORPUS_PATH)); Downloads.download(CORPUS_URL, CORPUS_PATH))
text = read(CORPUS_PATH, String)
chars = sort(unique(collect(text)))
stoi = Dict(c => i for (i, c) in enumerate(chars))
data = [stoi[c] for c in text]
n_train = floor(Int, 0.9 * length(data))
train_ids, val_ids = data[1:n_train], data[n_train+1:end]
vocab_size = length(chars)

const BLOCK_SIZE = 64
const DIM, N_HEADS, HIDDEN_DIM, N_LAYERS = 64, 2, 128, 3
const SEED = 1
const STEP_EARLY = 500
const STEP_LATE = 2500
const POST_GRAFT_STEPS = 1200
const MEASURE_EVERY = 25   # résolution augmentée vs 100 dans verify_attribution_curve.jl
const LR = 1f-3

sample_window(rng, ids, bs) = (i = rand(rng, 1:(length(ids)-bs)); (ids[i:i+bs-1], ids[i+1:i+bs]))

const VAL_TOK, VAL_LAB = let i = 1
    val_ids[i:i+BLOCK_SIZE-1], val_ids[i+1:i+BLOCK_SIZE]
end

function build_graph(ns)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.set!(g, :token_ids, ones(Int, BLOCK_SIZE); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:BLOCK_SIZE); atom_type=NeuroDSL.Datom, namespace=ns)
    tok_emb = NeuroDSL.Embedding(vocab_size, DIM)(g, :token_ids, :tok; namespace=ns)
    pos_emb = NeuroDSL.Embedding(BLOCK_SIZE, DIM)(g, :pos_ids, :pos; namespace=ns)
    x = :embed_sum
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(x, [tok_emb, pos_emb], :add; namespace=ns))
    out = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM; batched_attn=true)(g, x; namespace=ns)
    logits = NeuroDSL.Linear(DIM, vocab_size)(g, out, :lm_head; namespace=ns)
    NeuroDSL.set!(g, :labels, ones(Int, BLOCK_SIZE); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [logits, :labels], :cross_entropy; namespace=ns))
    return g, out
end

function set_batch!(g, ns, tok, lab)
    NeuroDSL.set!(g, :token_ids, tok; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:BLOCK_SIZE); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :labels, lab; atom_type=NeuroDSL.Datom, namespace=ns)
end

function ablation_A(g, ns, alpha_sym)
    set_batch!(g, ns, VAL_TOK, VAL_LAB)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    alpha_now = Array(NeuroDSL.demand!(g, alpha_sym; namespace=ns))[1]
    loss_now = Float64(sum(Array(NeuroDSL.demand!(g, :loss; namespace=ns))))
    NeuroDSL.set!(g, alpha_sym, Float32[0f0]; is_param=true, namespace=ns)
    loss_ablated = Float64(sum(Array(NeuroDSL.demand!(g, :loss; namespace=ns))))
    NeuroDSL.set!(g, alpha_sym, Float32[alpha_now]; is_param=true, namespace=ns)
    NeuroDSL.demand!(g, :loss; namespace=ns)
    return loss_ablated - loss_now, alpha_now, loss_now
end

v_stats(m2, syms) = isempty(syms) ? (0.0, 0.0) : begin
    vals = Float64[]
    for s in syms
        append!(vals, Float64.(Array(m2[s])))
    end
    (mean(vals), maximum(vals))
end

function run_arm(graft_step::Int, arm_name::String)
    ns = Symbol(:desync_, arm_name)
    Random.seed!(SEED)
    g, out_sym = build_graph(ns)
    m1 = Dict{Symbol,Any}(); m2 = Dict{Symbol,Any}()
    sync!() = for nd in NeuroDSL.params(g; namespace=ns)
        haskey(m1, nd.name) && continue
        m1[nd.name] = NeuroDSL.Backend.zeros32(dev, size(nd.value)...)
        m2[nd.name] = NeuroDSL.Backend.zeros32(dev, size(nd.value)...)
    end
    sync!()
    rng = MersenneTwister(SEED)
    handle = nothing
    branch_syms = Symbol[]
    lmhead_syms = Symbol[:lm_head_W]
    trajectory = NamedTuple[]
    total_steps = graft_step + POST_GRAFT_STEPS
    t0 = time()
    for step in 1:total_steps
        if step == graft_step + 1
            out_sym, handle = NeuroDSL.graft_shadow_block!(g, ns, out_sym, DIM, N_HEADS, HIDDEN_DIM;
                                                             alpha0=1f0, zero_out_proj=true,
                                                             prefix=Symbol(:shadow_, arm_name))
            sync!()
            branch_syms = [p.name for p in NeuroDSL.params(g; namespace=ns)
                            if startswith(String(p.name), String(handle.prefix))]
        end
        tok, lab = sample_window(rng, train_ids, BLOCK_SIZE)
        set_batch!(g, ns, tok, lab)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.demand!(g, :loss; namespace=ns)
        NeuroDSL.backward_graph!(g, :loss; namespace=ns)
        ps = NeuroDSL.params(g; namespace=ns)
        m1f = [m1[p.name] for p in ps]; m2f = [m2[p.name] for p in ps]
        NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps], [p.gradient for p in ps],
                                      m1f, m2f, LR, 0.9f0, 0.999f0, 1f-8, step, 1f0, 0f0)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        if handle !== nothing && (step - graft_step) % MEASURE_EVERY == 0
            A, alpha_t, loss_t = ablation_A(g, ns, handle.alpha_sym)
            rest_syms = [p.name for p in NeuroDSL.params(g; namespace=ns)
                         if p.name ∉ branch_syms && p.name ∉ lmhead_syms]
            v_branch_mean, v_branch_max = v_stats(m2, branch_syms)
            v_lmhead_mean, v_lmhead_max = v_stats(m2, lmhead_syms)
            v_rest_mean, v_rest_max = v_stats(m2, rest_syms)
            push!(trajectory, (t=step - graft_step, A=A, alpha=alpha_t, loss=loss_t,
                                v_branch_mean=v_branch_mean, v_branch_max=v_branch_max,
                                v_lmhead_mean=v_lmhead_mean, v_lmhead_max=v_lmhead_max,
                                v_rest_mean=v_rest_mean, v_rest_max=v_rest_max))
            @printf "[%s] t=%-4d A=%8.5f alpha=%6.4f v_branch=%9.6f v_lmhead=%9.6f v_rest=%9.6f\n" arm_name (step-graft_step) A alpha_t v_branch_mean v_lmhead_mean v_rest_mean
        end
    end
    dt = time() - t0
    println("[$arm_name] terminé en $(round(dt/60, digits=1)) min")
    return trajectory
end

println("="^70, "\n>>> Bras EARLY (greffe au pas $STEP_EARLY)\n", "="^70)
traj_early = run_arm(STEP_EARLY, "early")

println("\n", "="^70, "\n>>> Bras LATE (greffe au pas $STEP_LATE)\n", "="^70)
traj_late = run_arm(STEP_LATE, "late")

open(joinpath(@__DIR__, "e6_optimizer_desync_results.json"), "w") do io
    JSON.print(io, Dict(
        "step_early"=>STEP_EARLY, "step_late"=>STEP_LATE,
        "traj_early"=>[Dict(pairs(t)) for t in traj_early],
        "traj_late"=>[Dict(pairs(t)) for t in traj_late],
    ))
end
println("\nÉcrit -> notebook/e6_optimizer_desync_results.json")
