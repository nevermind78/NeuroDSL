# ══════════════════════════════════════════════════════════════════════════════
# Test contrefactuel de l'hypothèse "désynchronisation AdamW cause le rejet" :
# verify_e6_optimizer_desync.jl a montré une vraie désynchronisation (v_lmhead
# 100-200x plus grand que v_branch au moment de la greffe) mais dans le mauvais
# sens pour expliquer le timing (le bras tardif, qui rejette PLUS VITE, a un
# ratio de désynchronisation PLUS PETIT). Ici : on "préchauffe" artificiellement
# le second moment v de la branche fraîche au niveau de v_lmhead (son unique
# consommateur aval direct) AU MOMENT de la greffe, ce qui annule l'avantage de
# "pas effectif géant" qu'un v quasi nul donne à AdamW pour un gradient donné.
# Si la désynchronisation cause le rejet : lambda devrait baisser (rejet plus
# lent) une fois préchauffé, et l'écart early/late devrait se réduire. Si non :
# lambda et l'écart early/late devraient rester quasi inchangés vs. le bras
# froid déjà mesuré (E6 / verify_e6_optimizer_desync.jl).
#
# Même protocole exact que verify_e6_optimizer_desync.jl (même tâche, mêmes
# deux bras, même construction Net2Net-style alpha0=1, zero_out_proj=true),
# seule différence : juste après graft_shadow_block!, m2[sym] pour chaque
# paramètre de la branche est réinitialisé à mean(v_lmhead) au lieu de 0
# (m1 reste à 0, choix standard -- seul le second moment, responsable du "pas
# effectif géant" à v quasi nul, est préchauffé).
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
const MEASURE_EVERY = 25
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

function run_arm(graft_step::Int, arm_name::String; warm_start::Bool)
    ns = Symbol(:warm_, arm_name)
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
    v_warm_used = NaN
    t0 = time()
    for step in 1:total_steps
        if step == graft_step + 1
            out_sym, handle = NeuroDSL.graft_shadow_block!(g, ns, out_sym, DIM, N_HEADS, HIDDEN_DIM;
                                                             alpha0=1f0, zero_out_proj=true,
                                                             prefix=Symbol(:shadow_, arm_name))
            sync!()
            branch_syms = [p.name for p in NeuroDSL.params(g; namespace=ns)
                            if startswith(String(p.name), String(handle.prefix))]
            if warm_start
                v_warm_used = mean(Float64.(Array(m2[:lm_head_W])))
                for s in branch_syms
                    shp = size(m2[s])
                    m2[s] = NeuroDSL.Backend.to_device(dev, fill(Float32(v_warm_used), shp))
                end
                println("  [warm-start] v_lmhead (mean) = ", v_warm_used, " -> injecté dans ", length(branch_syms), " params de la branche")
            end
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
            v_branch_mean, v_branch_max = v_stats(m2, branch_syms)
            v_lmhead_mean, v_lmhead_max = v_stats(m2, lmhead_syms)
            push!(trajectory, (t=step - graft_step, A=A, alpha=alpha_t, loss=loss_t,
                                v_branch_mean=v_branch_mean, v_branch_max=v_branch_max,
                                v_lmhead_mean=v_lmhead_mean, v_lmhead_max=v_lmhead_max))
            @printf "[%s/%s] t=%-4d A=%8.5f alpha=%6.4f v_branch_max=%.6f v_lmhead_max=%.6f\n" arm_name (warm_start ? "warm" : "cold") (step-graft_step) A alpha_t v_branch_max v_lmhead_max
        end
    end
    dt = time() - t0
    println("[$arm_name/$(warm_start ? "warm" : "cold")] terminé en $(round(dt/60, digits=1)) min")
    return trajectory, v_warm_used
end

println("="^70, "\n>>> Bras EARLY, WARM-START (greffe au pas $STEP_EARLY)\n", "="^70)
traj_early_warm, vwarm_early = run_arm(STEP_EARLY, "early"; warm_start=true)

println("\n", "="^70, "\n>>> Bras LATE, WARM-START (greffe au pas $STEP_LATE)\n", "="^70)
traj_late_warm, vwarm_late = run_arm(STEP_LATE, "late"; warm_start=true)

open(joinpath(@__DIR__, "e6_v_warmstart_results.json"), "w") do io
    JSON.print(io, Dict(
        "step_early"=>STEP_EARLY, "step_late"=>STEP_LATE,
        "v_warm_early"=>vwarm_early, "v_warm_late"=>vwarm_late,
        "traj_early_warm"=>[Dict(pairs(t)) for t in traj_early_warm],
        "traj_late_warm"=>[Dict(pairs(t)) for t in traj_late_warm],
    ))
end
println("\nÉcrit -> notebook/e6_v_warmstart_results.json")
