# ══════════════════════════════════════════════════════════════════════════════
# SECONDE OPINION (agent indépendant) -- sonde sur l'INITIALISATION des poids.
# NE PAS confondre avec les fichiers de l'autre agent : ce script écrit
# UNIQUEMENT dans le scratchpad de session (GROK_PROBE_OUT), jamais dans
# notebook/grok_ckpt ni dans les logs *_diag_*.
#
# Hypothèse (angle NON exploré : wd/eps/arch/précision l'ont déjà été) :
#   NeuroDSL initialise TOUS les poids en UNIFORME U[-1/sqrt(fan_in), +...],
#   donc std = (1/sqrt(fan_in))/sqrt(3). Pour DIM=128 : std ~ 0.051 partout,
#   et ~0.0255 pour W2 du MLP (fan_in=512).
#   TransformerLens (HookedTransformer init_weights=True, init_mode "gpt2" par
#   défaut) initialise TOUS les W_ en NORMAL de std = 0.8/sqrt(d_model), soit
#   ~0.0707 -- MÊME std pour toutes les matrices, distribution normale.
#   => NeuroDSL part ~1.4x plus petit partout, ~2.8x plus petit pour W2.
#   Omnigrok (Liu et al. 2022) : la dynamique de grokking dépend fortement de
#   la NORME initiale des poids et de son interaction avec le weight decay.
#
# GROK_INIT = "default" (uniforme NeuroDSL, inchangé) | "tl" (normal 0.8/sqrt(DIM)).
# Tout le reste est IDENTIQUE au script principal (grokking_demo_neurodsl.jl),
# y compris le chemin CPU float64 de cross_entropy_grad (correctif de l'autre
# agent, déjà en place dans src/kernels.jl).
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, LinearAlgebra

const P          = parse(Int, get(ENV, "GROK_P", "7"))
const N_LAYERS   = 1
const DIM        = parse(Int, get(ENV, "GROK_DIM", "128"))
const N_HEADS    = parse(Int, get(ENV, "GROK_HEADS", "4"))
const D_HEAD     = DIM ÷ N_HEADS
const D_MLP      = parse(Int, get(ENV, "GROK_MLP", "512"))
const FRAC_TRAIN = parse(Float64, get(ENV, "GROK_FRAC", "0.3"))
const LR         = parse(Float32, get(ENV, "GROK_LR", "1e-3"))
const WD         = parse(Float32, get(ENV, "GROK_WD", "1.0"))
const BETA2      = 0.98f0
const N_STEPS    = parse(Int, get(ENV, "GROK_STEPS", "8000"))
const CLIP       = parse(Float32, get(ENV, "GROK_CLIP", "1e6"))
const EPS        = parse(Float32, get(ENV, "GROK_EPS", "1e-8"))
const INIT_MODE  = get(ENV, "GROK_INIT", "default")
const SEQ_LEN    = 3
const VOCAB_IN   = P + 1
const EQ_TOK     = P + 1

function build_dataset()
    pairs = [(a, b) for a in 1:P for b in 1:P]
    labels = [((a - 1 + b - 1) % P) + 1 for (a, b) in pairs]
    return pairs, labels
end

function train_test_split(pairs, labels; seed)
    rng = MersenneTwister(seed)
    idx = shuffle(rng, 1:length(pairs))
    cutoff = round(Int, length(pairs) * FRAC_TRAIN)
    return idx[1:cutoff], idx[cutoff+1:end]
end

function build_grok_graph(dev, nsx::Symbol)
    g = NeuroDSL.NeuroGraph(namespace=nsx, device=dev)
    NeuroDSL.set!(g, :token_ids, ones(Int, SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=nsx)
    tok_emb = NeuroDSL.Embedding(VOCAB_IN, DIM)(g, :token_ids, :tok; namespace=nsx)
    pos_emb = NeuroDSL.Embedding(SEQ_LEN, DIM)(g, :pos_ids, :pos; namespace=nsx)
    x0 = :embed_sum
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(x0, [tok_emb, pos_emb], :add; namespace=nsx))

    ao = NeuroDSL.MultiHeadAttention(DIM, N_HEADS; batched=true)(g, x0, :blk_mha; namespace=nsx)
    r1 = :blk_res1
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(r1, [x0, ao], :add; namespace=nsx))

    k1 = 1f0 / sqrt(Float32(DIM))
    W1 = (NeuroDSL.Backend.rand32(dev, D_MLP, DIM) .- 0.5f0) .* (2k1)
    NeuroDSL.set!(g, :blk_mlp_w1, W1; is_param=true, namespace=nsx)
    k2 = 1f0 / sqrt(Float32(D_MLP))
    W2 = (NeuroDSL.Backend.rand32(dev, DIM, D_MLP) .- 0.5f0) .* (2k2)
    NeuroDSL.set!(g, :blk_mlp_w2, W2; is_param=true, namespace=nsx)
    pre = :blk_mlp_pre
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(pre, [r1, :blk_mlp_w1], :matmul;
                       attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=nsx))
    post = :blk_mlp_post
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(post, [pre], :relu; namespace=nsx))
    mo = :blk_mlp_out
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(mo, [post, :blk_mlp_w2], :matmul;
                       attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=nsx))
    out = :blk_out
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out, [r1, mo], :add; namespace=nsx))

    logits = NeuroDSL.Linear(DIM, P; bias=false)(g, out, :unembed; namespace=nsx)

    sel = zeros(Float32, 1, SEQ_LEN); sel[1, SEQ_LEN] = 1f0
    NeuroDSL.set!(g, :sel_last, sel; atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:final_logits, [:sel_last, logits], :matmul; namespace=nsx))
    NeuroDSL.set!(g, :final_label, [1]; atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:final_logits, :final_label], :cross_entropy; namespace=nsx))
    return g, logits, mo, post
end

# Override d'init : réécrit TOUS les paramètres en normal std=0.8/sqrt(DIM)
# (fidèle à TransformerLens init_mode="gpt2"). Zéro empreinte sur src/.
function apply_tl_init!(g, nsx)
    dev = g.device
    std = 0.8f0 / sqrt(Float32(DIM))
    ps = NeuroDSL.params(g; namespace=nsx)
    for p in ps
        p.value .= NeuroDSL.Backend.randn32(dev, size(p.value)...) .* std
    end
    return std, length(ps)
end

function set_input!(g, nsx, a, b)
    NeuroDSL.set!(g, :token_ids, [a, b, EQ_TOK]; atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=nsx)
end

function param_norm(g, nsx)
    ps = NeuroDSL.params(g; namespace=nsx)
    s = 0.0
    for p in ps; s += sum(abs2, Array(p.value)); end
    return sqrt(s)
end

function train_grok!(g, nsx, pairs, labels, train_idx, test_idx; n_steps, io)
    dev = g.device
    ps = NeuroDSL.params(g; namespace=nsx)
    m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    accs = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    n_train = length(train_idx)
    train_losses = Float64[]; test_losses = Float64[]
    for t in 1:n_steps
        for acc in accs; fill!(acc, 0f0); end
        step_loss = 0.0
        for idx in train_idx
            a, b = pairs[idx]; v = labels[idx]
            set_input!(g, nsx, a, b)
            NeuroDSL.set!(g, :final_label, [v]; atom_type=NeuroDSL.Datom, namespace=nsx)
            NeuroDSL.invalidate_all!(g; namespace=nsx)
            loss_val = NeuroDSL.demand!(g, :loss; namespace=nsx)
            step_loss += Float64(sum(Array(loss_val)))
            NeuroDSL.backward_graph!(g, :loss; namespace=nsx)
            for (i, p) in enumerate(ps)
                p.gradient === nothing && continue
                accs[i] .+= p.gradient
            end
        end
        push!(train_losses, step_loss / n_train)
        for (i, p) in enumerate(ps)
            p.gradient === nothing && continue
            p.gradient .= accs[i] ./ Float32(n_train)
            NeuroDSL.adamw_step!(dev, p.value, p.gradient, m1s[i], m2s[i], LR, 0.9f0, BETA2, EPS, t, CLIP, WD)
        end
        NeuroDSL.invalidate_all!(g; namespace=nsx)
        if t % 100 == 0 || t == n_steps || t == 1
            test_loss = 0.0
            for idx in test_idx
                a, b = pairs[idx]; v = labels[idx]
                set_input!(g, nsx, a, b)
                NeuroDSL.set!(g, :final_label, [v]; atom_type=NeuroDSL.Datom, namespace=nsx)
                NeuroDSL.invalidate_all!(g; namespace=nsx)
                test_loss += Float64(sum(Array(NeuroDSL.demand!(g, :loss; namespace=nsx))))
            end
            push!(test_losses, test_loss / length(test_idx))
            pn = param_norm(g, nsx)
            line = "  pas $t/$n_steps  train=$(round(train_losses[end], digits=5))  test=$(round(test_losses[end], digits=5))  |W|=$(round(pn, digits=3))"
            println(line); flush(stdout)
            println(io, "$t,$(train_losses[end]),$(test_losses[end]),$pn"); flush(io)
        end
    end
    return train_losses, test_losses
end

dev = get(ENV, "GROK_DEVICE", "cpu") == "cpu" ?
      NeuroDSL.Backend.CPUDevice() : NeuroDSL.Backend.CUDADevice()
nsx = :grokprobe
Random.seed!(parse(Int, get(ENV, "GROK_SEED", "0")))
g, logits, mlp_out_sym, mlp_post_sym = build_grok_graph(dev, nsx)

if INIT_MODE == "tl"
    std, np = apply_tl_init!(g, nsx)
    println("INIT=tl : $np paramètres réécrits en normal std=$(round(std, digits=5))")
else
    println("INIT=default (uniforme NeuroDSL, inchangé)")
end

pairs, labels = build_dataset()
train_idx, test_idx = train_test_split(pairs, labels; seed=598)
println("P=$P  INIT=$INIT_MODE  |train|=$(length(train_idx))  |test|=$(length(test_idx))  budget=$N_STEPS pas")
println("Uniform loss (log P) = ", log(P), "   |W|_init = ", round(param_norm(g, nsx), digits=3))
flush(stdout)

out = get(ENV, "GROK_PROBE_OUT", "grok_probe_out.csv")
io = open(out, "w")
println(io, "step,train_loss,test_loss,param_norm")
t0 = time()
train_losses, test_losses = train_grok!(g, nsx, pairs, labels, train_idx, test_idx; n_steps=N_STEPS, io=io)
close(io)
println("Temps : ", round(time() - t0, digits=1), " s")
println("Final : train=$(train_losses[end])  test=$(test_losses[end])  min_test=$(minimum(test_losses))")
