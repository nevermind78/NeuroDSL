# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic ciblé : le run à 16000 pas (seed2) montre une oscillation stable
# (limit cycle, période ~1300-1400 pas, pics/creux STATIONNAIRES sur 11 cycles
# -- PAS de tendance à la baisse), pas un grokking en cours. Ceci teste
# directement le mécanisme "slingshot"/edge-of-stability (Thilak et al. 2022,
# "The Slingshot Mechanism") : la norme des poids de la dernière couche
# (unembed_W) et du second moment Adam (v/m2) croît-elle jusqu'à un seuil
# juste avant chaque pic de train_loss, puis s'effondre juste après ? Log de
# ces normes à CHAQUE pas de vérification (au lieu du seul train/test loss).
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, LinearAlgebra

const P          = parse(Int, get(ENV, "GROK_P", "17"))
const DIM        = 128
const N_HEADS    = 4
const D_MLP      = 512
const FRAC_TRAIN = 0.3
const LR         = parse(Float32, get(ENV, "GROK_LR", "1e-3"))
const WD         = parse(Float32, get(ENV, "GROK_WD", "1.0"))
const BETA2      = 0.98f0
const N_STEPS    = parse(Int, get(ENV, "GROK_STEPS", "4000"))
const CLIP       = parse(Float32, get(ENV, "GROK_CLIP", "1e6"))
const EPS        = parse(Float32, get(ENV, "GROK_EPS", "1e-8"))
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

set_input!(g, nsx, a, b) = begin
    NeuroDSL.set!(g, :token_ids, [a, b, EQ_TOK]; atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=nsx)
end

# CORRECTIF : ce script ignorait GROK_DEVICE et tournait TOUJOURS sur GPU,
# alors que grokking_demo_neurodsl.jl (le run principal qu'il est censé
# suivre à la trace, même graine) respecte GROK_DEVICE=cpu -- découvert en
# comparant les deux trajectoires (censées être identiques) et en trouvant
# un écart net dès les premiers pas. CPU et GPU ne sont PAS garantis
# bit-identiques (ordre de sommation BLAS différent), et ce système s'est
# révélé extrêmement sensible à des écarts numériques infimes (tout ce
# diagnostic grokking le montre). Corrigé pour matcher exactement le run
# principal.
dev = get(ENV, "GROK_DEVICE", "cuda") == "cpu" ?
      NeuroDSL.Backend.CPUDevice() : NeuroDSL.Backend.CUDADevice()
nsx = :grokdiag
Random.seed!(parse(Int, get(ENV, "GROK_SEED", "3")))
g, logits, mlp_out_sym, mlp_post_sym = build_grok_graph(dev, nsx)
pairs, labels = build_dataset()
train_idx, test_idx = train_test_split(pairs, labels; seed=598)
println("P=$P  |train|=$(length(train_idx))  |test|=$(length(test_idx))  budget=$N_STEPS pas -- DIAGNOSTIC INSTRUMENTÉ")
flush(stdout)

ps = NeuroDSL.params(g; namespace=nsx)
names = [p.name for p in ps]
i_unembed = findfirst(==(:unembed_W), names)
i_w2      = findfirst(==(:blk_mlp_w2), names)
i_tokE    = findfirst(==(:tok_E), names)
println("Paramètres suivis : unembed_W(idx=$i_unembed) blk_mlp_w2(idx=$i_w2) tok_E(idx=$i_tokE)")
flush(stdout)

m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
accs = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
n_train = length(train_idx)

const DIAG_OUT = joinpath(@__DIR__, "grok_diag_norms_eps$(EPS).txt")
open(DIAG_OUT, "w") do io
    println(io, "step,train_loss,test_loss,norm_unembed_W,norm_w2,norm_tokE,m2_unembed_mean,m2_w2_mean,global_param_norm")
    for t in 1:N_STEPS
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
        train_loss = step_loss / n_train
        for (i, p) in enumerate(ps)
            p.gradient === nothing && continue
            p.gradient .= accs[i] ./ Float32(n_train)
            NeuroDSL.adamw_step!(dev, p.value, p.gradient, m1s[i], m2s[i], LR, 0.9f0, BETA2, EPS, t, CLIP, WD)
        end
        NeuroDSL.invalidate_all!(g; namespace=nsx)
        if t % 20 == 0 || t == N_STEPS
            test_loss = 0.0
            for idx in test_idx
                a, b = pairs[idx]; v = labels[idx]
                set_input!(g, nsx, a, b)
                NeuroDSL.set!(g, :final_label, [v]; atom_type=NeuroDSL.Datom, namespace=nsx)
                NeuroDSL.invalidate_all!(g; namespace=nsx)
                test_loss += Float64(sum(Array(NeuroDSL.demand!(g, :loss; namespace=nsx))))
            end
            test_loss /= length(test_idx)
            nu  = Float64(norm(Array(ps[i_unembed].value)))
            nw2 = Float64(norm(Array(ps[i_w2].value)))
            nte = Float64(norm(Array(ps[i_tokE].value)))
            m2u = Float64(sum(Array(m2s[i_unembed])) / length(m2s[i_unembed]))
            m2w = Float64(sum(Array(m2s[i_w2])) / length(m2s[i_w2]))
            gpn = sqrt(sum(Float64(norm(Array(p.value)))^2 for p in ps))
            println(io, "$t,$train_loss,$test_loss,$nu,$nw2,$nte,$m2u,$m2w,$gpn")
            flush(io)
            println("  pas $t/$N_STEPS  train=$(round(train_loss,digits=5))  test=$(round(test_loss,digits=3))  |unembed|=$(round(nu,digits=3))  |w2|=$(round(nw2,digits=3))  m2_unembed=$(round(m2u,sigdigits=3))")
            flush(stdout)
        end
    end
end
println("Diagnostic terminé -- ", DIAG_OUT)
