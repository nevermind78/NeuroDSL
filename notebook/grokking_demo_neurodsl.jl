# ══════════════════════════════════════════════════════════════════════════════
# Port NeuroDSL du "Grokking Demo" de Neel Nanda/TransformerLens
# (artilce/Grokking_Demo.ipynb) -- addition modulaire, transition de grokking,
# circuit de Fourier (cos/sin, identité trigonométrique d'addition d'angles).
#
# BUT : (1) valider indépendamment que NeuroDSL reproduit un résultat
# d'interprétabilité mécaniste canonique déjà établi par la littérature ;
# (2) construire/valider la boîte à outils "perte restreinte/exclue" sur un
# cas où la vraie réponse (circuit trig) est déjà connue, AVANT de l'appliquer
# à la tâche du marqueur (notebook/marker_task_restricted_loss.jl, fait sans
# Fourier -- pas de structure cyclique dans le vocabulaire du marqueur).
#
# ÉCARTS DOCUMENTÉS PAR RAPPORT AU NOTEBOOK ORIGINAL (pas cachés) :
#   - p=17 au lieu de 113 : NeuroDSL traite UNE séquence à la fois (pas de
#     dimension batch native dans les tenseurs, contrairement à PyTorch) --
#     "full batch" littéral (accumuler sur tout le train set à chaque pas)
#     coûterait p²*0.3 passes forward-backward PAR pas d'optimiseur. À p=113
#     (~3830 exemples/pas), même 5000 pas donnerait ~19M passes -- hors budget
#     raisonnable. À p=17 (~86 exemples/pas), le même "full batch" (esprit
#     fidèle : tout le train set, pas de sous-échantillonnage) reste
#     praticable (8000 pas * 86 ≈ 688k passes, comparable à ce qui a déjà été
#     fait cette session pour V=16/dim=128 de math4.tex).
#   - Architecture IDENTIQUE sinon : 1 couche, 4 têtes, d_head=32, d_model=128,
#     MLP relu simple largeur 512 (PAS de gating SwiGLU -- construit à la main
#     via addrule!, LlamaBlock ne convient pas), AUCUNE norme (contrairement à
#     LlamaBlock qui a RMSNorm en dur), biais désactivés.
#   - Fréquences clés IDENTIFIÉES sur le modèle entraîné (data-driven), PAS
#     copiées de Nanda et al. (qui sont spécifiques à p=113, sans rapport
#     avec p=17).
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, LinearAlgebra

const P          = parse(Int, get(ENV, "GROK_P", "17"))
const N_LAYERS   = 1
# GROK_DIM/GROK_HEADS/GROK_MLP : hypothèse de surparamétrisation extrême --
# 128 dims/4 têtes/512 MLP pour 3 à 87 exemples d'entraînement (p petit) est
# un ratio paramètres/données sans rapport avec le notebook Nanda réel (même
# largeur, mais p=113 -> 3831 exemples). Défauts inchangés pour ne rien casser.
const DIM        = parse(Int, get(ENV, "GROK_DIM", "128"))
const N_HEADS    = parse(Int, get(ENV, "GROK_HEADS", "4"))
const D_HEAD     = DIM ÷ N_HEADS
const D_MLP      = parse(Int, get(ENV, "GROK_MLP", "512"))
const FRAC_TRAIN = 0.3
const LR         = parse(Float32, get(ENV, "GROK_LR", "1e-3"))
const WD         = parse(Float32, get(ENV, "GROK_WD", "1.0"))
const BETA2      = 0.98f0
const N_STEPS    = parse(Int, get(ENV, "GROK_STEPS", "8000"))
const CLIP       = parse(Float32, get(ENV, "GROK_CLIP", "1e6"))
const EPS        = parse(Float32, get(ENV, "GROK_EPS", "1e-8"))
# EPS : diagnostic instrumenté (notebook/grok_diag_norms.txt, notebook/
# grokking_diag_instrumented.jl) -- confirmé sur pièce : en phase de
# mémorisation parfaite le gradient réel est EXACTEMENT nul (m2 décroît par
# EMA pure, ratio EXACT beta2^20=0.98^20=0.667 sur ~1300 pas), donc m2 chute
# jusqu'à ~1e-15/1e-16, i.e. sqrt(m2)~1e-8 -- l'ordre de grandeur d'EPS
# lui-même. Quand un gradient résiduel minuscule réapparaît (le weight decay
# ayant érodé la marge mémorisée), m2 est recalculé à partir de CE seul
# gradient (l'ancien m2 est négligeable), et le pas devient ~lr quelle que
# soit la taille réelle du gradient -- survol disproportionné = pic de
# train_loss = mécanisme "slingshot" (Thilak et al., "The Slingshot
# Mechanism", 2022 -- littérature documentant précisément ce phénomène sur ce
# type de petit modèle/tâche algorithmique). Remède mécaniquement justifié :
# relever EPS pour qu'il domine le dénominateur PENDANT la phase calme (où
# sqrt(vh) s'effondrait vers ~1e-8), sans perturber la phase d'apprentissage
# active (gradients ~1e-2 à 1, 100-1000x plus grands qu'EPS=1e-4).
# CLIP : le notebook PyTorch de Nanda (artilce/Grokking_Demo.ipynb, cellule 28,
# `optimizer = torch.optim.AdamW(...)`) N'APPLIQUE AUCUN clipping de gradient
# nulle part dans la boucle d'entraînement (cellule 33) -- torch.optim.AdamW
# ne clippe pas par défaut. La 1re version de ce script utilisait clip=1f0
# par analogie avec marker_task_experiment.jl, sans le vérifier contre le vrai
# notebook -- c'est un écart réel, corrigé ici (CLIP=1e6 rend le clamp inerte,
# on GARDE l'appel à adamw_step! inchangé plutôt que de modifier le kernel
# partagé src/kernels.jl).
const SEQ_LEN    = 3          # [a, b, =]
const VOCAB_IN   = P + 1      # tokens 1..P pour les valeurs (1-indexé), P+1 pour "="
const EQ_TOK     = P + 1

# ── Tâche : addition modulaire (1-indexée : valeurs 1..P représentent 0..P-1) ─
function build_dataset()
    pairs = [(a, b) for a in 1:P for b in 1:P]
    labels = [((a - 1 + b - 1) % P) + 1 for (a, b) in pairs]   # (a+b) mod P, ré-indexé à 1..P
    return pairs, labels
end

function train_test_split(pairs, labels; seed)
    rng = MersenneTwister(seed)
    idx = shuffle(rng, 1:length(pairs))
    cutoff = round(Int, length(pairs) * FRAC_TRAIN)
    return idx[1:cutoff], idx[cutoff+1:end]
end

# ── Graphe : 1 bloc SANS norme, attention standard, MLP relu simple ──────────
function build_grok_graph(dev, nsx::Symbol)
    g = NeuroDSL.NeuroGraph(namespace=nsx, device=dev)
    NeuroDSL.set!(g, :token_ids, ones(Int, SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=nsx)
    tok_emb = NeuroDSL.Embedding(VOCAB_IN, DIM)(g, :token_ids, :tok; namespace=nsx)
    pos_emb = NeuroDSL.Embedding(SEQ_LEN, DIM)(g, :pos_ids, :pos; namespace=nsx)
    x0 = :embed_sum
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(x0, [tok_emb, pos_emb], :add; namespace=nsx))

    # Attention (sans norme -- appelée directement sur x0, batched pour
    # cohérence avec le reste de la session ; aucun biais, comme le demo).
    ao = NeuroDSL.MultiHeadAttention(DIM, N_HEADS; batched=true)(g, x0, :blk_mha; namespace=nsx)
    r1 = :blk_res1
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(r1, [x0, ao], :add; namespace=nsx))

    # MLP relu SIMPLE (pas de gating/SwiGLU) : mo = relu(r1 * W1') * W2'
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

    # Sortie : Linear(dim, P, bias=false) -- vocabulaire de sortie = P (pas P+1)
    logits = NeuroDSL.Linear(DIM, P; bias=false)(g, out, :unembed; namespace=nsx)

    sel = zeros(Float32, 1, SEQ_LEN); sel[1, SEQ_LEN] = 1f0
    NeuroDSL.set!(g, :sel_last, sel; atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:final_logits, [:sel_last, logits], :matmul; namespace=nsx))
    NeuroDSL.set!(g, :final_label, [1]; atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:final_logits, :final_label], :cross_entropy; namespace=nsx))
    return g, logits, mo, post  # renvoie aussi les noeuds MLP pour l'analyse ultérieure
end

function set_input!(g, nsx, a, b)
    NeuroDSL.set!(g, :token_ids, [a, b, EQ_TOK]; atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=nsx)
end

function train_grok!(g, nsx, pairs, labels, train_idx, test_idx; n_steps)
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
        # "full batch" -- accumulation sur TOUT le train set à chaque pas
        # (esprit fidèle au choix de Nanda et al., à échelle réduite -- voir
        # bannière en tête de fichier).
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
        if t % 20 == 0 || t == n_steps
            test_loss = 0.0
            for idx in test_idx
                a, b = pairs[idx]; v = labels[idx]
                set_input!(g, nsx, a, b)
                NeuroDSL.set!(g, :final_label, [v]; atom_type=NeuroDSL.Datom, namespace=nsx)
                NeuroDSL.invalidate_all!(g; namespace=nsx)
                test_loss += Float64(sum(Array(NeuroDSL.demand!(g, :loss; namespace=nsx))))
            end
            push!(test_losses, test_loss / length(test_idx))
            println("  pas $t/$n_steps  train_loss=$(round(train_losses[end], digits=5))  test_loss=$(round(test_losses[end], digits=5))")
            flush(stdout)
        end
    end
    return train_losses, test_losses
end

# ── Smoke test (petite échelle) puis run complet ─────────────────────────────
# GROK_DEVICE=cpu : pour les tâches minuscules (p<=7, 3 à 15 exemples train),
# le lancement de noyau GPU domine le temps de calcul réel -- le CPU est plus
# rapide ET permet un vrai parallélisme (plusieurs processus OS concurrents)
# sans contention avec un run GPU déjà en cours (ex. le run p=17/25000 pas).
# Défaut inchangé (CUDA) pour ne rien casser des invocations existantes.
dev = get(ENV, "GROK_DEVICE", "cuda") == "cpu" ?
      NeuroDSL.Backend.CPUDevice() : NeuroDSL.Backend.CUDADevice()
nsx = :grok
Random.seed!(parse(Int, get(ENV, "GROK_SEED", "0")))
g, logits, mlp_out_sym, mlp_post_sym = build_grok_graph(dev, nsx)
pairs, labels = build_dataset()
train_idx, test_idx = train_test_split(pairs, labels; seed=598)
println("P=$P  |train|=$(length(train_idx))  |test|=$(length(test_idx))  budget=$N_STEPS pas (full-batch)")
println("Uniform loss (log P) = ", log(P))
flush(stdout)

t0 = time()
train_losses, test_losses = train_grok!(g, nsx, pairs, labels, train_idx, test_idx; n_steps=N_STEPS)
println("Temps d'entraînement : ", round(time() - t0, digits=1), " s")

# Sauvegarde pour l'analyse Fourier/perte-restreinte (script séparé).
ckpt = joinpath(@__DIR__, "grok_ckpt", "p$(P)_seed$(get(ENV, "GROK_SEED", "0"))")
mkpath(dirname(ckpt))
NeuroDSL.save_graph!(g, nsx, ckpt)
open(ckpt * "_losses.txt", "w") do io
    println(io, "train_losses,test_losses")
    for i in 1:length(test_losses)
        step = i * 20
        tl = train_losses[min(step, length(train_losses))]
        println(io, "$step,$tl,$(test_losses[i])")
    end
end
println("Checkpoint + courbes de perte sauvegardés : ", ckpt)
println("Perte finale : train=$(train_losses[end])  test=$(test_losses[end])")
