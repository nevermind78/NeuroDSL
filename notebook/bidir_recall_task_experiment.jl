# ══════════════════════════════════════════════════════════════════════════════
# "Tâche de rappel bidirectionnel" -- SECOND banc, INDÉPENDANT de la "tâche du
# marqueur" (notebook/marker_task_experiment.jl), construit pour fournir à
# artilce/code4.tex une réplication sur une tâche ET une architecture
# GENUINEMENT différentes (demande explicite de l'utilisateur/coordinateur :
# pas un simple rejeu de la même tâche avec de nouvelles graines).
#
# CE QUI DIFFÈRE DE LA TÂCHE DU MARQUEUR (pas une nouvelle graine du même
# mécanisme) :
#   - Mécanisme conditionnel : la tâche du marqueur applique une PERMUTATION
#     EXTERNE FIXE sigma au token affiché puis relit dans le MÊME sens
#     (clé -> valeur) ; il n'y a ici NI permutation NI table apprise. Le
#     marqueur choisit à la place le SENS de lecture d'une correspondance
#     clé<->valeur déjà présente dans le contexte : en avant (clé affichée ->
#     retrouver sa valeur, rappel associatif standard) ou en arrière (valeur
#     affichée -> retrouver la clé qui lui correspond, un rappel INVERSE que
#     rien dans la tâche du marqueur n'exige jamais -- toutes ses requêtes
#     remontent dans le même sens clé->valeur).
#   - Vocabulaire : V=12 (contenu) + 2 marqueurs = 14, contre V=8+2=10.
#   - Architecture : dim=48, 3 têtes (d_head=16), hidden=96, 3 couches, contre
#     dim=64, 4 têtes, hidden=128, 4 couches.
#   - Nombre de paires : 4 (contre 3).
#
# L'axe de contraste partagé (l'analogue de "littéral" / "inversé" via sigma)
# est ici directement les deux tokens de la paire-cible (k_t, v_t) : le
# rappel AVANT doit favoriser v_t (la bonne réponse) sur k_t (copier
# tel-quel le token affiché -- l'erreur "sans op" typique d'un circuit
# d'induction qui échoue à chercher) ; le rappel ARRIÈRE doit favoriser k_t
# sur v_t, exactement symétrique. Un seul sop(x) := logit(v_t) - logit(k_t)
# sert aux deux formats, avec la polarité requise opposée -- structure
# analogue à sop(x)=logit(lit)-logit(inv) dans marker_task, mais sans aucune
# permutation à apprendre.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random

const V2 = parse(Int, get(ENV, "BIDIR_V", "12"))
const MARKER_FWD = V2 + 1
const MARKER_BWD = V2 + 2
const VOCAB_SIZE2 = V2 + 2
const N_PAIRS2 = parse(Int, get(ENV, "BIDIR_PAIRS", "4"))
const SEQ_LEN2 = 2 * N_PAIRS2 + 2
const N_STEPS2 = parse(Int, get(ENV, "BIDIR_STEPS", "6000"))
const BATCH2   = parse(Int, get(ENV, "BIDIR_BATCH", "64"))
const LR2      = parse(Float32, get(ENV, "BIDIR_LR", "1e-3"))
# Constantes d'architecture -- délibérément HORS du bloc `if` d'entrée
# principal ci-dessous, pour que les scripts qui `include`nt ce fichier (les
# scripts de vérification, qui n'entraînent rien) les voient aussi.
const N_LAYERS2 = parse(Int, get(ENV, "BIDIR_LAYERS", "3"))
const DIM2      = parse(Int, get(ENV, "BIDIR_DIM", "48"))
const N_HEADS2  = parse(Int, get(ENV, "BIDIR_HEADS", "3"))

function distinct_sample(rng, vmax::Int, n::Int)
    out = Int[]
    while length(out) < n
        c = rand(rng, 1:vmax)
        c in out || push!(out, c)
    end
    return out
end

"""
    sample_bidir_sequence(rng, fmt=nothing) -> (tokens, labels, fmt, k_t, v_t)

`N_PAIRS2` paires (clé,valeur) -- clés distinctes entre elles, valeurs
distinctes entre elles (indépendamment) -- mélangées dans le contexte, puis
une requête [marqueur, token]. Format `:F` (avant, marqueur `MARKER_FWD`) :
token affiché = la clé `k_t` d'une paire cible tirée au hasard, réponse = sa
valeur `v_t`. Format `:R` (arrière, marqueur `MARKER_BWD`) : token affiché =
`v_t`, réponse = `k_t` -- rappel inverse, jamais requis par le format avant.
"""
function sample_bidir_sequence(rng, fmt::Union{Nothing,Symbol}=nothing)
    local keys, vals, t, k_t, v_t
    while true
        keys = distinct_sample(rng, V2, N_PAIRS2)
        vals = distinct_sample(rng, V2, N_PAIRS2)
        t = rand(rng, 1:N_PAIRS2)
        k_t, v_t = keys[t], vals[t]
        k_t != v_t && break  # évite la paire-cible dégénérée k_t==v_t (le
        # contraste sop=logit(v_t)-logit(k_t) serait alors identiquement nul
        # -- ni un format ni l'autre n'aurait de marge définie)
    end
    pair_order = shuffle(rng, 1:N_PAIRS2)
    tokens = Int[]
    for i in pair_order
        push!(tokens, keys[i]); push!(tokens, vals[i])
    end
    f = fmt === nothing ? rand(rng, (:F, :R)) : fmt
    label = f == :F ? v_t : k_t
    if f == :F
        push!(tokens, MARKER_FWD); push!(tokens, k_t)
    else
        push!(tokens, MARKER_BWD); push!(tokens, v_t)
    end
    labels = vcat(tokens[2:end], [label])
    return tokens, labels, f, k_t, v_t
end

function build_bidir_graph(dev, ns::Symbol; dim::Int, n_heads::Int, hidden_dim::Int, n_layers::Int)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.set!(g, :token_ids, ones(Int, SEQ_LEN2); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN2); atom_type=NeuroDSL.Datom, namespace=ns)
    tok_emb = NeuroDSL.Embedding(VOCAB_SIZE2, dim)(g, :token_ids, :tok; namespace=ns)
    pos_emb = NeuroDSL.Embedding(SEQ_LEN2, dim)(g, :pos_ids, :pos; namespace=ns)
    x = :embed_sum
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(x, [tok_emb, pos_emb], :add; namespace=ns))
    out = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim; batched_attn=true)(g, x; namespace=ns)
    logits = NeuroDSL.Linear(dim, VOCAB_SIZE2)(g, out, :lm_head; namespace=ns)
    sel = zeros(Float32, 1, SEQ_LEN2); sel[1, SEQ_LEN2] = 1f0
    NeuroDSL.set!(g, :sel_last, sel; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:final_logits, [:sel_last, logits], :matmul; namespace=ns))
    NeuroDSL.set!(g, :final_label, [1]; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:final_logits, :final_label], :cross_entropy; namespace=ns))
    return g, logits
end

function train_bidir!(g, ns; n_steps, batch=BATCH2, seed=123, lr=LR2, warmup=max(50, n_steps ÷ 20),
                       evalcb=nothing, eval_every=250)
    dev = g.device
    ps = NeuroDSL.params(g; namespace=ns)
    m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    accs = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    rng = MersenneTwister(seed)
    losses = Float64[]
    for t in 1:n_steps
        for a in accs; fill!(a, 0f0); end
        step_loss = 0.0
        for _ in 1:batch
            tokens, labels, _, _, _ = sample_bidir_sequence(rng)
            NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN2); atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.set!(g, :final_label, [labels[end]]; atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.invalidate_all!(g; namespace=ns)
            loss_val = NeuroDSL.demand!(g, :loss; namespace=ns)
            step_loss += Float64(sum(Array(loss_val)))
            NeuroDSL.backward_graph!(g, :loss; namespace=ns)
            for (i, p) in enumerate(ps)
                p.gradient === nothing && continue
                accs[i] .+= p.gradient
            end
        end
        push!(losses, step_loss / batch)
        lr_t = if t <= warmup
            lr * Float32(t) / Float32(warmup)
        else
            lr  # décroissance désactivée par défaut, comme MARKER_DECAY=0
        end
        for (i, p) in enumerate(ps)
            p.gradient === nothing && continue
            p.gradient .= accs[i] ./ Float32(batch)
            NeuroDSL.adamw_step!(dev, p.value, p.gradient, m1s[i], m2s[i], lr_t, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
        end
        NeuroDSL.invalidate_all!(g; namespace=ns)
        if t % 100 == 0
            println("  pas $t/$n_steps  loss(100 derniers) = ", round(sum(losses[t-99:t])/100, digits=4))
            flush(stdout)
        end
        if evalcb !== nothing && t % eval_every == 0
            r = evalcb()
            println("  [eval @ pas $t]  acc_F = ", r.acc_F, "   acc_R = ", r.acc_R)
            flush(stdout)
            if r.acc_F >= 0.97 && r.acc_R >= 0.97
                println("  Early-stop : les deux accuracies >= 0.97 au pas $t.")
                flush(stdout)
                break
            end
        end
    end
    return losses
end

function evaluate_bidir(g, logits, ns; n_eval=200, seed=999)
    eval_rng = MersenneTwister(seed)
    acc = Dict(:F => (0, 0), :R => (0, 0))
    for _ in 1:n_eval
        tokens, labels, fmt, k_t, v_t = sample_bidir_sequence(eval_rng)
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN2); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        lg = Array(NeuroDSL.demand!(g, logits; namespace=ns))
        pred = argmax(lg[end, :])
        target = labels[end]
        ok, tot = acc[fmt]
        acc[fmt] = (ok + (pred == target ? 1 : 0), tot + 1)
    end
    return (; acc_F = acc[:F][1]/acc[:F][2], acc_R = acc[:R][1]/acc[:R][2])
end

if abspath(PROGRAM_FILE) == @__FILE__
    dev = NeuroDSL.Backend.CUDADevice()
    ns = :bidir_task
    const INIT_SEED2 = parse(Int, get(ENV, "BIDIR_INIT_SEED", "1"))
    Random.seed!(INIT_SEED2)
    NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.seed!(INIT_SEED2)
    g, logits = build_bidir_graph(dev, ns; dim=DIM2, n_heads=N_HEADS2, hidden_dim=2*DIM2, n_layers=N_LAYERS2)

    println("Séquence exemple : ", sample_bidir_sequence(MersenneTwister(7)))
    println("="^70)
    println("Config bidir : V=$V2 N_PAIRS=$N_PAIRS2 dim=$DIM2 n_heads=$N_HEADS2 n_layers=$N_LAYERS2 | $N_STEPS2 pas x batch $BATCH2 | lr=$LR2")
    flush(stdout)
    t0 = time()
    losses = train_bidir!(g, ns; n_steps=N_STEPS2,
                           seed=parse(Int, get(ENV, "BIDIR_TRAIN_SEED", "123")),
                           evalcb=() -> evaluate_bidir(g, logits, ns; n_eval=200))
    println("Temps d'entraînement : ", round(time() - t0, digits=1), " s")

    result = evaluate_bidir(g, logits, ns; n_eval=400)
    println("P1(bidir) -- acc_F = ", result.acc_F, "   acc_R = ", result.acc_R)
    println("Seuil requis : les deux >= 0.95")

    if get(ENV, "BIDIR_SAVE", "") != ""
        if (result.acc_F >= 0.95 && result.acc_R >= 0.95) || get(ENV, "BIDIR_SAVE_ALWAYS", "0") == "1"
            ckpt = ENV["BIDIR_SAVE"]
            isempty(dirname(ckpt)) || mkpath(dirname(ckpt))
            NeuroDSL.save_graph!(g, ns, ckpt)
            println("Checkpoint sauvegardé : ", ckpt, ".json/.bin")
        else
            println("BIDIR_SAVE demandé mais seuil non atteint -- pas de sauvegarde.")
        end
    end
end
