# ══════════════════════════════════════════════════════════════════════════════
# "Tâche du marqueur" -- protocole conçu par Fable pour tester si un sous-graphe
# causal minimal peut être isolé pour la fonction f de mise en correspondance
# entre deux formulations équivalentes d'une même question (par opposition au
# "contenu" partagé). Design complet et 4 prédictions falsifiables (P1-P4) --
# voir la discussion de session pour le formalisme (factorisation
# encodeur/cœur/décodeur, causal abstraction) et les 4 défauts du protocole
# naïf que ce design évite.
#
# Tâche : n paires (clé, valeur) mélangées dans le contexte, puis une requête à
# deux tokens [marqueur, token_clé] :
#   - format A (marqueur m_A) : token_clé = k directement -> réponse v(k)
#   - format B (marqueur m_B) : token_clé = sigma(k) (sigma = permutation FIXE
#     de 1..V, pas apprise par paire) -> vraie clé = sigma^{-1}(sigma(k)) = k
#     -> réponse v(k) (même réponse que le format A, T = identité)
# sigma(k) apparaît AUSSI comme clé ordinaire ailleurs dans le contexte -> sa
# signification dépend du marqueur, donc f ne peut pas être absorbée dans la
# table d'embedding seule (le piège qui rendrait le résultat trivial).
#
# ÉTAPE 1 (ce script) : construire la tâche, l'entraîner, vérifier P1 (le
# prérequis empirique -- le modèle doit réussir les deux formats, sinon rien
# de ce qui suit n'a de sens). L'analyse causale (P2-P4) est un script séparé,
# à écrire seulement si P1 passe.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random

# Paramétrable par ENV pour smoke-test vs run complet, sans éditer le fichier.
# Les défauts ci-dessous = config VALIDÉE le 2026-07-19 : P1 atteint avec
# acc_A = 1.0, acc_B = 1.0 (early-stop au pas 3500/5000, transition de phase
# entre les pas 3000 et 3250 après un long plateau à ~0.47 -- d'où lr CONSTANT
# après warmup (MARKER_DECAY=0) et batch 64 : la décroissance cosinus tuait le
# lr avant la transition, cause de l'échec des runs à 2500-6000 pas décayés).
# V=8/3 paires ≈ 1M combinaisons distinctes : non mémorisable par cœur, le
# modèle implémente donc le vrai algorithme (recherche associative + inversion
# conditionnelle de sigma) -- exactement ce que P2-P4 doivent disséquer.
const V = parse(Int, get(ENV, "MARKER_V", "8"))          # vocabulaire des clés/valeurs
const MARKER_A = V + 1
const MARKER_B = V + 2
const VOCAB_SIZE = V + 2
const N_PAIRS = parse(Int, get(ENV, "MARKER_PAIRS", "3"))
const SEQ_LEN = 2 * N_PAIRS + 2  # paires + [marqueur, token_clé]
const N_STEPS = parse(Int, get(ENV, "MARKER_STEPS", "5000"))   # pas d'OPTIMISEUR (chacun = BATCH séquences)
const BATCH   = parse(Int, get(ENV, "MARKER_BATCH", "64"))     # micro-batch par accumulation de gradients
const LR      = parse(Float32, get(ENV, "MARKER_LR", "1e-3"))

# sigma fixe, sans point fixe si possible (derangement), générée une fois pour
# toute la session (seed dédiée, indépendante du reste de l'expérience).
const SIGMA = let rng = MersenneTwister(42)
    perm = shuffle(rng, collect(1:V))
    # évite un point fixe accidentel (rare à V=12 mais on vérifie) -- sinon retire
    while any(perm[i] == i for i in 1:V)
        perm = shuffle(rng, collect(1:V))
    end
    perm
end
const SIGMA_INV = let inv = zeros(Int, V)
    for k in 1:V; inv[SIGMA[k]] = k; end
    inv
end

println("sigma = ", SIGMA)

function sample_marker_sequence(rng, fmt::Union{Nothing,Symbol}=nothing)
    keys = Int[]
    while length(keys) < N_PAIRS
        cand = rand(rng, 1:V)
        cand in keys || push!(keys, cand)
    end
    vals = [rand(rng, 1:V) for _ in 1:N_PAIRS]
    pair_order = shuffle(rng, 1:N_PAIRS)
    tokens = Int[]
    for i in pair_order
        push!(tokens, keys[i])
        push!(tokens, vals[i])
    end
    target_idx = rand(rng, 1:N_PAIRS)
    k = keys[target_idx]
    v = vals[target_idx]
    f = fmt === nothing ? rand(rng, (:A, :B)) : fmt
    if f == :A
        push!(tokens, MARKER_A)
        push!(tokens, k)
    else
        push!(tokens, MARKER_B)
        push!(tokens, SIGMA[k])
    end
    labels = vcat(tokens[2:end], [v])
    return tokens, labels, f, k, v
end

function build_marker_graph(dev, ns::Symbol; dim::Int, n_heads::Int, hidden_dim::Int, n_layers::Int)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.set!(g, :token_ids, ones(Int, SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, dim)(g, :token_ids, :tok; namespace=ns)
    pos_emb = NeuroDSL.Embedding(SEQ_LEN, dim)(g, :pos_ids, :pos; namespace=ns)
    x = :embed_sum
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(x, [tok_emb, pos_emb], :add; namespace=ns))
    out = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim; batched_attn=true)(g, x; namespace=ns)
    logits = NeuroDSL.Linear(dim, VOCAB_SIZE)(g, out, :lm_head; namespace=ns)
    # Perte restreinte à la POSITION FINALE seule (diagnostic : la CE moyennée
    # sur toute la séquence consacre 5/6 du gradient à des positions dont la
    # cible est non-prédictible par construction -- entropie irréductible
    # ~1.10 nat/position ; une fois les marginales apprises, ce gradient est
    # du bruit pur de moyenne nulle qui noie le signal de la position finale,
    # surtout à batch 1). Extraction de la dernière ligne des logits par une
    # matrice de sélection constante (Datom -> accum_grad! ignore son gradient)
    # via l'op :matmul existante -- aucun nouvel op ni masque nécessaire.
    sel = zeros(Float32, 1, SEQ_LEN); sel[1, SEQ_LEN] = 1f0
    NeuroDSL.set!(g, :sel_last, sel; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:final_logits, [:sel_last, logits], :matmul; namespace=ns))
    NeuroDSL.set!(g, :final_label, [1]; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:final_logits, :final_label], :cross_entropy; namespace=ns))
    return g, logits
end

# Diagnostic des tentatives 1-5 : batch efficace = 1 séquence/pas + lr=3e-3
# sans warmup -> gradient extrêmement bruité (plafonnement ~0.55-0.59) et
# divergence NaN à dim=128 (la CE est stable numériquement, max-shift vérifié
# dans src/kernels.jl -- l'instabilité venait de la dynamique d'optimisation).
# Correctifs : micro-batch par ACCUMULATION externe (backward_graph! remet les
# gradients de paramètres à zéro au début de chaque appel, src/backward.jl,
# donc l'accumulation inter-séquences doit vivre dans des tampons dédiés),
# moyenne, warmup linéaire, lr abaissé à 1e-3.
function train_marker!(g, ns; n_steps, batch=BATCH, seed=123, lr=LR, warmup=max(50, n_steps ÷ 20),
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
            tokens, _, _, _, v = sample_marker_sequence(rng)
            NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.set!(g, :final_label, [v]; atom_type=NeuroDSL.Datom, namespace=ns)
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
        # Warmup linéaire puis décroissance cosinus vers 0.1*lr (le bruit de
        # gradient à batch fini impose un plancher de loss à lr constant).
        # MARKER_DECAY=0 désactive la décroissance (lr constant après warmup) --
        # utile si la tâche a une transition de phase tardive que le cosinus tue.
        lr_t = if t <= warmup
            lr * Float32(t) / Float32(warmup)
        elseif get(ENV, "MARKER_DECAY", "0") == "0"
            lr
        else
            prog = Float32(t - warmup) / Float32(max(1, n_steps - warmup))
            lr * (0.1f0 + 0.45f0 * (1f0 + cos(pi * prog)))
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
            println("  [eval @ pas $t]  acc_A = ", r.acc_A, "   acc_B = ", r.acc_B)
            flush(stdout)
            if r.acc_A >= 0.97 && r.acc_B >= 0.97
                println("  Early-stop : les deux accuracies >= 0.97 au pas $t.")
                flush(stdout)
                break
            end
        end
    end
    return losses
end

function evaluate_marker(g, logits, ns; n_eval=200, seed=999)
    eval_rng = MersenneTwister(seed)
    acc = Dict(:A => (0, 0), :B => (0, 0))
    literal_reading = 0  # nb de fois où le format B répond v(sigma(k)) au lieu de v(k) (erreur systématique attendue si f échoue)
    for _ in 1:n_eval
        tokens, labels, fmt, k, v = sample_marker_sequence(eval_rng)
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        lg = Array(NeuroDSL.demand!(g, logits; namespace=ns))
        pred = argmax(lg[end, :])
        ok, tot = acc[fmt]
        acc[fmt] = (ok + (pred == v ? 1 : 0), tot + 1)
    end
    return (; acc_A = acc[:A][1]/acc[:A][2], acc_B = acc[:B][1]/acc[:B][2])
end

# ── Smoke test (petite échelle) puis run complet ─────────────────────────────
dev = NeuroDSL.Backend.CUDADevice()
ns = :marker_task
# Graine d'init OPT-IN (matrice de graines 2026-07-19) : MARKER_INIT_SEED
# contrôle l'init des poids. Attention : sur CUDADevice, Backend.rand32 tire
# de CUDA.rand (RNG CUDA propre, src/backend.jl:16), PAS du RNG Julia global
# -- il faut donc seeder les DEUX. CUDA.seed! n'est appelé que si la variable
# est explicitement posée (le chemin par défaut historique reste inchangé,
# non-reproductible côté GPU comme avant).
const INIT_SEED = parse(Int, get(ENV, "MARKER_INIT_SEED", "1"))
Random.seed!(INIT_SEED)
haskey(ENV, "MARKER_INIT_SEED") && NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.seed!(INIT_SEED)
const N_LAYERS = parse(Int, get(ENV, "MARKER_LAYERS", "4"))
const DIM = parse(Int, get(ENV, "MARKER_DIM", "64"))
g, logits = build_marker_graph(dev, ns; dim=DIM, n_heads=4, hidden_dim=2*DIM, n_layers=N_LAYERS)

println("Séquence exemple : ", sample_marker_sequence(MersenneTwister(7)))

println("="^70)
println("Config : V=$V N_PAIRS=$N_PAIRS | $N_STEPS pas d'optimiseur x batch $BATCH | lr=$LR")
flush(stdout)
t0 = time()
losses = train_marker!(g, ns; n_steps=N_STEPS,
                       seed=parse(Int, get(ENV, "MARKER_TRAIN_SEED", "123")),  # opt-in, défaut historique
                       evalcb=() -> evaluate_marker(g, logits, ns; n_eval=200))
println("Temps d'entraînement : ", round(time() - t0, digits=1), " s")
let w = max(1, min(100, N_STEPS ÷ 10)), n_done = length(losses)
    for chunk in unique(clamp.(round.(Int, N_STEPS .* [0.05, 0.125, 0.25, 0.5, 0.75, 1.0]), 1, n_done))
        lo = max(1, chunk - w + 1)
        println("  loss moyenne (position finale) des pas $lo-$chunk : ", sum(losses[lo:chunk])/(chunk-lo+1))
    end
end

result = evaluate_marker(g, logits, ns; n_eval=400)
println("P1 -- acc_A = ", result.acc_A, "   acc_B = ", result.acc_B)
println("Seuil requis (P1) : les deux >= 0.95")

# Checkpoint OPT-IN (2026-07-19) : MARKER_SAVE=<path_prefix> sauvegarde le
# modèle entraîné via save_graph! (feuilles + règles, src/serialization.jl:83)
# une fois le gate P1 passé -- AUCUN effet sans la variable d'ENV, les défauts
# figés de ce script restent inchangés. MARKER_SMOKE=1 force la sauvegarde
# même sans P1 (test du chemin de code sur modèle non entraîné).
if get(ENV, "MARKER_SAVE", "") != ""
    if (result.acc_A >= 0.95 && result.acc_B >= 0.95) || get(ENV, "MARKER_SMOKE", "0") == "1"
        let ckpt = ENV["MARKER_SAVE"]
            isempty(dirname(ckpt)) || mkpath(dirname(ckpt))
            NeuroDSL.save_graph!(g, ns, ckpt)
            println("Checkpoint sauvegardé : ", ckpt, ".json/.bin")
        end
    else
        println("MARKER_SAVE demandé mais P1 non atteint -- pas de sauvegarde.")
    end
end
