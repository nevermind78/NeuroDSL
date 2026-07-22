# ══════════════════════════════════════════════════════════════════════════════
# "Tâche du marqueur" -- P4-bis : ablation de poids AFFINÉE.
#
# Motivation (run du 2026-07-19, notebook/marker_task_causal_analysis.log) :
# P3 validé (circuit d'activation = [:layer_1_mlp_out], r médian 1.043,
# bascule 0.85 sur 20 paires) mais P4 non validé (zero-ablation de mlp_w2
# entier : acc_A 0.857, littéral_B 0.423) -- AVEC deux artefacts identifiés :
#   (1) le témoin (layer_1_mha_ao_h1) n'était PAS neutre (r individuel 1.024
#       sur la paire 1 -- c'est un carrier redondant du même drapeau) ;
#   (2) zéroter mlp_w2 ENTIER supprime tout le MLP de couche 1 à toutes les
#       positions (contenu inclus) et envoie l'aval hors distribution --
#       bien plus brutal que le patch d'activation qui a validé P3.
# Observation clé du run : post-ablation, correct+littéral = 0.993 (le TYPE
# d'erreur prédit réussit, seul le TAUX échoue) -> hypothèse de redondance :
# le drapeau de format est une direction ~linéaire du flux résiduel de
# couche 1, alimentée additivement par plusieurs composants (mlp_out + têtes).
#
# Protocole P4-bis (échelle d'escalade, chaque étape ~2-3 min sur checkpoint) :
#   B   -- baseline (checkpoint tel quel).
#   C   -- témoin PROPRE : tête de couche 4 avec r≈0 dans le sweep individuel
#          (neutralité vérifiée AU PATCH avant l'ablation), zero-ablation ->
#          attendu ≈ baseline. Requalifie le témoin invalide du run précédent.
#   M   -- mean-ablation de layer_1_mlp_out SEUL : sortie du composant forcée
#          à sa MOYENNE sur la distribution de la tâche (activation constante,
#          indépendante de l'entrée -- standard interp, évite l'artefact
#          hors-distribution du zérotage). Niveau activation mais
#          input-indépendant : le composant ne peut plus calculer f.
#   D1  -- ablation DIRECTIONNELLE de mlp_w2 seul : sous-espace de format U
#          estimé par SVD des deltas donneur-receveur (positions marqueur +
#          requête, k composantes pour >= 90% d'énergie, cap 8), replié dans
#          les poids : mlp_w2 <- (I - UU')·mlp_w2. Vraie édition de poids,
#          bas rang, indépendante de la position.
#   DJ  -- D1 + projection directionnelle des têtes carriers de COUCHE 1
#          (dans l'espace d'activation de chaque tête, replié dans les
#          colonnes correspondantes de output_W).
#   DJA -- (si DJ échoue) directionnelle jointe sur TOUS les carriers
#          (toutes couches, seuil r moyen >= 0.25 au sweep).
#   MJ  -- (si DJA échoue) mean-ablation jointe de tous les carriers --
#          diagnostic final : sépare "estimation de direction insuffisante"
#          de "mécanisme distribué au-delà des carriers identifiés".
#
# Prédiction falsifiable INCHANGÉE : une config directionnelle valide P4-bis
# ssi littéral_B >= 0.9 ET acc_A >= 0.9 (quasi intacte).
#
# Gestion du checkpoint : si <ckpt>.json/.bin existent -> chargement
# (load_graph! overwrite=true, src/serialization.jl:166) et l'entraînement de
# l'include est réduit à 1 pas symbolique ; sinon -> entraînement complet via
# marker_task_experiment.jl (config gagnante par défaut) + sauvegarde opt-in
# (MARKER_SAVE) après le gate P1.
#
# Usage :
#   smoke (2 passes recommandées : sauvegarde puis rechargement) :
#     MARKER_SMOKE=1 MARKER_CKPT=<scratch>/ck julia --project=. notebook/marker_task_p4bis.jl
#   run complet (1er lancement ~75 min ; suivants ~3-5 min) :
#     julia --project=. notebook/marker_task_p4bis.jl
# ══════════════════════════════════════════════════════════════════════════════

const SMOKE = get(ENV, "MARKER_SMOKE", "0") == "1"
const CKPT  = get(ENV, "MARKER_CKPT", joinpath(@__DIR__, "marker_ckpt", "marker_task"))
const HAVE_CKPT = isfile(CKPT * ".json") && isfile(CKPT * ".bin")

if HAVE_CKPT
    # L'include va "entraîner" 1 pas symbolique puis on écrase par le checkpoint.
    ENV["MARKER_STEPS"] = "1"
    ENV["MARKER_BATCH"] = "2"
else
    ENV["MARKER_SAVE"] = CKPT
    if SMOKE
        get!(ENV, "MARKER_STEPS", "40")
        get!(ENV, "MARKER_BATCH", "8")
    end
end

include(joinpath(@__DIR__, "marker_task_experiment.jl"))

using LinearAlgebra   # svd, I -- stdlib, déjà dans Project.toml

if HAVE_CKPT
    println("\nChargement du checkpoint : ", CKPT)
    NeuroDSL.load_graph!(g, ns, CKPT; overwrite=true)
end

# ── Gate P1 sur le modèle courant (chargé ou fraîchement entraîné) ───────────
resP1 = evaluate_marker(g, logits, ns; n_eval=400)
println("\n", "═"^70)
println("P1 (modèle courant) : acc_A = ", resP1.acc_A, "   acc_B = ", resP1.acc_B)
if SMOKE
    println("[SMOKE] gate P1 ignoré.")
elseif !(resP1.acc_A >= 0.95 && resP1.acc_B >= 0.95)
    error("P1 non satisfait sur le modèle courant -- arrêt.")
end
flush(stdout)

# ── Helpers (mêmes définitions que marker_task_causal_analysis.jl) ───────────
const N_HEADS = 4
const D_HEAD  = DIM ÷ N_HEADS
const N_CTX   = 2 * N_PAIRS

head_site(l, h) = Symbol("layer_$(l)_mha_ao_h$(h)")
mlp_site(l)     = Symbol("layer_$(l)_mlp_out")

const CANDIDATES = Symbol[]
for l in 1:N_LAYERS
    for h in 1:N_HEADS; push!(CANDIDATES, head_site(l, h)); end
    push!(CANDIDATES, mlp_site(l))
end
for s in CANDIDATES
    haskey(g.nodes[ns], s) || error("Nœud candidat $s introuvable")
end

function run_forward!(g, ns, tokens; const_patches=nothing)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    if const_patches !== nothing
        # patch_nodes! trie topologiquement (src/patching.jl:393) -- sûr même
        # avec plusieurs sites en relation ancêtre-descendant.
        NeuroDSL.patch_nodes!(g, collect(keys(const_patches)), const_patches; namespace=ns)
    end
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))  # (1, VOCAB)
end

function run_and_capture!(g, ns, tokens)
    out = run_forward!(g, ns, tokens)
    return out, NeuroDSL.capture_activations(g, ns)
end

function full_reset!(g, ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, :final_logits; namespace=ns)
    return g
end

function sample_contrast(rng)
    while true
        k  = rand(rng, 1:V)
        sk = SIGMA[k]
        others = collect(setdiff(1:V, (k, sk)))
        rest = length(others) >= N_PAIRS - 2 ? shuffle(rng, others)[1:N_PAIRS-2] : Int[]
        ks = vcat([k, sk], rest)
        vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS)
            push!(ctx, ks[i]); push!(ctx, vs[i])
        end
        return (; ctx, k, sk, vk = vs[1], vsk = vs[2])
    end
end

donor_tokens(c)    = vcat(c.ctx, [MARKER_A, c.sk])
receiver_tokens(c) = vcat(c.ctx, [MARKER_B, c.sk])
delta_logit(out, c) = Float64(out[1, c.vsk] - out[1, c.vk])

function draw_clean_pair!(g, ns, rng; max_tries=200)
    for _ in 1:max_tries
        c = sample_contrast(rng)
        d_out = run_forward!(g, ns, donor_tokens(c))
        r_out = run_forward!(g, ns, receiver_tokens(c))
        dd, dr = delta_logit(d_out, c), delta_logit(r_out, c)
        if SMOKE
            abs(dd - dr) > 1e-6 && return c
        else
            (argmax(vec(d_out)) == c.vsk && argmax(vec(r_out)) == c.vk && dd > 0 > dr) && return c
        end
    end
    error("Pas de paire de contraste exploitable en $max_tries essais")
end

# ══════════════════════════════════════════════════════════════════════════════
# Étape 1 -- sweep individuel : r moyen par site -> carriers + témoin neutre
# ══════════════════════════════════════════════════════════════════════════════
println("\n", "═"^70)
println("Étape 1 -- sweep individuel (r moyen par site)")
flush(stdout)

function single_site_sweep!(g, ns; n_pairs, seed)
    rng = MersenneTwister(seed)
    sums = Dict{Symbol,Float64}(s => 0.0 for s in CANDIDATES)
    for _ in 1:n_pairs
        c = draw_clean_pair!(g, ns, rng)
        d_out, dcache = run_and_capture!(g, ns, donor_tokens(c))
        r_out, rcache = run_and_capture!(g, ns, receiver_tokens(c))
        dd, dr = delta_logit(d_out, c), delta_logit(r_out, c)
        for s in CANDIDATES
            NeuroDSL.patch_node!(g, s, dcache; namespace=ns)
            out = Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
            sums[s] += (delta_logit(out, c) - dr) / (dd - dr)
            NeuroDSL.patch_node!(g, s, rcache; namespace=ns)
            NeuroDSL.demand!(g, :final_logits; namespace=ns)
        end
        full_reset!(g, ns)
    end
    return Dict{Symbol,Float64}(s => sums[s] / n_pairs for s in CANDIDATES)
end

const N_SWEEP = SMOKE ? 2 : 8
MEAN_R = single_site_sweep!(g, ns; n_pairs=N_SWEEP, seed=4242)
for l in 1:N_LAYERS
    println("  layer_$l : ", [(String(s), round(MEAN_R[s], digits=3))
                              for s in vcat([head_site(l,h) for h in 1:N_HEADS], [mlp_site(l)])])
end

const CARRIER_THRESHOLD = 0.25
CARRIERS    = sort([s for s in CANDIDATES if MEAN_R[s] >= CARRIER_THRESHOLD]; by=s -> -MEAN_R[s])
CARRIERS_L1 = [s for s in CARRIERS if startswith(String(s), "layer_1_")]
println("  Carriers (r moyen >= $CARRIER_THRESHOLD)  : ", [(String(s), round(MEAN_R[s], digits=3)) for s in CARRIERS])
println("  Carriers de couche 1                : ", CARRIERS_L1)

# Témoin : tête de couche 4 de |r| minimal, neutralité exigée au patch.
CONTROL_SITE = let l4 = [head_site(4, h) for h in 1:N_HEADS]
    l4[argmin([abs(MEAN_R[s]) for s in l4])]
end
println("  Témoin choisi : ", CONTROL_SITE, "  (r moyen au patch = ", round(MEAN_R[CONTROL_SITE], digits=4), ")")
CONTROL_NEUTRAL = abs(MEAN_R[CONTROL_SITE]) < 0.1
CONTROL_NEUTRAL || println("  ⚠ |r| >= 0.1 : témoin imparfaitement neutre -- à interpréter avec prudence.")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════════════
# Étape 2 -- deltas donneur-receveur (positions marqueur + requête) et
#            sous-espaces de format par carrier (SVD)
# ══════════════════════════════════════════════════════════════════════════════
println("\n", "═"^70)
println("Étape 2 -- estimation des sous-espaces de format (SVD des deltas)")
flush(stdout)

function collect_deltas!(g, ns, sites; n_pairs, seed)
    rng = MersenneTwister(seed)
    deltas = Dict{Symbol,Vector{Vector{Float64}}}(s => Vector{Vector{Float64}}() for s in sites)
    for _ in 1:n_pairs
        c = draw_clean_pair!(g, ns, rng)
        _, dcache = run_and_capture!(g, ns, donor_tokens(c))
        _, rcache = run_and_capture!(g, ns, receiver_tokens(c))
        for s in sites
            da = Array(dcache[s]); ra = Array(rcache[s])
            for row in (SEQ_LEN - 1, SEQ_LEN)   # position du marqueur + position de la requête
                push!(deltas[s], vec(Float64.(da[row, :] .- ra[row, :])))
            end
        end
    end
    return deltas
end

# k = plus petit nombre de composantes couvrant >= `energy` de l'énergie
# (borné par kcap) ; retourne (U, k, énergie couverte, alignement moyen).
function format_subspace(delta_vecs; energy=0.90, kcap::Int)
    X = reduce(hcat, delta_vecs)                       # d × n
    F = svd(X)
    e = F.S .^ 2
    ce = cumsum(e) ./ sum(e)
    k = min(something(findfirst(>=(energy), ce), length(ce)), kcap)
    U = F.U[:, 1:k]
    # Alignement : fraction moyenne de la norme de chaque delta capturée par U.
    align = sum(norm(U' * v) / (norm(v) + 1e-12) for v in delta_vecs) / length(delta_vecs)
    return U, k, ce[k], align
end

const N_DIR_PAIRS = SMOKE ? 4 : 32
ALL_DIR_SITES = union(CARRIERS, [mlp_site(1)])   # mlp_site(1) toujours inclus (circuit P3)
DELTAS = collect_deltas!(g, ns, collect(ALL_DIR_SITES); n_pairs=N_DIR_PAIRS, seed=1618)

SUBSPACES = Dict{Symbol,Any}()
for s in collect(ALL_DIR_SITES)
    kcap = endswith(String(s), "_mlp_out") ? 8 : 4
    U, k, en, align = format_subspace(DELTAS[s]; kcap=kcap)
    SUBSPACES[s] = (; U, k, en, align)
    println("  $s : k = $k, énergie couverte = ", round(en, digits=3),
            ", alignement moyen des deltas = ", round(align, digits=3))
end
flush(stdout)

# ══════════════════════════════════════════════════════════════════════════════
# Étape 3 -- constructions d'ablation
# ══════════════════════════════════════════════════════════════════════════════

_current_W(wsym) = Matrix{Float32}(Array(NeuroDSL.node(g, wsym; namespace=ns).value))

# Zero-ablation d'une tête (comme le P4 initial, pour le témoin).
function zero_head_weights(site)
    m = match(r"^layer_(\d+)_mha_ao_h(\d+)$", String(site))
    m === nothing && error("zero_head_weights : $site n'est pas une tête")
    l = parse(Int, m.captures[1]); h = parse(Int, m.captures[2])
    wsym = Symbol("layer_$(l)_mha_output_W")
    W = _current_W(wsym)
    W[:, (h-1)*D_HEAD+1 : h*D_HEAD] .= 0f0
    return Dict{Symbol,Matrix{Float32}}(wsym => W)
end

# Ablation directionnelle repliée dans les poids :
#  - site MLP  : mlp_out = sg·W2' -> projeter l'ESPACE DE SORTIE :
#                W2 <- (I - UU')·W2   (U ∈ R^{dim×k}).
#  - site tête : contribution = ao_h·W[:,cols]'  -> projeter l'espace
#                d'activation de la tête : W[:,cols] <- W[:,cols]·(I - U_h U_h')
#                (U_h ∈ R^{d_head×k}, (I-UU') symétrique donc équivaut à
#                projeter ao_h avant lecture).
function directional_weights(sites)
    w = Dict{Symbol,Matrix{Float32}}()
    for s in sites
        U = Float32.(SUBSPACES[s].U)
        m = match(r"^layer_(\d+)_mha_ao_h(\d+)$", String(s))
        if m !== nothing
            l = parse(Int, m.captures[1]); h = parse(Int, m.captures[2])
            wsym = Symbol("layer_$(l)_mha_output_W")
            W = get!(() -> _current_W(wsym), w, wsym)   # accumule si plusieurs têtes/couche
            cols = (h-1)*D_HEAD+1 : h*D_HEAD
            P = Matrix{Float32}(I, D_HEAD, D_HEAD) .- U * U'
            W[:, cols] = W[:, cols] * P
        else
            m2 = match(r"^layer_(\d+)_mlp_out$", String(s))
            m2 === nothing && error("directional_weights : site inattendu $s")
            wsym = Symbol("layer_$(m2.captures[1])_mlp_w2")
            P = Matrix{Float32}(I, DIM, DIM) .- U * U'
            w[wsym] = P * _current_W(wsym)
        end
    end
    return w
end

# Activation moyenne (sur la distribution d'entraînement, formats mélangés).
function mean_activations!(g, ns, sites; n, seed)
    rng = MersenneTwister(seed)
    acc = Dict{Symbol,Any}(s => nothing for s in sites)
    for _ in 1:n
        tokens, _, _, _, _ = sample_marker_sequence(rng)
        run_forward!(g, ns, tokens)
        for s in sites
            v = Float64.(Array(NeuroDSL.node(g, s; namespace=ns).value))
            acc[s] = acc[s] === nothing ? v : acc[s] .+ v
        end
    end
    return Dict{Symbol,Any}(s => Float32.(acc[s] ./ n) for s in sites)
end

# ══════════════════════════════════════════════════════════════════════════════
# Étape 4 -- évaluation des configurations (échelle d'escalade)
# ══════════════════════════════════════════════════════════════════════════════
println("\n", "═"^70)
println("Étape 4 -- évaluation des configurations")
println("  Prédiction falsifiable (validation P4-bis) : littéral_B >= 0.9 ET acc_A >= 0.9")
flush(stdout)

function eval_condition(g, ns; n, seed, const_patches=nothing)
    rngA = MersenneTwister(seed)
    okA = 0
    for _ in 1:n
        tokens, _, _, _, v = sample_marker_sequence(rngA, :A)
        okA += (argmax(vec(run_forward!(g, ns, tokens; const_patches=const_patches))) == v) ? 1 : 0
    end
    rngB = MersenneTwister(seed + 1)
    okB = 0; lit = 0
    for _ in 1:n
        c = sample_contrast(rngB)
        pred = argmax(vec(run_forward!(g, ns, receiver_tokens(c); const_patches=const_patches)))
        okB += (pred == c.vk)  ? 1 : 0
        lit += (pred == c.vsk) ? 1 : 0
    end
    return (; acc_A = okA / n, acc_B = okB / n, literal_B = lit / n)
end

const N_EVAL = SMOKE ? 30 : 300
RESULTS = Dict{String,Any}()

function run_config!(name, desc; weights=nothing, const_patches=nothing)
    if weights !== nothing
        restore = Dict{Symbol,Matrix{Float32}}(wsym => _current_W(wsym) for wsym in keys(weights))
        NeuroDSL.set_params!(g, ns, weights)
        r = eval_condition(g, ns; n=N_EVAL, seed=555, const_patches=const_patches)
        NeuroDSL.set_params!(g, ns, restore)
    else
        r = eval_condition(g, ns; n=N_EVAL, seed=555, const_patches=const_patches)
    end
    full_reset!(g, ns)
    RESULTS[name] = r
    pass = r.acc_A >= 0.9 && r.literal_B >= 0.9
    println("  [$name] $desc")
    println("        acc_A = $(r.acc_A)  acc_B = $(r.acc_B)  littéral_B = $(r.literal_B)",
            "  correct+littéral = ", round(r.acc_B + r.literal_B, digits=3),
            name in ("B", "C") ? "" : (pass ? "   << PRÉDICTION VALIDÉE" : ""))
    flush(stdout)
    return r
end

# B -- baseline
base = run_config!("B", "baseline (checkpoint tel quel)")

# C -- témoin propre (zero-ablation, comme le P4 initial, mais site neutre)
ctrl = run_config!("C", "témoin propre : zero-ablation de $CONTROL_SITE";
                   weights=zero_head_weights(CONTROL_SITE))
ctrl_ok = abs(ctrl.acc_A - base.acc_A) <= 0.05 && abs(ctrl.acc_B - base.acc_B) <= 0.05
println("        Témoin neutre à l'ablation (|Δacc| <= 0.05 vs baseline) : ", ctrl_ok ? "OUI" : "NON")

# M -- mean-ablation de layer_1_mlp_out seul
const MEAN_ACTS_MLP = mean_activations!(g, ns, [mlp_site(1)]; n=(SMOKE ? 20 : 200), seed=777)
mres = run_config!("M", "mean-ablation de layer_1_mlp_out seul (activation constante)";
                   const_patches=MEAN_ACTS_MLP)

# D1 -- directionnelle mlp_w2 seul
d1 = run_config!("D1", "directionnelle : mlp_w2 <- (I-UU')·mlp_w2 (k=$(SUBSPACES[mlp_site(1)].k))";
                 weights=directional_weights([mlp_site(1)]))

# DJ -- directionnelle jointe couche 1
dj = run_config!("DJ", "directionnelle JOINTE couche 1 : $(CARRIERS_L1)";
                 weights=directional_weights(CARRIERS_L1))

# Escalade conditionnelle
dja = nothing
mj  = nothing
if !(dj.acc_A >= 0.9 && dj.literal_B >= 0.9)
    if length(CARRIERS) > length(CARRIERS_L1)
        global dja = run_config!("DJA", "directionnelle JOINTE tous carriers : $(CARRIERS)";
                                 weights=directional_weights(CARRIERS))
    else
        println("  [DJA] sans objet (tous les carriers sont en couche 1)")
    end
    last_dir = dja === nothing ? dj : dja
    if !(last_dir.acc_A >= 0.9 && last_dir.literal_B >= 0.9)
        mean_all = mean_activations!(g, ns, collect(union(CARRIERS, [mlp_site(1)]));
                                     n=(SMOKE ? 20 : 200), seed=777)
        global mj = run_config!("MJ", "mean-ablation JOINTE de tous les carriers (diagnostic)";
                                const_patches=mean_all)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Verdict
# ══════════════════════════════════════════════════════════════════════════════
println("\n", "═"^70)
println("VERDICT P4-BIS ", SMOKE ? "[SMOKE -- chiffres NON significatifs]" : "")
println("  P1 : acc_A = $(resP1.acc_A)  acc_B = $(resP1.acc_B)")
println("  Témoin propre ($CONTROL_SITE, r patch = ", round(MEAN_R[CONTROL_SITE], digits=3), ") : ",
        ctrl_ok ? "NEUTRE à l'ablation" : "NON NEUTRE",
        "  (acc_A $(ctrl.acc_A), acc_B $(ctrl.acc_B) vs base $(base.acc_A)/$(base.acc_B))")
for (nm, r) in sort(collect(RESULTS); by=first)
    nm in ("B", "C") && continue
    pass = r.acc_A >= 0.9 && r.literal_B >= 0.9
    println("  [$nm] acc_A = $(r.acc_A)  littéral_B = $(r.literal_B)  acc_B = $(r.acc_B)  -> ",
            pass ? "PRÉDICTION VALIDÉE" : "non validée")
end
passing = sort([nm for (nm, r) in RESULTS
                if !(nm in ("B", "C")) && r.acc_A >= 0.9 && r.literal_B >= 0.9])
println("  P4-BIS GLOBAL : ", isempty(passing) ?
        "NON VALIDÉ (aucune config n'atteint littéral_B >= 0.9 avec acc_A >= 0.9)" :
        "VALIDÉ (config(s) $(join(passing, ",")))")
println("═"^70)
