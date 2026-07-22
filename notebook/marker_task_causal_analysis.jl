# ══════════════════════════════════════════════════════════════════════════════
# "Tâche du marqueur" -- ÉTAPE 2 : analyse causale P2/P3/P4.
#
# Prérequis : P1 confirmé (voir marker_task_experiment.jl, config gagnante figée
# dans ses défauts : V=8, N_PAIRS=3, 4 couches, dim=64, 4 têtes, batch=64,
# lr constant 1e-3 après warmup, 5000 pas budget / early-stop ~3500).
# Ce script `include` le script d'entraînement tel quel (aucune duplication de
# config -- un lancement nu reproduit exactement le run P1 validé), vérifie P1,
# puis exécute :
#
#   P2 -- profil delta(n) : distance d'activation entre la trace (M, [m_A, k])
#         et la trace (M, [m_B, sigma(k)]) -- même contenu M, même réponse
#         v(k) -- à la position de la requête, pour chaque sortie de tête
#         d'attention, sortie de MLP et flux résiduel par couche. Prédiction :
#         delta grand aux couches précoces, petit après une "frontière de
#         convergence" située AVANT la dernière couche.
#         Critère pré-enregistré : frontière = première couche l telle que
#         delta(layer_l_out) < 0.3 × max_l' delta(layer_l'_out) ; P2 validé si
#         frontière <= N_LAYERS - 1.
#
#   P3 -- contraste causal propre : donneur = (M, [m_A, sigma(k)]) (lecture
#         littérale, réponse v(sigma(k))), receveur = (M, [m_B, sigma(k)])
#         (réponse correcte v(k)) -- SEUL le marqueur diffère. Tirages filtrés :
#         sigma(k) présent dans M comme clé ordinaire ET v(k) != v(sigma(k)).
#         Métrique custom (kwarg `metric` de greedy_patch_search!/
#         backward_prune!, src/patching.jl) : différence de logits normalisée
#         entre v(sigma(k)) et v(k) à la position finale --
#         r(out) = (dl(out) - dl_receveur) / (dl_donneur - dl_receveur),
#         0 = comportement receveur intact, 1 = bascule complète vers la
#         lecture littérale du donneur. PAS recovery_metric (denom fragile).
#         Prédiction : ensemble minimal <= 3 sites, r > 0.9, stable sous
#         backward_prune! et à travers des paires de recherche indépendantes.
#
#   P4 -- validation au niveau des POIDS : ablation (mise à zéro) de la tranche
#         de layer_l_mha_output_W portée par la tête identifiée (ou mlp_w2
#         entier pour un site MLP). Prédiction discriminante : acc_A >= 0.9
#         conservée, ET en format B le modèle répond SYSTÉMATIQUEMENT
#         v(sigma(k)) (lecture littérale) dans >= 0.9 des cas filtrés -- une
#         erreur PRÉVISIBLE, pas un effondrement vers du bruit. Ablation témoin
#         (tête hors circuit) incluse pour la spécificité.
#
# Notes d'API vérifiées sur src/patching.jl, src/graph_surgery.jl,
# src/layers.jl, src/graph_api.jl, src/dispatch.jl :
#   - noms de nœuds (LlamaModel, préfixe layer_{l}) : layer_{l}_mha_ao_h{h}
#     (sortie de tête, vue :head_view en batched_attn -- patchable comme
#     n'importe quel nœud, patch_node! remplace .value par un tampon possédé),
#     layer_{l}_mha_output_out (projection de sortie), layer_{l}_mlp_out,
#     layer_{l}_out (flux résiduel), lm_head_out, :final_logits (1×VOCAB).
#   - :scale_mask applique causal_mask(dev, seqlen) (src/dispatch.jl:543-548) :
#     les positions de contexte (1..2*N_PAIRS) sont donc identiques entre
#     donneur et receveur (mêmes tokens, causalité) -- patcher le nœud ENTIER
#     équivaut à un patch restreint aux positions de la requête. Vérifié
#     empiriquement ci-dessous (check "fuite anti-causale") plutôt que supposé.
#   - invalidate_all! remet valid=false sur tout nœud non-paramètre : c'est le
#     reset global propre entre deux entrées (efface tout patch actif).
#   - poids : layer_{l}_mha_output_W est (dim, dim), out = concat * W'
#     (:matmul trans_b=true, src/layers.jl:44-45) -- la tête h occupe les
#     COLONNES (h-1)*d_head+1 : h*d_head de W. layer_{l}_mlp_w2 est
#     (dim, hidden_dim).
#
# Usage :
#   smoke (syntaxe/API, modèle quasi non entraîné, ~2 min) :
#     MARKER_SMOKE=1 julia --project=. notebook/marker_task_causal_analysis.jl
#   run complet (~50-70 min d'entraînement + quelques minutes d'analyse) :
#     julia --project=. notebook/marker_task_causal_analysis.jl
# ══════════════════════════════════════════════════════════════════════════════

const SMOKE = get(ENV, "MARKER_SMOKE", "0") == "1"
if SMOKE
    # Réduit l'entraînement au minimum syntaxique -- get! ne touche pas une
    # variable déjà posée explicitement par l'utilisateur.
    get!(ENV, "MARKER_STEPS", "40")
    get!(ENV, "MARKER_BATCH", "8")
end

# Entraîne (config gagnante par défaut) et évalue -> g, logits, result, V,
# MARKER_A/B, N_PAIRS, SEQ_LEN, SIGMA, SIGMA_INV, sample_marker_sequence,
# evaluate_marker, dev, ns, N_LAYERS, DIM.
include(joinpath(@__DIR__, "marker_task_experiment.jl"))

const N_HEADS = 4                 # doit rester aligné sur l'appel build_marker_graph
const D_HEAD  = DIM ÷ N_HEADS
const N_CTX   = 2 * N_PAIRS       # positions de contexte (avant [marqueur, clé])

# ── P1 (gate) ────────────────────────────────────────────────────────────────
println()
println("═"^70)
println("P1 (gate) : acc_A = ", result.acc_A, "   acc_B = ", result.acc_B)
if SMOKE
    println("[SMOKE] gate P1 ignoré (modèle quasi non entraîné, on ne teste que le code).")
elseif !(result.acc_A >= 0.95 && result.acc_B >= 0.95)
    error("P1 non satisfait (seuil 0.95 sur les deux formats) -- l'analyse causale n'a pas de sens, arrêt.")
else
    println("P1 CONFIRMÉ -- on continue vers P2/P3/P4.")
end
flush(stdout)

# ── Helpers communs ──────────────────────────────────────────────────────────

function run_forward!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))  # (1, VOCAB)
end

function run_and_capture!(g, ns, tokens)
    out = run_forward!(g, ns, tokens)
    cache = NeuroDSL.capture_activations(g, ns)
    return out, cache
end

# Reset global propre : efface tout patch actif et reconverge sur l'entrée
# courante (les Datoms token_ids/pos_ids ne sont pas touchés par invalidate_all!).
function full_reset!(g, ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, :final_logits; namespace=ns)
    return g
end

# Sites candidats (spécification P3) : sorties de tête + sortie MLP, par couche.
head_site(l, h) = Symbol("layer_$(l)_mha_ao_h$(h)")
mlp_site(l)     = Symbol("layer_$(l)_mlp_out")
resid_site(l)   = Symbol("layer_$(l)_out")
attnproj_site(l)= Symbol("layer_$(l)_mha_output_out")

const CANDIDATES = Symbol[]
for l in 1:N_LAYERS
    for h in 1:N_HEADS
        push!(CANDIDATES, head_site(l, h))
    end
    push!(CANDIDATES, mlp_site(l))
end
for s in CANDIDATES
    haskey(g.nodes[ns], s) || error("Nœud candidat $s introuvable -- nommage à revérifier dans src/layers.jl")
end
println("Candidats P3 (", length(CANDIDATES), ") : ", CANDIDATES)
flush(stdout)

# Tirage filtré pour P3/P4 : contexte M contenant k ET sigma(k) comme clés
# ordinaires, avec v(k) != v(sigma(k)) (sinon la lecture littérale et la
# lecture correcte coïncident et le contraste est dégénéré).
function sample_contrast(rng)
    while true
        k  = rand(rng, 1:V)
        sk = SIGMA[k]                       # != k (SIGMA est un dérangement)
        others = collect(setdiff(1:V, (k, sk)))
        rest = length(others) >= N_PAIRS - 2 ? shuffle(rng, others)[1:N_PAIRS-2] : Int[]
        ks = vcat([k, sk], rest)
        vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue          # impose v(k) != v(sigma(k))
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS)
            push!(ctx, ks[i]); push!(ctx, vs[i])
        end
        return (; ctx, k, sk, vk = vs[1], vsk = vs[2])
    end
end

donor_tokens(c)    = vcat(c.ctx, [MARKER_A, c.sk])   # lecture littérale attendue : v(sigma(k)) = c.vsk
receiver_tokens(c) = vcat(c.ctx, [MARKER_B, c.sk])   # lecture correcte attendue : v(k) = c.vk

delta_logit(out, c) = Float64(out[1, c.vsk] - out[1, c.vk])

# ══════════════════════════════════════════════════════════════════════════════
# P2 -- profil delta(n)
# ══════════════════════════════════════════════════════════════════════════════
println()
println("═"^70)
println("P2 -- profil delta(n) entre (M,[m_A,k]) et (M,[m_B,sigma(k)])")
flush(stdout)

const P2_PROBES = Symbol[:embed_sum]
for l in 1:N_LAYERS
    for h in 1:N_HEADS; push!(P2_PROBES, head_site(l, h)); end
    push!(P2_PROBES, attnproj_site(l))
    push!(P2_PROBES, mlp_site(l))
    push!(P2_PROBES, resid_site(l))
end
push!(P2_PROBES, :lm_head_out)
for s in P2_PROBES
    haskey(g.nodes[ns], s) || error("Sonde P2 $s introuvable")
end

# delta relatif à la position finale : ||a-b|| / (0.5(||a||+||b||) + eps)
function _rel_delta_lastrow(cA, cB, sym)
    a = Array(cA[sym]); b = Array(cB[sym])
    ra = vec(Float64.(a[end, :])); rb = vec(Float64.(b[end, :]))
    num = sqrt(sum(abs2, ra .- rb))
    den = 0.5 * (sqrt(sum(abs2, ra)) + sqrt(sum(abs2, rb))) + 1e-8
    return num / den
end

let n_p2 = SMOKE ? 4 : 64
    rng = MersenneTwister(2026)
    sums = Dict{Symbol,Float64}(s => 0.0 for s in P2_PROBES)
    logit_gap = 0.0     # écart des logits finaux (même réponse attendue des deux côtés)
    n_used = 0
    for _ in 1:n_p2
        # même M, même k, deux formats -- réponse identique v(k) des deux côtés.
        tokens, _, _, k, v = sample_marker_sequence(rng, :A)   # se termine par [m_A, k]
        tokA = tokens
        tokB = vcat(tokens[1:N_CTX], [MARKER_B, SIGMA[k]])
        outA, cacheA = run_and_capture!(g, ns, tokA)
        outB, cacheB = run_and_capture!(g, ns, tokB)
        for s in P2_PROBES
            sums[s] += _rel_delta_lastrow(cacheA, cacheB, s)
        end
        logit_gap += sqrt(sum(abs2, Float64.(outA[1, :]) .- Float64.(outB[1, :]))) /
                     (0.5 * (sqrt(sum(abs2, Float64.(outA[1, :]))) + sqrt(sum(abs2, Float64.(outB[1, :])))) + 1e-8)
        n_used += 1
    end
    global P2_DELTAS = Dict{Symbol,Float64}(s => sums[s] / n_used for s in P2_PROBES)
    global P2_LOGIT_GAP = logit_gap / n_used

    println("  (moyenne sur $n_used paires, delta relatif à la position finale)")
    println("  embed_sum                 : ", round(P2_DELTAS[:embed_sum], digits=4))
    for l in 1:N_LAYERS
        heads = [round(P2_DELTAS[head_site(l,h)], digits=4) for h in 1:N_HEADS]
        println("  layer_$l  ao_h1..$N_HEADS       : ", heads)
        println("  layer_$l  attn_proj / mlp / resid : ",
                round(P2_DELTAS[attnproj_site(l)], digits=4), " / ",
                round(P2_DELTAS[mlp_site(l)], digits=4), " / ",
                round(P2_DELTAS[resid_site(l)], digits=4))
    end
    println("  lm_head_out (pos finale)  : ", round(P2_DELTAS[:lm_head_out], digits=4))
    println("  final_logits (delta rel.) : ", round(P2_LOGIT_GAP, digits=4))

    resid = [P2_DELTAS[resid_site(l)] for l in 1:N_LAYERS]
    dmax = maximum(resid)
    frontier = findfirst(l -> resid[l] < 0.3 * dmax, 1:N_LAYERS)
    global P2_FRONTIER = frontier
    global P2_PASS = frontier !== nothing && frontier <= N_LAYERS - 1
    println("  Flux résiduel par couche : ", round.(resid, digits=4))
    println("  Frontière de convergence (delta < 0.3 × max) : ",
            frontier === nothing ? "AUCUNE" : "couche $frontier")
    println("  P2 (critère pré-enregistré : frontière <= $(N_LAYERS-1)) : ",
            P2_PASS ? "VALIDÉ" : "NON VALIDÉ")
end
flush(stdout)

# ══════════════════════════════════════════════════════════════════════════════
# P3 -- contraste causal (donneur m_A vs receveur m_B, même token_clé affiché)
# ══════════════════════════════════════════════════════════════════════════════
println()
println("═"^70)
println("P3 -- greedy_patch_search! + backward_prune! sur le contraste marqueur")
flush(stdout)

const N_SEARCH_PAIRS = SMOKE ? 1 : 3
const N_VAL_PAIRS    = SMOKE ? 3 : 20
const MAX_SITES      = SMOKE ? 2 : 6

# Tire une paire de recherche où le modèle se comporte comme attendu
# (donneur -> argmax = vsk, receveur -> argmax = vk) et où le contraste de
# logits est net. En SMOKE (modèle non entraîné) on relâche : premier tirage
# avec dénominateur non dégénéré.
function draw_search_pair!(g, ns, rng; max_tries=200)
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
    error("Impossible de tirer une paire de contraste exploitable en $max_tries essais")
end

pair_results = NamedTuple[]
rng_p3 = MersenneTwister(31415)

for ip in 1:N_SEARCH_PAIRS
    c = draw_search_pair!(g, ns, rng_p3)
    d_out, donor_cache = run_and_capture!(g, ns, donor_tokens(c))
    r_out, recv_cache  = run_and_capture!(g, ns, receiver_tokens(c))
    dd, dr = delta_logit(d_out, c), delta_logit(r_out, c)
    metric = out -> (delta_logit(Array(out), c) - dr) / (dd - dr)
    println("\n  Paire $ip : k=$(c.k) sigma(k)=$(c.sk) v(k)=$(c.vk) v(sigma(k))=$(c.vsk)")
    println("    dl_donneur = ", round(dd, digits=3), "   dl_receveur = ", round(dr, digits=3))

    if ip == 1
        # Check "fuite anti-causale" : les positions de contexte doivent être
        # identiques entre donneur et receveur (masque causal) -- sinon le
        # patch de nœud entier ne serait PAS équivalent à un patch restreint
        # à la requête.
        leak = maximum(maximum(abs.(Array(donor_cache[s])[1:N_CTX, :] .-
                                     Array(recv_cache[s])[1:N_CTX, :])) for s in CANDIDATES)
        println("    Fuite anti-causale max (positions 1..$N_CTX, tous candidats) : ", leak)
        leak > 1e-5 && println("    ⚠ fuite non nulle -- le patch de nœud entier dépasse la position de requête !")
    end

    # Plafond de faisabilité : tous les candidats patchés ensemble.
    NeuroDSL.patch_nodes!(g, CANDIDATES, donor_cache; namespace=ns)
    r_all = metric(NeuroDSL.demand!(g, :final_logits; namespace=ns))
    full_reset!(g, ns)   # retour à l'état receveur propre
    println("    Plafond (les $(length(CANDIDATES)) candidats patchés) : r = ", round(r_all, digits=4))

    # Balayage site par site (informatif ; restauration par patch corrompu,
    # patron patch_and_measure!).
    singles = NamedTuple[]
    for s in CANDIDATES
        NeuroDSL.patch_node!(g, s, donor_cache; namespace=ns)
        rs = metric(NeuroDSL.demand!(g, :final_logits; namespace=ns))
        NeuroDSL.patch_node!(g, s, recv_cache; namespace=ns)
        NeuroDSL.demand!(g, :final_logits; namespace=ns)
        push!(singles, (; site=s, r=rs))
    end
    full_reset!(g, ns)
    sort!(singles; by=x -> -x.r)
    println("    Top-5 sites individuels : ",
            [(String(x.site), round(x.r, digits=3)) for x in singles[1:min(5, end)]])

    # Recherche gloutonne + élagage arrière, métrique custom.
    selected, traj = NeuroDSL.greedy_patch_search!(g, :final_logits, CANDIDATES,
        donor_cache, recv_cache, copy(d_out), copy(r_out);
        max_sites=MAX_SITES, namespace=ns, metric=metric)
    println("    Greedy : ", [(String(t.site), round(t.cumulative_recovery, digits=3)) for t in traj])

    remaining, pruned = isempty(selected) ? (Symbol[], Symbol[]) :
        NeuroDSL.backward_prune!(g, :final_logits, selected,
            donor_cache, recv_cache, copy(d_out), copy(r_out);
            namespace=ns, tol=Float32(0.05), metric=metric)
    r_final = if isempty(remaining)
        0.0
    else
        # backward_prune! laisse le graphe avec `remaining` patché.
        metric(NeuroDSL.demand!(g, :final_logits; namespace=ns))
    end
    full_reset!(g, ns)
    println("    Après backward_prune! : ", remaining, "  (élagués : ", pruned, ")  r = ", round(r_final, digits=4))
    push!(pair_results, (; c, remaining, r_final, r_all, singles=singles[1:min(5, end)]))
    flush(stdout)
end

# Consensus inter-paires : sites présents dans une majorité des ensembles élagués.
site_counts = Dict{Symbol,Int}()
for pr in pair_results, s in pr.remaining
    site_counts[s] = get(site_counts, s, 0) + 1
end
need = cld(N_SEARCH_PAIRS + 1, 2)
CIRCUIT = sort([s for (s, n) in site_counts if n >= need]; by=String)
if isempty(CIRCUIT) && !isempty(pair_results) && !isempty(pair_results[1].remaining)
    CIRCUIT = pair_results[1].remaining
    println("\n  (pas de consensus majoritaire -- repli sur l'ensemble de la paire 1)")
end
println("\n  Circuit retenu (consensus) : ", CIRCUIT)

# Validation sur paires indépendantes : patch du circuit seul, r + taux de
# bascule argmax vers la lecture littérale.
P3_VAL = (; median_r = NaN, flip_rate = NaN, n = 0)
if !isempty(CIRCUIT)
    let rng_val = MersenneTwister(2718)
        rs = Float64[]; flips = 0; n_ok = 0
        for _ in 1:N_VAL_PAIRS
            c = sample_contrast(rng_val)
            d_out, donor_cache = run_and_capture!(g, ns, donor_tokens(c))
            r_out = run_forward!(g, ns, receiver_tokens(c))
            dd, dr = delta_logit(d_out, c), delta_logit(r_out, c)
            abs(dd - dr) < 1e-6 && continue
            NeuroDSL.patch_nodes!(g, CIRCUIT, donor_cache; namespace=ns)
            p_out = Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
            full_reset!(g, ns)
            push!(rs, (delta_logit(p_out, c) - dr) / (dd - dr))
            flips += (argmax(vec(p_out)) == c.vsk) ? 1 : 0
            n_ok += 1
        end
        med = isempty(rs) ? NaN : sort(rs)[cld(length(rs), 2)]
        global P3_VAL = (; median_r = med, flip_rate = n_ok == 0 ? NaN : flips / n_ok, n = n_ok)
        println("  Validation ($n_ok paires indépendantes) : r médian = ", round(P3_VAL.median_r, digits=4),
                "   taux de bascule argmax -> v(sigma(k)) = ", round(P3_VAL.flip_rate, digits=4))
    end
end

P3_PASS = !isempty(CIRCUIT) && length(CIRCUIT) <= 3 &&
          (P3_VAL.n > 0 && P3_VAL.median_r > 0.9)
println("  P3 (<= 3 sites, r médian > 0.9 en validation indépendante) : ",
        P3_PASS ? "VALIDÉ" : "NON VALIDÉ")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════════════
# P4 -- ablation au niveau des poids + erreur systématique prédite
# ══════════════════════════════════════════════════════════════════════════════
println()
println("═"^70)
println("P4 -- ablation de poids des sites du circuit")
flush(stdout)

# Sites à ablater : le circuit P3 ; repli (meilleur site individuel de la
# paire 1 si r > 0.5) sinon ; en SMOKE, site arbitraire pour exercer le code.
ABLATE_SITES = copy(CIRCUIT)
if isempty(ABLATE_SITES)
    best1 = isempty(pair_results) ? nothing : pair_results[1].singles[1]
    if best1 !== nothing && best1.r > 0.5
        ABLATE_SITES = [best1.site]
        println("  (circuit vide -- repli exploratoire sur le meilleur site individuel : $(best1.site))")
    elseif SMOKE
        ABLATE_SITES = [head_site(2, 1)]
        println("  [SMOKE] circuit vide -- site arbitraire pour exercer le chemin d'ablation.")
    end
end

if isempty(ABLATE_SITES)
    println("  Aucun site exploitable -- P4 sans objet (P3 a échoué).")
    global P4_PASS = false
else
    # Construit le dict de poids ablatés + sauvegarde des originaux.
    function ablation_weights(sites)
        w = Dict{Symbol,Matrix{Float32}}()
        for s in sites
            m = match(r"^layer_(\d+)_mha_ao_h(\d+)$", String(s))
            if m !== nothing
                l = parse(Int, m.captures[1]); h = parse(Int, m.captures[2])
                wsym = Symbol("layer_$(l)_mha_output_W")
                W = get!(() -> Matrix{Float32}(Array(NeuroDSL.node(g, wsym; namespace=ns).value)), w, wsym)
                W[:, (h-1)*D_HEAD+1 : h*D_HEAD] .= 0f0   # colonnes = tranche d'ENTRÉE de la tête h (trans_b=true)
            else
                m2 = match(r"^layer_(\d+)_mlp_out$", String(s))
                m2 === nothing && error("Site non ablatable au niveau des poids : $s")
                wsym = Symbol("layer_$(m2.captures[1])_mlp_w2")
                w[wsym] = zeros(Float32, size(NeuroDSL.node(g, wsym; namespace=ns).value))
            end
        end
        return w
    end

    abl_w   = ablation_weights(ABLATE_SITES)
    ctrl_site = something(findfirst(s -> !(s in ABLATE_SITES) &&
                                        match(r"_mha_ao_h\d+$", String(s)) !== nothing, CANDIDATES), 1)
    ctrl_w  = ablation_weights([CANDIDATES[ctrl_site]])
    touched = union(keys(abl_w), keys(ctrl_w))
    originals = Dict{Symbol,Matrix{Float32}}(
        wsym => Matrix{Float32}(Array(NeuroDSL.node(g, wsym; namespace=ns).value)) for wsym in touched)

    function eval_condition(g, ns; n, seed)
        # Format A : accuracy standard.
        rngA = MersenneTwister(seed)
        okA = 0
        for _ in 1:n
            tokens, _, _, _, v = sample_marker_sequence(rngA, :A)
            okA += (argmax(vec(run_forward!(g, ns, tokens))) == v) ? 1 : 0
        end
        # Format B, tirages filtrés (sigma(k) en contexte, v(k) != v(sigma(k))) :
        # correct = v(k), littéral = v(sigma(k)), autre = tout le reste.
        rngB = MersenneTwister(seed + 1)
        okB = 0; lit = 0
        for _ in 1:n
            c = sample_contrast(rngB)
            pred = argmax(vec(run_forward!(g, ns, receiver_tokens(c))))
            okB  += (pred == c.vk)  ? 1 : 0
            lit  += (pred == c.vsk) ? 1 : 0
        end
        return (; acc_A = okA / n, acc_B = okB / n, literal_B = lit / n)
    end

    n_eval = SMOKE ? 30 : 300
    base = eval_condition(g, ns; n=n_eval, seed=555)
    println("  Sites ablatés : ", ABLATE_SITES, "  (poids touchés : ", collect(keys(abl_w)), ")")
    println("  AVANT ablation      : acc_A = $(base.acc_A)  acc_B = $(base.acc_B)  littéral_B = $(base.literal_B)")

    NeuroDSL.set_params!(g, ns, abl_w)
    abl = eval_condition(g, ns; n=n_eval, seed=555)
    println("  APRÈS ablation      : acc_A = $(abl.acc_A)  acc_B = $(abl.acc_B)  littéral_B = $(abl.literal_B)")
    NeuroDSL.set_params!(g, ns, originals)

    NeuroDSL.set_params!(g, ns, ctrl_w)
    ctrl = eval_condition(g, ns; n=n_eval, seed=555)
    println("  Ablation TÉMOIN ($(CANDIDATES[ctrl_site])) : acc_A = $(ctrl.acc_A)  acc_B = $(ctrl.acc_B)  littéral_B = $(ctrl.literal_B)")
    NeuroDSL.set_params!(g, ns, originals)
    full_reset!(g, ns)

    global P4_PASS = abl.acc_A >= 0.9 && abl.literal_B >= 0.9
    println("  P4 (acc_A >= 0.9 conservée ET littéral_B >= 0.9 systématique) : ",
            P4_PASS ? "VALIDÉ" : "NON VALIDÉ")
    global P4_BASE, P4_ABL, P4_CTRL = base, abl, ctrl
end
flush(stdout)

# ── Verdict final ────────────────────────────────────────────────────────────
println()
println("═"^70)
println("VERDICT ", SMOKE ? "[SMOKE -- chiffres NON significatifs, test de code seulement]" : "")
println("  P1 : acc_A = $(result.acc_A)  acc_B = $(result.acc_B)")
println("  P2 : ", P2_PASS ? "VALIDÉ" : "NON VALIDÉ",
        "  (frontière = ", P2_FRONTIER === nothing ? "aucune" : P2_FRONTIER,
        ", résiduel = ", [round(P2_DELTAS[resid_site(l)], digits=3) for l in 1:N_LAYERS], ")")
println("  P3 : ", P3_PASS ? "VALIDÉ" : "NON VALIDÉ",
        "  (circuit = ", CIRCUIT, ", r médian validation = ",
        P3_VAL.n > 0 ? round(P3_VAL.median_r, digits=3) : "n/a",
        ", bascule = ", P3_VAL.n > 0 ? round(P3_VAL.flip_rate, digits=3) : "n/a", ")")
if @isdefined(P4_ABL)
    println("  P4 : ", P4_PASS ? "VALIDÉ" : "NON VALIDÉ",
            "  (post-ablation : acc_A = $(P4_ABL.acc_A), littéral_B = $(P4_ABL.literal_B), acc_B = $(P4_ABL.acc_B))")
else
    println("  P4 : sans objet (aucun site à ablater)")
end
println("═"^70)
