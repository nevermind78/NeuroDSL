# ══════════════════════════════════════════════════════════════════════════════
# Vérification de la Conjecture 1 (artilce/math4.tex, section 6, conj:polarity)
# -- pré-enregistrée dans l'article, testée ici pour la première fois.
# AUCUN réentraînement : passes forward sur les checkpoints existants.
#
# Quantités opérationnelles (par paire appariée (x_A, x_B) = même contexte,
# même token affiché t, seul le marqueur diffère) :
#   s_op(x)   = logit_{v(t)} - logit_{v(sigma^{-1}(t))}   (position finale)
#   g_A, g_B  = s_op(x_A), s_op(x_B)                      (marges)
#   q(x)      = s_op(x) - s_op(x | activations carriers projetées (I-P_i)c_i)
#               -- mesuré par INTERVENTION D'ACTIVATION ; identique à l'édition
#               de poids par linéarité exacte (mlp_out·(I-P) == sg·((I-P)W2)',
#               ao_h·(I-P) == colonnes de W_O projetées) ; l'écart numérique
#               |(g - q) - s_op(poids ablatés)| est VÉRIFIÉ explicitement.
#   Phi_op    = g_A - g_B ;  Phi_S = q_A - q_B ;  qbar = (q_A+q_B)/2 ;
#   sbar      = (g_A+g_B)/2.
#
# C1a : la prédiction par entrée (littéral ssi g - q > 0, inversé sinon)
#       concorde avec la réponse observée post-ablation (argmax catégorisé
#       littéral / inversé / autre ; "autre" compte comme désaccord) sur
#       >= 90% des entrées filtrées, pour chaque (instance, config).
#       NOTE DE STRUCTURE LOGIQUE (honnêteté) : l'équivalence exacte
#       activation/poids rend la prédiction équivalente à sign(s_op ablaté) ;
#       le contenu empirique de C1a est donc précisément "la réponse reste
#       confinée aux deux lectures candidates et suit le seuil" -- le taux
#       d'"autre" est rapporté séparément.
# C1b : taux agrégés prédits Pr[g_A - q_A < 0] (inversé-A) et
#       Pr[g_B - q_B > 0] (littéral-B) vs observés, à ±0.10.
# C1c : configs à retrait quasi complet/symétrique (med Phi_S >= 0.8 med Phi,
#       |med qbar| <= 0.2 |med sbar|) : polarité observée == signe(med sbar)
#       (littéral survit ssi sbar > 0) ; ET ablation CONÇUE ciblant qbar pour
#       basculer la polarité à volonté (graine 44).
#
# Carriers/sous-espaces reconstruits à l'identique des runs d'origine (mêmes
# graines 4242/1618, même seuil 0.25, mêmes kcaps) -- déterministe à poids
# fixés, donc mêmes (U_i) que marker_seed_matrix.jl / marker_task_p4bis.jl.
# ══════════════════════════════════════════════════════════════════════════════

const SMOKE = get(ENV, "MARKER_SMOKE", "0") == "1"
ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))
using LinearAlgebra

const CKPT_DIR = get(ENV, "MARKER_CKPT_DIR", joinpath(@__DIR__, "marker_ckpt"))
INSTANCES = if SMOKE
    [("seed_99", joinpath(get(ENV, "MARKER_SMOKE_CKPT_DIR", CKPT_DIR), "seed_99"))]
else
    [("inst2",   joinpath(CKPT_DIR, "marker_task")),
     ("seed_11", joinpath(CKPT_DIR, "seed_11")),
     ("seed_22", joinpath(CKPT_DIR, "seed_22")),
     ("seed_33", joinpath(CKPT_DIR, "seed_33")),
     ("seed_44", joinpath(CKPT_DIR, "seed_44"))]
end
for (nm, ck) in INSTANCES
    (isfile(ck * ".json") && isfile(ck * ".bin")) || error("Checkpoint manquant : $ck")
end

const N_HEADS = 4
const D_HEAD  = DIM ÷ N_HEADS
const N_PAIRS_FAM = SMOKE ? 10 : 150     # paires par famille (A-filtrée / B-filtrée)
const N_SWEEP     = SMOKE ? 2 : 8
const N_DELTA     = SMOKE ? 4 : 32

head_site(l, h) = Symbol("layer_$(l)_mha_ao_h$(h)")
mlp_site(l)     = Symbol("layer_$(l)_mlp_out")
site_layer(s)   = parse(Int, match(r"^layer_(\d+)_", String(s)).captures[1])
const CANDIDATES = Symbol[]
for l in 1:N_LAYERS
    for h in 1:N_HEADS; push!(CANDIDATES, head_site(l, h)); end
    push!(CANDIDATES, mlp_site(l))
end

function run_forward!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
end
run_and_capture!(g, ns, tokens) = (out = run_forward!(g, ns, tokens); (out, NeuroDSL.capture_activations(g, ns)))
full_reset!(g, ns) = (NeuroDSL.invalidate_all!(g; namespace=ns); NeuroDSL.demand!(g, :final_logits; namespace=ns); g)

function sample_contrast(rng)      # B-filtrée : t = sigma(k)
    while true
        k = rand(rng, 1:V); sk = SIGMA[k]
        rest = shuffle(rng, collect(setdiff(1:V, (k, sk))))[1:N_PAIRS-2]
        ks = vcat([k, sk], rest); vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS); push!(ctx, ks[i]); push!(ctx, vs[i]); end
        # t = sk : lecture littérale v(sk)=vs[2], lecture inversée v(k)=vs[1]
        return (; tokA = vcat(ctx, [MARKER_A, sk]), tokB = vcat(ctx, [MARKER_B, sk]),
                 lit = vs[2], inv = vs[1])
    end
end
function sample_contrast_A(rng)    # A-filtrée : t = k
    while true
        k = rand(rng, 1:V); ik = SIGMA_INV[k]
        rest = shuffle(rng, collect(setdiff(1:V, (k, ik))))[1:N_PAIRS-2]
        ks = vcat([k, ik], rest); vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS); push!(ctx, ks[i]); push!(ctx, vs[i]); end
        # t = k : littéral v(k)=vs[1], inversé v(sigma^{-1}(k))=v(ik)=vs[2]
        return (; tokA = vcat(ctx, [MARKER_A, k]), tokB = vcat(ctx, [MARKER_B, k]),
                 lit = vs[1], inv = vs[2])
    end
end

sop(out, p) = Float64(out[1, p.lit] - out[1, p.inv])
category(out, p) = (a = argmax(vec(out)); a == p.lit ? :lit : (a == p.inv ? :inv : :other))

function draw_clean_pair!(g, ns, rng)   # pour sweep/deltas : mêmes filtres que la matrice
    for _ in 1:200
        c = sample_contrast(rng)
        d_out = run_forward!(g, ns, c.tokA); r_out = run_forward!(g, ns, c.tokB)
        dd, dr = sop(d_out, c), sop(r_out, c)
        if SMOKE
            abs(dd - dr) > 1e-6 && return c
        else
            (argmax(vec(d_out)) == c.lit && argmax(vec(r_out)) == c.inv && dd > 0 > dr) && return c
        end
    end
    error("pas de paire propre")
end

function single_site_sweep!(g, ns; n_pairs, seed)
    rng = MersenneTwister(seed)
    sums = Dict{Symbol,Float64}(s => 0.0 for s in CANDIDATES)
    for _ in 1:n_pairs
        c = draw_clean_pair!(g, ns, rng)
        d_out, dcache = run_and_capture!(g, ns, c.tokA)
        r_out, rcache = run_and_capture!(g, ns, c.tokB)
        dd, dr = sop(d_out, c), sop(r_out, c)
        for s in CANDIDATES
            NeuroDSL.patch_node!(g, s, dcache; namespace=ns)
            out = Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
            sums[s] += (sop(out, c) - dr) / (dd - dr)
            NeuroDSL.patch_node!(g, s, rcache; namespace=ns)
            NeuroDSL.demand!(g, :final_logits; namespace=ns)
        end
        full_reset!(g, ns)
    end
    return Dict{Symbol,Float64}(s => sums[s] / n_pairs for s in CANDIDATES)
end

function collect_rows!(g, ns, sites; n_pairs, seed, which::Symbol)   # :delta ou :mu
    rng = MersenneTwister(seed)
    rows = Dict{Symbol,Vector{Vector{Float64}}}(s => Vector{Vector{Float64}}() for s in sites)
    for _ in 1:n_pairs
        c = draw_clean_pair!(g, ns, rng)
        _, dcache = run_and_capture!(g, ns, c.tokA)
        _, rcache = run_and_capture!(g, ns, c.tokB)
        for s in sites
            da = Array(dcache[s]); ra = Array(rcache[s])
            for row in (SEQ_LEN - 1, SEQ_LEN)
                v = which == :delta ? (da[row, :] .- ra[row, :]) : 0.5 .* (da[row, :] .+ ra[row, :])
                push!(rows[s], vec(Float64.(v)))
            end
        end
    end
    return rows
end

function format_subspace(vecs; energy=0.90, kcap::Int)
    X = reduce(hcat, vecs); F = svd(X)
    e = F.S .^ 2; ce = cumsum(e) ./ sum(e)
    k = min(something(findfirst(>=(energy), ce), length(ce)), kcap)
    return F.U[:, 1:k]
end

_current_W(g, ns, wsym) = Matrix{Float32}(Array(NeuroDSL.node(g, wsym; namespace=ns).value))

# Dict site => matrice de projection (I - U U') dans l'espace d'activation du site.
function projectors(subs::Dict{Symbol,<:AbstractMatrix})
    P = Dict{Symbol,Matrix{Float32}}()
    for (s, U) in subs
        d = size(U, 1)
        P[s] = Matrix{Float32}(I, d, d) .- Float32.(U) * Float32.(U)'
    end
    return P
end

function directional_weights(g, ns, subs::Dict{Symbol,<:AbstractMatrix})
    w = Dict{Symbol,Matrix{Float32}}()
    for (s, U) in subs
        Uf = Float32.(U)
        m = match(r"^layer_(\d+)_mha_ao_h(\d+)$", String(s))
        if m !== nothing
            l = parse(Int, m.captures[1]); h = parse(Int, m.captures[2])
            wsym = Symbol("layer_$(l)_mha_output_W")
            W = get!(() -> _current_W(g, ns, wsym), w, wsym)
            cols = (h-1)*D_HEAD+1 : h*D_HEAD
            W[:, cols] = W[:, cols] * (Matrix{Float32}(I, D_HEAD, D_HEAD) .- Uf * Uf')
        else
            wsym = Symbol("layer_$(match(r"^layer_(\d+)_mlp_out$", String(s)).captures[1])_mlp_w2")
            w[wsym] = (Matrix{Float32}(I, DIM, DIM) .- Uf * Uf') * _current_W(g, ns, wsym)
        end
    end
    return w
end

# g, q pour UNE entrée : baseline puis activations carriers projetées.
function measure_gq!(g, ns, tokens, p, P)
    out = run_forward!(g, ns, tokens)
    gval = sop(out, p)
    proj_vals = Dict{Symbol,Any}()
    for (s, Pm) in P
        proj_vals[s] = Array(NeuroDSL.node(g, s; namespace=ns).value) * Pm
    end
    NeuroDSL.patch_nodes!(g, collect(keys(P)), proj_vals; namespace=ns)
    out_p = Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
    return gval, gval - sop(out_p, p)
end

# ── Vérification C1a/C1b/C1c(stats) pour une config sur une instance ─────────
function verify_config!(g, ns, name, cfgname, subs, pairsA, pairsB)
    P = projectors(subs)
    # 1. mesures baseline + q par intervention d'activation
    meas = NamedTuple[]
    for (fam, plist) in ((:A, pairsA), (:B, pairsB))
        for p in plist
            gA, qA = measure_gq!(g, ns, p.tokA, p, P)
            gB, qB = measure_gq!(g, ns, p.tokB, p, P)
            push!(meas, (; fam, p, gA, qA, gB, qB))
        end
    end
    # 2. comportement observé sous ablation DE POIDS + équivalence numérique
    w = directional_weights(g, ns, subs)
    restore = Dict{Symbol,Matrix{Float32}}(k => _current_W(g, ns, k) for k in keys(w))
    NeuroDSL.set_params!(g, ns, w)
    obs = NamedTuple[]
    equiv_err = 0.0
    for m in meas
        outA = run_forward!(g, ns, m.p.tokA); outB = run_forward!(g, ns, m.p.tokB)
        equiv_err = max(equiv_err, abs((m.gA - m.qA) - sop(outA, m.p)), abs((m.gB - m.qB) - sop(outB, m.p)))
        push!(obs, (; catA = category(outA, m.p), catB = category(outB, m.p)))
    end
    NeuroDSL.set_params!(g, ns, restore); full_reset!(g, ns)
    # 3. C1a
    agree = 0; total = 0; nother = 0
    for (m, o) in zip(meas, obs)
        predA = (m.gA - m.qA) > 0 ? (:lit) : (:inv)
        predB = (m.gB - m.qB) > 0 ? (:lit) : (:inv)
        agree += (predA == o.catA) + (predB == o.catB); total += 2
        nother += (o.catA == :other) + (o.catB == :other)
    end
    c1a = agree / total; other_rate = nother / total
    # 4. C1b (x_A de la famille A ; x_B de la famille B -- conventions du tableau)
    mA = [(m, o) for (m, o) in zip(meas, obs) if m.fam == :A]
    mB = [(m, o) for (m, o) in zip(meas, obs) if m.fam == :B]
    pred_invA = sum(m.gA - m.qA < 0 for (m, _) in mA) / length(mA)
    obs_invA  = sum(o.catA == :inv for (_, o) in mA) / length(mA)
    pred_litB = sum(m.gB - m.qB > 0 for (m, _) in mB) / length(mB)
    obs_litB  = sum(o.catB == :lit for (_, o) in mB) / length(mB)
    c1b = abs(pred_invA - obs_invA) <= 0.10 && abs(pred_litB - obs_litB) <= 0.10
    # 5. stats C1c
    med(v) = isempty(v) ? NaN : sort(v)[cld(length(v), 2)]
    phi  = med([m.gA - m.gB for m in meas])
    phiS = med([m.qA - m.qB for m in meas])
    qbar = med([(m.qA + m.qB) / 2 for m in meas])
    sbar = med([(m.gA + m.gB) / 2 for m in meas])
    qual = phiS >= 0.8 * phi && abs(qbar) <= 0.2 * abs(sbar)
    pred_pol = sbar > 0 ? :literal_collapse : :inverted_collapse   # littéral survit ssi sbar>0 => B bascule
    obs_pol  = obs_litB > obs_invA ? :literal_collapse : :inverted_collapse
    c1c_sign = pred_pol == obs_pol
    # equiv_err : |(g - q_gelé) - s_op(poids ablatés)|. EXACTEMENT 0 attendu pour
    # une config mono-carrier (mêmes projections dans les deux routes). Pour une
    # config JOINTE, q est mesuré à activations gelées (spécification de l'article)
    # alors que la route poids propage l'ablation amont dans l'entrée des carriers
    # aval : l'écart mesure donc l'INTERACTION INTER-CARRIERS, c'est-à-dire l'erreur
    # d'idéalisation du modèle linéaire -- rapportée, pas supposée nulle.
    println("  [$name/$cfgname] n=$(total) interaction_max=", round(equiv_err, digits=5),
            "  C1a=", round(c1a, digits=3), " (autre=", round(other_rate, digits=3), ")")
    println("      C1b: pred_invA=", round(pred_invA, digits=3), " obs_invA=", round(obs_invA, digits=3),
            " | pred_litB=", round(pred_litB, digits=3), " obs_litB=", round(obs_litB, digits=3),
            " -> ", c1b ? "OK" : "ECHEC")
    println("      C1c-stats: Phi=", round(phi, digits=2), " PhiS=", round(phiS, digits=2),
            " qbar=", round(qbar, digits=2), " sbar=", round(sbar, digits=2),
            " qualif=", qual ? "OUI" : "non", " pred=", pred_pol, " obs=", obs_pol,
            qual ? (c1c_sign ? " -> SIGNE OK" : " -> SIGNE ECHEC") : "")
    flush(stdout)
    return (; name, cfgname, c1a, other_rate, c1b, pred_invA, obs_invA, pred_litB, obs_litB,
             phi, phiS, qbar, sbar, qual, pred_pol, obs_pol, c1c_sign, equiv_err)
end

# ── Pipeline par instance ────────────────────────────────────────────────────
function verify_instance!(g, ns, name, ckpt)
    println("\n", "═"^70); println("INSTANCE $name  ($ckpt)"); flush(stdout)
    NeuroDSL.load_graph!(g, ns, ckpt; overwrite=true)
    r = evaluate_marker(g, logits, ns; n_eval=200)
    println("  P1 : ", r)
    MEAN_R = single_site_sweep!(g, ns; n_pairs=N_SWEEP, seed=4242)
    carriers = sort([s for s in CANDIDATES if MEAN_R[s] >= 0.25]; by=s -> -MEAN_R[s])
    isempty(carriers) && (println("  AUCUN CARRIER -- instance sautée"); return NamedTuple[], nothing)
    main_layer = minimum(site_layer(s) for s in carriers)
    cmain = [s for s in carriers if site_layer(s) == main_layer]
    println("  Carriers : ", [(String(s), round(MEAN_R[s], digits=3)) for s in carriers])
    deltas = collect_rows!(g, ns, carriers; n_pairs=N_DELTA, seed=1618, which=:delta)
    subs_all = Dict{Symbol,Matrix{Float64}}()
    for s in carriers
        subs_all[s] = format_subspace(deltas[s]; kcap = endswith(String(s), "_mlp_out") ? 8 : 4)
    end
    rngA = MersenneTwister(90210); rngB = MersenneTwister(31337)
    pairsA = [sample_contrast_A(rngA) for _ in 1:N_PAIRS_FAM]
    pairsB = [sample_contrast(rngB)   for _ in 1:N_PAIRS_FAM]
    configs = Pair{String,Vector{Symbol}}["D1" => [cmain[1]], "DJ" => cmain]
    length(carriers) > length(cmain) && push!(configs, "DJA" => carriers)
    results = NamedTuple[]
    for (cfgname, sites) in configs
        subs = Dict{Symbol,Matrix{Float64}}(s => subs_all[s] for s in sites)
        push!(results, verify_config!(g, ns, name, cfgname, subs, pairsA, pairsB))
    end
    return results, (; carriers, cmain, subs_all, pairsA, pairsB)
end

ALL = NamedTuple[]
SEED44_CTX = Ref{Any}(nothing)
for (name, ckpt) in INSTANCES
    res, ctx = verify_instance!(g, ns, name, ckpt)
    append!(ALL, res)
    (name == "seed_44" || SMOKE) && (SEED44_CTX[] = (; name, ckpt, ctx))
end

# ── C1c partie actionnable : ablation CONÇUE pour piloter la polarité ────────
# Sur la graine 44 (ou l'instance smoke) : partir de DJ, augmenter les
# sous-espaces retirés avec des directions de RÉFÉRENCE (composantes
# principales des mu_i, orthogonalisées contre U_i) choisies par MESURE de
# leur effet sur qbar/s̃ sur un jeu de sondes, pour amener la médiane
# prédite de s̃ des DEUX formats du côté choisi -- puis vérifier en poids.
println("\n", "═"^70)
println("C1c -- ablation conçue (pilotage de la polarité)")
flush(stdout)

function designed_flip!(g, ns, ckpt, ctx, target::Symbol)   # :inverted_collapse ou :literal_collapse
    NeuroDSL.load_graph!(g, ns, ckpt; overwrite=true)
    cmain = ctx.cmain
    base_subs = Dict{Symbol,Matrix{Float64}}(s => ctx.subs_all[s] for s in cmain)
    mus = collect_rows!(g, ns, cmain; n_pairs=N_DELTA, seed=1618, which=:mu)
    # candidats : top-3 PCA des mu, orthogonalisés contre U_i
    cands = Tuple{Symbol,Vector{Float64}}[]
    for s in cmain
        U = base_subs[s]
        X = reduce(hcat, mus[s]); X = X .- U * (U' * X)      # orthogonalise
        F = svd(X)
        for j in 1:min(3, size(F.U, 2))
            push!(cands, (s, F.U[:, j]))
        end
    end
    probesA = ctx.pairsA[1:min(SMOKE ? 4 : 20, end)]
    probesB = ctx.pairsB[1:min(SMOKE ? 4 : 20, end)]
    med(v) = sort(v)[cld(length(v), 2)]
    function predicted_medians(subs)
        P = projectors(subs)
        sA = Float64[]; sB = Float64[]
        for p in vcat(probesA, probesB)
            gA, qA = measure_gq!(g, ns, p.tokA, p, P)
            gB, qB = measure_gq!(g, ns, p.tokB, p, P)
            push!(sA, gA - qA); push!(sB, gB - qB)
        end
        return med(sA), med(sB)
    end
    # cible : :inverted_collapse -> les deux médianes < 0 ; :literal -> > 0
    want_neg = target == :inverted_collapse
    cur = Dict{Symbol,Matrix{Float64}}(s => copy(base_subs[s]) for s in cmain)
    mA0, mB0 = predicted_medians(cur)
    println("  cible=$target | médianes s̃ de départ (DJ) : A=", round(mA0, digits=2), " B=", round(mB0, digits=2))
    used = Set{Int}()
    for round_i in 1:(SMOKE ? 2 : 6)
        okA, okB = want_neg ? (mA0 < 0, mB0 < 0) : (mA0 > 0, mB0 > 0)
        okA && okB && break
        best = nothing; best_score = want_neg ? max(mA0, mB0) : -min(mA0, mB0)
        for (ci, (s, wv)) in enumerate(cands)
            ci in used && continue
            trial = Dict(k => (k == s ? hcat(cur[k], wv) : cur[k]) for k in keys(cur))
            mA, mB = predicted_medians(trial)
            score = want_neg ? max(mA, mB) : -min(mA, mB)
            if score < best_score
                best_score = score; best = (ci, trial, mA, mB)
            end
        end
        best === nothing && (println("  aucun candidat n'améliore -- arrêt design"); break)
        ci, cur_new, mA0_, mB0_ = best
        push!(used, ci)
        cur = cur_new; mA0 = mA0_; mB0 = mB0_
        println("  + direction $(cands[ci][1]) (candidat $ci) -> médianes A=", round(mA0, digits=2),
                " B=", round(mB0, digits=2))
        flush(stdout)
    end
    # évaluation finale en POIDS sur les jeux complets
    subs = cur
    P = projectors(subs)
    w = directional_weights(g, ns, subs)
    restore = Dict{Symbol,Matrix{Float32}}(k => _current_W(g, ns, k) for k in keys(w))
    NeuroDSL.set_params!(g, ns, w)
    invA = 0; litB = 0; okA_c = 0; okB_c = 0
    for p in ctx.pairsA
        c = category(run_forward!(g, ns, p.tokA), p)
        invA += c == :inv; okA_c += c == :lit
    end
    for p in ctx.pairsB
        c = category(run_forward!(g, ns, p.tokB), p)
        litB += c == :lit; okB_c += c == :inv
    end
    NeuroDSL.set_params!(g, ns, restore); full_reset!(g, ns)
    nA = length(ctx.pairsA); nB = length(ctx.pairsB)
    obs_pol = (litB / nB) > (invA / nA) ? :literal_collapse : :inverted_collapse
    k_extra = sum(size(cur[s], 2) - size(base_subs[s], 2) for s in cmain)
    println("  RESULTAT design (cible $target, +$k_extra directions) : inversé_A=", round(invA/nA, digits=3),
            " (A correct=", round(okA_c/nA, digits=3), ")  littéral_B=", round(litB/nB, digits=3),
            " (B correct=", round(okB_c/nB, digits=3), ")  polarité observée=$obs_pol -> ",
            obs_pol == target ? "PILOTAGE RÉUSSI" : "pilotage ÉCHOUÉ")
    flush(stdout)
    return obs_pol == target
end

design_ok = Bool[]
if SEED44_CTX[] !== nothing && SEED44_CTX[].ctx !== nothing
    s44 = SEED44_CTX[]
    # DJ sur seed 44 collapse littéral (observé) -> cible 1 : le basculer en inversé.
    push!(design_ok, designed_flip!(g, ns, s44.ckpt, s44.ctx, :inverted_collapse))
    # cible 2 (contrôle de pilotage dans l'autre sens) : maintenir/forcer littéral.
    push!(design_ok, designed_flip!(g, ns, s44.ckpt, s44.ctx, :literal_collapse))
end

# ── Verdict global ───────────────────────────────────────────────────────────
println("\n", "═"^70)
println("VERDICT CONJECTURE 1 ", SMOKE ? "[SMOKE -- non significatif]" : "")
c1a_all = all(r.c1a >= 0.90 for r in ALL)
c1b_all = all(r.c1b for r in ALL)
qual_configs = [r for r in ALL if r.qual]
c1c_sign_all = all(r.c1c_sign for r in qual_configs)
println("  C1a (>=0.90 partout)   : ", c1a_all ? "CONFIRMÉ" : "ÉCHEC",
        "  (min = ", round(minimum(r.c1a for r in ALL), digits=3), ")")
println("  C1b (±0.10 partout)    : ", c1b_all ? "CONFIRMÉ" : "ÉCHEC",
        "  (max écart = ", round(maximum(max(abs(r.pred_invA - r.obs_invA), abs(r.pred_litB - r.obs_litB)) for r in ALL), digits=3), ")")
println("  C1c règle du signe     : ", isempty(qual_configs) ? "AUCUNE config qualifiante" :
        (c1c_sign_all ? "CONFIRMÉ ($(length(qual_configs)) configs qualifiantes)" : "ÉCHEC"))
println("  C1c pilotage conçu     : ", isempty(design_ok) ? "non testé" :
        (all(design_ok) ? "RÉUSSI (les deux cibles)" : "PARTIEL/ÉCHEC ($design_ok)"))
println("  Écart d'interaction inter-carriers (act. gelées vs poids) : max = ",
        round(maximum(r.equiv_err for r in ALL), digits=5),
        "  (0 exact attendu pour D1 ; >0 pour les configs jointes = erreur d'idéalisation mesurée)")
for r in ALL
    println("  RESUME_C1 inst=$(r.name) cfg=$(r.cfgname) C1a=$(round(r.c1a,digits=3)) autre=$(round(r.other_rate,digits=3)) ",
            "predinvA=$(round(r.pred_invA,digits=3)) obsinvA=$(round(r.obs_invA,digits=3)) ",
            "predlitB=$(round(r.pred_litB,digits=3)) obslitB=$(round(r.obs_litB,digits=3)) C1b=$(r.c1b) ",
            "qualif=$(r.qual) pred=$(r.pred_pol) obs=$(r.obs_pol)")
end
println("═"^70)
