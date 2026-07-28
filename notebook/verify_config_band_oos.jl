# ══════════════════════════════════════════════════════════════════════════════
# TEST HORS-ÉCHANTILLON DE LA BANDE DE SÉPARATION (artilce/code4.tex, §7,
# fig:interaction) -- aucun réentraînement, checkpoints existants uniquement.
#
# CE QUI EST TESTÉ
# ----------------
# L'article rapporte, sur 14 configurations (5 instances x {D1, DJ, [DJA]}),
# que la magnitude d'interaction SÉPARE les configurations où le modèle idéalisé
# tient de celles où il échoue : les 9 configurations jointes se scindent en
# "les deux critères passent" (interaction <= 6.78) et "au moins un échoue"
# (interaction >= 9.95), sans recouvrement, avec une bande vide entre les deux.
#
# PROBLÈME : cette bande est LUE SUR CES 14 POINTS. Elle est descriptive, pas
# validée. Ce script la GÈLE et la teste hors échantillon.
#
# PROTOCOLE (pré-enregistré dans ce fichier avant exécution)
# ----------------------------------------------------------
#   1. Bande GELÉE aux valeurs publiées : BAND_LO = 6.78, BAND_HI = 9.95.
#      Règle de décision figée : une config qui passe C1a ET C1b doit tomber
#      sous BAND_LO ; une config qui en échoue au moins un doit tomber au-dessus
#      de BAND_HI ; AUCUNE config ne doit tomber dans ]BAND_LO, BAND_HI[.
#   2. Seuils C1a >= 0.90 et C1b a +/-0.10 : identiques à
#      marker_conj1_verify.jl, non retouchés.
#   3. Jeu de configurations ÉLARGI : pour chaque instance, TOUS les sous-
#      ensembles non vides de l'ensemble de carriers (déterminé exactement comme
#      dans le pipeline d'origine : sweep graine 4242, seuil r >= 0.25). Les 3
#      sous-ensembles déjà publiés par instance (D1 = {carrier principal de la
#      couche porteuse}, DJ = carriers de la couche porteuse, DJA = tous les
#      carriers) sont marqués IN-SAMPLE ; tous les autres sont HORS ÉCHANTILLON
#      et n'ont jamais servi à fixer quoi que ce soit.
#   4. Sondes NEUVES : graines 20260728 / 20260729, distinctes de conj1
#      (90210/31337) ET de verify_d1_protocol_gap.jl (20260726/20260727).
#   5. CONTRÔLE DE STABILITÉ (split-half) : les mêmes sondes sont scindées en
#      deux moitiés disjointes ; C1a, C1b et le verdict de bande sont recalculés
#      sur chaque moitié. Si le verdict dépend de la moitié, la séparation est
#      un artefact d'échantillonnage et c'est ce qui sera rapporté.
#
# Les quantités mesurées (interaction_max, C1a, C1b) sont calculées par le MÊME
# code que marker_conj1_verify.jl, recopié sans modification fonctionnelle.
# ══════════════════════════════════════════════════════════════════════════════

const SMOKE = get(ENV, "MARKER_SMOKE", "0") == "1"
ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))
using LinearAlgebra, JSON

const CKPT_DIR = get(ENV, "MARKER_CKPT_DIR", joinpath(@__DIR__, "marker_ckpt"))
const INSTANCES = [("inst2",   joinpath(CKPT_DIR, "marker_task")),
                   ("seed_11", joinpath(CKPT_DIR, "seed_11")),
                   ("seed_22", joinpath(CKPT_DIR, "seed_22")),
                   ("seed_33", joinpath(CKPT_DIR, "seed_33")),
                   ("seed_44", joinpath(CKPT_DIR, "seed_44"))]

# ── Bande GELÉE (valeurs publiées, non recalculées ici) ──────────────────────
const BAND_LO = 6.78
const BAND_HI = 9.95
const C1A_MIN = 0.90
const C1B_TOL = 0.10

const N_HEADS = 4
const D_HEAD  = DIM ÷ N_HEADS
const N_PAIRS_FAM = SMOKE ? 6 : 75
const N_SWEEP     = SMOKE ? 2 : 8
const N_DELTA     = SMOKE ? 4 : 32
const SEED_SWEEP  = 4242
const SEED_DELTA  = 1618
const SEED_PROBE_A = 20260728
const SEED_PROBE_B = 20260729

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

function sample_contrast(rng)
    while true
        k = rand(rng, 1:V); sk = SIGMA[k]
        rest = shuffle(rng, collect(setdiff(1:V, (k, sk))))[1:N_PAIRS-2]
        ks = vcat([k, sk], rest); vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS); push!(ctx, ks[i]); push!(ctx, vs[i]); end
        return (; tokA = vcat(ctx, [MARKER_A, sk]), tokB = vcat(ctx, [MARKER_B, sk]),
                 lit = vs[2], inv = vs[1])
    end
end
function sample_contrast_A(rng)
    while true
        k = rand(rng, 1:V); ik = SIGMA_INV[k]
        rest = shuffle(rng, collect(setdiff(1:V, (k, ik))))[1:N_PAIRS-2]
        ks = vcat([k, ik], rest); vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS); push!(ctx, ks[i]); push!(ctx, vs[i]); end
        return (; tokA = vcat(ctx, [MARKER_A, k]), tokB = vcat(ctx, [MARKER_B, k]),
                 lit = vs[1], inv = vs[2])
    end
end

sop(out, p) = Float64(out[1, p.lit] - out[1, p.inv])
category(out, p) = (a = argmax(vec(out)); a == p.lit ? :lit : (a == p.inv ? :inv : :other))

function draw_clean_pair!(g, ns, rng)
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

function collect_deltas!(g, ns, sites; n_pairs, seed)
    rng = MersenneTwister(seed)
    rows = Dict{Symbol,Vector{Vector{Float64}}}(s => Vector{Vector{Float64}}() for s in sites)
    for _ in 1:n_pairs
        c = draw_clean_pair!(g, ns, rng)
        _, dcache = run_and_capture!(g, ns, c.tokA)
        _, rcache = run_and_capture!(g, ns, c.tokB)
        for s in sites
            da = Array(dcache[s]); ra = Array(rcache[s])
            for row in (SEQ_LEN - 1, SEQ_LEN)
                push!(rows[s], vec(Float64.(da[row, :] .- ra[row, :])))
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

# Évalue C1a/C1b sur un sous-ensemble d'indices (pour le contrôle split-half).
function criteria(meas, obs, idx)
    agree = 0; total = 0; nother = 0
    for i in idx
        m = meas[i]; o = obs[i]
        for (sv, cv) in ((m.gA - m.qA, o.catA), (m.gB - m.qB, o.catB))
            pred = sv > 0 ? :lit : :inv
            total += 1
            cv == :other ? (nother += 1) : (agree += (pred == cv))
        end
    end
    mA = [i for i in idx if meas[i].fam == :A]
    mB = [i for i in idx if meas[i].fam == :B]
    pred_invA = sum(meas[i].gA - meas[i].qA < 0 for i in mA) / length(mA)
    obs_invA  = sum(obs[i].catA == :inv for i in mA) / length(mA)
    pred_litB = sum(meas[i].gB - meas[i].qB > 0 for i in mB) / length(mB)
    obs_litB  = sum(obs[i].catB == :lit for i in mB) / length(mB)
    c1a = agree / total
    c1b = abs(pred_invA - obs_invA) <= C1B_TOL && abs(pred_litB - obs_litB) <= C1B_TOL
    return (; c1a, c1b, other_rate = nother/total, pass = (c1a >= C1A_MIN && c1b))
end

function eval_config!(g, ns, name, cfgname, in_sample, subs, pairsA, pairsB)
    P = projectors(subs)
    meas = NamedTuple[]
    for (fam, plist) in ((:A, pairsA), (:B, pairsB))
        for p in plist
            gA, qA = measure_gq!(g, ns, p.tokA, p, P)
            gB, qB = measure_gq!(g, ns, p.tokB, p, P)
            push!(meas, (; fam, p, gA, qA, gB, qB))
        end
    end
    w = directional_weights(g, ns, subs)
    restore = Dict{Symbol,Matrix{Float32}}(k => _current_W(g, ns, k) for k in keys(w))
    NeuroDSL.set_params!(g, ns, w)
    obs = NamedTuple[]; equiv_err = 0.0
    for m in meas
        outA = run_forward!(g, ns, m.p.tokA); outB = run_forward!(g, ns, m.p.tokB)
        equiv_err = max(equiv_err, abs((m.gA - m.qA) - sop(outA, m.p)), abs((m.gB - m.qB) - sop(outB, m.p)))
        push!(obs, (; catA = category(outA, m.p), catB = category(outB, m.p)))
    end
    NeuroDSL.set_params!(g, ns, restore); full_reset!(g, ns)

    n = length(meas)
    full = criteria(meas, obs, 1:n)
    h1   = criteria(meas, obs, 1:2:n)      # moitiés entrelacées : même mélange A/B
    h2   = criteria(meas, obs, 2:2:n)
    # verdict de bande, RÈGLE GELÉE
    inband = BAND_LO < equiv_err < BAND_HI
    consistent = full.pass ? (equiv_err <= BAND_LO) : (equiv_err >= BAND_HI)
    println("  [$name/$cfgname]", in_sample ? " (IN-SAMPLE)" : " (hors éch.)",
            " k=$(sum(size(U,2) for U in values(subs)))",
            " interaction=", round(equiv_err, digits=5),
            "  C1a=", round(full.c1a, digits=3), " C1b=", full.c1b,
            " -> ", full.pass ? "PASSE" : "ÉCHOUE",
            inband ? "  [DANS LA BANDE]" : (consistent ? "  [cohérent]" : "  [VIOLE LA RÈGLE]"),
            "  split-half: ", h1.pass == h2.pass == full.pass ? "stable" : "INSTABLE (h1=$(h1.pass) h2=$(h2.pass))")
    flush(stdout)
    return (; name, cfgname, in_sample, sites = sort(String.(collect(keys(subs)))),
             k_total = sum(size(U,2) for U in values(subs)),
             interaction = equiv_err, c1a = full.c1a, c1b = full.c1b,
             other_rate = full.other_rate, pass = full.pass,
             inband, consistent,
             c1a_h1 = h1.c1a, c1a_h2 = h2.c1a, pass_h1 = h1.pass, pass_h2 = h2.pass,
             split_stable = (h1.pass == h2.pass == full.pass), n_probes = 2n)
end

ALL = NamedTuple[]
for (name, ckpt) in INSTANCES
    println("\n", "═"^74); println("INSTANCE $name"); flush(stdout)
    NeuroDSL.load_graph!(g, ns, ckpt; overwrite=true)
    println("  P1 : ", evaluate_marker(g, logits, ns; n_eval=200))
    MEAN_R = single_site_sweep!(g, ns; n_pairs=N_SWEEP, seed=SEED_SWEEP)
    carriers = sort([s for s in CANDIDATES if MEAN_R[s] >= 0.25]; by=s -> -MEAN_R[s])
    isempty(carriers) && continue
    main_layer = minimum(site_layer(s) for s in carriers)
    cmain = [s for s in carriers if site_layer(s) == main_layer]
    println("  Carriers : ", [(String(s), round(MEAN_R[s], digits=3)) for s in carriers])
    deltas = collect_deltas!(g, ns, carriers; n_pairs=N_DELTA, seed=SEED_DELTA)
    subs_all = Dict{Symbol,Matrix{Float64}}(
        s => format_subspace(deltas[s]; kcap = endswith(String(s), "_mlp_out") ? 8 : 4) for s in carriers)

    # sous-ensembles PUBLIÉS (in-sample)
    published = Set{Vector{Symbol}}()
    push!(published, sort([cmain[1]]; by=String))
    push!(published, sort(copy(cmain); by=String))
    length(carriers) > length(cmain) && push!(published, sort(copy(carriers); by=String))

    rngA = MersenneTwister(SEED_PROBE_A); rngB = MersenneTwister(SEED_PROBE_B)
    pairsA = [sample_contrast_A(rngA) for _ in 1:N_PAIRS_FAM]
    pairsB = [sample_contrast(rngB)   for _ in 1:N_PAIRS_FAM]

    nc = length(carriers)
    for mask in 1:(2^nc - 1)
        sites = [carriers[i] for i in 1:nc if (mask >> (i-1)) & 1 == 1]
        key = sort(copy(sites); by=String)
        in_sample = key in published
        cfgname = join([replace(String(s), "layer_" => "L", "_mha_ao_h" => "h", "_mlp_out" => "mlp") for s in sites], "+")
        subs = Dict{Symbol,Matrix{Float64}}(s => subs_all[s] for s in sites)
        push!(ALL, eval_config!(g, ns, name, cfgname, in_sample, subs, pairsA, pairsB))
    end
end

# ── Verdict ──────────────────────────────────────────────────────────────────
println("\n", "═"^74)
println("VERDICT -- test hors échantillon de la bande GELÉE ]$BAND_LO, $BAND_HI[")
println("═"^74)
oos = [r for r in ALL if !r.in_sample]
ins = [r for r in ALL if r.in_sample]
println("  configurations évaluées : ", length(ALL), " (", length(ins), " in-sample, ", length(oos), " hors échantillon)")
println("  sondes par configuration : ", isempty(ALL) ? 0 : ALL[1].n_probes)

for (lab, set) in (("IN-SAMPLE", ins), ("HORS ÉCHANTILLON", oos))
    isempty(set) && continue
    ps = [r for r in set if r.pass]; fs = [r for r in set if !r.pass]
    println("\n  $lab (", length(set), " configs) :")
    println("    passent les deux critères : ", length(ps),
            isempty(ps) ? "" : "   interaction max = $(round(maximum(r.interaction for r in ps), digits=3))")
    println("    échouent au moins un      : ", length(fs),
            isempty(fs) ? "" : "   interaction min = $(round(minimum(r.interaction for r in fs), digits=3))")
    nb = count(r -> r.inband, set)
    nv = count(r -> !r.consistent, set)
    println("    dans la bande gelée       : ", nb)
    println("    violent la règle gelée    : ", nv)
    if !isempty(ps) && !isempty(fs)
        sep = minimum(r.interaction for r in fs) - maximum(r.interaction for r in ps)
        println("    séparation observée       : ", round(sep, digits=3),
                sep > 0 ? "  (ordre préservé : tout passant < tout échouant)" : "  (ORDRE VIOLÉ : recouvrement)")
    end
end

nunstable = count(r -> !r.split_stable, ALL)
println("\n  CONTRÔLE split-half : ", nunstable, " / ", length(ALL),
        " configurations dont le verdict passe/échoue diffère entre les deux moitiés",
        nunstable == 0 ? "  (aucune : classification stable)" : "  (instabilité d'échantillonnage réelle)")

allv = count(r -> !r.consistent, ALL)
println("\n  RÉSULTAT GLOBAL : ", allv == 0 ?
        "la règle gelée n'est violée par AUCUNE des $(length(ALL)) configurations." :
        "$allv / $(length(ALL)) configurations violent la règle gelée.")

println("\n  Détail des violations / points en bande :")
for r in ALL
    (r.inband || !r.consistent) && println("    $(r.name)/$(r.cfgname) interaction=$(round(r.interaction,digits=3)) C1a=$(round(r.c1a,digits=3)) C1b=$(r.c1b) pass=$(r.pass) in_sample=$(r.in_sample)")
end

out = Dict{String,Any}(
    "protocol" => Dict("band_lo"=>BAND_LO, "band_hi"=>BAND_HI, "c1a_min"=>C1A_MIN, "c1b_tol"=>C1B_TOL,
                        "sweep_seed"=>SEED_SWEEP, "delta_seed"=>SEED_DELTA,
                        "probe_seed_A"=>SEED_PROBE_A, "probe_seed_B"=>SEED_PROBE_B,
                        "n_pairs_per_family"=>N_PAIRS_FAM),
    "configs" => [Dict(String(k)=>getfield(r, k) for k in keys(r)) for r in ALL],
)
open(joinpath(@__DIR__, "config_band_oos_results.json"), "w") do io
    JSON.print(io, out)
end
println("\nÉcrit -> notebook/config_band_oos_results.json")
