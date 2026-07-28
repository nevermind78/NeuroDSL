# ══════════════════════════════════════════════════════════════════════════════
# RÉPLICATION SUR UNE SECONDE TÂCHE ET ARCHITECTURE, GENUINEMENT DIFFÉRENTES
# (artilce/code4.tex, §7 -- demande explicite du coordinateur : pas une
# nouvelle graine de la tâche du marqueur, une tâche et une architecture
# distinctes, entraînées FRAÎCHEMENT -- voir notebook/bidir_recall_task_experiment.jl
# pour ce qui diffère exactement).
#
# CE QUI EST TESTÉ ICI, sur le SEUL checkpoint bidir (aucun réentraînement dans
# CE script -- l'entraînement est fait, une fois, par bidir_recall_task_experiment.jl) :
#
#   (1) Exactitude patch = édition de poids sur le carrier D1 seul
#       (Proposition patchedit) -- gap mesuré entre la route "activations
#       gelées" et la route "poids réellement édités", sur des sondes neuves.
#   (2) La relation monotone entre magnitude d'interaction et fidélité du
#       modèle idéalisé (C1a), sur TOUS les sous-ensembles non vides de
#       carriers trouvés (comme verify_config_band_oos.jl, mais une seule
#       instance -- il n'y a qu'un seul checkpoint bidir, pas de découpage
#       in-sample/hors-échantillon à tester ici).
#   (3) (bonus, non forcé) un analogue du renversement de polarité de seed 44 :
#       deux sous-ensembles emboîtés qui font basculer la même paire appariée
#       vers des branches opposées.
#
# Corrélation de Spearman calculée avec la définition STANDARD (rangs moyens,
# StatsBase.corspearman) -- voir notebook/verify_interaction_spearman.jl pour
# pourquoi ce choix est explicite ici (un audit précédent a trouvé qu'une
# règle de rangs non standard avait été utilisée par erreur ailleurs).
# ══════════════════════════════════════════════════════════════════════════════

const SMOKE = get(ENV, "BIDIR_SMOKE", "0") == "1"
include(joinpath(@__DIR__, "bidir_recall_task_experiment.jl"))
using LinearAlgebra, JSON, StatsBase, Random

const CKPT = get(ENV, "BIDIR_CKPT", joinpath(@__DIR__, "bidir_ckpt", "bidir_task"))
(isfile(CKPT * ".json") && isfile(CKPT * ".bin")) || error("Checkpoint bidir manquant : $CKPT (lancez d'abord bidir_recall_task_experiment.jl avec BIDIR_SAVE=...)")

const N_HEADS_B = N_HEADS2
const D_HEAD_B  = DIM2 ÷ N_HEADS_B
const N_PAIRS_FAM = SMOKE ? 5 : 60
const N_SWEEP      = SMOKE ? 2 : 10
const N_DELTA       = SMOKE ? 4 : 40
const SEED_SWEEP   = 53421            # NEUF -- indépendant de tout script marker_*
const SEED_DELTA   = 91827
const SEED_PROBE_A = 20260810
const SEED_PROBE_B = 20260811
const SEED_DELTA_CONTROL = 20260812   # ré-estimation, contrôle de falsifiabilité D1

head_site_b(l, h) = Symbol("layer_$(l)_mha_ao_h$(h)")
mlp_site_b(l)     = Symbol("layer_$(l)_mlp_out")
site_layer_b(s)   = parse(Int, match(r"^layer_(\d+)_", String(s)).captures[1])
is_mlp_b(s)       = endswith(String(s), "_mlp_out")
const CANDIDATES_B = Symbol[]
for l in 1:N_LAYERS2
    for h in 1:N_HEADS_B; push!(CANDIDATES_B, head_site_b(l, h)); end
    push!(CANDIDATES_B, mlp_site_b(l))
end

dev = NeuroDSL.Backend.CUDADevice()
ns = :bidir_task
g, logits = build_bidir_graph(dev, ns; dim=DIM2, n_heads=N_HEADS2, hidden_dim=2*DIM2, n_layers=N_LAYERS2)
NeuroDSL.load_graph!(g, ns, CKPT; overwrite=true)
p1 = evaluate_bidir(g, logits, ns; n_eval=400)
println("P1(bidir, checkpoint) : ", p1)
(p1.acc_F >= 0.90 && p1.acc_R >= 0.90) || @warn "Checkpoint bidir sous le seuil attendu -- résultats à lire avec prudence"

function run_forward_b!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN2); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
end
run_and_capture_b!(g, ns, tokens) = (out = run_forward_b!(g, ns, tokens); (out, NeuroDSL.capture_activations(g, ns)))
full_reset_b!(g, ns) = (NeuroDSL.invalidate_all!(g; namespace=ns); NeuroDSL.demand!(g, :final_logits; namespace=ns); g)

# sop(x) := logit(v_t) - logit(k_t) ; format F correct <=> sop>0, format R
# correct <=> sop<0 -- exactement l'axe partagé décrit dans
# bidir_recall_task_experiment.jl.
sop_b(out, c) = Float64(out[1, c.v_t] - out[1, c.k_t])
category_b(out, c) = (a = argmax(vec(out)); a == c.v_t ? :chaseF : (a == c.k_t ? :chaseR : :other))

# Paire appariée : MÊME contexte de paires clé/valeur, les deux formats F et R
# partagent (k_t, v_t). "Propre" = le réseau répond juste dans les deux sens.
function draw_clean_bidir_pair!(g, ns, rng)
    for _ in 1:400
        keys_, vals_, t, k_t, v_t = let
            local keys, vals, t, k_t, v_t
            while true
                keys = distinct_sample(rng, V2, N_PAIRS2)
                vals = distinct_sample(rng, V2, N_PAIRS2)
                t = rand(rng, 1:N_PAIRS2)
                k_t, v_t = keys[t], vals[t]
                k_t != v_t && break
            end
            (keys, vals, t, k_t, v_t)
        end
        pair_order = shuffle(rng, 1:N_PAIRS2)
        ctx = Int[]
        for i in pair_order; push!(ctx, keys_[i]); push!(ctx, vals_[i]); end
        tokF = vcat(ctx, [MARKER_FWD, k_t])
        tokR = vcat(ctx, [MARKER_BWD, v_t])
        c = (; tokF, tokR, k_t, v_t)
        outF = run_forward_b!(g, ns, tokF); outR = run_forward_b!(g, ns, tokR)
        dF, dR = sop_b(outF, c), sop_b(outR, c)
        if SMOKE
            abs(dF - dR) > 1e-6 && return c
        else
            (argmax(vec(outF)) == v_t && argmax(vec(outR)) == k_t && dF > 0 > dR) && return c
        end
    end
    error("pas de paire bidir propre")
end

function single_site_sweep_b!(g, ns; n_pairs, seed)
    rng = MersenneTwister(seed)
    sums = Dict{Symbol,Float64}(s => 0.0 for s in CANDIDATES_B)
    for _ in 1:n_pairs
        c = draw_clean_bidir_pair!(g, ns, rng)
        dF, cacheF = run_and_capture_b!(g, ns, c.tokF)
        dR, cacheR = run_and_capture_b!(g, ns, c.tokR)
        ddF, ddR = sop_b(dF, c), sop_b(dR, c)
        for s in CANDIDATES_B
            NeuroDSL.patch_node!(g, s, cacheF; namespace=ns)
            out = Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
            sums[s] += (sop_b(out, c) - ddR) / (ddF - ddR)
            NeuroDSL.patch_node!(g, s, cacheR; namespace=ns)
            NeuroDSL.demand!(g, :final_logits; namespace=ns)
        end
        full_reset_b!(g, ns)
    end
    return Dict{Symbol,Float64}(s => sums[s] / n_pairs for s in CANDIDATES_B)
end

function collect_deltas_b!(g, ns, sites; n_pairs, seed)
    rng = MersenneTwister(seed)
    rows = Dict{Symbol,Vector{Vector{Float64}}}(s => Vector{Vector{Float64}}() for s in sites)
    for _ in 1:n_pairs
        c = draw_clean_bidir_pair!(g, ns, rng)
        _, cacheF = run_and_capture_b!(g, ns, c.tokF)
        _, cacheR = run_and_capture_b!(g, ns, c.tokR)
        for s in sites
            aF = Array(cacheF[s]); aR = Array(cacheR[s])
            for row in (SEQ_LEN2 - 1, SEQ_LEN2)
                push!(rows[s], vec(Float64.(aF[row, :] .- aR[row, :])))
            end
        end
    end
    return rows
end

function format_subspace_b(vecs; energy=0.90, kcap::Int)
    X = reduce(hcat, vecs); F = svd(X)
    e = F.S .^ 2; ce = cumsum(e) ./ sum(e)
    k = min(something(findfirst(>=(energy), ce), length(ce)), kcap)
    return F.U[:, 1:k]
end

_current_W_b(g, ns, wsym) = Matrix{Float32}(Array(NeuroDSL.node(g, wsym; namespace=ns).value))
projector_b(U) = (d = size(U, 1); Matrix{Float32}(I, d, d) .- Float32.(U) * Float32.(U)')

function projectors_b(subs::Dict{Symbol,<:AbstractMatrix})
    P = Dict{Symbol,Matrix{Float32}}()
    for (s, U) in subs; P[s] = projector_b(U); end
    return P
end

function directional_weights_b(g, ns, subs::Dict{Symbol,<:AbstractMatrix})
    w = Dict{Symbol,Matrix{Float32}}()
    for (s, U) in subs
        Uf = Float32.(U)
        m = match(r"^layer_(\d+)_mha_ao_h(\d+)$", String(s))
        if m !== nothing
            l = parse(Int, m.captures[1]); h = parse(Int, m.captures[2])
            wsym = Symbol("layer_$(l)_mha_output_W")
            W = get!(() -> _current_W_b(g, ns, wsym), w, wsym)
            cols = (h-1)*D_HEAD_B+1 : h*D_HEAD_B
            W[:, cols] = W[:, cols] * (Matrix{Float32}(I, D_HEAD_B, D_HEAD_B) .- Uf * Uf')
        else
            wsym = Symbol("layer_$(match(r"^layer_(\d+)_mlp_out$", String(s)).captures[1])_mlp_w2")
            w[wsym] = (Matrix{Float32}(I, DIM2, DIM2) .- Uf * Uf') * _current_W_b(g, ns, wsym)
        end
    end
    return w
end

function measure_gq_b!(g, ns, tokens, c, P)
    out = run_forward_b!(g, ns, tokens)
    gval = sop_b(out, c)
    proj_vals = Dict{Symbol,Any}()
    for (s, Pm) in P
        proj_vals[s] = Array(NeuroDSL.node(g, s; namespace=ns).value) * Pm
    end
    NeuroDSL.patch_nodes!(g, collect(keys(P)), proj_vals; namespace=ns)
    out_p = Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
    return gval, gval - sop_b(out_p, c)
end

med(v) = isempty(v) ? NaN : sort(v)[cld(length(v), 2)]

const CARRIER_THRESH = parse(Float64, get(ENV, "BIDIR_CARRIER_THRESH", "0.25"))

println("\n", "═"^74); println("Découverte des carriers (bidir)"); flush(stdout)
MEAN_R = single_site_sweep_b!(g, ns; n_pairs=N_SWEEP, seed=SEED_SWEEP)
println("Tous les sites candidats (r moyen de recovery, tri décroissant) :")
for s in sort(CANDIDATES_B; by=s -> -MEAN_R[s])
    println("    ", rpad(String(s), 22), round(MEAN_R[s], digits=4))
end
println("Seuil carrier utilisé : r >= $CARRIER_THRESH  (BIDIR_CARRIER_THRESH)")
carriers = sort([s for s in CANDIDATES_B if MEAN_R[s] >= CARRIER_THRESH]; by=s -> -MEAN_R[s])
println("Carriers : ", [(String(s), round(MEAN_R[s], digits=3)) for s in carriers])
if get(ENV, "BIDIR_SWEEP_ONLY", "0") == "1"
    println("BIDIR_SWEEP_ONLY=1 -- arrêt après la découverte des carriers.")
    exit(0)
end
isempty(carriers) && error("Aucun carrier trouvé -- checkpoint bidir insuffisamment entraîné ou tâche non résolue par un circuit compact.")
main_layer = minimum(site_layer_b(s) for s in carriers)
cmain = [s for s in carriers if site_layer_b(s) == main_layer]
d1 = cmain[1]
println("D1 = $d1  (", is_mlp_b(d1) ? "MLP" : "tête", ", r=", round(MEAN_R[d1], digits=3), ")")

deltas = collect_deltas_b!(g, ns, carriers; n_pairs=N_DELTA, seed=SEED_DELTA)
subs_all = Dict{Symbol,Matrix{Float64}}(
    s => format_subspace_b(deltas[s]; kcap = is_mlp_b(s) ? 8 : 4) for s in carriers)

rngA = MersenneTwister(SEED_PROBE_A); rngB = MersenneTwister(SEED_PROBE_B)
pairs = [draw_clean_bidir_pair!(g, ns, rngA) for _ in 1:N_PAIRS_FAM]

# ── (1) Exactitude patch = poids sur D1 seul (Proposition patchedit) ─────────
println("\n", "═"^74); println("(1) Exactitude D1 : patch (activations gelées) vs poids réellement édités")
println("═"^74)
Ud1 = subs_all[d1]
Pmd1 = projector_b(Ud1)
Ud1_ctrl = format_subspace_b(collect_deltas_b!(g, ns, [d1]; n_pairs=N_DELTA, seed=SEED_DELTA_CONTROL)[d1]; kcap = is_mlp_b(d1) ? 8 : 4)
Pmd1_ctrl = projector_b(Ud1_ctrl)

function sop_patched_b!(g, ns, tokens, c, s, Pm)
    run_forward_b!(g, ns, tokens)
    v = Array(NeuroDSL.node(g, s; namespace=ns).value) * Pm
    NeuroDSL.patch_nodes!(g, [s], Dict(s => v); namespace=ns)
    return sop_b(Array(NeuroDSL.demand!(g, :final_logits; namespace=ns)), c)
end

frozen_d1 = NamedTuple[]
for c in pairs
    push!(frozen_d1, (; c, sF = sop_patched_b!(g, ns, c.tokF, c, d1, Pmd1), sR = sop_patched_b!(g, ns, c.tokR, c, d1, Pmd1),
                       cF = sop_patched_b!(g, ns, c.tokF, c, d1, Pmd1_ctrl), cR = sop_patched_b!(g, ns, c.tokR, c, d1, Pmd1_ctrl)))
end
full_reset_b!(g, ns)
w_d1 = directional_weights_b(g, ns, Dict(d1 => Ud1))
restore_d1 = Dict{Symbol,Matrix{Float32}}(k => _current_W_b(g, ns, k) for k in keys(w_d1))
NeuroDSL.set_params!(g, ns, w_d1)
gaps_d1 = Float64[]; gaps_ctrl_d1 = Float64[]; sop_scale_d1 = Float64[]
for m in frozen_d1
    outF = run_forward_b!(g, ns, m.c.tokF); outR = run_forward_b!(g, ns, m.c.tokR)
    tF, tR = sop_b(outF, m.c), sop_b(outR, m.c)
    push!(gaps_d1, abs(m.sF - tF), abs(m.sR - tR))
    push!(gaps_ctrl_d1, abs(m.cF - tF), abs(m.cR - tR))
    push!(sop_scale_d1, abs(tF), abs(tR))
end
NeuroDSL.set_params!(g, ns, restore_d1); full_reset_b!(g, ns)
println("  gap D1 (patch vs poids édités), $(length(gaps_d1)) sondes : médiane=", med(gaps_d1),
        "  max=", maximum(gaps_d1), "  (échelle |sop| médiane=", round(med(sop_scale_d1), digits=3), ")")
println("  contrôle (sous-espace ré-estimé, patch seul) : médiane=", round(med(gaps_ctrl_d1), digits=4),
        "  max=", round(maximum(gaps_ctrl_d1), digits=4))

# ── (2) Interaction magnitude vs fidélité (C1a), tous les sous-ensembles ────
println("\n", "═"^74); println("(2) Interaction vs fidélité idéalisée sur tous les sous-ensembles de carriers")
println("═"^74)

function criteria_b(meas, obs)
    agree = 0; total = 0; nother = 0
    for i in eachindex(meas)
        m = meas[i]; o = obs[i]
        for (sv, cv) in ((m.gF - m.qF, o.cF), (m.gR - m.qR, o.cR))
            pred = sv > 0 ? :chaseF : :chaseR
            total += 1
            cv == :other ? (nother += 1) : (agree += (pred == cv))
        end
    end
    return (; c1a = agree/total, other_rate = nother/total)
end

function eval_bidir_config!(g, ns, cfgname, subs)
    P = projectors_b(subs)
    meas = NamedTuple[]
    for c in pairs
        gF, qF = measure_gq_b!(g, ns, c.tokF, c, P)
        gR, qR = measure_gq_b!(g, ns, c.tokR, c, P)
        push!(meas, (; c, gF, qF, gR, qR))
    end
    w = directional_weights_b(g, ns, subs)
    restore = Dict{Symbol,Matrix{Float32}}(k => _current_W_b(g, ns, k) for k in keys(w))
    NeuroDSL.set_params!(g, ns, w)
    obs = NamedTuple[]; equiv_err = 0.0
    for m in meas
        outF = run_forward_b!(g, ns, m.c.tokF); outR = run_forward_b!(g, ns, m.c.tokR)
        equiv_err = max(equiv_err, abs((m.gF - m.qF) - sop_b(outF, m.c)), abs((m.gR - m.qR) - sop_b(outR, m.c)))
        push!(obs, (; cF = category_b(outF, m.c), cR = category_b(outR, m.c)))
    end
    NeuroDSL.set_params!(g, ns, restore); full_reset_b!(g, ns)
    crit = criteria_b(meas, obs)
    layers = sort(unique(site_layer_b(s) for s in keys(subs)))
    println("  [$cfgname] couches=", layers, " k=", sum(size(U,2) for U in values(subs)),
            " interaction=", round(equiv_err, digits=4), "  C1a=", round(crit.c1a, digits=4),
            " autre=", round(crit.other_rate, digits=4))
    flush(stdout)
    return (; cfgname, layers, k_total = sum(size(U,2) for U in values(subs)),
             interaction = equiv_err, c1a = crit.c1a, other_rate = crit.other_rate,
             meas, obs)
end

nc = length(carriers)
CONFIGS = NamedTuple[]
for mask in 1:(2^nc - 1)
    sites = [carriers[i] for i in 1:nc if (mask >> (i-1)) & 1 == 1]
    cfgname = join([replace(String(s), "layer_" => "L", "_mha_ao_h" => "h", "_mlp_out" => "mlp") for s in sites], "+")
    subs = Dict{Symbol,Matrix{Float64}}(s => subs_all[s] for s in sites)
    push!(CONFIGS, eval_bidir_config!(g, ns, cfgname, subs))
end

interaction = [c.interaction for c in CONFIGS]
c1a_vals = [c.c1a for c in CONFIGS]
rho = length(CONFIGS) >= 3 ? corspearman(interaction, c1a_vals) : NaN
println("\n  $(length(CONFIGS)) configurations. Spearman(interaction, C1a) [rangs moyens, standard] = ", round(rho, digits=3))

# ── (3) bonus : recherche d'un renversement de polarité type seed 44 ────────
println("\n", "═"^74); println("(3) Recherche (non forcée) d'un renversement de polarité entre sous-ensembles emboîtés")
println("═"^74)
found_reversal = false
for i in 1:length(CONFIGS), j in 1:length(CONFIGS)
    i == j && continue
    si = Set(split(CONFIGS[i].cfgname, "+")); sj = Set(split(CONFIGS[j].cfgname, "+"))
    issubset(si, sj) || continue
    # vérifie si les deux collapsent la MÊME paire (parmi `pairs`) sur des branches opposées
    for (k, c) in enumerate(pairs)
        mi, oi = CONFIGS[i].meas[k], CONFIGS[i].obs[k]
        mj, oj = CONFIGS[j].meas[k], CONFIGS[j].obs[k]
        # collapse = les deux formats prédits du même côté (idéalisé) après ablation
        pi_F = sign(mi.gF - mi.qF); pi_R = sign(mi.gR - mi.qR)
        pj_F = sign(mj.gF - mj.qF); pj_R = sign(mj.gR - mj.qR)
        if pi_F == pi_R && pj_F == pj_R && pi_F != pj_F
            println("  RENVERSEMENT : $(CONFIGS[i].cfgname) (pol=", pi_F, ") vs $(CONFIGS[j].cfgname) (pol=", pj_F, ") sur la paire #$k")
            global found_reversal = true  # bloc `if` top-niveau : soft scope ambigu sans `global` (Julia)
        end
    end
end
found_reversal || println("  Aucun renversement de polarité trouvé sur les $(length(pairs)) paires sondées et $(length(CONFIGS)) configurations.")

out = Dict{String,Any}(
    "protocol" => Dict("sweep_seed"=>SEED_SWEEP, "delta_seed"=>SEED_DELTA, "control_delta_seed"=>SEED_DELTA_CONTROL,
                        "probe_seed_A"=>SEED_PROBE_A, "probe_seed_B"=>SEED_PROBE_B, "n_pairs"=>N_PAIRS_FAM,
                        "task"=>"bidir_recall", "dim"=>DIM2, "n_heads"=>N_HEADS2, "n_layers"=>N_LAYERS2, "vocab"=>VOCAB_SIZE2),
    "P1" => Dict("acc_F"=>p1.acc_F, "acc_R"=>p1.acc_R),
    "carriers" => [(String(s), MEAN_R[s]) for s in carriers],
    "d1" => Dict("site"=>String(d1), "kind"=>(is_mlp_b(d1) ? "mlp" : "head"), "r"=>MEAN_R[d1],
                 "gap_median"=>med(gaps_d1), "gap_max"=>maximum(gaps_d1), "sop_scale_median"=>med(sop_scale_d1),
                 "ctrl_gap_median"=>med(gaps_ctrl_d1), "ctrl_gap_max"=>maximum(gaps_ctrl_d1)),
    "configs" => [Dict("cfgname"=>c.cfgname, "layers"=>c.layers, "k_total"=>c.k_total,
                        "interaction"=>c.interaction, "c1a"=>c.c1a, "other_rate"=>c.other_rate) for c in CONFIGS],
    "spearman_rho" => rho,
    "polarity_reversal_found" => found_reversal,
)
open(joinpath(@__DIR__, "bidir_replication_results.json"), "w") do io
    JSON.print(io, out)
end
println("\nÉcrit -> notebook/bidir_replication_results.json")
