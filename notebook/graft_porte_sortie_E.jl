# ══════════════════════════════════════════════════════════════════════════════
# PORTE DE SORTIE E -- test court qui clôt la question de la greffe guidée par
# diagnostic.
#
# Contexte. Le run précédent (real_llm_graft_experiment_v2.jl, _run2.log) a
# étranglé `theta` du greffon à LR_B/500 avec wd=0.1 pour l'empêcher de dériver.
# Diagnostic accepté : dans cette configuration la greffe n'a plus qu'UN degré de
# liberté effectif (`alpha`) sur un bloc quasi gelé et quasi aléatoire -- le
# protocole ne testait donc plus l'hypothèse pour laquelle il avait été conçu.
#
# Test E. Ré-entraîner la greffe au MÊME site diagnostiqué avec `theta` à PLEIN
# LR, mais avec la protection faite correctement : correction de biais AdamW
# locale à l'âge du paramètre (catalogue 3.3), désormais implémentée de façon
# additive dans src/kernels.jl (kwarg `age`/`ages`, défaut = comportement
# inchangé). 1 graine, 1000 pas, backbone Phase A rechargé (pas de ré-entraînement).
#
# CRITÈRE PRÉ-ENREGISTRÉ (fixé avant le run) :
#   recovery finale OU meilleure recovery atteinte sur les 1000 pas  > 0.2
#     ⟹ la greffe fonctionne quand l'optimiseur est correctement réglé ; les trois
#        campagnes négatives précédentes s'expliquent par un défaut de réglage.
#   ≤ 0.2
#     ⟹ l'explication banale ("votre LR était mal réglé") est éliminée ; le
#        résultat négatif tient.
# Le verdict porte sur le BRAS PRÉ-ENREGISTRÉ (E_theta_full_agelocal) et sur lui
# seul. Les autres bras sont des diagnostics de contexte, explicitement étiquetés
# comme non pré-enregistrés :
#   - ctrl_theta_full_globalt : même chose SANS la correction d'âge (isole ce que
#     la correction fait réellement dans cette configuration) ;
#   - ladder_* : échelle de LR pour theta sur 3 ordres de grandeur -- si aucun
#     point de l'échelle ne franchit 0.2, "le LR était mal réglé" est éliminé bien
#     plus solidement que par un point unique ;
#   - prev_div500_wd01 : l'ancienne configuration prolongée à 1000 pas (elle était
#     à 0.2008 et ENCORE CROISSANTE au pas 295 dans diag_graft_v2_quick_fixB.log) ;
#   - temoin_full_agelocal : le bras E au site témoin aléatoire, pour calibrer.
#
# Signal d'alarme réinstrumenté (comme dans diag_graft_v2_quick.jl) : ||R(x;theta)||
# (RMS par élément de la sortie brute du greffon AVANT le gate alpha). Dans le run
# cassé il croissait ×26 en 300 pas sans borne.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics, Printf, LinearAlgebra, JSON

dev = NeuroDSL.Backend.CUDADevice()
export_dir = joinpath(@__DIR__, "graft_experiment_v2_export")
isdir(export_dir) || error("Backbone Phase A introuvable -- lancez d'abord real_llm_graft_experiment_v2.jl")
println("Device: ", dev)

# ── 1. Corpus + tokenizer (cache local, pas de téléchargement) ───────────────
const CORPUS_PATH = joinpath(@__DIR__, "data", "tinyshakespeare", "input.txt")
text = read(CORPUS_PATH, String)
chars = sort(unique(collect(text)))
stoi  = Dict(c => i for (i, c) in enumerate(chars))
encode(s::AbstractString) = [stoi[c] for c in s]
decode(ids::AbstractVector{<:Integer}) = String(chars[ids])
vocab_size = length(chars)

data    = encode(text)
n_train = floor(Int, 0.9 * length(data))
train_ids = data[1:n_train]
val_ids   = data[n_train+1:end]

block_size = 256
dim        = 256
n_heads     = 4
hidden_dim  = 512
n_layers    = 4
d_head      = dim ÷ n_heads

sample_window(rng, ids, bs) = begin
    i = rand(rng, 1:(length(ids) - bs))
    (ids[i:i+bs-1], ids[i+1:i+bs])
end

# ── 2. Reconstruction BIT-EXACTE de la fenêtre cible ARIEL du run précédent ──
# Le token de corruption avait été tiré par rng_scan=MersenneTwister(999) balayé
# séquentiellement sur candidates_raw. Ce tirage ne dépend QUE de la rng et de
# `orig_id` (aucun forward), donc on le rejoue à l'identique sur CPU seul.
function all_speaker_headers(val_ids, chars; min_name_len::Int=4)
    text_val = decode(val_ids)
    headers = NamedTuple[]
    for m in eachmatch(r"\n([A-Z][A-Z ]{2,})\:", text_val)
        name = String(strip(m.captures[1]))
        length(name) >= min_name_len || continue
        push!(headers, (; pos=m.offset + 1, name))
    end
    return headers
end
function build_candidates(headers, block_size::Int; k::Int=3, margin::Int=5)
    candidates = NamedTuple[]
    n = length(headers)
    for i in 1:n-1
        for jx in i+1:n
            gap = headers[jx].pos - headers[i].pos
            gap > block_size - 20 && break
            headers[i].name != headers[jx].name && continue
            kk = min(k, length(headers[i].name) - 1)
            kk < 1 && continue
            window_start = headers[i].pos - margin
            window_start < 1 && continue
            p1 = headers[i].pos - window_start + 1
            p2 = headers[jx].pos - window_start + 1
            p2 + kk > block_size && continue
            has_intervening_different = any(headers[m2].pos > headers[i].pos && headers[m2].pos < headers[jx].pos &&
                                             headers[m2].name != headers[i].name for m2 in i+1:jx-1)
            variant = has_intervening_different ? :long_gap : :adjacent
            push!(candidates, (; window_start, p1, p2, k=kk, name=headers[i].name, gap, variant))
        end
    end
    return candidates
end

headers        = all_speaker_headers(val_ids, chars)
candidates_raw = build_candidates(headers, block_size)
rng_scan = MersenneTwister(999)
target = nothing
for c in candidates_raw
    tokens_clean = val_ids[c.window_start:c.window_start+block_size-1]
    j = c.p2 + c.k - 1
    (j < 1 || j > block_size) && continue
    tokens_corrupt = copy(tokens_clean)
    orig_id = tokens_corrupt[c.p1 + c.k]
    new_id = orig_id
    while new_id == orig_id
        new_id = rand(rng_scan, 1:vocab_size)
    end
    tokens_corrupt[c.p1 + c.k] = new_id
    if c.name == "ARIEL" && c.gap == 64 && j == 72
        global target = (; c..., j, tokens_clean, tokens_corrupt, orig_id, new_id)
    end
end
target === nothing && error("Fenêtre ARIEL (gap=64, j=72) introuvable -- rejouer le screening")
tokens_clean, tokens_corrupt, j_target = target.tokens_clean, target.tokens_corrupt, target.j
@printf("Fenêtre cible ARIEL rejouée : variant=%s gap=%d j=%d | token corrompu %d -> %d\n",
        String(target.variant), target.gap, j_target, target.orig_id, target.new_id)

DOMINANT_SITE = :layer_1_mha_ao_h2   # site diagnostiqué du run précédent (patch=0.7463)
TEMOIN_SITE   = :layer_4_mha_ao_h4   # témoin aléatoire du run précédent (patch=0.0494)

# ── 3. Références clean/corrupted sur le backbone Phase A ────────────────────
ns_ref = :porteE_ref
g_ref = NeuroDSL.NeuroGraph(namespace=ns_ref, device=dev)
NeuroDSL.load_graph!(g_ref, ns_ref, joinpath(export_dir, "base"))
logits_sym = :lm_head_out
@assert haskey(g_ref.nodes[ns_ref], logits_sym) "symbole logits introuvable"

NeuroDSL.set!(g_ref, :token_ids, tokens_clean; atom_type=NeuroDSL.Datom, namespace=ns_ref)
NeuroDSL.set!(g_ref, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns_ref)
NeuroDSL.invalidate_all!(g_ref; namespace=ns_ref)
clean_output_ref = copy(Array(NeuroDSL.demand!(g_ref, logits_sym; namespace=ns_ref)))

NeuroDSL.set!(g_ref, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns_ref)
NeuroDSL.invalidate_all!(g_ref; namespace=ns_ref)
corrupted_output_ref = copy(Array(NeuroDSL.demand!(g_ref, logits_sym; namespace=ns_ref)))

denom0 = norm(corrupted_output_ref[j_target, :] .- clean_output_ref[j_target, :])
@printf("||corrupted_ref - clean_ref|| à j=%d : %.6f  (dénominateur de recovery_metric)\n", j_target, denom0)

row_metric_ref(out) = NeuroDSL.recovery_metric(Array(out)[j_target:j_target, :],
                                               clean_output_ref[j_target:j_target, :],
                                               corrupted_output_ref[j_target:j_target, :])

# Calibration : recovery obtenue par un PATCH exact du site diagnostiqué sur ce
# backbone -- le plafond que la greffe essaie d'atteindre par apprentissage.
NeuroDSL.set!(g_ref, :token_ids, tokens_clean; atom_type=NeuroDSL.Datom, namespace=ns_ref)
NeuroDSL.invalidate_all!(g_ref; namespace=ns_ref)
NeuroDSL.demand!(g_ref, logits_sym; namespace=ns_ref)
clean_cache_ref = NeuroDSL.capture_activations(g_ref, ns_ref)
NeuroDSL.set!(g_ref, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns_ref)
NeuroDSL.invalidate_all!(g_ref; namespace=ns_ref)
NeuroDSL.demand!(g_ref, logits_sym; namespace=ns_ref)
patch_recoveries = Dict{Symbol,Float64}()
for s in (DOMINANT_SITE, TEMOIN_SITE)
    NeuroDSL.patch_node!(g_ref, s, clean_cache_ref; namespace=ns_ref)
    patch_recoveries[s] = row_metric_ref(NeuroDSL.demand!(g_ref, logits_sym; namespace=ns_ref))
    NeuroDSL.set!(g_ref, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns_ref)
    NeuroDSL.invalidate_all!(g_ref; namespace=ns_ref)
    NeuroDSL.demand!(g_ref, logits_sym; namespace=ns_ref)
end
@printf("Calibration (patch exact sur ce backbone) : %s=%.4f | %s=%.4f\n",
        DOMINANT_SITE, patch_recoveries[DOMINANT_SITE], TEMOIN_SITE, patch_recoveries[TEMOIN_SITE])
g_ref = nothing; GC.gc()

# ── 4. Protocole commun aux bras ─────────────────────────────────────────────
N_STEPS      = 1000
LOG_EVERY    = 10
LR_ALPHA     = 1f-3
WD_ALPHA     = 0f0
B1, B2, EPSV = 0.9f0, 0.999f0, 1f-8
CLIP         = 1f0
DATA_SEED    = 9000
GRAFT_DIM, GRAFT_HEADS, GRAFT_HIDDEN = d_head, 2, 128
PREREG_THRESHOLD = 0.2

# Facteur d'inflation du pas effectif (catalogue 3.3) que la correction d'âge annule.
inflation(s, t) = ((1 - Float64(B1)^s) / (1 - Float64(B1)^t)) *
                  (sqrt(1 - Float64(B2)^t) / sqrt(1 - Float64(B2)^s))

seed_rng!(seed::Int) = (Random.seed!(seed);
                        NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.seed!(seed))

function freeze_backbone!(g, ns, keep_prefix)
    n = 0
    for (sym, nd) in g.nodes[ns]
        nd.is_param || continue
        startswith(String(sym), String(keep_prefix)) && continue
        nd.is_param = false
        n += 1
    end
    return n
end

grad_is_nonzero(gr) = Float64(sum(abs, gr)) > 0.0

function run_arm(arm)
    ns_b = Symbol(:porteE_, arm.label)
    g_b = NeuroDSL.NeuroGraph(namespace=ns_b, device=dev)
    NeuroDSL.load_graph!(g_b, ns_b, joinpath(export_dir, "base"))

    seed_rng!(arm.seed)
    prefix = Symbol(:shadow_, arm.label, :_, arm.site)
    _, handle = NeuroDSL.graft_shadow_block!(g_b, ns_b, arm.site, GRAFT_DIM, GRAFT_HEADS, GRAFT_HIDDEN;
                                             alpha0=0f0, zero_out_proj=false, prefix=prefix)
    n_frozen = freeze_backbone!(g_b, ns_b, handle.prefix)
    ps = NeuroDSL.params(g_b; namespace=ns_b)

    alpha_idx = findall(p -> p.name == handle.alpha_sym, ps)
    theta_idx = findall(p -> p.name != handle.alpha_sym, ps)
    @assert length(alpha_idx) == 1 "alpha introuvable"
    @printf("  [%s] site=%s | %d gelés, %d entraînables (%d alpha @lr=%.1e wd=%.2f | %d theta @lr=%.1e wd=%.2f) | correction d'âge=%s\n",
            arm.label, arm.site, n_frozen, length(ps), length(alpha_idx), LR_ALPHA, WD_ALPHA,
            length(theta_idx), arm.lr_theta, arm.wd_theta, arm.agelocal)

    m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]

    # ── Âge local par tenseur : nombre de pas pendant lesquels CE tenseur a reçu
    # un gradient non identiquement nul. `alpha0=0` rend grad_theta EXACTEMENT nul
    # au pas 1 (grad_R = dy·alpha = 0), donc l'âge de theta décroche du compteur
    # global -- c'est précisément ce décrochage que la correction annule.
    # On ne sonde `sum(abs, grad)` que tant qu'un tenseur n'a jamais reçu de
    # gradient : une fois éveillé il le reste (vérifié par le log des âges).
    ages = zeros(Int, length(ps))
    awake = falses(length(ps))

    rng_cont = MersenneTwister(DATA_SEED)
    hist = NamedTuple[]
    t_start = time()
    for t in 1:N_STEPS
        tokens, labels = sample_window(rng_cont, train_ids, block_size)
        NeuroDSL.set!(g_b, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns_b)
        NeuroDSL.set!(g_b, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns_b)
        NeuroDSL.set!(g_b, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns_b)
        NeuroDSL.invalidate_all!(g_b; namespace=ns_b)
        loss_val = NeuroDSL.demand!(g_b, :loss; namespace=ns_b)
        train_loss = Float64(sum(Array(loss_val)))
        NeuroDSL.backward_graph_sparse!(g_b, :loss; namespace=ns_b, prune_frozen=true)

        if !all(awake)
            for i in eachindex(ps)
                awake[i] && continue
                grad_is_nonzero(ps[i].gradient) || continue
                awake[i] = true
            end
        end
        for i in eachindex(ps)
            awake[i] && (ages[i] += 1)
        end

        for (idxs, lr, wd) in ((alpha_idx, LR_ALPHA, WD_ALPHA), (theta_idx, arm.lr_theta, arm.wd_theta))
            isempty(idxs) && continue
            NeuroDSL.adamw_step_batched!(dev, [ps[i].value for i in idxs], [ps[i].gradient for i in idxs],
                                         m1s[idxs], m2s[idxs], lr, B1, B2, EPSV, t, CLIP, wd;
                                         ages = arm.agelocal ? [max(ages[i], 1) for i in idxs] : nothing)
        end
        NeuroDSL.invalidate_all!(g_b; namespace=ns_b)

        if t % LOG_EVERY == 0 || t <= 3
            NeuroDSL.set!(g_b, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns_b)
            NeuroDSL.set!(g_b, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns_b)
            NeuroDSL.invalidate_all!(g_b; namespace=ns_b)
            out = NeuroDSL.demand!(g_b, logits_sym; namespace=ns_b)
            r = row_metric_ref(out)
            alpha_val = Float64(Array(NeuroDSL.node(g_b, handle.alpha_sym; namespace=ns_b).value)[1])
            R_val = Array(NeuroDSL.demand!(g_b, handle.R_sym; namespace=ns_b))
            R_rms = norm(R_val) / sqrt(length(R_val))
            theta_age = minimum(ages[i] for i in theta_idx)
            push!(hist, (; t, train_loss, recovery=r, alpha=alpha_val, R_rms,
                          theta_age, infl=inflation(theta_age, t)))
            NeuroDSL.invalidate_all!(g_b; namespace=ns_b)
        end
    end
    elapsed = time() - t_start

    recs = [h.recovery for h in hist]
    best_i = argmax(recs)
    res = (; label=String(arm.label), site=String(arm.site), lr_theta=Float64(arm.lr_theta),
            wd_theta=Float64(arm.wd_theta), agelocal=arm.agelocal, preregistered=arm.preregistered,
            elapsed, hist,
            final_recovery = recs[end], best_recovery = recs[best_i], best_step = hist[best_i].t,
            final_alpha = hist[end].alpha, final_R_rms = hist[end].R_rms,
            R_rms_init = hist[1].R_rms, R_growth = hist[end].R_rms / hist[1].R_rms,
            theta_age_final = hist[end].theta_age, infl_max = maximum(h.infl for h in hist))
    @printf("  [%s] %d pas en %.1f s | recovery finale=%+.4f | MEILLEURE=%+.4f (pas %d) | alpha final=%+.4f | ||R|| %.4f->%.4f (×%.2f)\n",
            arm.label, N_STEPS, elapsed, res.final_recovery, res.best_recovery, res.best_step,
            res.final_alpha, res.R_rms_init, res.final_R_rms, res.R_growth)
    return res
end

arms = [
    (; label=:E_theta_full_agelocal,   site=DOMINANT_SITE, seed=7001, lr_theta=1f-3, wd_theta=0f0,  agelocal=true,  preregistered=true),
    (; label=:ctrl_theta_full_globalt, site=DOMINANT_SITE, seed=7001, lr_theta=1f-3, wd_theta=0f0,  agelocal=false, preregistered=false),
    (; label=:ladder_theta_div10,      site=DOMINANT_SITE, seed=7001, lr_theta=1f-4, wd_theta=0f0,  agelocal=true,  preregistered=false),
    (; label=:ladder_theta_div100,     site=DOMINANT_SITE, seed=7001, lr_theta=1f-5, wd_theta=0f0,  agelocal=true,  preregistered=false),
    (; label=:prev_div500_wd01,        site=DOMINANT_SITE, seed=7001, lr_theta=2f-6, wd_theta=1f-1, agelocal=true,  preregistered=false),
    (; label=:temoin_full_agelocal,    site=TEMOIN_SITE,   seed=7002, lr_theta=1f-3, wd_theta=0f0,  agelocal=true,  preregistered=false),
]

results = Any[]
for a in arms
    println("\n", "="^94)
    println(a.preregistered ? "BRAS PRÉ-ENREGISTRÉ : $(a.label)" : "bras diagnostic (NON pré-enregistré) : $(a.label)")
    println("="^94)
    push!(results, run_arm(a))
end

# ── 5. Trajectoire détaillée du bras pré-enregistré ──────────────────────────
E = results[findfirst(r -> r.preregistered, results)]
println("\n", "="^94)
println("Trajectoire du bras pré-enregistré ", E.label, " (1 point sur 5 affiché)")
println("="^94)
@printf("%6s | %10s | %10s | %10s | %10s | %9s | %8s\n",
        "pas", "train_loss", "recovery", "alpha", "||R(x)||", "âge_theta", "I(s,t)")
for (k, h) in enumerate(E.hist)
    (k <= 5 || h.t % 50 == 0) || continue
    @printf("%6d | %10.4f | %+10.4f | %+10.4f | %10.4f | %9d | %8.4f\n",
            h.t, h.train_loss, h.recovery, h.alpha, h.R_rms, h.theta_age, h.infl)
end

# ── 6. Verdict ───────────────────────────────────────────────────────────────
println("\n", "="^94, "\nRÉCAPITULATIF (tous les bras)\n", "="^94)
@printf("%-26s %-20s %9s %6s %10s %10s %10s %9s\n",
        "bras", "site", "lr_theta", "âge?", "rec_fin", "rec_max", "alpha_fin", "×||R||")
for r in results
    @printf("%-26s %-20s %9.1e %6s %+10.4f %+10.4f %+10.4f %9.2f\n",
            r.label, r.site, r.lr_theta, r.agelocal ? "oui" : "non",
            r.final_recovery, r.best_recovery, r.final_alpha, r.R_growth)
end

passed = max(E.final_recovery, E.best_recovery) > PREREG_THRESHOLD
println("\n", "="^94, "\nVERDICT (bras pré-enregistré ", E.label, " uniquement)\n", "="^94)
@printf("recovery finale                 = %+.4f\n", E.final_recovery)
@printf("meilleure recovery sur %d pas = %+.4f (pas %d)\n", N_STEPS, E.best_recovery, E.best_step)
@printf("seuil pré-enregistré            = %.2f\n", PREREG_THRESHOLD)
println()
if passed
    println("→ > 0.2 : la greffe FONCTIONNE quand l'optimiseur est correctement réglé.")
    println("  Les trois campagnes négatives précédentes s'expliquent par un défaut de réglage,")
    println("  pas par une impossibilité structurelle.")
else
    println("→ ≤ 0.2 : c'est réglé. L'explication banale (\"votre LR était mal réglé\") est éliminée,")
    println("  et le résultat négatif tient.")
end

out = Dict(
    "protocol" => Dict("n_steps" => N_STEPS, "log_every" => LOG_EVERY, "lr_alpha" => LR_ALPHA,
                       "threshold" => PREREG_THRESHOLD, "data_seed" => DATA_SEED,
                       "graft_dim" => GRAFT_DIM, "graft_heads" => GRAFT_HEADS, "graft_hidden" => GRAFT_HIDDEN),
    "target" => Dict("name" => target.name, "variant" => String(target.variant), "gap" => target.gap,
                     "j" => j_target, "orig_id" => target.orig_id, "new_id" => target.new_id,
                     "denom" => denom0),
    "patch_calibration" => Dict(String(k) => v for (k, v) in patch_recoveries),
    "dominant_site" => String(DOMINANT_SITE), "temoin_site" => String(TEMOIN_SITE),
    "arms" => results,
    "preregistered_arm" => E.label,
    "preregistered_final_recovery" => E.final_recovery,
    "preregistered_best_recovery" => E.best_recovery,
    "preregistered_passed" => passed,
)
open(joinpath(@__DIR__, "graft_porte_sortie_E_results.json"), "w") do io
    JSON.print(io, out)
end
println("\nRésultats -> notebook/graft_porte_sortie_E_results.json")
