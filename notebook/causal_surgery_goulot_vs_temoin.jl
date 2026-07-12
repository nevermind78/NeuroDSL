# ══════════════════════════════════════════════════════════════════════════════
# Étape 3 (redesign chirurgie causale dirigée) — protocole goulot-vs-témoin.
#
# Base : LlamaModel 6 couches, tâche d'induction à 3 sauts (src/synthetic_circuits.jl),
# entraînée 10 000 pas -- même config que la calibration du 2026-07-11
# (query_acc≈0.907). Profil de recovery par couche mesuré sur cette base :
# couche 1 = 0.98 (biais de proximité au point de corruption, EXCLU du choix
# goulot/témoin -- confondu avec la profondeur, signalé par Fable) ; couches
# 2-6 = 0.13-0.26, PAS monotone avec la profondeur. Couche 3 (0.26, la plus
# haute parmi 2-6) = GOULOT ; couche 2 (0.13, la plus basse) = TÉMOIN --
# profondeurs voisines (2 vs 3), écart réel ~2x, sans le biais d'extrémité.
#
# Greffe : `graft_shadow_block!` (src/graph_surgery.jl), variante "rezero"
# (alpha0=0, zero_out_proj=false, theta aléatoire) -- déjà validée (bit-
# exactness E1, échappe au point de selle degenerate E2,
# notebook/experiments_surgery.ipynb). Bras A (goulot) greffé après
# `:layer_3_out` ; bras B (témoin) après `:layer_2_out`.
#
# Test d'abandon précoce (~10% du budget prévu, 1000 pas, UNE graine) avant
# tout engagement du budget complet (3 graines × 2 bras) -- protège contre le
# risque de "recrutement indiscriminé" (alpha croît partout, pas seulement au
# goulot) signalé par Fable.
#
# NOTE : bug de cache périmé dans `patch_node!` (non-batched) découvert lors
# de la calibration -- contourné ici (comme partout ailleurs) via
# `batched_attn=true` (défaut de `build_multihop_graph`). Correctif de fond
# dans `src/patching.jl` reporté à une session dédiée.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, LinearAlgebra, Printf, JSON

dev = NeuroDSL.Backend.CUDADevice()
vocab_size, dim, n_heads, hidden_dim, n_layers, n_hops = 30, 64, 4, 128, 6, 3
n_steps_A = 10_000
LR = 3f-3

GOULOT_AFTER = :layer_3_out
TEMOIN_AFTER = :layer_2_out

export_dir = joinpath(@__DIR__, "causal_surgery_export")
mkpath(export_dir)

function train_step!(g, ns, ps, m1s, m2s, t, rng)
    tokens, labels, _ = NeuroDSL.sample_multihop_sequence(rng, vocab_size, n_hops)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    loss_val = NeuroDSL.demand!(g, :loss; namespace=ns)
    NeuroDSL.backward_graph!(g, :loss; namespace=ns)
    for (i, p) in enumerate(ps)
        NeuroDSL.adamw_step!(dev, p.value, p.gradient, m1s[i], m2s[i], LR, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
    end
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Float64(sum(Array(loss_val)))
end

# ═══ Phase A : base pré-entraînée (une seule fois, sauvegardée) ═══
println("="^70, "\nPhase A -- pré-entraînement de la base (n_hops=", n_hops, ", ", n_steps_A, " pas)\n", "="^70)
Random.seed!(n_hops)
ns_base = :base
g_base, logits_base = NeuroDSL.build_multihop_graph(dev, ns_base; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                                      hidden_dim=hidden_dim, n_layers=n_layers, n_hops=n_hops)
rng_base = MersenneTwister(123)
ps_base = NeuroDSL.params(g_base; namespace=ns_base)
m1s_base = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps_base]
m2s_base = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps_base]
for t in 1:n_steps_A
    train_step!(g_base, ns_base, ps_base, m1s_base, m2s_base, t, rng_base)
end
acc_base = NeuroDSL.evaluate_multihop(g_base, logits_base, ns_base; vocab_size=vocab_size, n_hops=n_hops, n_eval=300)
@printf("Base entraînée : query_acc=%.3f  body_acc=%.3f\n", acc_base.query_acc, acc_base.body_acc)
NeuroDSL.save_graph!(g_base, ns_base, joinpath(export_dir, "base"))
println("Base sauvegardée -> ", joinpath(export_dir, "base"))

# ═══ Fonction : greffe + entraînement continué à partir de la base ═══
function run_arm(seed::Int, graft_after::Symbol, n_steps_B::Int; t0::Int=n_steps_A, log_every::Int=100)
    ns = Symbol(:arm_, graft_after, :_seed, seed)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.load_graph!(g, ns, joinpath(export_dir, "base"))
    logits_sym = :lm_head_out

    # `Backend.rand32(::CUDADevice,...) = CUDA.rand(...)` -- utilise le RNG cuRAND,
    # PAS le RNG Julia global : `Random.seed!` n'aurait aucun effet ici.
    NeuroDSL.Backend.CUDA_AVAILABLE ? NeuroDSL.CUDA.seed!(1000 + seed) : Random.seed!(1000 + seed)
    new_out, handle = NeuroDSL.graft_shadow_block!(g, ns, graft_after, dim, n_heads, hidden_dim;
                                                    alpha0=0f0, zero_out_proj=false,
                                                    prefix=Symbol(:shadow_seed, seed, :_, graft_after))
    gated_sym = Symbol(handle.prefix, :_gated)

    rng_cont = MersenneTwister(2000 + seed)
    ps = NeuroDSL.params(g; namespace=ns)
    m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]

    alpha_history = Float64[]
    grad_alpha_history = Float64[]
    for t in 1:n_steps_B
        train_step!(g, ns, ps, m1s, m2s, t0 + t, rng_cont)
        if t % log_every == 0 || t == n_steps_B
            alpha_val = Array(NeuroDSL.node(g, handle.alpha_sym; namespace=ns).value)[1]
            grad_val = NeuroDSL.node(g, handle.alpha_sym; namespace=ns).gradient
            grad_norm = grad_val === nothing ? 0.0 : abs(Float64(Array(grad_val)[1]))
            push!(alpha_history, alpha_val)
            push!(grad_alpha_history, grad_norm)
        end
    end
    final_loss = mean_recent_loss(g, ns, ps, rng_cont)
    return (; g, ns, logits_sym, handle, gated_sym, alpha_history, grad_alpha_history,
            final_alpha=alpha_history[end], final_grad=grad_alpha_history[end], final_loss)
end

function mean_recent_loss(g, ns, ps, rng; n_eval::Int=100)
    total = 0.0
    for _ in 1:n_eval
        tokens, labels, _ = NeuroDSL.sample_multihop_sequence(rng, vocab_size, n_hops)
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        loss_val = NeuroDSL.demand!(g, :loss; namespace=ns)
        total += Float64(sum(Array(loss_val)))
    end
    return total / n_eval
end

# ═══ Pilote (1 graine, budget réduit) -- test d'abandon précoce ═══
n_steps_pilot = 1000
println("\n", "="^70, "\nPilote (graine 1, ", n_steps_pilot, " pas) -- test d'abandon précoce\n", "="^70)
pilot_A = run_arm(1, GOULOT_AFTER, n_steps_pilot)
pilot_B = run_arm(1, TEMOIN_AFTER, n_steps_pilot)
@printf("Bras A (goulot, après %s)  : alpha final=%.4f  |grad(alpha)|=%.6f\n",
        GOULOT_AFTER, pilot_A.final_alpha, pilot_A.final_grad)
@printf("Bras B (témoin, après %s)  : alpha final=%.4f  |grad(alpha)|=%.6f\n",
        TEMOIN_AFTER, pilot_B.final_alpha, pilot_B.final_grad)

ratio = pilot_B.final_grad > 1e-8 ? pilot_A.final_grad / pilot_B.final_grad : Inf
@printf("\nRatio |grad(alpha)_A| / |grad(alpha)_B| = %.3f  (seuil d'abandon : < 1.5)\n", ratio)

if ratio < 1.5
    println("\n⚠️  ABANDON : recrutement indiscriminé détecté (le gradient d'alpha ne distingue pas")
    println("    goulot et témoin dès le pilote) -- ne pas engager le budget complet sans durcir le")
    println("    protocole (alpha0 plus petit, ou pénalité L1 sur alpha).")
else
    println("\n✅ Pilote passé -- le gradient d'alpha distingue déjà goulot et témoin. Engagement du")
    println("    budget complet (3 graines × 2 bras).")

    # ═══ Budget complet : 3 graines × 2 bras ═══
    n_steps_B_full = 8000   # au-delà des 1000 pas déjà faits par le pilote -> 9000 pas de Phase B au total
    results = Dict{String,Any}()
    arms_final = NamedTuple[]
    for seed in 1:3
        println("\n--- Graine ", seed, " ---")
        armA = run_arm(seed, GOULOT_AFTER, n_steps_B_full + n_steps_pilot)
        armB = run_arm(seed, TEMOIN_AFTER, n_steps_B_full + n_steps_pilot)
        @printf("  A (goulot) : alpha=%.4f  loss=%.4f\n", armA.final_alpha, armA.final_loss)
        @printf("  B (témoin) : alpha=%.4f  loss=%.4f\n", armB.final_alpha, armB.final_loss)
        push!(arms_final, (; seed, alpha_A=armA.final_alpha, alpha_B=armB.final_alpha,
                            loss_A=armA.final_loss, loss_B=armB.final_loss))

        # ── Re-diagnostic : la greffe apparaît-elle dans le top-3 du circuit causal ? ──
        for (label, arm) in (("A", armA), ("B", armB))
            rng_eval = MersenneTwister(999)
            tokens, labels, q_pos = NeuroDSL.sample_multihop_sequence(rng_eval, vocab_size, n_hops)
            tokens_corrupt = copy(tokens)
            orig = tokens_corrupt[2]
            new_id = orig
            while new_id == orig || new_id in tokens
                new_id = rand(rng_eval, 1:vocab_size)
            end
            tokens_corrupt[2] = new_id

            g, ns, logits_sym = arm.g, arm.ns, arm.logits_sym
            NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.invalidate_all!(g; namespace=ns)
            clean_output = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))
            clean_cache = NeuroDSL.capture_activations(g, ns)

            NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.invalidate_all!(g; namespace=ns)
            corrupted_output = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))
            corrupted_cache = NeuroDSL.capture_activations(g, ns)

            row_metric(out) = NeuroDSL.recovery_metric(Array(out)[q_pos:q_pos, :], clean_output[q_pos:q_pos, :], corrupted_output[q_pos:q_pos, :])
            candidates = sort(collect(filter(s -> occursin(r"_mha_ao_h\d+$", String(s)), keys(g.nodes[ns]))))
            push!(candidates, arm.gated_sym)   # la greffe elle-même comme candidat de patch

            NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
            NeuroDSL.invalidate_all!(g; namespace=ns)
            NeuroDSL.demand!(g, logits_sym; namespace=ns)
            selected, trajectory = NeuroDSL.greedy_patch_search!(g, logits_sym, candidates, clean_cache, corrupted_cache,
                                                                   clean_output, corrupted_output;
                                                                   namespace=ns, max_sites=3, metric=row_metric)
            rank = findfirst(==(arm.gated_sym), selected)
            @printf("    [%s] top-3 du re-diagnostic = %s  (greffe classée : %s)\n",
                    label, selected, rank === nothing ? "absente" : "rang $rank")
        end
    end

    println("\n", "="^70, "\nCritères pré-enregistrés\n", "="^70)
    alpha_A_all = [a.alpha_A for a in arms_final]; alpha_B_all = [a.alpha_B for a in arms_final]
    loss_delta = [a.loss_B - a.loss_A for a in arms_final]
    for a in arms_final
        @printf("  graine %d : |alpha_A|=%.4f |alpha_B|=%.4f  ratio=%.2f  loss_A=%.4f loss_B=%.4f  Δ(B-A)=%.4f\n",
                a.seed, abs(a.alpha_A), abs(a.alpha_B), abs(a.alpha_A)/max(abs(a.alpha_B),1e-8),
                a.loss_A, a.loss_B, a.loss_B - a.loss_A)
    end
    crit_i = all(abs(a.alpha_A) > 2*abs(a.alpha_B) for a in arms_final)
    crit_iii = all(a.loss_B > a.loss_A for a in arms_final)
    println("\nCritère (i)  |alpha_A| > 2×|alpha_B| sur les 3 graines : ", crit_i)
    println("Critère (iii) loss_A < loss_B sur les 3 graines (A avantagé)   : ", crit_iii)
    println("Critère (ii) voir le rang de la greffe dans le re-diagnostic ci-dessus, par bras/graine.")
end
