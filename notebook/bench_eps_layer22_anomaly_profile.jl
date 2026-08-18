# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC : POURQUOI L'ANCRE layer_22 (B(t)=14, PLUS élaguée que layer_16,
# B(t)=26) EST-ELLE PLUS LENTE (ratio 0.79) QUE layer_16 (ratio 0.28) APRÈS LE
# CORRECTIF GEMM (src/backward.jl, GRAD_RULES[:matmul]) ?
# ══════════════════════════════════════════════════════════════════════════════
# CONTEXTE : bench_eps_vocab_projection_profile_results.txt (APRÈS correctif) a
# mesuré, sur Qwen2.5-1.5B-Instruct réel :
#   layer_16 (is_param depuis couche 16, 13 couches entraînables) : 146.602 ms
#   layer_22 (is_param depuis couche 22,  7 couches entraînables) : 408.996 ms
# layer_22 gèle STRICTEMENT PLUS de couches (21 vs 15) -- la prédiction
# structurelle est qu'elle devrait être PLUS RAPIDE, pas 2.8x plus lente.
# Ce script teste 3 hypothèses SANS EN SUPPOSER AUCUNE VRAIE A PRIORI :
#   (A) la frontière d'élagage (_compute_needs_bwd) est mal calculée à cette
#       profondeur -- moins de nœuds exclus qu'attendu ;
#   (B) un AUTRE coût fixe (pas lm_head) domine sur ce chemin précis --
#       profil par op/par couche du VRAI temps GPU passé dans chaque
#       GRAD_RULES lors d'une passe élaguée à t=22 vs t=16 ;
#   (C) effet d'ordonnancement/warm-up GPU : la mesure layer_22 du script
#       précédent a eu lieu APRÈS layer_16 dans la même boucle -- on la
#       rejoue ICI à froid (juste après le chargement, AVANT tout backward
#       à layer_16) et on compare.
#
# USAGE : julia --project=. notebook/bench_eps_layer22_anomaly_profile.jl
# ÉCRIT  notebook/bench_eps_layer22_anomaly_profile_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, Statistics, LinearAlgebra
using CUDA

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const PROMPTS_F = joinpath(@__DIR__, "qwen_sweep_prompts.json")
const OUT       = joinpath(@__DIR__, "bench_eps_layer22_anomaly_profile_results.txt")

const N_TIMING = 6
const N_WARMUP = 3

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))

    emit("DIAGNOSTIC ANOMALIE layer_22 -- pourquoi plus élagué (B(t)=14) != plus rapide,")
    emit("sur Qwen2.5-1.5B-Instruct réel (modèle entraîné, checkpoint natif).")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))

    dev = NeuroDSL.Backend.CUDADevice()
    ns, L = :qwen2, 28
    logits, loss = :lm_head_out, :ce_loss

    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    emit("\nChargement du checkpoint natif...")
    t_load0 = time_ns()
    NeuroDSL.load_graph!(g, ns, CKPT)
    emit(@sprintf("Chargé en %.1f s", (time_ns()-t_load0)/1e9))
    reclaim()

    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss, [logits, :labels], :cross_entropy;
                                            namespace=ns))

    function layer_of(sym::Symbol)
        m = match(r"^layer_(\d+)_", String(sym))
        m === nothing ? nothing : parse(Int, m.captures[1])
    end
    param_syms_by_layer = Dict{Int,Vector{Symbol}}()
    for (s_, nd) in g.nodes[ns]
        li = layer_of(s_)
        if li !== nothing && !haskey(g.rules[ns], s_)
            push!(get!(param_syms_by_layer, li, Symbol[]), s_)
        end
    end
    all_param_syms = vcat(values(param_syms_by_layer)...)
    function set_trainable_from!(from_layer::Int)
        for s_ in all_param_syms
            li = layer_of(s_)
            g.nodes[ns][s_].is_param = (li >= from_layer)
        end
    end
    # Lecture seule pour tout ce script : lm_head/tok_E/final_norm_gamma jamais
    # entraînés (même config que le scénario "APRÈS" du script précédent, celui
    # où l'anomalie a été mesurée).
    g.nodes[ns][:lm_head_W].is_param = false
    haskey(g.nodes[ns], :final_norm_gamma) && (g.nodes[ns][:final_norm_gamma].is_param = false)
    haskey(g.nodes[ns], :tok_E) && (g.nodes[ns][:tok_E].is_param = false)

    function set_input!(ids::Vector{Int})
        n = length(ids)
        NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, vcat(ids[2:end], ids[end]);
                      atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
    end
    function backward!()
        NeuroDSL.demand!(g, loss; namespace=ns)
        NeuroDSL.backward_graph!(g, loss; namespace=ns, prune_frozen=true)
    end

    prompts = JSON.parsefile(PROMPTS_F)
    order = sortperm([length(p["token_ids"]) for p in prompts]; rev=true)
    ids1 = Int.(prompts[order[1]]["token_ids"]) .+ 1
    set_input!(ids1)
    emit(@sprintf("Prompt : \"%s\" (n=%d tokens)", prompts[order[1]]["prompt"][1:min(40,end)], length(ids1)))

    # ═══════════════════════════════════════════════════════════════════════
    # HYPOTHÈSE (A) -- frontière d'élagage : compter needs_bwd=true par couche
    # ═══════════════════════════════════════════════════════════════════════
    emit("\n" * "="^92)
    emit("HYPOTHÈSE (A) -- _compute_needs_bwd : la frontière exclut-elle bien ce qu'elle devrait ?")

    function report_needs_bwd(T_LAYER::Int)
        set_trainable_from!(T_LAYER)
        needs_bwd = NeuroDSL._compute_needs_bwd(g, ns)
        n_true = count(values(needs_bwd))
        n_total = length(needs_bwd)
        by_layer_true = Dict{Int,Int}()
        by_layer_total = Dict{Int,Int}()
        n_nolayer_true = 0
        for (sym, v) in needs_bwd
            li = layer_of(sym)
            if li === nothing
                v && (n_nolayer_true += 1)
                continue
            end
            by_layer_total[li] = get(by_layer_total, li, 0) + 1
            v && (by_layer_true[li] = get(by_layer_true, li, 0) + 1)
        end
        emit(@sprintf("  T_LAYER=%d : needs_bwd=true pour %d/%d nœuds au total (+ %d nœuds hors-couche [lm_head/loss/embeddings])",
                      T_LAYER, n_true, n_total, n_nolayer_true))
        lowest_true_layer = minimum(keys(by_layer_true); init=999)
        highest_frozen_with_true = [li for li in sort(collect(keys(by_layer_true))) if li < T_LAYER]
        if isempty(highest_frozen_with_true)
            emit(@sprintf("  Couche needs_bwd=true la plus BASSE = %d (attendu >= %d) -- frontière correcte, aucune couche gelée ne fuit",
                          lowest_true_layer, T_LAYER))
        else
            emit(@sprintf("  ❌ ANOMALIE : couches < %d avec needs_bwd=true : %s (la frontière fuit vers l'amont)",
                          T_LAYER, string(highest_frozen_with_true)))
        end
        # Détail complet, couche par couche, pour comparaison visuelle directe
        for li in sort(collect(keys(by_layer_total)))
            nt = get(by_layer_true, li, 0)
            tot = by_layer_total[li]
            emit(@sprintf("    couche %2d : %d/%d nœuds needs_bwd=true", li, nt, tot))
        end
        return needs_bwd
    end

    nb16 = report_needs_bwd(16)
    nb22 = report_needs_bwd(22)

    # ═══════════════════════════════════════════════════════════════════════
    # HYPOTHÈSE (C) -- ordre/warm-up : mesurer layer_22 À FROID EN PREMIER
    # (juste après le chargement, layer_16 n'a encore JAMAIS été touché),
    # PUIS layer_16, PUIS layer_22 à nouveau (reproduit l'ordre du script
    # précédent) -- si l'écart persiste dans les DEUX ordres, ce n'est pas
    # un artefact d'ordonnancement.
    # ═══════════════════════════════════════════════════════════════════════
    emit("\n" * "="^92)
    emit("HYPOTHÈSE (C) -- effet d'ordre/warm-up GPU (réordonnancement des ancres)")

    function time_pruned!(T_LAYER::Int; n=N_TIMING, warmup=N_WARMUP)
        set_trainable_from!(T_LAYER)
        for _ in 1:warmup
            backward!(); CUDA.synchronize()
        end
        ts = Float64[]
        for _ in 1:n
            t0 = time_ns(); backward!(); CUDA.synchronize(); push!(ts, (time_ns()-t0)/1e6)
        end
        return median(ts), ts
    end

    ms22_cold, ts22_cold = time_pruned!(22)
    emit(@sprintf("  layer_22 EN PREMIER (à froid, jamais de backward à layer_16 avant) : médiane sur %d = %.3f ms",
                  N_TIMING, ms22_cold))
    emit(@sprintf("    échantillons (ms) : %s", string(round.(ts22_cold, digits=1))))

    ms16_second, ts16_second = time_pruned!(16)
    emit(@sprintf("  layer_16 EN SECOND (après layer_22)                                : médiane sur %d = %.3f ms",
                  N_TIMING, ms16_second))
    emit(@sprintf("    échantillons (ms) : %s", string(round.(ts16_second, digits=1))))

    ms22_repeat, ts22_repeat = time_pruned!(22)
    emit(@sprintf("  layer_22 REJOUÉ (ordre original : après layer_16)                  : médiane sur %d = %.3f ms",
                  N_TIMING, ms22_repeat))
    emit(@sprintf("    échantillons (ms) : %s", string(round.(ts22_repeat, digits=1))))

    if abs(ms22_cold - ms22_repeat) / ms22_repeat < 0.15
        emit(@sprintf("  => layer_22 stable entre les deux ordres (%.1f ms vs %.1f ms, écart %.1f%%) : PAS un artefact d'ordre/warm-up.",
                      ms22_cold, ms22_repeat, 100*abs(ms22_cold-ms22_repeat)/ms22_repeat))
    else
        emit(@sprintf("  => layer_22 CHANGE selon l'ordre (%.1f ms vs %.1f ms, écart %.1f%%) : effet d'ordre/warm-up réel, à creuser.",
                      ms22_cold, ms22_repeat, 100*abs(ms22_cold-ms22_repeat)/ms22_repeat))
    end

    # ═══════════════════════════════════════════════════════════════════════
    # HYPOTHÈSE (B) -- profil par op RÉEL (temps GPU mesuré par appel
    # GRAD_RULES) pour layer_16 vs layer_22, pour trouver quel(s) nœud(s)
    # dominent le temps d'horloge sur chacun des deux chemins.
    # ═══════════════════════════════════════════════════════════════════════
    emit("\n" * "="^92)
    emit("HYPOTHÈSE (B) -- profil par op (temps GPU réel par nœud, GRAD_RULES instrumentées)")

    # Instrumentation NON-INTRUSIVE : on enveloppe chaque GRAD_RULES existante
    # dans un chronomètre GPU-synchrone, sans toucher src/backward.jl. On
    # mesure PAR TYPE D'OP (agrégé sur toute la passe) -- suffisant pour
    # répondre à l'hypothèse (quel TYPE d'op domine le delta), pas besoin de
    # distinguer les nœuds individuels d'un même type. GRAD_RULES est un
    # `const Dict` (non réassignable) : on mute son CONTENU en place, jamais
    # le binding lui-même, et on restaure les fonctions ORIGINALES à
    # l'identique juste après mesure -- zéro effet de bord après ce script.
    orig_rules = copy(NeuroDSL.GRAD_RULES)
    TIMING = Dict{Symbol,Float64}()
    for (op, f) in orig_rules
        NeuroDSL.GRAD_RULES[op] = (dev, dy, ctx, inputs) -> begin
            CUDA.synchronize()
            t0 = time_ns()
            r = f(dev, dy, ctx, inputs)
            CUDA.synchronize()
            TIMING[op] = get(TIMING, op, 0.0) + (time_ns()-t0)/1e6
            r
        end
    end

    function profile_pruned(T_LAYER::Int; n=3, warmup=2)
        set_trainable_from!(T_LAYER)
        for _ in 1:warmup; backward!(); end
        empty!(TIMING)
        for _ in 1:n; backward!(); end
        return Dict(op => t/n for (op, t) in TIMING)   # moyenne par passe
    end

    prof16 = profile_pruned(16)
    prof22 = profile_pruned(22)

    empty!(NeuroDSL.GRAD_RULES)
    merge!(NeuroDSL.GRAD_RULES, orig_rules)   # restauration en place (Dict const, jamais réassigné)

    emit("  Temps GPU cumulé par TYPE d'op (moyenne/passe, ms), layer_16 vs layer_22 :")
    all_ops = union(keys(prof16), keys(prof22))
    rows = [(op, get(prof16, op, 0.0), get(prof22, op, 0.0)) for op in all_ops]
    sort!(rows; by = r -> -(r[3]-r[2]))   # triés par écart layer_22 - layer_16 décroissant
    emit(@sprintf("    %-24s %12s %12s %12s", "op", "layer_16(ms)", "layer_22(ms)", "delta(ms)"))
    for (op, t16, t22) in rows
        emit(@sprintf("    %-24s %12.3f %12.3f %12.3f", string(op), t16, t22, t22-t16))
    end
    tot16 = sum(v for (_,v) in prof16); tot22 = sum(v for (_,v) in prof22)
    emit(@sprintf("  TOTAL instrumenté : layer_16=%.3f ms, layer_22=%.3f ms (delta=%.3f ms)",
                  tot16, tot22, tot22-tot16))

    emit("\n" * "="^92)
    emit("FIN DU DIAGNOSTIC -- voir HYPOTHÈSE (A)/(B)/(C) ci-dessus pour la conclusion par hypothèse.")
end
println("\nÉcrit : ", OUT)
