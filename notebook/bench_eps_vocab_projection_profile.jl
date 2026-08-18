# ══════════════════════════════════════════════════════════════════════════════
# PROFIL : LA PROJECTION VOCABULAIRE (lm_head) + LA PERTE DOMINENT-ELLES LE
# COÛT D'HORLOGE DU PROTOCOLE D'ABLATION SUR QWEN2.5-1.5B ?
# ══════════════════════════════════════════════════════════════════════════════
# CONTEXTE : bench_eps_jvp_relay_cost.jl a mesuré un gain d'horloge RÉEL
# (1.0x-1.2x) très en-dessous du gain STRUCTUREL prédit (2x-4x) pour le
# mécanisme prune_frozen=true combiné au relais JVP, sur Qwen2.5-1.5B-Instruct.
# HYPOTHÈSE DE TRAVAIL (PAS ENCORE VÉRIFIÉE) : un coût FIXE -- la projection
# vocabulaire (`lm_head`, 1536 -> ~152k) + la cross-entropy calculée dessus --
# domine le temps d'horloge quel que soit l'élagage du reste du réseau, car
# la séquence de test est courte (16 tokens) donc le calcul du backbone est
# petit relativement à cette projection.
#
# CE SCRIPT :
#   1. PROFILE D'ABORD (sans supposer) : sépare, en forward ET en backward, le
#      coût de "backbone" (couches Llama) du coût de "lm_head+loss".
#   2. Localise PRÉCISÉMENT la source du coût fixe : GRAD_RULES[:matmul]
#      (src/backward.jl) calcule INCONDITIONNELLEMENT dA (gradient vers
#      dec_final_norm) ET dB (gradient vers lm_head_W, poids [vocab,dim]),
#      pour n'en jeter qu'un via le `continue` de prune_frozen -- le calcul
#      (le vrai GEMM ~vocab-large), lui, avait DÉJÀ eu lieu. lm_head_W reste
#      is_param=true du chargement du checkpoint jusqu'à la fin du script
#      bench_eps_jvp_relay_cost.jl (jamais touché par set_trainable_from!, qui
#      ne modifie que les feuilles layer_i_*) alors que RIEN dans ce protocole
#      de LECTURE de profil n'utilise jamais lm_head_W.gradient.
#   3. CORRECTIF appliqué à la racine (src/backward.jl) : GRAD_RULES[:matmul]
#      saute maintenant le calcul (pas seulement l'accumulation) de dA ou dB
#      quand `needs_bwd` (prune_frozen=true) prouve que son résultat serait de
#      toute façon jeté -- comportement strictement inchangé quand
#      prune_frozen=false ou que les deux gradients sont nécessaires (dont
#      TOUTE passe backward existante de ce projet où lm_head_W reste
#      is_param=true). Validé par 472/472 tests ciblés (test_backward.jl,
#      test_ctx_rebuild.jl, test_grad_pool.jl, test_backward_release.jl,
#      test_backward_sparse.jl, test_kernels.jl, test_layers.jl,
#      test_patching.jl) + suite complète en tâche de fond.
#   4. RE-MESURE le gain d'horloge (anchors layer_16, layer_22 -- layer_26
#      écarté par consigne, il avait bloqué la dernière fois) avec en plus
#      lm_head_W/tok_E/final_norm_gamma mis à is_param=false (véritable
#      intention de ce protocole de LECTURE, qui ne forme jamais ces poids) --
#      c'est la COMBINAISON des deux (le vrai correctif moteur + ce
#      changement de configuration du benchmark) qui active le gain, ni l'un
#      ni l'autre seul.
#
# USAGE : julia --project=. notebook/bench_eps_vocab_projection_profile.jl
# ÉCRIT  notebook/bench_eps_vocab_projection_profile_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, Statistics, LinearAlgebra
using CUDA

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const PROMPTS_F = joinpath(@__DIR__, "qwen_sweep_prompts.json")
const OUT       = joinpath(@__DIR__, "bench_eps_vocab_projection_profile_results.txt")

const T_LAYERS  = [16, 22]     # layer_26 écarté (a bloqué la dernière fois)
const N_TIMING  = 6
const N_WARMUP  = 2
const N_FWD     = 5

const EPSV     = Dict{Symbol,Float32}()
const CAPTURED = Dict{Symbol,Array{Float32}}()

function ensure_eps_op!(branch::Symbol)
    op = Symbol("epsop_", branch)
    EPSV[branch] = 1.0f0
    haskey(NeuroDSL.CUSTOM_OPS, op) && return op
    NeuroDSL.register_op!(op,
        (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> (out_buf .= inputs[1]))
    NeuroDSL.CUSTOM_SHAPE_RULES[op] = (inputs, attrs) -> size(inputs[1])
    NeuroDSL.GRAD_RULES[op] = (dev, dy, ctx, inputs) -> begin
        e = EPSV[branch]
        gr = e .* dy
        CAPTURED[branch] = copy(Array(gr))
        return (gr,)
    end
    return op
end

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))

    emit("PROFIL VOCAB-PROJECTION + LOSS -- coût réel (GPU chronométré) du forward")
    emit("ET du backward de lm_head/cross_entropy, isolé du reste du backbone,")
    emit("sur Qwen2.5-1.5B-Instruct (modèle entraîné réel).")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))

    dev = NeuroDSL.Backend.CUDADevice()
    ns, L = :qwen2, 28
    logits, loss = :lm_head_out, :ce_loss
    FINAL_NORM_GUESS = :dec_final_norm  # non utilisé -- corrigé ci-dessous d'après le graphe réel

    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    emit("\nChargement du checkpoint natif...")
    NeuroDSL.load_graph!(g, ns, CKPT)
    reclaim()

    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss, [logits, :labels], :cross_entropy;
                                            namespace=ns))

    # ─── STEP 0 : confirmer vocab_size / hidden_dim RÉELS depuis le checkpoint
    # chargé (pas depuis config.json, pas une supposition) ───
    W_lmhead = g.nodes[ns][:lm_head_W].value
    vocab_size, hidden_dim = size(W_lmhead)
    emit(@sprintf("\nlm_head_W (chargé) : shape = (%d, %d)  ->  vocab_size=%d, hidden_dim=%d",
                  vocab_size, hidden_dim, vocab_size, hidden_dim))
    fn_rule = g.rules[ns][logits]
    FINAL_NORM = fn_rule.inputs[1]   # nœud RÉEL juste avant lm_head (lu depuis le graphe, pas supposé)
    emit(@sprintf("Règle :%s = matmul(%s, %s ; trans_b=%s)", logits, fn_rule.inputs[1],
                  fn_rule.inputs[2], get(fn_rule.attrs, :trans_b, false)))
    emit(@sprintf("Nœud juste avant lm_head (lu depuis le graphe) : :%s", FINAL_NORM))

    branches = Symbol[]
    for i in 1:L
        push!(branches, Symbol("layer_", i, "_mha_output_out"))
        push!(branches, Symbol("layer_", i, "_mlp_out"))
    end
    pos = Dict(b => k for (k, b) in enumerate(branches))
    Bs = length(branches)

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
    function set_lmhead_trainable!(flag::Bool)
        g.nodes[ns][:lm_head_W].is_param = flag
        haskey(g.nodes[ns], :final_norm_gamma) && (g.nodes[ns][:final_norm_gamma].is_param = flag)
        haskey(g.nodes[ns], :tok_E) && (g.nodes[ns][:tok_E].is_param = flag)
    end

    BRANCH_TO_JOIN = Dict{Symbol,Symbol}()
    for i in 1:L
        for (join_sym, branch_sym) in ((Symbol("layer_", i, "_res1"),
                                        Symbol("layer_", i, "_mha_output_out")),
                                       (Symbol("layer_", i, "_out"),
                                        Symbol("layer_", i, "_mlp_out")))
            r = g.rules[ns][join_sym]
            r.inputs[2] == branch_sym || error("$join_sym : branche inattendue")
            eps_sym = Symbol("eps_", branch_sym)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(eps_sym, [branch_sym],
                                                    ensure_eps_op!(branch_sym); namespace=ns))
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(join_sym, [r.inputs[1], eps_sym], r.op;
                                                    attrs=r.attrs, namespace=ns,
                                                    atom_type=r.atom_type))
            NeuroDSL._invalidate_downstream!(g, join_sym, ns)
            BRANCH_TO_JOIN[branch_sym] = join_sym
        end
    end
    for b in branches; EPSV[b] = 1.0f0; end

    function set_input!(ids::Vector{Int})
        n = length(ids)
        NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, vcat(ids[2:end], ids[end]);
                      atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
    end
    function backward!()
        empty!(CAPTURED)
        NeuroDSL.demand!(g, loss; namespace=ns)
        NeuroDSL.backward_graph!(g, loss; namespace=ns, prune_frozen=true)
        return CAPTURED
    end

    prompts = JSON.parsefile(PROMPTS_F)
    order = sortperm([length(p["token_ids"]) for p in prompts]; rev=true)
    ids1 = Int.(prompts[order[1]]["token_ids"]) .+ 1
    set_input!(ids1)
    emit(@sprintf("Prompt : \"%s\" (n=%d tokens)", prompts[order[1]]["prompt"][1:min(40,end)], length(ids1)))

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 1a -- PROFIL FORWARD : backbone seul VS backbone+lm_head+loss
    # ═══════════════════════════════════════════════════════════════════════
    emit("\n" * "="^92)
    emit("ÉTAPE 1a -- PROFIL FORWARD (médiane sur $N_FWD, +$N_WARMUP warmup, invalidate_all! à chaque tir)")

    function time_forward!(target_sym::Symbol; n=N_FWD, warmup=N_WARMUP)
        for _ in 1:warmup
            NeuroDSL.invalidate_all!(g; namespace=ns)
            NeuroDSL.demand!(g, target_sym; namespace=ns)
            CUDA.synchronize()
        end
        ts = Float64[]
        for _ in 1:n
            NeuroDSL.invalidate_all!(g; namespace=ns)
            t0 = time_ns()
            NeuroDSL.demand!(g, target_sym; namespace=ns)
            CUDA.synchronize()
            push!(ts, (time_ns()-t0)/1e6)
        end
        return median(ts)
    end
    ms_fwd_backbone = time_forward!(FINAL_NORM)
    ms_fwd_full     = time_forward!(loss)
    ms_fwd_vocab    = ms_fwd_full - ms_fwd_backbone
    frac_fwd = 100 * ms_fwd_vocab / ms_fwd_full

    emit(@sprintf("  Forward backbone seul (jusqu'à :%s)         : %.3f ms", FINAL_NORM, ms_fwd_backbone))
    emit(@sprintf("  Forward complet (jusqu'à :%s)                : %.3f ms", loss, ms_fwd_full))
    emit(@sprintf("  => Forward lm_head+loss (soustractif)         : %.3f ms  (%.1f%% du forward complet)",
                  ms_fwd_vocab, frac_fwd))
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, loss; namespace=ns)   # recache tout avant la suite

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 1b -- PROFIL BACKWARD : coût isolé de lm_head+loss (backbone
    # ENTIÈREMENT gelé), AVANT/APRÈS le correctif moteur (lm_head_W.is_param
    # true = comportement historique inchangé ; false = correctif actif)
    # ═══════════════════════════════════════════════════════════════════════
    emit("\n" * "="^92)
    emit("ÉTAPE 1b -- BACKWARD ISOLÉ (backbone entièrement gelé, is_param depuis couche $(L+1))")

    function time_backward!(; n=N_TIMING, warmup=N_WARMUP)
        for _ in 1:warmup; backward!(); end
        ts = Float64[]
        for _ in 1:n
            t0 = time_ns(); backward!(); CUDA.synchronize(); push!(ts, (time_ns()-t0)/1e6)
        end
        return median(ts)
    end

    set_trainable_from!(L+1)             # aucune couche entraînable
    set_lmhead_trainable!(true)          # comportement HISTORIQUE (avant correctif)
    ms_isolated_before = time_backward!()
    set_lmhead_trainable!(false)         # comportement APRÈS correctif (lecture seule)
    ms_isolated_after  = time_backward!()

    emit(@sprintf("  AVANT (lm_head_W/tok_E/final_norm_gamma is_param=true, comportement historique) : %.3f ms", ms_isolated_before))
    emit(@sprintf("  APRÈS (is_param=false -- correctif actif, dB jamais calculé)                      : %.3f ms", ms_isolated_after))
    emit(@sprintf("  => coût du GEMM dB (gradient de lm_head_W, [%d,%d]) isolé : %.3f ms (%.1f%% du AVANT)",
                  vocab_size, hidden_dim, ms_isolated_before - ms_isolated_after,
                  100*(ms_isolated_before-ms_isolated_after)/ms_isolated_before))

    frac_bwd_vs_full_before = nothing  # rempli après le calcul de ms_full ci-dessous

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 2/3 -- RE-MESURE DU GAIN, ancres layer_16 / layer_22, AVANT vs
    # APRÈS le correctif + le changement de config is_param sur lm_head/etc.
    # ═══════════════════════════════════════════════════════════════════════
    emit("\n" * "="^92)
    emit("ÉTAPE 2/3 -- RE-MESURE DU GAIN prune_frozen (AVANT = comportement historique reproduit sur")
    emit("ce même binaire corrigé, avec lm_head_W is_param=true ; APRÈS = correctif + config)")

    for (label, lmhead_flag) in (("AVANT (is_param(lm_head_W)=true, historique)", true),
                                  ("APRÈS (is_param(lm_head_W)=false, correctif+config)", false))
        emit("\n" * "-"^92)
        emit("  Scénario : " * label)
        set_lmhead_trainable!(lmhead_flag)

        set_trainable_from!(1)
        ms_full = time_backward!()
        emit(@sprintf("  Passe COMPLÈTE (is_param depuis couche 1)  : médiane sur %d = %.3f ms", N_TIMING, ms_full))

        for T_LAYER in T_LAYERS
            T_SYM = Symbol("layer_", T_LAYER, "_mha_output_out")
            cone_t = [b for b in branches if pos[b] >= pos[T_SYM]]
            Bt = length(cone_t)

            set_trainable_from!(1)
            g_t_full = Float64.(vec(backward!()[T_SYM]))
            set_trainable_from!(T_LAYER)
            g_t_pruned = Float64.(vec(backward!()[T_SYM]))
            max_err = maximum(abs.(g_t_full .- g_t_pruned))
            gate_ok = max_err < 1f-4 ? "OK" : "❌ ÉCHEC"
            emit(@sprintf("    ANCRE t=%s (B(t)=%d)  PORTE élagué==plein : max|diff|=%.3e %s",
                          T_SYM, Bt, max_err, gate_ok))
            max_err < 1f-4 || error("❌ correctif casse la valeur du gradient à $T_SYM -- arrêt")

            ms_pruned = time_backward!()
            cost_direct = (Bt+1) * ms_full
            cost_relay  = ms_full + Bt * ms_pruned   # comparaison structurelle simple (sans relais JVP ici)
            emit(@sprintf("    Passe ÉLAGUÉE (is_param depuis couche %d) : médiane sur %d = %.3f ms  (ratio vs pleine = %.4f)",
                          T_LAYER, N_TIMING, ms_pruned, ms_pruned/ms_full))
        end
    end

    emit("\n" * "="^92)
    emit("RÉSUMÉ : voir notebook/bench_eps_vocab_projection_profile_results.txt pour le détail")
    emit("numérique complet AVANT/APRÈS ci-dessus (ratios ms_pruned/ms_full par ancre).")
end
println("\nÉcrit : ", OUT)
