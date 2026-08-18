# ══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION NUMÉRIQUE DE L'IDENTITÉ h_s = J_Phi(x_s) g^(s) (Proposition
# prop:transport, dernière clause : q_j^(s) = <beta_j^(t), h_s>/<g^(t), h_s>)
# PAR DIFFÉRENCE FINIE CENTRÉE SUR LE MODÈLE ENTRAÎNÉ RÉEL.
#
# CE QUE ÇA TESTE, ET POURQUOI CE N'EST PAS TAUTOLOGIQUE
# --------------------------------------------------------
# bench_eps_transport_exact_check.jl vérifie <beta_j^(t), u> = q_j^(s) - q_j^(t)
# par moindres carrés sur u -- système sous-déterminé (D >> B(t)+1), donc un u
# EXISTE toujours et le résidu tombe à la précision machine QUELLE QUE SOIT LA
# VÉRITÉ de la proposition : ce test-là ne peut pas échouer.
#
# Ici on ne résout PAS pour u : on le CALCULE indépendamment, par une méthode
# qui n'a jamais vu q_j^(s) ni q_j^(t). h_s := T'g^(s) = J_Phi(x_s) g^(s) est
# une JVP (dérivée directionnelle) du vrai calcul FORWARD Phi_{s->t} (les
# couches ordinaires entre s et t), dans la direction u = g^(s)/||g^(s)||,
# évaluée au point d'activation réel x_s. L'autodiff en mode reverse
# (implémenté ici) ne donne QUE des VJP -- jamais une JVP directement -- donc
# h_s est estimé par différence finie centrée :
#
#   h_s_hat = ||g^(s)|| * (Phi(x_s + eps*u) - Phi(x_s - eps*u)) / (2*eps)
#
# en ne ré-exécutant QUE le segment forward de s à t (set! sur le nœud de
# branche s, demand! sur le nœud de branche t -- aucun backward déclenché).
# Puis on forme la prédiction q_j^(s)_hat := <beta_j^(t), h_s_hat>/<g^(t), h_s_hat>
# et on la compare à q_j^(s) MESURÉ DIRECTEMENT (ablation ordinaire au site s,
# sans aucune machinerie de transport). Si l'identité tient, l'erreur doit
# décroître en O(eps^2) (différence centrée) jusqu'au plancher flottant, et
# CE FAIT-LÀ peut échouer : rien ne garantit a priori que h_s_hat (calculé par
# un chemin FORWARD indépendant) prédise correctement 32 scalaires mesurés par
# un chemin BACKWARD totalement différent.
#
# SITE PAIR : s = layer_1_mha_output_out (cône plein), t = layer_13_mha_output_out
# (cône B(t)=32) -- identiques à bench_eps_transport_exact_check.jl.
#
# USAGE : julia --project=. notebook/bench_eps_jvp_transport_check.jl
# ÉCRIT  notebook/bench_eps_jvp_transport_check_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, Statistics, LinearAlgebra

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const PROMPTS_F = joinpath(@__DIR__, "qwen_sweep_prompts.json")
const OUT       = joinpath(@__DIR__, "bench_eps_jvp_transport_check_results.txt")

const N_PROMPTS  = parse(Int, get(ENV, "JVP_NPROMPTS", "1"))
const T_SITE_POS = parse(Int, get(ENV, "JVP_TPOS", "25"))   # deep site
const S_SITE_POS = 1                                         # shallow site (full cone)
const EPS_GRID   = Float64[1e-2, 1e-3, 1e-4]

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

    emit("VÉRIFICATION PAR DIFFÉRENCE FINIE (JVP) DE L'IDENTITÉ h_s = J_Phi(x_s) g^(s)")
    emit("SUR LE MODÈLE ENTRAÎNÉ RÉEL (Qwen2.5-1.5B-Instruct).")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))

    dev = NeuroDSL.Backend.CUDADevice()
    ns, L = :qwen2, 28
    logits, loss = :lm_head_out, :ce_loss

    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    emit("\nChargement du checkpoint natif...")
    NeuroDSL.load_graph!(g, ns, CKPT)
    reclaim()

    witnesses = vcat([Symbol("layer_", i, "_norm1_gamma") for i in 1:L],
                     [Symbol("layer_", i, "_norm2_gamma") for i in 1:L])
    keep = Set(vcat(witnesses, [:final_norm_gamma]))
    for (s, nd) in g.nodes[ns]
        nd.is_param && !(s in keep) && (nd.is_param = false)
    end
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss, [logits, :labels], :cross_entropy;
                                            namespace=ns))

    branches = Symbol[]
    for i in 1:L
        push!(branches, Symbol("layer_", i, "_mha_output_out"))
        push!(branches, Symbol("layer_", i, "_mlp_out"))
    end
    pos = Dict(b => k for (k, b) in enumerate(branches))
    S_SYM = branches[S_SITE_POS]
    T_SYM = branches[T_SITE_POS]
    cone_t = [b for b in branches if pos[b] >= T_SITE_POS]
    Bt = length(cone_t)
    emit(@sprintf("\nSite peu profond s = %s (site_pos=%d, cône entier B=%d)",
                  S_SYM, S_SITE_POS, length(branches)))
    emit(@sprintf("Site profond     t = %s (site_pos=%d, cône B(t)=%d)",
                  T_SYM, T_SITE_POS, Bt))
    emit(@sprintf("Grille eps : %s", join(EPS_GRID, ", ")))

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
        NeuroDSL.backward_graph!(g, loss; namespace=ns)
        return CAPTURED
    end

    prompts = JSON.parsefile(PROMPTS_F)

    # ─── enregistrer les eps-ops sur toutes les branches (comme le script existant) ───
    ids1 = Int.(prompts[1]["token_ids"]) .+ 1
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

    # ─── CORRECTIF MATHÉMATIQUE (relecture de Definition~def:gate, lignes 123-134) ───
    # Le gate est défini SUR LE JOIN y=x+f(x) : franchir le gate transporte le
    # cotangent de la SORTIE du join (ȳ) vers son ENTRÉE (x̄) via x̄=(I+εK)ȳ,
    # K=J_f^T. En dépliant l'index sur T = F_2···F_{25} (branches strictement
    # après s, jusqu'À ET Y COMPRIS t), T = J_Φ(x_s)^T où Φ va de la SORTIE DU
    # JOIN de s (flux résiduel COMPLET après addition, layer_1_res1) à la
    # SORTIE DU JOIN de t (layer_13_res1) -- PAS de la sortie brute de la
    # branche (layer_1_mha_output_out / layer_13_mha_output_out) comme dans les
    # runs précédents, qui ignoraient donc le chemin de saut résiduel direct de
    # s vers l'aval (contournant t). g_s/g_t restent corrects tels quels : à
    # gate ouvert (e=1), le cotangent capturé à la branche est numériquement
    # IDENTIQUE au cotangent au join (les nœuds "+" et eps-op intermédiaires
    # sont des passages identité à e=1) -- seuls les nœuds FORWARD perturbés/lus
    # doivent changer.
    S_FWD_SYM = BRANCH_TO_JOIN[S_SYM]
    T_FWD_SYM = BRANCH_TO_JOIN[T_SYM]
    emit(@sprintf("\n[correctif] nœud forward perturbé  s_fwd = %s (pas %s)", S_FWD_SYM, S_SYM))
    emit(@sprintf("[correctif] nœud forward lu         t_fwd = %s (pas %s)", T_FWD_SYM, T_SYM))

    order = sortperm([length(p["token_ids"]) for p in prompts]; rev=true)
    to_process = order[1:min(N_PROMPTS, length(order))]

    emit(@sprintf("\nPromts traités : %d (indices %s)", length(to_process),
                  join([prompts[k]["prompt"][1:min(20,end)] for k in to_process], " | ")))
    emit("\n" * "="^92)

    # médiane / max des erreurs |q_hat - q| par eps, agrégés sur tous les prompts
    err_med_by_eps = Dict{Float64,Vector{Float64}}(e => Float64[] for e in EPS_GRID)
    err_max_by_eps = Dict{Float64,Vector{Float64}}(e => Float64[] for e in EPS_GRID)

    for k in to_process
        pr = prompts[k]
        ids = Int.(pr["token_ids"]) .+ 1
        set_input!(ids)

        emit(@sprintf("\n--- prompt %d (n=%d tokens) : \"%s\" ---",
                      k, length(ids), pr["prompt"][1:min(40,end)]))

        # ─── 1) passe de base + boucle d'ablation exacte : g_s, g_t, Bmat (beta_j^(t)), q_j^(s), q_j^(t) ───
        for b in branches; EPSV[b] = 1.0f0; end
        base = backward!()
        g_s = Float64.(vec(base[S_SYM]))
        g_t = Float64.(vec(base[T_SYM]))
        D = length(g_t)
        Ds = length(g_s)

        # copie GPU de l'activation forward réelle AU JOIN de s (flux résiduel complet,
        # PAS la sortie brute de la branche -- voir correctif mathématique ci-dessus)
        x_s_orig = copy(NeuroDSL.demand!(g, S_FWD_SYM; namespace=ns))
        x_s_norm = norm(Float64.(Array(vec(x_s_orig))))
        t_orig = Float64.(vec(Array(NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns))))

        # ─── BUG D'USAGE TROUVÉ ET CORRIGÉ (indépendant du correctif mathématique) ──
        # `set!(g, S_FWD_SYM, ...)` appelle `_invalidate_downstream!(g, S_FWD_SYM, ns)`,
        # dont le PREMIER élément de la file est S_FWD_SYM LUI-MÊME -- donc
        # `nd.valid = false` s'applique aussi à S_FWD_SYM, pas seulement à ses
        # consommateurs. Comme S_FWD_SYM garde intacte sa règle ORIGINALE dans
        # g.rules[ns] (set! ne la touche pas, seul un vrai nœud d'entrée comme
        # token_ids n'a jamais de règle), le `demand!(T_FWD_SYM)` qui suit, en
        # parcourant ses ancêtres, retombe sur S_FWD_SYM invalide + une règle
        # existante -> `execute_rule!` le RECALCULE depuis ses VRAIS ancêtres amont,
        # écrasant silencieusement notre valeur perturbée AVANT même qu'elle ne se
        # propage. D'où le 0.0 bit-exact observé sur TOUS les tests (décalage
        # constant ET bruit gaussien) dans les diagnostics précédents.
        #
        # CORRECTIF (pas de modification du moteur -- juste l'usage) : retirer
        # temporairement la règle de S_FWD_SYM de g.rules[ns] pendant tout le segment
        # perturb+demand!, pour que demand! n'ait plus rien pour le "recalculer" et
        # traite S_FWD_SYM comme un vrai nœud d'entrée le temps de la mesure JVP --
        # puis la restaurer avant de passer au prompt suivant.
        orig_rule_s = g.rules[ns][S_FWD_SYM]
        delete!(g.rules[ns], S_FWD_SYM)

        # ─── vérification étroite AVANT le balayage complet (bruit non-dégénéré) ───
        Random.seed!(12345)
        noise_cpu = Float32.(randn(size(x_s_orig)) .* (x_s_norm / sqrt(Ds)))
        noise_dev = NeuroDSL.Backend.to_device(dev, noise_cpu)
        NeuroDSL.set!(g, S_FWD_SYM, x_s_orig .+ noise_dev; namespace=ns)
        _t_noise = Float64.(vec(Array(NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns))))
        _diff_noise = norm(_t_noise .- t_orig)
        emit(@sprintf("  [vérif] ||Phi(x_s+bruit)-Phi(x_s)|| après correctif (doit être >> 0) = %.6e",
                      _diff_noise))
        NeuroDSL.set!(g, S_FWD_SYM, x_s_orig; namespace=ns)
        g.rules[ns][S_FWD_SYM] = orig_rule_s   # restaurer avant la boucle d'ablation (backward! normal)
        if _diff_noise == 0.0
            error("❌ correctif insuffisant : la perturbation n'atteint toujours pas T_FWD_SYM")
        end

        Bmat = Matrix{Float64}(undef, Bt, D)
        qs_all = Vector{Float64}(undef, Bt)
        qt_all = Vector{Float64}(undef, Bt)

        for (row, bj) in enumerate(cone_t)
            EPSV[bj] = 0.0f0
            abl = backward!()
            EPSV[bj] = 1.0f0
            abl_s = Float64.(vec(abl[S_SYM]))
            abl_t = Float64.(vec(abl[T_SYM]))
            beta_t = g_t .- abl_t
            qs = 1.0 - dot(abl_s, g_s)/dot(g_s, g_s)
            qt = 1.0 - dot(abl_t, g_t)/dot(g_t, g_t)
            Bmat[row, :] = beta_t
            qs_all[row] = qs; qt_all[row] = qt
            (row % 8 == 0) && reclaim()
        end

        emit(@sprintf("  ||g_s||=%.4e (D_s=%d)  ||g_t||=%.4e (D=%d)  ||x_s||=%.4e",
                      norm(g_s), Ds, norm(g_t), D, x_s_norm))
        emit(@sprintf("  q_j^(s) mesuré directement : médiane=%.4e  max|.|=%.4e",
                      median(qs_all), maximum(abs.(qs_all))))

        # ─── 2) direction unitaire u = g_s / ||g_s|| (mise en forme GPU, même shape que x_s) ───
        u_cpu = Float32.(reshape(g_s ./ norm(g_s), size(x_s_orig)))
        u_dev = NeuroDSL.Backend.to_device(dev, u_cpu)

        emit(@sprintf("\n  %-10s %14s %14s %10s", "eps", "err médiane", "err max", "eps/||x_s||"))
        emit("  " * "-"^54)

        delete!(g.rules[ns], S_FWD_SYM)   # même correctif que la vérification étroite ci-dessus
        for eps in EPS_GRID
            x_plus  = x_s_orig .+ Float32(eps) .* u_dev
            x_minus = x_s_orig .- Float32(eps) .* u_dev

            # Array(...) rapatrie explicitement sur l'hôte AVANT toute arithmétique
            # Float64/dot/matvec avec g_t, Bmat (hôte) -- mélanger CuArray et Array
            # dans dot()/* déclenche une indexation scalaire GPU interdite par défaut.
            NeuroDSL.set!(g, S_FWD_SYM, x_plus; namespace=ns)
            t_plus = Float64.(vec(Array(NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns))))

            NeuroDSL.set!(g, S_FWD_SYM, x_minus; namespace=ns)
            t_minus = Float64.(vec(Array(NeuroDSL.demand!(g, T_FWD_SYM; namespace=ns))))

            hs_hat = norm(g_s) .* (t_plus .- t_minus) ./ (2.0*eps)
            denom = dot(g_t, hs_hat)
            cosang = denom / (norm(g_t) * norm(hs_hat) + 1e-300)
            qhat_all = (Bmat * hs_hat) ./ denom

            errs = abs.(qhat_all .- qs_all)
            emed, emax = median(errs), maximum(errs)
            push!(err_med_by_eps[eps], emed)
            push!(err_max_by_eps[eps], emax)

            emit(@sprintf("  %-10.1e %14.4e %14.4e %10.2e   ||hs_hat||=%.3e  cos(g_t,hs_hat)=%.4f",
                          eps, emed, emax, eps/x_s_norm, norm(hs_hat), cosang))
        end

        # ─── 3) restaurer la règle ET le nœud s à sa vraie valeur forward avant le prompt suivant ───
        g.rules[ns][S_FWD_SYM] = orig_rule_s
        NeuroDSL.set!(g, S_FWD_SYM, x_s_orig; namespace=ns)
        reclaim()
    end

    emit("\n" * "="^92)
    emit("RÉSUMÉ AGRÉGÉ SUR TOUS LES PROMPTS")
    emit("="^92)
    emit(@sprintf("\n%-10s %14s %14s", "eps", "err médiane", "err max"))
    emit("-"^40)
    logs_eps = Float64[]
    logs_med = Float64[]
    for eps in EPS_GRID
        m = median(err_med_by_eps[eps])
        mx = maximum(err_max_by_eps[eps])
        emit(@sprintf("%-10.1e %14.4e %14.4e", eps, m, mx))
        if m > 0
            push!(logs_eps, log10(eps)); push!(logs_med, log10(m))
        end
    end

    emit("\nOrdre empirique de convergence (pente de log10(erreur médiane) vs log10(eps),")
    emit("régression linéaire sur toute la grille -- attendu proche de 2 avant plancher flottant) :")
    if length(logs_eps) >= 2
        n = length(logs_eps)
        mx_, my_ = mean(logs_eps), mean(logs_med)
        slope = sum((logs_eps .- mx_).*(logs_med .- my_)) / sum((logs_eps .- mx_).^2)
        emit(@sprintf("  pente globale (tous points) = %.3f", slope))
        # pente locale entre chaque paire consécutive (plus informatif si le plancher domine à petit eps)
        for i in 1:(length(EPS_GRID)-1)
            e1, e2 = EPS_GRID[i], EPS_GRID[i+1]
            m1, m2 = median(err_med_by_eps[e1]), median(err_med_by_eps[e2])
            if m1 > 0 && m2 > 0
                loc_slope = (log10(m2) - log10(m1)) / (log10(e2) - log10(e1))
                emit(@sprintf("  pente locale %8.1e -> %8.1e : %.3f", e1, e2, loc_slope))
            end
        end
    else
        emit("  (pas assez de points non-nuls pour une régression)")
    end

    emit("\nSi la pente locale est proche de 2 pour eps grand puis s'effondre (voire devient")
    emit("négative/instable) pour eps petit, c'est la signature attendue d'une différence")
    emit("centrée O(eps^2) qui rencontre son plancher de précision flottante -- PAS un échec")
    emit("de l'identité. Si l'erreur au meilleur eps est petite (proche de la précision d'un")
    emit("backward Float32, ~1e-3 à 1e-5 relatif) et diminue avec eps avant ce plancher, c'est")
    emit("un support numérique NON-TAUTOLOGIQUE de l'identité de transport : h_s a été estimé")
    emit("par un chemin FORWARD indépendant qui ne connaît ni q_j^(s) ni q_j^(t), et prédit")
    emit("correctement 32 scalaires mesurés par ablation BACKWARD directe, sur le modèle réel.")
end
println("\nÉcrit : ", OUT)
