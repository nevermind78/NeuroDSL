# ══════════════════════════════════════════════════════════════════════════════
# EXPÉRIENCE EPSILON SUR POIDS ENTRAÎNÉS : Qwen2.5-1.5B-Instruct
#
# POURQUOI CE SCRIPT EXISTE
# -------------------------
# `bench_eps_branch_order.jl` a mesuré l'ordre de branche moyen d'un readout
# arrière à l'INITIALISATION ALÉATOIRE, et a trouvé n_bar ~= 0.1 * ordre max
# (4.69 pour un maximum de 47 à L=24). Cette limite était déclarée en tête de
# l'artefact : la distribution de masse dépend des POIDS, donc un résultat à
# l'init ne dit rien d'un modèle entraîné. Ce script lève cette limite -- même
# mesure, mêmes portes, poids réels de Qwen2.5-1.5B-Instruct (L=28, H=12,
# GQA n_kv=2, dim=1536, vocab=151936).
#
# CE QUI CHANGE PAR RAPPORT À LA VERSION SYNTHÉTIQUE
#   - Amorçage par une VRAIE cross-entropy sur un vrai prompt tokenisé, au lieu
#     d'une mse contre une cible aléatoire. C'est le seed d'attribution standard
#     (dCE/dh), donc la mesure devient directement pertinente pour une lens.
#   - Un seul graphe en mémoire : 7.1 Go de poids ne tiennent pas en double sur
#     16 Go. Les portes comparent donc AVANT et APRÈS recâblage sur le même
#     graphe, séquentiellement, au lieu de deux graphes côte à côte.
#   - is_param mis à FALSE sur les gros tenseurs après chargement. Les gradients
#     de paramètres sont les seuls que le moteur conserve ; les garder coûterait
#     7.1 Go de plus et ferait OOM. `prune_frozen` vaut false par défaut, donc
#     la passe arrière continue de propager PARTOUT -- seule la rétention change.
#     Les 56 gammas de RMSNorm (1536 flottants chacun) restent params : ce sont
#     les témoins des portes, pour un coût mémoire nul.
#
# LA PORTE G3 EST BILATÉRALE, ET C'EST LE POINT
# ---------------------------------------------
# Tout gamma d'un BLOC siège dans une branche (norm1 -> q/k/v -> ... ->
# mha_output_out, ou norm2 -> mlp -> mlp_out) : son seul accès à la perte
# traverse un nœud eps, donc à eps=0 son gradient doit être EXACTEMENT nul.
# Mais `final_norm_gamma` est APRÈS la dernière couche, hors de toute branche :
# à eps=0 son gradient doit être NON NUL, puisque le chemin de skip est intact.
# Un recâblage qui aurait attrapé le skip par erreur annulerait tout, et la
# version unilatérale du test ne le verrait pas.
#
# CRITÈRE, repris de la version synthétique sans le modifier
#   n_bar borné (<= 3) alors que l'ordre max atteint 1+2*27 = 55
#     => masse découplée du compte de chemins.
#   n_bar croissant avec (L-i) => la masse suit la profondeur.
# Le seuil absolu de 3 est un mauvais critère (il ne s'échelonne pas avec L) --
# faiblesse déjà consignée pour la version synthétique. On rapporte donc AUSSI
# n_bar/ordre_max, la quantité qui, elle, se compare entre profondeurs.
#
# USAGE
#   julia --project=. notebook/bench_eps_branch_order_qwen.jl
# Écrit notebook/bench_eps_branch_order_qwen_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, LinearAlgebra

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const OUT       = joinpath(@__DIR__, "bench_eps_branch_order_qwen_results.txt")

const EPS      = Ref{Float32}(1.0f0)
const CAPTURED = Dict{Symbol,Array{Float32}}()

function ensure_eps_op!(branch::Symbol)
    op = Symbol("epsop_", branch)
    haskey(NeuroDSL.CUSTOM_OPS, op) && return op
    NeuroDSL.register_op!(op,
        (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> (out_buf .= inputs[1]))
    NeuroDSL.CUSTOM_SHAPE_RULES[op] = (inputs, attrs) -> size(inputs[1])
    NeuroDSL.GRAD_RULES[op] = (dev, dy, ctx, inputs) -> begin
        gr = EPS[] .* dy
        CAPTURED[branch] = copy(Array(gr))
        return (gr,)
    end
    return op
end

vram_free() = NeuroDSL.Backend.CUDA_AVAILABLE ?
    NeuroDSL.CUDA.free_memory() / 2^30 : NaN
function reclaim()
    GC.gc()
    NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim()
end

open(OUT, "w") do io
    # NE PAS nommer ce helper `log` : il masquerait Base.log, utilisé plus bas
    # pour la dérivée logarithmique (constaté 2026-08-11).
    emit(s) = (println(io, s); println(s); flush(io))

    emit("EXPÉRIENCE EPSILON SUR POIDS ENTRAÎNÉS -- Qwen2.5-1.5B-Instruct")
    emit("Amorçage : vraie cross-entropy next-token sur un prompt tokenisé.")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("")

    dev = NeuroDSL.Backend.CUDADevice()
    ns  = :qwen2
    @printf(io, "VRAM libre au départ : %.2f Go\n", vram_free())

    # ─── Chargement ────────────────────────────────────────────────────────
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    emit("Chargement du checkpoint natif (pas de re-parse safetensors)...")
    NeuroDSL.load_graph!(g, ns, CKPT)
    reclaim()
    L, H = 28, 12
    logits = :lm_head_out
    @printf(io, "Nœuds : %d   règles : %d   VRAM libre : %.2f Go\n",
            length(g.nodes[ns]), length(g.rules[ns]), vram_free())

    # ─── Prompt réel ───────────────────────────────────────────────────────
    prompts = JSON.parsefile(joinpath(MODEL_DIR, "adhoc_prompts.json"))
    pr  = prompts[1]
    ids = Int.(pr["token_ids"]) .+ 1     # HF 0-indexé -> Embedding 1-indexé
    n   = length(ids)
    emit("\nPrompt : " * repr(pr["prompt"]) * "  ($n tokens)")

    # Labels next-token. La dernière position n'a pas de suivant : elle se
    # prédit elle-même. 1 position sur $n est donc dégénérée -- sans effet sur
    # la STRUCTURE des chemins arrière, qui est la question ici, mais dit plutôt
    # que passé sous silence.
    labels = vcat(ids[2:end], ids[end])

    NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
    loss = :ce_loss
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss, [logits, :labels], :cross_entropy;
                                            namespace=ns))
    NeuroDSL.invalidate_all!(g; namespace=ns)

    # ─── Mémoire : ne conserver que les gammas comme témoins ───────────────
    witnesses = vcat([Symbol("layer_", i, "_norm1_gamma") for i in 1:L],
                     [Symbol("layer_", i, "_norm2_gamma") for i in 1:L])
    keep = Set(vcat(witnesses, [:final_norm_gamma]))
    n_demoted = 0
    for (s, nd) in g.nodes[ns]
        if nd.is_param && !(s in keep)
            nd.is_param = false
            n_demoted += 1
        end
    end
    @printf(io, "\nTenseurs rétrogradés is_param=false : %d (gradients non conservés)\n", n_demoted)
    @printf(io, "Témoins conservés : %d gammas de couche + final_norm_gamma\n", length(witnesses))

    grab_w(gr) = Dict(s => copy(Array(g.nodes[ns][s].gradient))
                      for s in keep if g.nodes[ns][s].gradient !== nothing)

    function backward_at(eps::Float32)
        EPS[] = eps
        empty!(CAPTURED)
        NeuroDSL.demand!(g, loss; namespace=ns)
        NeuroDSL.backward_graph!(g, loss; namespace=ns)
        return copy(CAPTURED), grab_w(g)
    end

    # ─── Référence AVANT recâblage ─────────────────────────────────────────
    emit("\nPasse de référence (graphe non recâblé)...")
    v_ref = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))
    l_ref = copy(Array(NeuroDSL.demand!(g, loss; namespace=ns)))
    _, w_ref = backward_at(1.0f0)
    reclaim()
    @printf(io, "  perte CE = %.6f   VRAM libre : %.2f Go   témoins avec gradient : %d\n",
            l_ref[1], vram_free(), length(w_ref))

    # ─── Recâblage des 2L jonctions résiduelles ────────────────────────────
    n_rw = 0
    for i in 1:L
        for (join_sym, branch_sym) in ((Symbol("layer_", i, "_res1"),
                                        Symbol("layer_", i, "_mha_output_out")),
                                       (Symbol("layer_", i, "_out"),
                                        Symbol("layer_", i, "_mlp_out")))
            rule = g.rules[ns][join_sym]
            length(rule.inputs) == 2 || error("$join_sym : arité $(length(rule.inputs))")
            rule.inputs[2] == branch_sym ||
                error("$join_sym : branche attendue en 2e position, trouvé $(rule.inputs[2])")
            eps_sym = Symbol("eps_", branch_sym)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(eps_sym, [branch_sym],
                                                    ensure_eps_op!(branch_sym); namespace=ns))
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(join_sym, [rule.inputs[1], eps_sym],
                                                    rule.op; attrs=rule.attrs, namespace=ns,
                                                    atom_type=rule.atom_type))
            NeuroDSL._invalidate_downstream!(g, join_sym, ns)
            n_rw += 1
        end
    end
    @printf(io, "\nJonctions recâblées : %d (attendu %d = 2L)\n", n_rw, 2 * L)

    sites = vcat([Symbol("layer_", i, "_mha_output_out") for i in 1:L],
                 [Symbol("layer_", i, "_mlp_out") for i in 1:L])

    # ─── Portes ────────────────────────────────────────────────────────────
    emit("\n" * "-"^78)
    EPS[] = 1.0f0
    v_rw = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))
    g1 = (v_rw == v_ref)
    @printf(io, "G1  logits bit-identiques après recâblage : %s   (max|diff| = %.3e)\n",
            g1 ? "OUI" : "NON", maximum(abs.(v_rw .- v_ref)))

    br1, w1 = backward_at(1.0f0)
    nbit = count(s -> haskey(w_ref, s) && w1[s] == w_ref[s], keys(w1))
    worst = maximum(maximum(abs.(w1[s] .- w_ref[s]); init=0.0) for s in keys(w1))
    g2 = (nbit == length(w_ref) && length(w1) == length(w_ref))
    @printf(io, "G2  gradients témoins à eps=1 bit-identiques : %d/%d   (max|diff| = %.3e)\n",
            nbit, length(w_ref), worst)
    @printf(io, "    branches capturées : %d/%d\n", length(br1), length(sites))

    _, w0 = backward_at(0.0f0)
    nz_layer = count(s -> haskey(w0, s) && any(!=(0.0f0), w0[s]), witnesses)
    fin_nz = haskey(w0, :final_norm_gamma) && any(!=(0.0f0), w0[:final_norm_gamma])
    g3a, g3b = (nz_layer == 0), fin_nz
    @printf(io, "G3a gammas de COUCHE à eps=0 exactement nuls : %d/%d non nuls\n",
            nz_layer, length(witnesses))
    @printf(io, "G3b final_norm_gamma à eps=0 NON nul (skip intact) : %s\n", g3b ? "OUI" : "NON")

    if !(g1 && g2 && g3a && g3b)
        emit("\n✗ PORTE ÉCHOUÉE -- aucun chiffre de masse n'est lisible.")
        return
    end
    emit("\n✓ G1, G2, G3a, G3b franchies sur poids entraînés.")

    # ─── Mesure ────────────────────────────────────────────────────────────
    function nbar(h::Float64)
        bp, _ = backward_at(Float32(exp(h)))
        bm, _ = backward_at(Float32(exp(-h)))
        Dict(s => (log(sum(abs2, bp[s])) - log(sum(abs2, bm[s]))) / (4h) for s in sites)
    end
    nb_a = nbar(0.05)
    nb_b = nbar(0.025)
    reclaim()

    emit("\n" * "-"^78)
    emit("ORDRE DE BRANCHE MOYEN n_bar -- POIDS ENTRAÎNÉS")
    emit("  ordre max = 1 + 2(L-i) ;  n_bar >= 1 toujours (a_0 = 0)")
    emit("-"^78)
    @printf(io, "\n%6s %5s %6s %10s %9s %10s %9s\n",
            "couche", "L-i", "type", "max ordre", "n_bar", "n_bar/max", "|h-h/2|")
    rows = Tuple{Int,Int,Symbol,Float64,Float64}[]
    # Ordre MAXIMAL atteignable, par type de branche. Une branche MLP de la
    # couche i traverse son propre nœud eps puis 2 par couche aval : 1+2(L-i).
    # Une branche ATTN en traverse un de plus -- celui du MLP de SA PROPRE
    # couche, via res1 -> norm2 -> mlp -> eps -> out : 2+2(L-i).
    # Erreur d'étiquetage révélée par l'ancrage : le site le plus superficiel
    # donnait n_bar = 1.073 alors qu'un ordre max de 1 impose exactement 1.
    for i in 1:L, (tag, s) in ((:attn, Symbol("layer_", i, "_mha_output_out")),
                               (:mlp,  Symbol("layer_", i, "_mlp_out")))
        mx = (tag === :attn ? 2 : 1) + 2 * (L - i)
        @printf(io, "%6d %5d %6s %10d %9.3f %10.4f %9.2e\n",
                i, L - i, tag, mx, nb_a[s], nb_a[s] / mx, abs(nb_a[s] - nb_b[s]))
        push!(rows, (i, L - i, tag, nb_a[s], nb_a[s] / mx))
    end

    nb_max = maximum(r[4] for r in rows)
    rich   = maximum(abs(nb_a[s] - nb_b[s]) for s in sites)
    i1     = findfirst(r -> r[1] == 1 && r[3] === :attn, rows)
    @printf(io, "\n  n_bar maximal            : %.3f\n", nb_max)
    @printf(io, "  ordre max à la couche 1  : %d\n", 2 + 2 * (L - 1))
    @printf(io, "  ratio à la couche 1      : %.4f\n", rows[i1][5])
    @printf(io, "  écart Richardson max     : %.3e\n", rich)

    emit("\n" * "="^78)
    emit("VERDICT -- poids entraînés")
    emit("="^78)
    @printf(io, "\n  n_bar dépasse-t-il 3 (ordre max %d) ? %s\n",
            2 + 2 * (L - 1), nb_max > 3 ? "OUI" : "NON")
    emit(nb_max > 3 ?
        "  => la masse suit la profondeur sur poids entraînés aussi." :
        "  => la masse reste en bas ordre sur poids entraînés : le compte de chemins\n" *
        "     ne gouverne pas la dégradation, contrairement au régime à l'init.")
    emit("\n  Comparaison au régime à l'initialisation (bench_eps_branch_order.jl,")
    emit("  L=24 : n_bar max 4.691, ordre max 47, ratio 0.0998) -- même mesure,")
    emit("  mêmes portes, seuls les poids changent.")
    @printf(io, "\nVRAM libre en fin de run : %.2f Go\n", vram_free())
end

println("\nÉcrit : ", OUT)
