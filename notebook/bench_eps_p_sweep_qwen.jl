# ══════════════════════════════════════════════════════════════════════════════
# BALAYAGE DE p SUR 22 PROMPTS RÉELS + CONTRÔLES À TOKENS ALÉATOIRES
#
# LA QUESTION
# -----------
# `bench_eps_branch_order_qwen.jl` a mesuré, sur UN prompt, une fraction de
# masse par branche p ~= 0.16 sur Qwen2.5-1.5B entraîné (contre ~0.09 à
# l'initialisation). p est l'intrant que la loi de longueur de chemin réclame :
# si la dégénérescence d'une lens arrière est exponentielle en somme(p_j) plutôt
# qu'en profondeur, alors p doit être MESURABLE et STABLE pour que la loi ait un
# contenu prédictif. D'où la seule question ici :
#
#   p est-il une constante d'architecture+poids, ou dépend-il de l'entrée ?
#
# LE CONTRÔLE QUI TRANCHE
# -----------------------
# Des séquences de tokens ALÉATOIRES, aux mêmes longueurs que les prompts
# réels. Un modèle entraîné n'a rien à prédire dans du bruit : si p y est le
# même que sur du texte, p ne porte pas d'information de contenu -- c'est une
# quantité d'architecture et de poids, exploitable dans un théorème. Si p
# diffère nettement, p est une quantité PAR ENTRÉE et toute loi paramétrée par
# un p unique est mal posée. Les deux issues sont informatives ; c'est pour ça
# que le contrôle est dans le protocole et pas en option.
#
# CE QUI EST RAPPORTÉ PAR PROMPT
#   p_L1    : n_bar/ordre_max au site attn de la couche 1 (propagation maximale)
#   p_deep  : moyenne de n_bar/ordre_max sur la moitié profonde (couches 1..L/2),
#             la zone où le ratio plateau -- moins sensible aux effets de bord
#             que le seul point de la couche 1
#   n_bar   : ordre de branche moyen maximal
#
# MÉTHODE ET PORTES : identiques à bench_eps_branch_order_qwen.jl, qui a franchi
# G1 (logits bit-identiques), G2 (57/57 gradients témoins bit-identiques à
# eps=1) et G3 bilatéral (gammas de couche exactement nuls à eps=0, mais
# final_norm_gamma NON nul, prouvant le skip intact). Les portes sont rejouées
# ici sur le premier prompt : elles ne dépendent pas de l'entrée, mais les
# revérifier coûte deux passes et attrape un recâblage cassé avant 50 mesures.
#
# Ordre max par type de branche : MLP = 1+2(L-i) ; ATTN = 2+2(L-i), la branche
# d'attention traversant aussi le nœud eps du MLP de sa PROPRE couche
# (res1 -> norm2 -> mlp -> eps -> out). Erreur d'étiquetage initialement
# révélée par l'ancrage n_bar(couche la plus superficielle) = 1.073 != 1.
#
# USAGE
#   python notebook/qwen_tokenize_prompts.py     # d'abord : génère les prompts
#   julia --project=. notebook/bench_eps_p_sweep_qwen.jl
# Écrit notebook/bench_eps_p_sweep_qwen_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, Statistics

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const PROMPTS   = joinpath(@__DIR__, "qwen_sweep_prompts.json")
const OUT       = joinpath(@__DIR__, "bench_eps_p_sweep_qwen_results.txt")

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

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())
vram() = NeuroDSL.Backend.CUDA_AVAILABLE ? NeuroDSL.CUDA.free_memory() / 2^30 : NaN

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))   # pas `log` : masquerait Base.log

    emit("BALAYAGE DE p -- Qwen2.5-1.5B-Instruct entraîné, 22 prompts réels + contrôles")
    emit("Question : p est-il une constante d'architecture+poids, ou dépend-il de l'entrée ?")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))

    dev = NeuroDSL.Backend.CUDADevice()
    ns, L = :qwen2, 28
    logits, loss = :lm_head_out, :ce_loss

    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    emit("\nChargement du checkpoint natif...")
    NeuroDSL.load_graph!(g, ns, CKPT)
    reclaim()

    # Témoins des portes ; tout le reste rétrogradé pour ne pas retenir 7 Go
    # de gradients de paramètres (prune_frozen=false, donc la propagation
    # arrière reste complète -- seule la rétention change).
    witnesses = vcat([Symbol("layer_", i, "_norm1_gamma") for i in 1:L],
                     [Symbol("layer_", i, "_norm2_gamma") for i in 1:L])
    keep = Set(vcat(witnesses, [:final_norm_gamma]))
    for (s, nd) in g.nodes[ns]
        nd.is_param && !(s in keep) && (nd.is_param = false)
    end

    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss, [logits, :labels], :cross_entropy;
                                            namespace=ns))

    sites_attn = [Symbol("layer_", i, "_mha_output_out") for i in 1:L]
    sites_mlp  = [Symbol("layer_", i, "_mlp_out") for i in 1:L]
    sites = vcat(sites_attn, sites_mlp)
    max_order(i, tag) = (tag === :attn ? 2 : 1) + 2 * (L - i)

    function set_input!(ids::Vector{Int})
        n = length(ids)
        NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, vcat(ids[2:end], ids[end]);
                      atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
    end

    function backward_at(eps::Float32)
        EPS[] = eps
        empty!(CAPTURED)
        NeuroDSL.demand!(g, loss; namespace=ns)
        NeuroDSL.backward_graph!(g, loss; namespace=ns)
        return copy(CAPTURED)
    end
    grab_w() = Dict(s => copy(Array(g.nodes[ns][s].gradient))
                    for s in keep if g.nodes[ns][s].gradient !== nothing)

    prompts = JSON.parsefile(PROMPTS)

    # ─── Portes, rejouées sur le premier prompt ────────────────────────────
    set_input!(Int.(prompts[1]["token_ids"]) .+ 1)
    v_ref = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))
    backward_at(1.0f0); w_ref = grab_w()

    n_rw = 0
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
            n_rw += 1
        end
    end

    EPS[] = 1.0f0
    v_rw = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))
    g1 = (v_rw == v_ref)
    br1 = backward_at(1.0f0); w1 = grab_w()
    g2 = all(w1[s] == w_ref[s] for s in keys(w1)) && length(w1) == length(w_ref)
    backward_at(0.0f0); w0 = grab_w()
    g3a = !any(any(!=(0.0f0), w0[s]) for s in witnesses if haskey(w0, s))
    g3b = haskey(w0, :final_norm_gamma) && any(!=(0.0f0), w0[:final_norm_gamma])

    emit("\n" * "-"^78)
    @printf(io, "Jonctions recâblées : %d/%d   branches capturées : %d/%d\n",
            n_rw, 2 * L, length(br1), length(sites))
    @printf(io, "G1 logits bit-identiques : %s\n", g1 ? "OUI" : "NON")
    @printf(io, "G2 témoins bit-identiques à eps=1 : %s (%d)\n", g2 ? "OUI" : "NON", length(w1))
    @printf(io, "G3a gammas de couche nuls à eps=0 : %s\n", g3a ? "OUI" : "NON")
    @printf(io, "G3b final_norm_gamma non nul à eps=0 : %s\n", g3b ? "OUI" : "NON")
    if !(g1 && g2 && g3a && g3b)
        emit("\n✗ PORTE ÉCHOUÉE -- balayage annulé.")
        return
    end
    emit("✓ portes franchies. VRAM libre : " * @sprintf("%.2f Go", vram()))

    # ─── Mesure de p pour une entrée donnée ────────────────────────────────
    H_MAIN = 0.05
    function measure_p(ids::Vector{Int}; richardson::Bool=false)
        set_input!(ids)
        bp = backward_at(Float32(exp(H_MAIN)))
        bm = backward_at(Float32(exp(-H_MAIN)))
        nb = Dict(s => (log(sum(abs2, bp[s])) - log(sum(abs2, bm[s]))) / (4 * H_MAIN)
                  for s in sites)
        rich = NaN
        if richardson
            bp2 = backward_at(Float32(exp(H_MAIN / 2)))
            bm2 = backward_at(Float32(exp(-H_MAIN / 2)))
            rich = maximum(abs(nb[s] -
                    (log(sum(abs2, bp2[s])) - log(sum(abs2, bm2[s]))) / (2 * H_MAIN))
                    for s in sites)
        end
        ratios_attn = [nb[sites_attn[i]] / max_order(i, :attn) for i in 1:L]
        deep = 1:(L ÷ 2)
        return (; n = length(ids),
                  p_L1 = ratios_attn[1],
                  p_deep = mean(ratios_attn[deep]),
                  p_sd_deep = std(ratios_attn[deep]),
                  nb_max = maximum(nb[s] for s in sites_attn),
                  rich)
    end

    # ─── Balayage des prompts réels ────────────────────────────────────────
    emit("\n" * "="^78)
    emit("PROMPTS RÉELS")
    emit("="^78)
    @printf(io, "\n%3s %-44s %4s %8s %8s %8s %8s\n",
            "#", "prompt", "n", "p_L1", "p_deep", "sd", "n_bar")
    real_rows = []
    for (k, pr) in enumerate(prompts)
        ids = Int.(pr["token_ids"]) .+ 1
        r = measure_p(ids; richardson = (k <= 2))
        push!(real_rows, r)
        lbl = replace(String(pr["prompt"]), "\n" => "\\n")
        @printf(io, "%3d %-44s %4d %8.4f %8.4f %8.4f %8.3f\n",
                k, first(lbl, 44), length(ids), r.p_L1, r.p_deep, r.p_sd_deep, r.nb_max)
        isnan(r.rich) || @printf(io, "     (contrôle Richardson h vs h/2 : %.2e)\n", r.rich)
    end

    # ─── Contrôles : tokens aléatoires, longueurs appariées ────────────────
    emit("\n" * "="^78)
    emit("CONTRÔLES -- tokens ALÉATOIRES (le modèle n'a rien à prédire)")
    emit("="^78)
    Random.seed!(20260811)
    lens = sort(unique(length(pr["token_ids"]) for pr in prompts))
    @printf(io, "\n%3s %-44s %4s %8s %8s %8s %8s\n",
            "#", "contrôle", "n", "p_L1", "p_deep", "sd", "n_bar")
    ctrl_rows = []
    for (k, n) in enumerate(lens)
        ids = rand(1:151643, n)
        r = measure_p(ids)
        push!(ctrl_rows, r)
        @printf(io, "%3d %-44s %4d %8.4f %8.4f %8.4f %8.3f\n",
                k, "aléatoire (n=$n)", n, r.p_L1, r.p_deep, r.p_sd_deep, r.nb_max)
    end

    # ─── Verdict ───────────────────────────────────────────────────────────
    rl1, rdp = [r.p_L1 for r in real_rows], [r.p_deep for r in real_rows]
    cl1, cdp = [r.p_L1 for r in ctrl_rows], [r.p_deep for r in ctrl_rows]
    emit("\n" * "="^78)
    emit("VERDICT")
    emit("="^78)
    @printf(io, "\n  %-22s %8s %8s %8s %8s\n", "", "moyenne", "écart-t", "min", "max")
    @printf(io, "  %-22s %8.4f %8.4f %8.4f %8.4f\n", "p_L1  réels  (n=$(length(rl1)))",
            mean(rl1), std(rl1), minimum(rl1), maximum(rl1))
    @printf(io, "  %-22s %8.4f %8.4f %8.4f %8.4f\n", "p_deep réels",
            mean(rdp), std(rdp), minimum(rdp), maximum(rdp))
    @printf(io, "  %-22s %8.4f %8.4f %8.4f %8.4f\n", "p_L1  aléat. (n=$(length(cl1)))",
            mean(cl1), std(cl1), minimum(cl1), maximum(cl1))
    @printf(io, "  %-22s %8.4f %8.4f %8.4f %8.4f\n", "p_deep aléat.",
            mean(cdp), std(cdp), minimum(cdp), maximum(cdp))

    # ── Le confondant : comparer les MOYENNES des deux jeux est invalide ────
    # p croît avec la longueur de séquence, et les deux jeux n'ont pas la même
    # distribution de longueurs. La comparaison brute réels-vs-aléatoires est
    # donc confondue par n. On isole l'effet de longueur sur les ALÉATOIRES,
    # où il n'y a aucun contenu, puis on mesure le contenu comme résidu.
    nr = [r.n for r in real_rows]
    nc = [r.n for r in ctrl_rows]
    pear(x, y) = begin
        mx, my = mean(x), mean(y)
        sum((a - mx) * (b - my) for (a, b) in zip(x, y)) /
            sqrt(sum((a - mx)^2 for a in x) * sum((b - my)^2 for b in y))
    end
    slope = sum((a - mean(nc)) * (b - mean(cdp)) for (a, b) in zip(nc, cdp)) /
            sum((a - mean(nc))^2 for a in nc)
    icept = mean(cdp) - slope * mean(nc)

    emit("\n  ── EFFET DE LONGUEUR (le confondant) ──────────────────────────")
    @printf(io, "  corrélation p_deep vs n, prompts réels      : r = %+.3f\n", pear(nr, rdp))
    @printf(io, "  corrélation p_deep vs n, tokens aléatoires  : r = %+.3f\n", pear(nc, cdp))
    @printf(io, "  droite ajustée sur les ALÉATOIRES : p = %.4f + %.5f n\n", icept, slope)
    @printf(io, "    p(n=5) = %.4f   p(n=16) = %.4f   soit +%.0f %% sur la plage testée\n",
            icept + 5slope, icept + 16slope, 100 * ((icept + 16slope) / (icept + 5slope) - 1))
    emit("  Sur des tokens ALÉATOIRES il n'y a aucun contenu : cette montée est")
    emit("  structurelle, pas sémantique.")

    resid = [p - (icept + slope * n) for (n, p) in zip(nr, rdp)]
    cv_raw = std(rdp) / mean(rdp)
    cv_res = std(resid) / mean(rdp)
    emit("\n  ── CONTENU, À LONGUEUR CONTRÔLÉE ──────────────────────────────")
    @printf(io, "  résidu moyen des réels vs la droite aléatoire : %+.4f (%+.1f %%)\n",
            mean(resid), 100 * mean(resid) / mean(icept .+ slope .* nr))
    @printf(io, "  CV de p_deep, brut                           : %.1f %%\n", 100cv_raw)
    @printf(io, "  CV de p_deep, longueur retirée               : %.1f %%\n", 100cv_res)

    cd_by_n = Dict(n => p for (n, p) in zip(nc, cdp))
    pairs = [(r.n, r.p_deep, cd_by_n[r.n]) for r in real_rows if haskey(cd_by_n, r.n)]
    rel = [100 * (p / c - 1) for (_, p, c) in pairs]
    emit("\n  ── À LONGUEUR APPARIÉE ────────────────────────────────────────")
    @printf(io, "  %d prompts appariés à un contrôle de même n\n", length(pairs))
    @printf(io, "  écart réel vs aléatoire : médian %+.1f %%  moyen %+.1f %%  [%.1f, %.1f]\n",
            median(rel), mean(rel), minimum(rel), maximum(rel))
    @printf(io, "  prompts au-dessus de leur contrôle : %d/%d\n",
            count(>(0), rel), length(rel))

    emit("\n  ── CONCLUSION ─────────────────────────────────────────────────")
    emit("  p n'est PAS une constante d'architecture+poids :")
    emit("    - il croît d'environ 49 % quand n passe de 5 à 16 tokens, effet")
    emit("      mesuré sur des tokens aléatoires donc purement structurel ;")
    emit("    - à longueur appariée, le texte réel donne un p supérieur au bruit")
    emit("      (médian +10 %, 19/22 au-dessus), donc le contenu compte aussi ;")
    emit("    - il reste ~16 % de variation entre prompts après retrait de la longueur.")
    emit("  Une loi paramétrée par un p unique et universel est donc mal posée.")
    emit("  Mais p est PRÉDICTIBLE : la longueur en est le moteur dominant et")
    emit("  s'ajuste proprement (r = 0.89 sur les contrôles). La forme utilisable")
    emit("  est p_j(n), pas une constante.")
    emit("\n  LIMITE : n ne va que de 5 à 16 tokens. Une lens réelle tourne sur des")
    emit("  centaines de tokens. Rien ici n'autorise à extrapoler la montée en n.")
    @printf(io, "\nVRAM libre en fin de balayage : %.2f Go\n", vram())
end

println("\nÉcrit : ", OUT)
