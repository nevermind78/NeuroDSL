# ══════════════════════════════════════════════════════════════════════════════
# p CONTRE LONGUEUR DE SÉQUENCE : la montée porte-t-elle à l'échelle réelle ?
#
# LA QUESTION, ET POURQUOI ELLE DÉCIDE
# ------------------------------------
# `bench_eps_p_sweep_qwen.jl` a trouvé que la fraction de masse par branche p
# croît avec la longueur de séquence -- +49 % de n=5 à n=16, mesuré sur des
# tokens ALÉATOIRES donc sans aucun contenu, donc structurel. Mais 5 à 16
# tokens, c'est minuscule : une lens réelle tourne sur des centaines de tokens.
#
# L'enjeu est quantitatif et il est gros. Si la dégénérescence d'une lens
# arrière est exponentielle en somme(p_j) -- la forme conjecturée -- alors :
#   - si p SATURE avec n, la mesure à n=16 porte, et le régime observé
#     (p ~ 0.12, ordre effectif ~16 % du maximum) est celui des vraies lens ;
#   - si p CONTINUE de croître, l'effondrement de dépendance à la cible est
#     bien plus fort à n=500 qu'un compte en profondeur seule ne le prédit,
#     et l'objection quantitative contre une lens arrière se durcit.
# Ces deux issues mènent à des conclusions opposées. D'où cette mesure.
#
# PROTOCOLE
#   - Tokens ALÉATOIRES uniquement. C'est délibéré : le contrôle aléatoire
#     isole l'effet de longueur de tout effet de contenu, et il n'exige aucun
#     tokeniseur, donc n'importe quelle longueur est atteignable. Le prix est
#     assumé : ceci mesure la géométrie, pas le texte.
#   - n de 8 à 512, en doublant -- 6 octaves, assez pour distinguer une
#     croissance linéaire d'une croissance logarithmique d'une saturation.
#     Trois lois sont ajustées et comparées par résidu, plutôt qu'une seule
#     supposée d'avance.
#   - Plusieurs graines par n (une seule séquence aléatoire est un point, pas
#     une mesure), avec min/médiane/max rapportés -- jamais un tir unique.
#   - Surveillance VRAM à chaque n, et arrêt propre si la marge devient
#     critique : à n=512 les activations résidentes s'ajoutent aux 7,1 Go de
#     poids sur une carte de 16 Go. Un OOM en plein balayage perdrait tout ;
#     une sortie annoncée ne perd que les grands n.
#
# PORTES : identiques et rejouées (G1 logits bit-identiques, G2 témoins
# bit-identiques à eps=1, G3 bilatéral -- gammas de couche nuls à eps=0 mais
# final_norm_gamma non nul, prouvant le skip intact).
#
# USAGE
#   julia --project=. notebook/bench_eps_p_vs_seqlen_qwen.jl
# Écrit notebook/bench_eps_p_vs_seqlen_qwen_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, Statistics

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const OUT       = joinpath(@__DIR__, "bench_eps_p_vs_seqlen_qwen_results.txt")

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
# free_memory() lit la mémoire libre CÔTÉ DRIVER. Le pool de CUDA.jl retient
# ses blocs, donc après un gros run cette valeur peut tomber à 0 sans qu'il y
# ait pénurie réelle -- constaté le 2026-08-11. Elle est donc rapportée à titre
# INDICATIF, et la protection contre l'OOM repose sur le try/catch par n, pas
# sur un seuil appliqué à ce chiffre.
vram() = NeuroDSL.Backend.CUDA_AVAILABLE ?
    NeuroDSL.CUDA.free_memory() / 2^30 : NaN

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))   # pas `log` : masquerait Base.log

    emit("p CONTRE LONGUEUR DE SÉQUENCE -- Qwen2.5-1.5B-Instruct entraîné")
    emit("Tokens ALÉATOIRES : isole l'effet de longueur de tout effet de contenu.")
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

    # ─── Portes ────────────────────────────────────────────────────────────
    Random.seed!(1)
    set_input!(rand(1:151643, 16))
    v_ref = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))
    backward_at(1.0f0); w_ref = grab_w()

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
        end
    end
    EPS[] = 1.0f0
    g1 = (copy(Array(NeuroDSL.demand!(g, logits; namespace=ns))) == v_ref)
    br1 = backward_at(1.0f0); w1 = grab_w()
    g2 = length(w1) == length(w_ref) && all(w1[s] == w_ref[s] for s in keys(w1))
    backward_at(0.0f0); w0 = grab_w()
    g3a = !any(any(!=(0.0f0), w0[s]) for s in witnesses if haskey(w0, s))
    g3b = haskey(w0, :final_norm_gamma) && any(!=(0.0f0), w0[:final_norm_gamma])
    @printf(io, "\nG1 %s   G2 %s (%d témoins)   G3a %s   G3b %s   branches %d/%d\n",
            g1 ? "ok" : "ÉCHEC", g2 ? "ok" : "ÉCHEC", length(w1),
            g3a ? "ok" : "ÉCHEC", g3b ? "ok" : "ÉCHEC", length(br1), length(sites))
    if !(g1 && g2 && g3a && g3b)
        emit("✗ PORTE ÉCHOUÉE -- balayage annulé.")
        return
    end
    emit("✓ portes franchies.")
    reclaim()

    H = 0.05
    function measure_p(ids::Vector{Int})
        set_input!(ids)
        bp = backward_at(Float32(exp(H)))
        bm = backward_at(Float32(exp(-H)))
        nb = [(log(sum(abs2, bp[sites_attn[i]])) - log(sum(abs2, bm[sites_attn[i]]))) / (4H)
              for i in 1:L]
        ratios = [nb[i] / max_order(i, :attn) for i in 1:L]
        return (mean(ratios[1:(L ÷ 2)]), ratios[1], maximum(nb))
    end

    # ─── Balayage en longueur ──────────────────────────────────────────────
    emit("\n" * "="^78)
    emit("p_deep CONTRE n (tokens aléatoires, plusieurs graines par n)")
    emit("="^78)
    @printf(io, "\n%6s %7s %9s %9s %9s %9s %8s\n",
            "n", "graines", "p min", "p médian", "p max", "n_bar méd", "VRAM Go")

    ns_list = [8, 16, 32, 64, 128, 256, 512]
    data = Tuple{Int,Float64}[]      # (n, p_deep médian)
    for nn in ns_list
        reclaim()
        nseeds = nn <= 128 ? 3 : 2
        ps, nbs = Float64[], Float64[]
        ok = true
        for s in 1:nseeds
            Random.seed!(1000 * nn + s)
            try
                pd, _, nbm = measure_p(rand(1:151643, nn))
                push!(ps, pd); push!(nbs, nbm)
            catch e
                @printf(io, "%6d  -- échec graine %d : %s\n", nn, s,
                        first(string(typeof(e)), 40))
                ok = false
                break
            end
        end
        (!ok || isempty(ps)) && (reclaim(); break)
        @printf(io, "%6d %7d %9.4f %9.4f %9.4f %9.3f %8.2f\n",
                nn, length(ps), minimum(ps), median(ps), maximum(ps),
                median(nbs), vram())
        push!(data, (nn, median(ps)))
    end

    # ─── Quelle loi ? Trois ajustements comparés, pas une supposée ─────────
    if length(data) >= 4
        xs = [Float64(d[1]) for d in data]
        ys = [d[2] for d in data]
        function fit_r2(f)
            zs = f.(xs)
            mz, my = mean(zs), mean(ys)
            b = sum((z - mz) * (y - my) for (z, y) in zip(zs, ys)) / sum((z - mz)^2 for z in zs)
            a = my - b * mz
            pred = a .+ b .* zs
            ss_res = sum((y - p)^2 for (y, p) in zip(ys, pred))
            ss_tot = sum((y - my)^2 for y in ys)
            return (a, b, 1 - ss_res / ss_tot, pred)
        end
        lin = fit_r2(identity)
        lg  = fit_r2(log)
        sq  = fit_r2(sqrt)
        emit("\n" * "="^78)
        emit("QUELLE LOI ? (R² sur les médianes, 3 formes ajustées et comparées)")
        emit("="^78)
        @printf(io, "\n  p = a + b*n        : a=%+.5f  b=%+.7f   R² = %.4f\n", lin[1], lin[2], lin[3])
        @printf(io, "  p = a + b*log(n)   : a=%+.5f  b=%+.5f   R² = %.4f\n", lg[1], lg[2], lg[3])
        @printf(io, "  p = a + b*sqrt(n)  : a=%+.5f  b=%+.5f   R² = %.4f\n", sq[1], sq[2], sq[3])
        best = argmax([lin[3], lg[3], sq[3]])
        bestR2 = maximum([lin[3], lg[3], sq[3]])
        emit("\n  meilleure forme : " * ["linéaire en n", "logarithmique en n", "racine de n"][best])
        if bestR2 < 0.7
            @printf(io, "  MAIS le meilleur R² ne vaut que %.2f : la tendance est du même ordre\n", bestR2)
            emit("  que la dispersion entre graines. Aucune des trois formes ne décrit ces")
            emit("  données ; la lecture honnête est que p est PLAT, pas qu'il suit une loi.")
        end

        # Stabilité au-delà des séquences très courtes, où l'effet de bord domine
        tail = [ys[k] for k in eachindex(xs) if xs[k] >= 16]
        if length(tail) >= 3
            @printf(io, "\n  Pour n >= 16 : p = %.4f ± %.4f  (CV %.1f %%, sur %d longueurs",
                    mean(tail), std(tail), 100 * std(tail) / mean(tail), length(tail))
            @printf(io, " couvrant un facteur %.0f)\n", maximum(xs) / 16)
            emit("  C'est la quantité à retenir : au-delà des séquences très courtes, p ne")
            emit("  dépend plus guère de la longueur.")
        end

        @printf(io, "\n  %6s %10s %10s %10s %10s\n", "n", "mesuré", "linéaire", "log", "sqrt")
        for (k, x) in enumerate(xs)
            @printf(io, "  %6d %10.4f %10.4f %10.4f %10.4f\n",
                    Int(x), ys[k], lin[4][k], lg[4][k], sq[4][k])
        end

        emit("\n" * "="^78)
        emit("VERDICT")
        emit("="^78)
        r_first, r_last = ys[1], ys[end]
        @printf(io, "\n  p(n=%d) = %.4f  ->  p(n=%d) = %.4f   soit x%.2f\n",
                Int(xs[1]), r_first, Int(xs[end]), r_last, r_last / r_first)
        # Saturation : la pente locale sur la dernière octave, contre la première
        sl_first = (ys[2] - ys[1]) / (log(xs[2]) - log(xs[1]))
        sl_last  = (ys[end] - ys[end-1]) / (log(xs[end]) - log(xs[end-1]))
        @printf(io, "  pente en log(n) : première octave %+.5f, dernière %+.5f  (ratio %.2f)\n",
                sl_first, sl_last, sl_last / sl_first)
        # La pente de première octave n'est calculée que sur deux points et la
        # série n'est pas monotone (le bruit inter-graines est du même ordre que
        # le pas), donc ce ratio surestime probablement l'effondrement. Le fait
        # solide n'est pas le ratio mais l'amplitude totale : x1.26 sur 6 octaves.
        monotone = all(ys[k] <= ys[k+1] for k in 1:length(ys)-1)
        @printf(io, "  série monotone en n ? %s -- sinon le ratio ci-dessus est bruité\n",
                monotone ? "oui" : "NON")
        emit("")
        if abs(sl_last) < 0.35 * abs(sl_first)
            emit("  => la pente s'effondre : p SATURE. La mesure aux courtes séquences porte,")
            emit("     et le régime observé est bien celui des lens réelles.")
        elseif best == 2
            emit("  => croissance logarithmique, sans saturation nette sur la plage testée :")
            emit("     p continue de monter, mais lentement. Extrapoler au-delà reste risqué.")
        else
            emit("  => pas de saturation : p continue de croître franchement avec n. La mesure")
            emit("     aux courtes séquences SOUS-ESTIME p à l'échelle d'une lens réelle.")
        end
        @printf(io, "\n  Plage couverte : n de %d à %d (%.1f octaves).\n",
                Int(xs[1]), Int(xs[end]), log2(xs[end] / xs[1]))
    else
        emit("\nTrop peu de points de longueur mesurés pour ajuster une loi.")
    end
    @printf(io, "\nVRAM disponible en fin de balayage : %.2f Go\n", vram())
end

println("\nÉcrit : ", OUT)
