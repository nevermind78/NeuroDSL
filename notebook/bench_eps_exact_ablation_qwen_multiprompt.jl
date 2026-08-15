# ══════════════════════════════════════════════════════════════════════════════
# IDENTITÉ EXACTE D'ABLATION SUR 22 PROMPTS -- referme le n=1, et teste la
# dérive d'alignement à travers les entrées
#
# CE QUE CE SCRIPT REFERME
# ------------------------
# bench_eps_exact_ablation_qwen.jl a mesuré la matrice q COMPLÈTE (56x56), la
# loi de signe, l'enveloppe et le seuil rho=1 -- mais sur UNE SEULE entrée.
# Toute figure qui en dérive (16,2 % de coefficients négatifs, borne
# rho >= 0.169, quantiles de dépendance au site) est donc une mesure à n=1.
# Ce script rejoue la MÊME mesure exacte sur les 22 prompts réels déjà validés
# (qwen_sweep_prompts.json, ceux de bench_eps_p_sweep_qwen.jl -- aucune entrée
# fabriquée), en rechargeant le checkpoint UNE fois.
#
# DEUXIÈME OBJET, NOUVEAU : LA DÉRIVE D'ALIGNEMENT
# ------------------------------------------------
# Sur le prompt unique, au site le plus profond, l'alignement moyen vaut
# c_j = -0.197 avec 85 % de branches négatives, contre +0.037 et 56 % positives
# À L'INITIALISATION (bench_eps_convergence_general_results.txt) : l'entraînement
# pousserait l'alignement systématiquement vers le NÉGATIF. Sur une seule entrée
# on ne peut pas distinguer une propriété du MODÈLE d'une propriété du PROMPT.
# C'est exactement ce que 22 prompts tranchent, et cela exige d'exporter
# rho_j et c_j par branche -- ce que ce script fait (colonnes rho, cos).
#
# ÉCRITURE INCRÉMENTALE, IMPOSÉE PAR L'EXPÉRIENCE
# -----------------------------------------------
# Le balayage en profondeur a perdu 55 minutes de calcul parce que les mesures
# s'accumulaient en mémoire et n'atteignaient l'artefact que par des boucles de
# rapport finales qui n'ont jamais tourné. Ici CHAQUE prompt est écrit (ligne de
# résumé + lignes CSV) DÈS qu'il est calculé, avec flush. Une mort au prompt 15
# laisse 14 prompts de données exploitables.
#
# USAGE : julia --project=. notebook/bench_eps_exact_ablation_qwen_multiprompt.jl
# ÉCRIT  notebook/bench_eps_exact_ablation_qwen_multiprompt_results.txt
#        notebook/bench_eps_exact_ablation_qwen_multiprompt_matrix.csv
# NE TOUCHE PAS bench_eps_exact_ablation_qwen_results.txt ni son CSV.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, Statistics, LinearAlgebra

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const PROMPTS_F = joinpath(@__DIR__, "qwen_sweep_prompts.json")
const OUT       = joinpath(@__DIR__, "bench_eps_exact_ablation_qwen_multiprompt_results.txt")
const CSV       = joinpath(@__DIR__, "bench_eps_exact_ablation_qwen_multiprompt_matrix.csv")

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

# ─── REPRISE PAR LOTS ──────────────────────────────────────────────────────
# CAUSE RÉELLE DES MORTS SUCCESSIVES, ET CORRECTION D'UN DIAGNOSTIC FAUX.
# Ce script est mort six fois, à des points imprévisibles (0, 2, 7, 9, 10
# prompts). J'ai d'abord écrit ici que c'était une FUITE par prompt, puis une
# FRAGMENTATION du pool CUDA. Les deux étaient faux, et le commentaire les
# présentait comme établis. La cause réelle, trouvée en interrogeant
# `nvidia-smi --query-compute-apps` PENDANT une exécution : la VRAM était
# détenue par EpicGamesLauncher.exe et des processus associés, dont l'usage
# fluctuait -- il restait par moments 35 Mio libres sur 16384. Les appels à
# reclaim() fonctionnaient correctement ; Julia rendait bien la mémoire, elle
# était simplement reprise par d'autres applications. D'où l'imprévisibilité
# des points de mort, qui est justement ce qui aurait dû m'alerter : une fuite
# déterministe tue au même endroit.
# À retenir aussi : un OOM côté pilote peut tuer le processus SANS lever
# d'exception Julia, donc ni la capture de stderr ni un try/catch n'attrapent
# quoi que ce soit -- les journaux vides étaient une donnée, pas une absence.
#
# Le traitement par LOTS reste utile indépendamment de la cause : il borne le
# coût d'une mort à un lot, quelle qu'en soit l'origine. Un processus neuf par
# lot, et on reprend où on s'est arrêté. Deux variables d'environnement :
#   MP_SKIP  = indices deja mesures, separes par des virgules
#   MP_LIMIT = nombre max de prompts a traiter dans CE processus
# Le mode d'ouverture passe en ajout des qu'on reprend, pour ne pas ecraser.
const SKIP = Set(parse.(Int, filter(!isempty, split(get(ENV, "MP_SKIP", ""), ","))))
const LIMIT = parse(Int, get(ENV, "MP_LIMIT", "999"))
const RESUME = !isempty(SKIP)

csv_io = open(CSV, RESUME ? "a" : "w")
RESUME || (println(csv_io, "prompt_idx,site,branch,site_pos,branch_pos,q,rho,cos"); flush(csv_io))

open(OUT, RESUME ? "a" : "w") do io
    emit(s) = (println(io, s); println(s); flush(io))

    emit("IDENTITÉ EXACTE D'ABLATION SUR 22 PROMPTS -- Qwen2.5-1.5B-Instruct")
    emit("Referme le n=1 de la matrice q, et teste la dérive d'alignement c_j")
    emit("à travers les entrées. Écriture incrémentale : chaque prompt est écrit")
    emit("dès qu'il est calculé.")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))

    dev = NeuroDSL.Backend.CUDADevice()
    ns, L = :qwen2, 28
    logits, loss = :lm_head_out, :ce_loss

    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    emit("\nChargement du checkpoint natif (une seule fois)...")
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
    B_of(b) = length(branches) - pos[b] + 1
    s1 = Symbol("layer_1_mha_output_out")
    B1 = B_of(s1)

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
        return copy(CAPTURED)
    end
    grab_w() = Dict(s => copy(Array(g.nodes[ns][s].gradient))
                    for s in keep if g.nodes[ns][s].gradient !== nothing)

    prompts = JSON.parsefile(PROMPTS_F)
    emit(@sprintf("\n%d prompts chargés depuis %s (mêmes que le balayage p par DF).",
                  length(prompts), basename(PROMPTS_F)))

    # ─── portes de correction structurelles, une fois, sur le prompt 1 ──────
    ids1 = Int.(prompts[1]["token_ids"]) .+ 1
    set_input!(ids1)
    v_ref = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))
    backward!(); w_ref = grab_w()

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

    for b in branches; EPSV[b] = 1.0f0; end
    g1ok = (copy(Array(NeuroDSL.demand!(g, logits; namespace=ns))) == v_ref)
    backward!(); w1 = grab_w()
    g2ok = length(w1) == length(w_ref) && all(w1[s] == w_ref[s] for s in keys(w1))
    for b in branches; EPSV[b] = 0.0f0; end
    backward!(); w0 = grab_w()
    for b in branches; EPSV[b] = 1.0f0; end
    g3a = !any(any(!=(0.0f0), w0[s]) for s in witnesses if haskey(w0, s))
    g3b = haskey(w0, :final_norm_gamma) && any(!=(0.0f0), w0[:final_norm_gamma])
    emit(@sprintf("\nPortes structurelles (une fois) : G1 %s  G2 %s (%d)  G3a %s  G3b %s",
                  g1ok ? "ok" : "ÉCHEC", g2ok ? "ok" : "ÉCHEC", length(w1),
                  g3a ? "ok" : "ÉCHEC", g3b ? "ok" : "ÉCHEC"))
    (g1ok && g2ok && g3a && g3b) || (emit("✗ PORTE ÉCHOUÉE -- arrêt."); return)

    emit("\n" * "-"^78)
    emit("PAR PROMPT -- matrice complète (56 sites x 56 branches, 1596 paires en cône)")
    emit("Colonnes : P1 = max|q_jj - 1| ; P2 = max|q_j| en amont ; 2sc = max écart")
    emit("à la forme à deux scalaires ; sgn = violations de la loi de signe ;")
    emit("puis, AU SITE DE COUCHE 1 (55 branches contournables) : %neg, moyenne et")
    emit("médiane de c_j, et part de branches avec c_j < -rho_j.")
    emit("-"^78)
    emit(@sprintf("\n%3s %4s %9s %9s %9s %4s | %6s %8s %8s %8s",
                  "#", "n", "P1", "P2", "2sc", "sgn", "%neg", "moy c", "méd c", "c<-rho"))

    # ORDRE DE TRAITEMENT : longueur DÉCROISSANTE. Les tenseurs du graphe sont
    # redimensionnés à chaque changement de n ; commencer par le plus long
    # (n=16) alloue le pic UNE fois et tout prompt plus court réutilise ces
    # tampons. C'est une bonne pratique en soi, mais ce n'est PAS ce qui a
    # causé les morts : celles-ci venaient d'une contention VRAM externe (voir
    # l'en-tête). Gardé parce que réduire les réallocations reste utile quand
    # la carte est partagée, pas parce que le diagnostic initial était bon.
    order = sortperm([length(p["token_ids"]) for p in prompts]; rev=true)
    emit(@sprintf("\nOrdre de traitement : longueur décroissante (n = %d ... %d).",
                  length(prompts[order[1]]["token_ids"]),
                  length(prompts[order[end]]["token_ids"])))
    emit("La colonne # est l'indice ORIGINAL du prompt, pas l'ordre d'exécution.")
    emit("")

    rows = NamedTuple[]
    n_done_here = 0
    for k in order
        k in SKIP && continue                  # deja mesure dans un lot precedent
        n_done_here >= LIMIT && break          # lot plein : sortie propre
        n_done_here += 1
        pr = prompts[k]
        ids = Int.(pr["token_ids"]) .+ 1
        set_input!(ids)

        for b in branches; EPSV[b] = 1.0f0; end
        base = backward!()
        nrm2 = Dict(b => Float64(sum(abs2, base[b])) for b in branches)

        Q    = Dict{Tuple{Symbol,Symbol},Float64}()
        RHO  = Dict{Tuple{Symbol,Symbol},Float64}()
        COSC = Dict{Tuple{Symbol,Symbol},Float64}()
        for (bidx, bj) in enumerate(branches)
            EPSV[bj] = 0.0f0
            abl = backward!()
            EPSV[bj] = 1.0f0
            # Reclaim périodique : rend la mémoire au plus vite plutôt qu'à la
            # discrétion du GC. Utile sur une carte PARTAGÉE, où tout retard à
            # libérer est une fenêtre pour qu'un autre processus prenne la
            # place. Ce n'était pas le correctif de la panne -- celle-ci venait
            # d'une contention externe -- mais cela réduit la surface exposée.
            (bidx % 8 == 0) && reclaim()
            for si in branches
                g0 = abl[si]; gg = base[si]
                # trois produits scalaires suffisent -- ne PAS matérialiser beta
                # (56x56 tableaux épuiseraient le pool CUDA). dot() évite en plus
                # le temporaire hôte de sum(g0 .* gg), 3136 fois par prompt.
                dg = Float64(dot(vec(g0), vec(gg))); n0 = Float64(dot(vec(g0), vec(g0))); ng = nrm2[si]
                Q[(si,bj)] = 1.0 - dg/ng
                n2b = ng - 2dg + n0
                nb, nsg = sqrt(max(n2b, 0.0)), sqrt(n0)
                RHO[(si,bj)]  = nsg > 0 ? nb/nsg : Inf
                COSC[(si,bj)] = (nb > 0 && nsg > 0) ? (dg - n0)/(nb*nsg) : NaN
            end
        end

        # ─── portes par prompt, conséquences du théorème ───────────────────
        p1err = maximum(abs(Q[(s,s)] - 1.0) for s in branches)
        ups = [(si,bj) for si in branches for bj in branches if pos[bj] < pos[si]]
        p2err = isempty(ups) ? 0.0 : maximum(abs(Q[t]) for t in ups)
        inside = [(si,bj) for si in branches for bj in branches if pos[bj] >= pos[si]]
        two = 0.0; sgn = 0
        for t in inside
            q, r, c = Q[t], RHO[t], COSC[t]
            (isfinite(r) && isfinite(c)) || continue
            two = max(two, abs(q - r*(r+c)/(1 + 2r*c + r^2)))
            (sign(q) == sign(r+c) || abs(r+c) < 1e-6 || abs(q) < 1e-6) || (sgn += 1)
        end

        # ─── statistiques au site de couche 1 (branches contournables) ─────
        byp = [bj for bj in branches if pos[bj] >= pos[s1] && bj != s1]
        qs1 = [Q[(s1,bj)] for bj in byp]
        cs1 = [COSC[(s1,bj)] for bj in byp if isfinite(COSC[(s1,bj)])]
        rs1 = [RHO[(s1,bj)] for bj in byp]
        nbar = sum(Q[(s1,bj)] for bj in branches if pos[bj] >= pos[s1])
        pneg = 100 * count(<(0), qs1) / length(qs1)
        frac_cneg = 100 * count(t -> isfinite(COSC[(s1,t)]) && isfinite(RHO[(s1,t)]) &&
                                     COSC[(s1,t)] < -RHO[(s1,t)], byp) / length(byp)
        allneg = 100 * count(<(0), [Q[t] for t in inside]) / length(inside)

        push!(rows, (n=length(ids), nbar=nbar, p1=nbar/B1, pneg=pneg, allneg=allneg,
                     meanc=mean(cs1), medc=median(cs1), fcneg=frac_cneg,
                     p1err=p1err, p2err=p2err, two=two, sgn=sgn,
                     prompt=pr["prompt"]))

        emit(@sprintf("%3d %4d %9.2e %9.2e %9.2e %4d | %6.1f %8.4f %8.4f %8.1f",
                      k, length(ids), p1err, p2err, two, sgn,
                      pneg, mean(cs1), median(cs1), frac_cneg))

        # ─── CSV incrémental : écrit MAINTENANT, pas à la fin ──────────────
        for (si,bj) in inside
            println(csv_io, k, ",", si, ",", bj, ",", pos[si], ",", pos[bj], ",",
                    Q[(si,bj)], ",", RHO[(si,bj)], ",", COSC[(si,bj)])
        end
        flush(csv_io)
        reclaim()
    end

    # L'agrégat ne porte QUE sur les prompts de ce lot. Il n'est donc écrit que
    # si ce lot termine les 22 ; sinon il serait un résumé partiel présenté
    # comme un résumé global. L'agrégat définitif sur les 22 prompts est
    # recalculé depuis le CSV par bench_eps_multiprompt_aggregate.jl.
    # Un processus qui REPREND ne connaît que ses propres prompts : son agrégat
    # ne porterait que sur son lot, même s'il termine les 22. L'agrégat sur
    # l'ensemble est donc TOUJOURS recalculé depuis le CSV par
    # bench_eps_multiprompt_aggregate.jl, et n'est imprimé ici que dans le cas
    # d'un run unique couvrant tout.
    if RESUME || length(rows) < length(prompts)
        emit(@sprintf("\n[lot : %d prompts mesurés dans ce processus, %d déjà faits, %d au total.",
                      length(rows), length(SKIP), length(prompts)))
        emit(" Pas d'agrégat ici : il ne porterait que sur ce lot. Agrégat complet :")
        emit(" julia --project=. notebook/bench_eps_multiprompt_aggregate.jl]")
        close(csv_io)
        return
    end

    emit("\n" * "="^78)
    emit("AGRÉGAT SUR LES $(length(rows)) PROMPTS")
    emit("="^78)
    f(v) = (minimum(v), median(v), maximum(v))
    nbs = [r.nbar for r in rows]; p1s = [r.p1 for r in rows]
    pns = [r.pneg for r in rows]; ans_ = [r.allneg for r in rows]
    mcs = [r.meanc for r in rows]; fcs = [r.fcneg for r in rows]
    emit(@sprintf("\n  n_bar (site couche 1) : min %.4f  médian %.4f  max %.4f",  f(nbs)...))
    emit(@sprintf("  p_L1                  : min %.5f  médian %.5f  max %.5f", f(p1s)...))
    emit(@sprintf("  %% négatifs, cône du site 1 (55 br.) : min %.1f  médian %.1f  max %.1f", f(pns)...))
    emit(@sprintf("  %% négatifs, matrice entière (1596)  : min %.1f  médian %.1f  max %.1f", f(ans_)...))
    emit(@sprintf("  moyenne de c_j (site couche 1)      : min %+.4f  médiane %+.4f  max %+.4f", f(mcs)...))
    emit(@sprintf("  %% de branches avec c_j < -rho_j     : min %.1f  médian %.1f  max %.1f", f(fcs)...))
    emit(@sprintf("\n  portes, pires valeurs sur tous les prompts :"))
    emit(@sprintf("    P1 max %.3e   P2 max %.3e   2-scalaires max %.3e   violations de signe %d",
                  maximum(r.p1err for r in rows), maximum(r.p2err for r in rows),
                  maximum(r.two for r in rows), sum(r.sgn for r in rows)))

    emit("\n" * "-"^78)
    emit("DÉRIVE D'ALIGNEMENT : ENTRAÎNÉ CONTRE INITIALISATION")
    emit("-"^78)
    nneg_prompts = count(<(0), mcs)
    emit(@sprintf("\n  moyenne de c_j négative sur %d des %d prompts", nneg_prompts, length(mcs)))
    emit(@sprintf("  moyenne des moyennes : %+.4f", mean(mcs)))
    emit("  Référence à l'INITIALISATION (modèle pré-norm synthétique,")
    emit("  bench_eps_convergence_general_results.txt) : moyenne c_j = +0.037,")
    emit("  56 % de valeurs positives.")
    emit("  Une moyenne négative sur la grande majorité des prompts indiquerait")
    emit("  une propriété du MODÈLE ENTRAÎNÉ et non du prompt unique déjà mesuré.")

    emit("\nMatrice complète exportée : " * basename(CSV))
end
close(csv_io)
println("\nÉcrit : ", OUT)
