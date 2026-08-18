# ══════════════════════════════════════════════════════════════════════════════
# RAFFINEMENT PAR CHEMIN (K PAS) DE L'ESTIMATEUR AtP -- tâche du marqueur.
#
# CONTEXTE : bench_eps_atp_comparison_marker.jl montre que AtP_j (terme d'ordre 1
# en eps=1, UN seul point : le gradient au receveur) rate largement le vrai effet
# de patching quand Delta_j = a_j^donneur - a_j^receveur est grand. Le correctif
# à UN point de courbure (bench_eps_atp_certified_interval_marker.jl, eps=0.5)
# aide dans 20/24 cas mais SUR-corrige catastrophiquement sur la branche à plus
# gros écart (layer_1_mha_output_out), car il ne voit la courbure qu'EN UN point
# et extrapole sur tout le segment [0,1] -- pas de garantie si phi''' varie sur
# ce segment (saturation softmax, coudes GELU/SiLU).
#
# IDÉE TESTÉE ICI : subdiviser [0,1] en K sous-pas de taille 1/K. Phi(eps) :=
# métrique(a_receveur + eps*Delta_j), phi(0)=0 par construction (dl_receveur
# retranché), vrai_j = phi(1)-phi(0) EXACTEMENT (théorème fondamental du calcul,
# phi absolument continue -- réseau lisse par morceaux). AtP_j = phi'(0)*1 est
# l'approximation la plus grossière de l'intégrale ∫_0^1 phi'(eps) d(eps) = vrai_j
# (régle du rectangle à GAUCHE, K=1). On la raffine par la RÈGLE DU POINT MILIEU
# composite à K pas -- c'est le schéma centré analogue en esprit à la différence
# centrée du Corollaire cor:jvp de branch_order_theorem.tex, et c'est exactement
# la discrétisation de Integrated Gradients (Sundararajan et al. 2017) :
#
#   IG_K := (1/K) * sum_{k=1}^K phi'(eps_k),   eps_k = (k-0.5)/K
#
# BORNE : la règle du point milieu composite a une erreur EXACTE et standard
#   |IG_K - vrai_j| <= sup_{eps in [0,1]} |phi'''(eps)| / (24 K^2)
# -- même terme de courbure (dérivée d'ordre 3) que le Corollaire cor:jvp, mais
# maintenant chaque sous-pas a une taille 1/K qu'on choisit, donc l'erreur est
# PROUVABLEMENT O(1/K^2) SI on dispose d'une borne fiable sur sup|phi'''|.
#
# HONNÊTETÉ SUR CE QUI EST RIGOUREUX ET CE QUI NE L'EST PAS :
#   - La formule d'erreur ci-dessus EST un théorème de calcul standard (exacte,
#     pas approximative) UNE FOIS qu'on a une borne sur sup|phi'''| sur [0,1].
#   - Obtenir une VRAIE borne sup (pas un échantillon) sur phi''' d'un vrai
#     transformer entraîné est hors de portée en pratique (bornes de Lipschitz
#     certifiées sur softmax/GELU/LayerNorm composées sur plusieurs couches ->
#     bornes vides ou astronomiques). Donc ci-dessous M3 est un PROXY EMPIRIQUE :
#     une différence finie de phi' à 3 points consécutifs de la grille des K
#     points milieux déjà calculés (AUCUN passage supplémentaire -- on réutilise
#     les K gradients déjà obtenus pour IG_K), donnant K-2 estimations
#     ponctuelles de phi''' dont on prend le MAX. C'est un échantillonnage fini
#     d'une fonction continue : ça ne certifie PAS un sup rigoureux (un pic entre
#     deux points serait invisible), et les 2 sous-intervalles aux bords ne sont
#     pas couverts par cette estimation (limite déclarée, pas cachée). C'est un
#     bien meilleur proxy que le point unique de la Partie 2 précédente parce
#     que chaque sous-segment est petit (1/K, pas 1), mais ce n'est toujours PAS
#     un certificat au sens mathématique strict.
#
# COÛT (important, honnête) : phi'(eps_k) pour la branche b nécessite de patcher
# b à CETTE valeur précise (toutes les autres branches restant au receveur), donc
# UN passage avant + UN passage arrière PAR (branche, sous-pas) -- contrairement
# au passage arrière unique à eps=0 qui donne AtP_j pour LES 8 branches À LA FOIS
# (c'est précisément pourquoi AtP est "gratuit" : un seul état avant partagé).
# Les K points intérieurs sont des états avant PRIVÉS à chaque branche (patcher
# b1 seul != patcher b2 seul), donc ILS NE PEUVENT PAS être partagés entre
# branches : coût = O(B*K) passages arrière, PAS O(K). Sur ce banc (B=8), pour
# K=8 c'est 8*8=64 passages arrière par paire -- nettement plus cher que la
# méthode d'ablation EXACTE de Théorème thm:ablation (B+1=9 passages) et même
# plus cher que le calcul du vrai effet lui-même (O(B) passages AVANT SEULS,
# sans rétropropagation, moins coûteux qu'un passage arrière). Autrement dit :
# dans CE régime (patch d'une seule branche à la fois), calculer la vérité
# exacte est déjà moins cher que ce raffinement -- le raffinement n'a de sens
# économique que quand la vérité exacte coûte elle-même un passage arrière complet
# ou plus (pas le cas ici). On le teste quand même pour vérifier SI la borne
# tient, question distincte de son coût.
#
# Usage : julia --project=. notebook/bench_eps_atp_path_integral_certified_marker.jl
# Écrit : notebook/bench_eps_atp_path_integral_certified_marker_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, LinearAlgebra

const OUT  = joinpath(@__DIR__, "bench_eps_atp_path_integral_certified_marker_results.txt")
const CKPT = joinpath(@__DIR__, "marker_ckpt", "marker_task")
(isfile(CKPT * ".json") && isfile(CKPT * ".bin")) ||
    error("Checkpoint introuvable : $CKPT -- ce script ne réentraîne JAMAIS.")

ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))

const N_HEADS = 4
const N_CTX   = 2 * N_PAIRS
const K_STEPS = parse(Int, get(ENV, "IG_K", "8"))

const DEV_CPU = NeuroDSL.Backend.CPUDevice()
const NS = :marker_path
G = NeuroDSL.NeuroGraph(namespace=NS, device=DEV_CPU)
NeuroDSL.load_graph!(G, NS, CKPT; overwrite=true)

function run_forward!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
end

function check_p1(g, ns; n_eval=200, seed=999)
    rng = MersenneTwister(seed)
    okA = totA = okB = totB = 0
    for _ in 1:n_eval
        tokens, _, fmt, _, v = sample_marker_sequence(rng)
        pred = argmax(vec(run_forward!(g, ns, tokens)))
        if fmt == :A; totA += 1; okA += (pred == v) ? 1 : 0
        else;         totB += 1; okB += (pred == v) ? 1 : 0
        end
    end
    return (; acc_A = okA / totA, acc_B = okB / totB)
end
p1 = check_p1(G, NS)
p1_ok = p1.acc_A >= 0.95 && p1.acc_B >= 0.95
p1_ok || error("P1 non confirmé sur le checkpoint rechargé -- arrêt (p1=$p1)")

function sample_contrast(rng)
    while true
        k  = rand(rng, 1:V)
        sk = SIGMA[k]
        others = collect(setdiff(1:V, (k, sk)))
        rest = length(others) >= N_PAIRS - 2 ? shuffle(rng, others)[1:N_PAIRS-2] : Int[]
        ks = vcat([k, sk], rest)
        vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS)
            push!(ctx, ks[i]); push!(ctx, vs[i])
        end
        return (; ctx, k, sk, vk = vs[1], vsk = vs[2])
    end
end
donor_tokens(c)    = vcat(c.ctx, [MARKER_A, c.sk])
receiver_tokens(c) = vcat(c.ctx, [MARKER_B, c.sk])
delta_logit(out, c) = Float64(out[1, c.vsk] - out[1, c.vk])

function draw_search_pair!(g, ns, rng; max_tries=200)
    for _ in 1:max_tries
        c = sample_contrast(rng)
        d_out = run_forward!(g, ns, donor_tokens(c))
        r_out = run_forward!(g, ns, receiver_tokens(c))
        dd, dr = delta_logit(d_out, c), delta_logit(r_out, c)
        (argmax(vec(d_out)) == c.vsk && argmax(vec(r_out)) == c.vk && dd > 0 > dr) && return c
    end
    error("Impossible de tirer une paire de contraste exploitable en $max_tries essais")
end

# Même graine que P3/q_j/AtP/certificat -- mêmes 3 paires exactement.
rng_p3 = MersenneTwister(31415)
PAIRS = [draw_search_pair!(G, NS, rng_p3) for _ in 1:3]

const VKI  = Ref(0)
const VSKI = Ref(0)
if !haskey(NeuroDSL.CUSTOM_OPS, :dl_logit_diff)
    NeuroDSL.register_op!(:dl_logit_diff,
        (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> begin
            fl = inputs[1]
            out_buf[1] = fl[1, VSKI[]] - fl[1, VKI[]]
            out_buf
        end)
    NeuroDSL.CUSTOM_SHAPE_RULES[:dl_logit_diff] = (inputs, attrs) -> (1,)
    NeuroDSL.GRAD_RULES[:dl_logit_diff] = (dev, dy, ctx, inputs) -> begin
        fl = inputs[1]
        gr = zeros(Float32, size(fl))
        gr[1, VSKI[]] = dy[1]
        gr[1, VKI[]]  = -dy[1]
        (NeuroDSL.Backend.to_device(dev, gr),)
    end
end
NeuroDSL.addrule!(G, NeuroDSL.GraphRule(:dl_obj, [:final_logits], :dl_logit_diff; namespace=NS))

BRANCHES = Symbol[]
for l in 1:N_LAYERS
    push!(BRANCHES, Symbol("layer_$(l)_mha_output_out"))
    push!(BRANCHES, Symbol("layer_$(l)_mlp_out"))
end
@assert length(BRANCHES) == 8

const GROUND_TRUTH = :layer_1_mlp_out

# ─── phi(eps) : passage AVANT SEUL (aucune rétropropagation) ───
function phi_value_at!(g, ns, c, b, val)
    NeuroDSL.patch_node!(g, b, Dict(b => Float32.(val)); namespace=ns)
    out = Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
    return delta_logit(out, c)
end

# ─── phi'(eps) : passage AVANT (patch) + passage ARRIÈRE (gradient AU point patché) ───
function phi_prime_at!(g, ns, b, val, delta)
    NeuroDSL.patch_node!(g, b, Dict(b => Float32.(val)); namespace=ns)
    g.nodes[ns][b].is_param = true
    NeuroDSL.demand!(g, :dl_obj; namespace=ns)
    NeuroDSL.backward_graph!(g, :dl_obj; namespace=ns)
    grad = Float64.(copy(Array(g.nodes[ns][b].gradient)))
    g.nodes[ns][b].is_param = false
    return sum(grad .* delta)
end

function restore!(g, ns, b, recv_val)
    NeuroDSL.patch_node!(g, b, Dict(b => Float32.(recv_val)); namespace=ns)
    NeuroDSL.demand!(g, :final_logits; namespace=ns)
end

open(OUT, "w") do io
    emit(s="") = (println(io, s); println(s); flush(io))
    emitf(fmt, args...) = emit(Printf.format(Printf.Format(fmt), args...))

    emit("RAFFINEMENT PAR CHEMIN (K=$K_STEPS PAS, RÈGLE DU POINT MILIEU) DE AtP")
    emit("IG_K = (1/K) * sum_k phi'(eps_k), eps_k=(k-0.5)/K -- discrétisation de")
    emit("Integrated Gradients ; borne théorique |IG_K-vrai| <= sup|phi'''|/(24K^2).")
    emit("M3 = proxy EMPIRIQUE (différence finie à 3 points sur la grille phi' déjà")
    emit("calculée, AUCUN passage supplémentaire) -- PAS un sup rigoureux, voir en-tête.")
    emit("Checkpoint réutilisé tel quel, AUCUN réentraînement. Mêmes 3 paires que P3/q_j/AtP/certificat.")
    emit()
    emitf("P1 (checkpoint) : acc_A=%.4f  acc_B=%.4f", p1.acc_A, p1.acc_B)
    emit()
    emit("Branches (B=8, ordre en profondeur) : " * join(string.(BRANCHES), ", "))
    emitf("Circuit P3 (cité) : %s", string(GROUND_TRUTH))
    emit()

    all_err_atp = Float64[]
    all_err_igK = Float64[]
    all_true    = Float64[]
    all_covered = Bool[]
    layer1_mha_rows = NamedTuple[]

    for (ip, c) in enumerate(PAIRS)
        VKI[] = c.vk; VSKI[] = c.vsk

        r_out = run_forward!(G, NS, receiver_tokens(c))
        recv_act = Dict(b => copy(Array(G.nodes[NS][b].value)) for b in BRANCHES)
        dr = delta_logit(r_out, c)

        # AtP baseline (K=1, eps=0) : UN SEUL passage arrière donne les 8 gradients à la fois.
        for b in BRANCHES; G.nodes[NS][b].is_param = true; end
        NeuroDSL.demand!(G, :dl_obj; namespace=NS)
        NeuroDSL.backward_graph!(G, :dl_obj; namespace=NS)
        grad_b = Dict(b => copy(Array(G.nodes[NS][b].gradient)) for b in BRANCHES)
        for b in BRANCHES; G.nodes[NS][b].is_param = false; end

        d_out = run_forward!(G, NS, donor_tokens(c))
        donor_act = Dict(b => copy(Array(G.nodes[NS][b].value)) for b in BRANCHES)
        dd = delta_logit(d_out, c)

        run_forward!(G, NS, receiver_tokens(c))  # retour à l'état receveur propre

        emitf("Paire %d : k=%d sigma(k)=%d v(k)=%d v(sigma(k))=%d  dl_donneur=%.3f  dl_receveur=%.3f",
              ip, c.k, c.sk, c.vk, c.vsk, dd, dr)

        rows = NamedTuple[]
        for b in BRANCHES
            delta = Float64.(donor_act[b]) .- Float64.(recv_act[b])
            atp = sum(Float64.(grad_b[b]) .* delta)

            # vrai effet (passage AVANT SEUL, patch complet eps=1)
            true_b = phi_value_at!(G, NS, c, b, donor_act[b]) - dr
            restore!(G, NS, b, recv_act[b])

            # K passages arrière PRIVÉS à cette branche (points milieux)
            h = 1.0 / K_STEPS
            phis = Float64[]
            for k in 1:K_STEPS
                eps_k = (k - 0.5) / K_STEPS
                val = Float64.(recv_act[b]) .+ eps_k .* delta
                push!(phis, phi_prime_at!(G, NS, b, val, delta))
            end
            restore!(G, NS, b, recv_act[b])

            IG_K = h * sum(phis)

            # M3 : proxy empirique de sup|phi'''| par différence finie à 3 points
            # consécutifs de la grille phi' (sous-intervalles de bord NON couverts).
            m3_vals = Float64[abs((phis[k+1] - 2*phis[k] + phis[k-1]) / h^2)
                               for k in 2:(K_STEPS-1)]
            M3 = isempty(m3_vals) ? NaN : maximum(m3_vals)
            pred_bound = M3 / (24 * K_STEPS^2)

            err_atp = abs(true_b - atp)
            err_igK = abs(true_b - IG_K)
            covered = isfinite(pred_bound) && err_igK <= pred_bound

            push!(rows, (; b, atp, IG_K, true_b, err_atp, err_igK, M3, pred_bound, covered))
            push!(all_err_atp, err_atp); push!(all_err_igK, err_igK); push!(all_true, true_b)
            push!(all_covered, covered)
            b == :layer_1_mha_output_out && push!(layer1_mha_rows, (; ip, atp, IG_K, true_b, err_atp, err_igK, M3, pred_bound, covered))
        end

        sort!(rows; by = r -> -abs(r.true_b))
        for r in rows
            emitf("  %-24s AtP=%+.4f  IG_%d=%+.4f  vrai=%+.4f  |err_AtP|=%.4f  |err_IG_%d|=%.4f  M3=%.2f  borne=%.4f  %s  %s",
                  string(r.b), r.atp, K_STEPS, r.IG_K, r.true_b, r.err_atp, K_STEPS, r.err_igK,
                  r.M3, r.pred_bound,
                  r.err_igK < r.err_atp ? "AMÉLIORE" : "N'AMÉLIORE PAS",
                  r.covered ? "COUVERT" : "NON COUVERT")
        end
        emit()
    end

    emit("═"^92)
    emit("AGRÉGAT SUR LES 3 PAIRES × 8 BRANCHES (24 mesures), K=$K_STEPS")
    emit("═"^92)
    n_improve = count(all_err_igK .< all_err_atp)
    n_covered = count(all_covered)
    emitf("  Médiane |err_AtP|  (K=1, un point)             : %.4f", sort(all_err_atp)[cld(length(all_err_atp),2)])
    emitf("  Médiane |err_IG_%d|                             : %.4f", K_STEPS, sort(all_err_igK)[cld(length(all_err_igK),2)])
    emitf("  Moyenne |err_AtP|                               : %.4f", sum(all_err_atp)/length(all_err_atp))
    emitf("  Moyenne |err_IG_%d|                              : %.4f", K_STEPS, sum(all_err_igK)/length(all_err_igK))
    emitf("  Cas où IG_%d AMÉLIORE sur AtP                    : %d / %d", K_STEPS, n_improve, length(all_err_atp))
    emitf("  Cas où l'erreur réelle tombe SOUS la borne M3    : %d / %d", n_covered, length(all_covered))
    ratio = (sum(all_err_igK)/length(all_err_igK)) / (sum(all_err_atp)/length(all_err_atp))
    emitf("  Ratio erreur moyenne IG_%d / AtP                 : %.4f  (< 1 = raffinement utile)", K_STEPS, ratio)
    emit()

    emit("═"^92)
    emit("FOCUS layer_1_mha_output_out (la branche qui a fait sur-corriger le")
    emit("certificat à un point dans bench_eps_atp_certified_interval_marker.jl)")
    emit("═"^92)
    for r in layer1_mha_rows
        emitf("  Paire %d : AtP=%+.4f (|err|=%.4f)  IG_%d=%+.4f (|err|=%.4f)  vrai=%+.4f  borne=%.4f  %s",
              r.ip, r.atp, r.err_atp, K_STEPS, r.IG_K, r.err_igK, r.true_b, r.pred_bound,
              r.covered ? "COUVERT" : "NON COUVERT")
    end
    emit()

    emit("═"^92)
    emit("VERDICT")
    emit("═"^92)
    if ratio < 0.5 && n_improve >= 20
        emit("Le raffinement à K pas réduit fortement l'erreur par rapport à AtP seul,")
        emit("y compris (à vérifier ligne par ligne ci-dessus) sur layer_1_mha_output_out.")
    elseif ratio < 1.0
        emit("Le raffinement aide en moyenne mais n'élimine pas toute l'erreur -- vérifier")
        emit("si layer_1_mha_output_out reste un cas dégradé malgré le raffinement.")
    else
        emit("Le raffinement à K pas N'améliore PAS (ratio >= 1) l'estimation en moyenne")
        emit("sur ce banc -- le sous-échantillonnage ne suffit pas à K=$K_STEPS.")
    end
    if n_covered == length(all_covered)
        emit("La borne M3/(24K^2), bien que fondée sur un proxy empirique et non un sup")
        emit("rigoureux, couvre l'erreur réelle observée sur les 24 mesures.")
    else
        emitf("La borne M3/(24K^2) est violée sur %d/%d mesures : le proxy empirique",
              length(all_covered)-n_covered, length(all_covered))
        emit("(différence finie à 3 points, sous-intervalles de bord non couverts) sous-estime")
        emit("par endroits la vraie courbure -- confirmerait que ce n'est PAS un certificat")
        emit("rigoureux, juste un proxy mieux testé que celui à un seul point.")
    end
end

println("\nÉcrit : ", OUT)
