# ══════════════════════════════════════════════════════════════════════════════
# CERTIFICAT DE 2e ORDRE POUR AtP -- tâche du marqueur.
#
# CONTEXTE (voir bench_eps_atp_comparison_marker_results.txt et
# branch_order_theorem.tex, Corollaire cor:jvp) : AtP_j est le terme d'ordre 1
# d'un développement de Taylor de la métrique dl le long du segment
# [receveur, receveur + Delta_j] où Delta_j = a_j^donneur - a_j^receveur (UNE
# branche patchée seule, les 7 autres restant à leur valeur receveur). Le
# reste exact de ce développement est un terme de courbure (Hessien) intégré
# sur TOUT le segment -- pas un O(eps^2) local, car eps=1 n'est PAS petit ici
# (c'est exactement l'écart donneur/receveur qui a déjà fait échouer AtP dans
# bench_eps_atp_comparison_marker.jl).
#
# CE SCRIPT teste si un correctif de courbure LOCALE (mesuré près du point
# receveur seul, via UN point supplémentaire à eps=0.5, donc SANS jamais
# regarder le point eps=1 pendant le calcul du correctif) suffit à prédire le
# vrai reste sur tout le segment jusqu'à eps=1. C'est un test falsifiable de
# l'hypothèse "gradient exact + courbure locale = certificat utilisable" :
#   - reste_brut   = |vrai_j - AtP_j|                (erreur si on ignore la courbure)
#   - reste_corrigé = |vrai_j - (AtP_j + correction)| (erreur après correctif)
# Si reste_corrigé << reste_brut sur la majorité des (paire, branche) : la
# courbure locale est un bon proxy du reste global -- combinaison saine.
# Si reste_corrigé ~ reste_brut ou pire : la courbure locale ne prédit pas la
# courbure intégrée sur le grand pas donneur/receveur -- confirme
# empiriquement que le "certificat" de la Partie 2 n'est pas exploitable tel
# quel sur ce banc d'essai.
#
# Coût : par paire, 1 passe arrière (donne les 8 AtP_j) + pour chacune des 8
# branches, 2 passes avant supplémentaires (patch complet eps=1, patch
# eps=0.5) + 2 passes de restauration -- O(B) passes avant, PAS de passe
# arrière supplémentaire, CPU uniquement, checkpoint réutilisé tel quel
# (AUCUN réentraînement).
#
# Usage : julia --project=. notebook/bench_eps_atp_certified_interval_marker.jl
# Écrit : notebook/bench_eps_atp_certified_interval_marker_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, LinearAlgebra

const OUT  = joinpath(@__DIR__, "bench_eps_atp_certified_interval_marker_results.txt")
const CKPT = joinpath(@__DIR__, "marker_ckpt", "marker_task")
(isfile(CKPT * ".json") && isfile(CKPT * ".bin")) ||
    error("Checkpoint introuvable : $CKPT -- ce script ne réentraîne JAMAIS.")

ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))

const N_HEADS = 4
const N_CTX   = 2 * N_PAIRS

const DEV_CPU = NeuroDSL.Backend.CPUDevice()
const NS = :marker_cert
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

# Même graine que P3/q_j/AtP -- mêmes 3 paires exactement.
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

open(OUT, "w") do io
    emit(s="") = (println(io, s); println(s); flush(io))
    emitf(fmt, args...) = emit(Printf.format(Printf.Format(fmt), args...))

    emit("CERTIFICAT DE 2e ORDRE POUR AtP -- tâche du marqueur")
    emit("AtP_j = terme d'ordre 1 (1 passe arrière/paire, 8 branches à la fois).")
    emit("correction_j = terme de courbure LOCALE estimé à partir d'UN point")
    emit("supplémentaire à eps=0.5 SEUL (jamais eps=1 pendant le calcul du correctif).")
    emit("vrai_j = effet RÉEL du patch complet (eps=1), mesuré ensuite pour vérification.")
    emit("Checkpoint réutilisé tel quel, AUCUN réentraînement. Mêmes 3 paires que P3/q_j/AtP.")
    emit()
    emitf("P1 (checkpoint) : acc_A=%.4f  acc_B=%.4f", p1.acc_A, p1.acc_B)
    emit()
    emit("Branches (B=8, ordre en profondeur) : " * join(string.(BRANCHES), ", "))
    emitf("Circuit P3 (cité) : %s", string(GROUND_TRUTH))
    emit()

    all_raw = Float64[]      # |vrai_j - AtP_j|
    all_corr = Float64[]     # |vrai_j - (AtP_j + correction_j)|
    all_true = Float64[]
    all_atp = Float64[]

    for (ip, c) in enumerate(PAIRS)
        VKI[] = c.vk; VSKI[] = c.vsk

        r_out = run_forward!(G, NS, receiver_tokens(c))
        recv_act = Dict(b => copy(Array(G.nodes[NS][b].value)) for b in BRANCHES)
        dr = delta_logit(r_out, c)

        for b in BRANCHES; G.nodes[NS][b].is_param = true; end
        NeuroDSL.demand!(G, :dl_obj; namespace=NS)
        NeuroDSL.backward_graph!(G, :dl_obj; namespace=NS)
        grad_b = Dict(b => copy(Array(G.nodes[NS][b].gradient)) for b in BRANCHES)
        for b in BRANCHES; G.nodes[NS][b].is_param = false; end

        d_out = run_forward!(G, NS, donor_tokens(c))
        donor_act = Dict(b => copy(Array(G.nodes[NS][b].value)) for b in BRANCHES)
        dd = delta_logit(d_out, c)

        # Retour à l'état receveur propre avant de patcher branche par branche.
        run_forward!(G, NS, receiver_tokens(c))

        emitf("Paire %d : k=%d sigma(k)=%d v(k)=%d v(sigma(k))=%d  dl_donneur=%.3f  dl_receveur=%.3f",
              ip, c.k, c.sk, c.vk, c.vsk, dd, dr)

        rows = NamedTuple[]
        for b in BRANCHES
            g_ = Float64.(grad_b[b])
            delta = Float64.(donor_act[b]) .- Float64.(recv_act[b])
            atp = sum(g_ .* delta)

            # eps = 0.5 : patch au milieu du segment (SEUL point utilisé pour le correctif).
            mid = Float32.(Float64.(recv_act[b]) .+ 0.5 .* delta)
            NeuroDSL.patch_node!(G, b, Dict(b => mid); namespace=NS)
            out_half = Array(NeuroDSL.demand!(G, :final_logits; namespace=NS))
            dl_half = delta_logit(out_half, c)
            NeuroDSL.patch_node!(G, b, Dict(b => Float32.(recv_act[b])); namespace=NS)
            NeuroDSL.demand!(G, :final_logits; namespace=NS)

            # eps = 1 : patch complet -- le vrai effet, retenu SEULEMENT pour vérification.
            NeuroDSL.patch_node!(G, b, Dict(b => Float32.(donor_act[b])); namespace=NS)
            out_full = Array(NeuroDSL.demand!(G, :final_logits; namespace=NS))
            true_b = delta_logit(out_full, c) - dr
            NeuroDSL.patch_node!(G, b, Dict(b => Float32.(recv_act[b])); namespace=NS)
            NeuroDSL.demand!(G, :final_logits; namespace=NS)

            # Courbure locale (points eps=0 et eps=0.5 uniquement) :
            #   phi(0.5) ≈ phi(0) + 0.5*AtP + (0.5)^2/2 * phi''(0)
            #   => phi''(0) ≈ 8*(phi(0.5) - phi(0) - 0.5*AtP)
            curv0 = 8.0 * ((dl_half - dr) - 0.5 * atp)
            # Extrapolation au pas complet eps=1 sous courbure ~constante :
            correction = 0.5 * curv0
            pred_corrected = atp + correction

            raw_err  = abs(true_b - atp)
            corr_err = abs(true_b - pred_corrected)
            push!(rows, (; b, atp, true_b, correction, pred_corrected, raw_err, corr_err))
            push!(all_raw, raw_err); push!(all_corr, corr_err)
            push!(all_true, true_b); push!(all_atp, atp)
        end

        sort!(rows; by = r -> -abs(r.true_b))
        for r in rows
            emitf("  %-24s AtP=%+.4f  correction=%+.4f  AtP+corr=%+.4f  vrai=%+.4f  |err_brut|=%.4f  |err_corrigé|=%.4f  %s",
                  string(r.b), r.atp, r.correction, r.pred_corrected, r.true_b,
                  r.raw_err, r.corr_err, r.corr_err < r.raw_err ? "AMÉLIORE" : "N'AMÉLIORE PAS")
        end
        emit()
    end

    emit("═"^78)
    emit("AGRÉGAT SUR LES 3 PAIRES × 8 BRANCHES (24 mesures)")
    emit("═"^78)
    n_improve = count(all_corr .< all_raw)
    emitf("  Médiane |err_brut|   (AtP seul)              : %.4f", sort(all_raw)[cld(length(all_raw),2)])
    emitf("  Médiane |err_corrigé| (AtP + courbure locale) : %.4f", sort(all_corr)[cld(length(all_corr),2)])
    emitf("  Moyenne |err_brut|                            : %.4f", sum(all_raw)/length(all_raw))
    emitf("  Moyenne |err_corrigé|                         : %.4f", sum(all_corr)/length(all_corr))
    emitf("  Cas où le correctif AMÉLIORE la prédiction     : %d / %d", n_improve, length(all_raw))
    ratio = (sum(all_corr)/length(all_corr)) / (sum(all_raw)/length(all_raw))
    emitf("  Ratio erreur moyenne corrigée / brute          : %.4f  (< 1 = correctif utile)", ratio)
    emit()

    emit("═"^78)
    emit("VERDICT")
    emit("═"^78)
    if ratio < 0.5 && n_improve >= 18
        emit("Le correctif de courbure LOCALE (un seul point à eps=0.5, sans jamais voir")
        emit("le point eps=1) réduit fortement l'erreur : la courbure mesurée près du point")
        emit("receveur reste représentative du reste sur tout le segment jusqu'au donneur.")
        emit("=> le certificat 'gradient exact + courbure locale' est mathématiquement sain")
        emit("ET utile empiriquement sur ce banc d'essai précis.")
    elseif ratio < 1.0
        emit("Le correctif aide partiellement mais n'élimine pas la majorité de l'erreur :")
        emit("la courbure varie sensiblement entre eps=0.5 et eps=1, donc une sonde locale")
        emit("sous-estime le reste réel sur le grand pas donneur/receveur -- cohérent avec")
        emit("l'hypothèse que le segment traverse plusieurs régions non-linéaires distinctes.")
    else
        emit("Le correctif de courbure locale N'améliore PAS (ratio >= 1) la prédiction du")
        emit("vrai effet de patching par rapport à AtP seul -- confirme empiriquement qu'une")
        emit("sonde de courbure près du point receveur ne borne PAS le reste réel sur un pas")
        emit("de la taille donneur/receveur : le certificat de la Partie 2 n'est PAS exploitable")
        emit("tel quel sur ce banc d'essai, la courbure locale n'étant pas un proxy fiable de")
        emit("la courbure intégrée sur tout le segment.")
    end
end

println("\nÉcrit : ", OUT)
