# ══════════════════════════════════════════════════════════════════════════════
# Filtre Δ_j gratuit -- version RESTREINTE AU ROUND 1 SEULEMENT.
#
# Suite directe de notebook/bench_eps_delta_prefilter_check.jl. Ce script-là
# avait montré : un seuil global "sûr" (protège tout le monde y compris les
# choix tardifs marginaux) = min sur les 3 paires des seuils sûrs par paire
# = 1.372 -> 7/60 candidats filtrables, 42/315 évaluations économisées sur
# toute la recherche gloutonne (13.3%). Un seuil plus agressif (médiane par
# paire) produit de VRAIS faux-négatifs -- toujours sur des ajouts TARDIFS et
# MARGINAUX (gain +0.01 a +0.03, recovery deja >= 0.87-1.0), jamais sur le
# site dominant du round 1 (le plus gros gain, evalue sur le sweep complet
# des 20 candidats -- structurellement le round le plus cher).
#
# Idée testée ici : n'appliquer le filtre Δ_j QU'AU ROUND 1, avec un seuil
# choisi pour ne jamais exclure le site RÉELLEMENT choisi au round 1 (pour
# aucune des 3 paires) -- tous les rounds suivants restent SANS FILTRE,
# exactement comme aujourd'hui. Le risque de faux-négatif tardif observé
# dans le script précédent est éliminé PAR CONSTRUCTION (le filtre ne touche
# jamais que candidats/round 1).
#
# Modification nécessaire de src/patching.jl (justifiée) : un wrapper PUR
# (aucune modification de patching.jl) ne peut PAS obtenir cet effet.
# greedy_patch_search! initialise `selected = Symbol[]` en interne et ne
# permet pas de le réamorcer entre deux appels séparés. Scinder l'appel en
# "round 1 seul (candidats filtrés, max_sites=1)" puis "rounds suivants
# (candidats complets)" recréerait exactement le bug corrigé le 2026-07-10
# (patch d'un site déjà retenu silencieusement effacé pendant la mesure
# d'un candidat amont) car le round 2 ne pourrait plus épingler conjointement
# le site du round 1 pendant l'évaluation de chaque nouveau candidat.
# Solution retenue : un paramètre additif pur `round1_candidates=nothing`
# ajouté à greedy_patch_search! (défaut = comportement 100% inchangé -- voir
# docstring et test/test_patching.jl, aucun test existant ne passe ce kwarg,
# donc aucune régression possible). Quand fourni, restreint UNIQUEMENT le
# balayage du round 1 ; `remaining` lui-même n'est jamais réduit, donc les
# candidats écartés du round 1 redeviennent éligibles dès le round 2.
#
# Comptage des évaluations forward : un compteur enveloppe `metric` (appelé
# exactement une fois par évaluation candidat, y compris pendant
# backward_prune!) -- mesure réelle, pas une formule supposée.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, LinearAlgebra

const CKPT = joinpath(@__DIR__, "marker_ckpt", "marker_task")
const HAVE_CKPT = isfile(CKPT * ".json") && isfile(CKPT * ".bin")
println("Checkpoint : ", CKPT, "  (existant : ", HAVE_CKPT, ")")
if HAVE_CKPT
    ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
end

include(joinpath(@__DIR__, "marker_task_experiment.jl"))

if HAVE_CKPT
    println("Chargement du checkpoint : ", CKPT)
    NeuroDSL.load_graph!(g, ns, CKPT; overwrite=true)
end

const N_HEADS = 4
const D_HEAD  = DIM ÷ N_HEADS
const N_CTX   = 2 * N_PAIRS

result = evaluate_marker(g, logits, ns; n_eval=400)
println("P1 (gate, checkpoint) : acc_A = ", result.acc_A, "   acc_B = ", result.acc_B)
(result.acc_A >= 0.95 && result.acc_B >= 0.95) ||
    error("P1 non satisfait sur ce checkpoint -- arrêt, l'analyse n'aurait pas de sens.")

function run_forward!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
end
function run_and_capture!(g, ns, tokens)
    out = run_forward!(g, ns, tokens)
    cache = NeuroDSL.capture_activations(g, ns)
    return out, cache
end
function full_reset!(g, ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, :final_logits; namespace=ns)
    return g
end

head_site(l, h) = Symbol("layer_$(l)_mha_ao_h$(h)")
mlp_site(l)     = Symbol("layer_$(l)_mlp_out")

const CANDIDATES = Symbol[]
for l in 1:N_LAYERS
    for h in 1:N_HEADS; push!(CANDIDATES, head_site(l, h)); end
    push!(CANDIDATES, mlp_site(l))
end
println("Candidats (", length(CANDIDATES), ") : ", CANDIDATES)

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

const N_SEARCH_PAIRS = 3
const MAX_SITES = 6

# ── ÉTAPE 1 : seuil sûr round-1-seul, par paire, dérivé du site RÉELLEMENT
#    choisi au round 1 dans le script précédent (bench_eps_delta_prefilter_check
#    .jl, même seed 31415, mêmes 3 paires) : layer_1_mlp_out à chaque fois,
#    avec Δ_j = 12.944 (paire k=1), 26.533 (paire k=2), 12.416 (paire k=7).
#    Seuil sûr par paire = Δ_j de ce site (le plus grand seuil qui le laisse
#    encore >= seuil). Seuil global sûr = min des 3 = 12.416 (paire k=7).
#    Recalculé ici sur les VRAIS Δ_j de CETTE exécution pour ne rien supposer.
round1_winner_delta = Dict{Int,Float64}()  # rempli au fil des paires

# Compteur d'évaluations forward (incrémenté une fois par appel metric()).
mutable struct EvalCounter
    n::Int
end
function counted_metric(base_metric, counter::EvalCounter)
    return out -> begin
        counter.n += 1
        return base_metric(out)
    end
end

pair_results = NamedTuple[]
rng_p3 = MersenneTwister(31415)

for ip in 1:N_SEARCH_PAIRS
    c = draw_search_pair!(g, ns, rng_p3)
    d_out, donor_cache = run_and_capture!(g, ns, donor_tokens(c))   # clean
    r_out, recv_cache  = run_and_capture!(g, ns, receiver_tokens(c)) # corrupted
    dd, dr = delta_logit(d_out, c), delta_logit(r_out, c)
    metric = out -> (delta_logit(Array(out), c) - dr) / (dd - dr)
    println("\n  Paire $ip : k=$(c.k) sigma(k)=$(c.sk) v(k)=$(c.vk) v(sigma(k))=$(c.vsk)")

    deltas = Dict{Symbol,Float64}()
    for s in CANDIDATES
        a = Array(donor_cache[s]); b = Array(recv_cache[s])
        deltas[s] = sqrt(sum(abs2, Float64.(a) .- Float64.(b)))
    end

    # ── Référence : recherche gloutonne SANS FILTRE (baseline exacte, telle
    #    que produite en production aujourd'hui). ──────────────────────────
    ctr_base = EvalCounter(0)
    selected_base, traj_base = NeuroDSL.greedy_patch_search!(g, :final_logits, CANDIDATES,
        donor_cache, recv_cache, copy(d_out), copy(r_out);
        max_sites=MAX_SITES, namespace=ns, metric=counted_metric(metric, ctr_base))
    println("    [BASELINE] Greedy : ", [(String(t.site), round(t.cumulative_recovery, digits=3)) for t in traj_base])
    println("    [BASELINE] Évaluations forward (phase greedy) : ", ctr_base.n)

    remaining_base, pruned_base = isempty(selected_base) ? (Symbol[], Symbol[]) :
        NeuroDSL.backward_prune!(g, :final_logits, selected_base,
            donor_cache, recv_cache, copy(d_out), copy(r_out);
            namespace=ns, tol=Float32(0.05), metric=metric)
    full_reset!(g, ns)
    println("    [BASELINE] Après backward_prune! : ", remaining_base, "  (élagués : ", pruned_base, ")")

    round1_winner = traj_base[1].site
    round1_winner_delta[ip] = deltas[round1_winner]
    println("    Site retenu au round 1 (baseline) : $round1_winner, Δ_j = ", round(deltas[round1_winner], digits=3))

    push!(pair_results, (; c, deltas, selected_base, traj_base, remaining_base, pruned_base,
                            round1_winner, round1_winner_delta = deltas[round1_winner],
                            n_eval_base = ctr_base.n))
    flush(stdout)
end

println("\n", "═"^70)
println("SEUIL ROUND-1-SEUL")
for ip in 1:N_SEARCH_PAIRS
    println("  Paire $ip (k=$(pair_results[ip].c.k)) : site round 1 = ",
            pair_results[ip].round1_winner, "  Δ_j = ", round(pair_results[ip].round1_winner_delta, digits=3))
end
const SAFE_ROUND1_THRESHOLD = minimum(pr.round1_winner_delta for pr in pair_results)
println("  Seuil global sûr round-1-seul (min des 3) = ", round(SAFE_ROUND1_THRESHOLD, digits=3))
println("═"^70)

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2+3 : rejoue les 3 paires avec le filtre round-1-seul appliqué via le
# nouveau kwarg round1_candidates (src/patching.jl) -- rounds suivants
# strictement sans filtre. Vérifie l'identité du circuit final + mesure
# l'économie réelle d'évaluations forward.
# ══════════════════════════════════════════════════════════════════════════════
println("\nRejeu avec filtre round-1-seul (seuil = ", round(SAFE_ROUND1_THRESHOLD, digits=3), ") ...")

rng_p3b = MersenneTwister(31415)  # même seed -> mêmes 3 paires, reproductibles

all_identical = true
total_saved = 0
total_base = 0

for ip in 1:N_SEARCH_PAIRS
    c = draw_search_pair!(g, ns, rng_p3b)
    d_out, donor_cache = run_and_capture!(g, ns, donor_tokens(c))
    r_out, recv_cache  = run_and_capture!(g, ns, receiver_tokens(c))
    dd, dr = delta_logit(d_out, c), delta_logit(r_out, c)
    metric = out -> (delta_logit(Array(out), c) - dr) / (dd - dr)

    deltas = Dict{Symbol,Float64}()
    for s in CANDIDATES
        a = Array(donor_cache[s]); b = Array(recv_cache[s])
        deltas[s] = sqrt(sum(abs2, Float64.(a) .- Float64.(b)))
    end
    round1_set = [s for s in CANDIDATES if deltas[s] >= SAFE_ROUND1_THRESHOLD]
    println("\n  Paire $ip : k=$(c.k) -- round1_candidates = ", length(round1_set), "/", length(CANDIDATES),
            " (filtre appliqué UNIQUEMENT au round 1)")

    ctr_filt = EvalCounter(0)
    selected_filt, traj_filt = NeuroDSL.greedy_patch_search!(g, :final_logits, CANDIDATES,
        donor_cache, recv_cache, copy(d_out), copy(r_out);
        max_sites=MAX_SITES, namespace=ns, metric=counted_metric(metric, ctr_filt),
        round1_candidates=round1_set)
    println("    [FILTRÉ]   Greedy : ", [(String(t.site), round(t.cumulative_recovery, digits=3)) for t in traj_filt])
    println("    [FILTRÉ]   Évaluations forward (phase greedy) : ", ctr_filt.n)

    remaining_filt, pruned_filt = isempty(selected_filt) ? (Symbol[], Symbol[]) :
        NeuroDSL.backward_prune!(g, :final_logits, selected_filt,
            donor_cache, recv_cache, copy(d_out), copy(r_out);
            namespace=ns, tol=Float32(0.05), metric=metric)
    full_reset!(g, ns)
    println("    [FILTRÉ]   Après backward_prune! : ", remaining_filt, "  (élagués : ", pruned_filt, ")")

    pr = pair_results[ip]
    identical_selected  = pr.selected_base == selected_filt
    identical_remaining = Set(pr.remaining_base) == Set(remaining_filt)
    identical = identical_selected && identical_remaining
    global all_identical &= identical
    println("    Identique à la baseline ? selected: ", identical_selected,
            "   remaining (circuit final) : ", identical_remaining)

    global total_base += pr.n_eval_base
    global total_saved += (pr.n_eval_base - ctr_filt.n)
    println("    Économie (phase greedy) : ", pr.n_eval_base - ctr_filt.n, "/", pr.n_eval_base,
            " évaluations (", round(100*(pr.n_eval_base - ctr_filt.n)/pr.n_eval_base, digits=1), "%).")
end

println("\n", "═"^70)
println("BILAN FINAL")
println("  Circuit final identique à la baseline sur les 3 paires : ", all_identical ? "OUI" : "NON")
println("  TOTAL économies (round-1-seul, phase greedy) : $total_saved / $total_base ",
        "(", round(100*total_saved/total_base, digits=1), "%).")
println("  Rappel -- seuil global naïf (script précédent, protège tous les rounds) : 42/315 (13.3%).")
println("═"^70)
