# ══════════════════════════════════════════════════════════════════════════════
# Filtre de présélection "gratuit" pour greedy_patch_search! -- vérification.
#
# Idée testée (proposée par l'utilisateur, PAS prescriptive) : clean_cache et
# corrupted_cache (src/patching.jl, capture_activations) sont déjà des
# snapshots COMPLETS de toutes les activations, pris via DEUX forward passes
# (un propre, un corrompu) AVANT que greedy_patch_search! ne démarre sa boucle
# -- confirmé en lisant capture_activations (src/patching.jl:38-59) et
# run_and_capture! (notebook/marker_task_causal_analysis.jl:107-111) : aucun
# forward pass supplémentaire n'est nécessaire pour lire ces caches.
#
# Pour tout candidat j, ||Δ_j|| = ||a_j^clean - a_j^corrupted|| (norme sur la
# valeur ENTIÈRE du nœud, telle que cachée) est une condition NÉCESSAIRE (pas
# suffisante) de pertinence causale : si l'activation ne change presque pas
# entre les deux runs, la PATCHER ne peut presque rien changer à la sortie.
# Un Δ_j petit peut donc servir à retirer j de `remaining` AVANT le round 1
# (Δ_j est fixe, calculé une seule fois sur les caches déjà en mémoire -- pas
# besoin de le recalculer à chaque round), sans jamais influencer laquelle
# des candidats retenus la recherche gloutonne choisit réellement.
#
# Protocole : réutilise TEL QUEL le setup P3 de marker_task_causal_analysis.jl
# (mêmes 20 candidats, mêmes 3 paires de recherche, seed rng 31415 identique,
# même draw_search_pair!/sample_contrast). Réentraîner coûterait ~75 min
# (voir marker_task_causal_analysis.log) -- on charge à la place le
# checkpoint déjà validé P1 (notebook/marker_ckpt/marker_task, config
# gagnante figée), motif identique à marker_seed_matrix.jl. CPU-only n'est
# pas atteignable ici : build_marker_graph est câblé sur CUDADevice() en dur
# dans marker_task_experiment.jl (partagé par tous les scripts marker_task_*,
# non modifié ici) -- mais le modèle est minuscule (dim=64, 4 couches,
# V=8, seq_len=8) donc le coût réel est de l'ordre de la seconde.
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

pair_results = NamedTuple[]
rng_p3 = MersenneTwister(31415)

for ip in 1:N_SEARCH_PAIRS
    c = draw_search_pair!(g, ns, rng_p3)
    d_out, donor_cache = run_and_capture!(g, ns, donor_tokens(c))   # clean
    r_out, recv_cache  = run_and_capture!(g, ns, receiver_tokens(c)) # corrupted
    dd, dr = delta_logit(d_out, c), delta_logit(r_out, c)
    metric = out -> (delta_logit(Array(out), c) - dr) / (dd - dr)
    println("\n  Paire $ip : k=$(c.k) sigma(k)=$(c.sk) v(k)=$(c.vk) v(sigma(k))=$(c.vsk)")
    println("    dl_donneur = ", round(dd, digits=3), "   dl_receveur = ", round(dr, digits=3))

    # ── Filtre gratuit : Δ_j = ||a_j^clean - a_j^corrupted|| sur les caches
    #    déjà calculés ci-dessus (AUCUN forward supplémentaire). ─────────────
    deltas = Dict{Symbol,Float64}()
    for s in CANDIDATES
        a = Array(donor_cache[s]); b = Array(recv_cache[s])
        deltas[s] = sqrt(sum(abs2, Float64.(a) .- Float64.(b)))
    end
    ranked = sort(CANDIDATES; by = s -> -deltas[s])
    rank_mlp1 = findfirst(==(mlp_site(1)), ranked)
    println("    Δ_j classés (desc.) : ", [(String(s), round(deltas[s], digits=3)) for s in ranked])
    println("    Rang de layer_1_mlp_out par Δ_j : $rank_mlp1 / $(length(CANDIDATES))")

    # Recherche gloutonne + élagage arrière réels (référence -- coûte les
    # vrais forward passes, sert à savoir qui était RÉELLEMENT sélectionné).
    selected, traj = NeuroDSL.greedy_patch_search!(g, :final_logits, CANDIDATES,
        donor_cache, recv_cache, copy(d_out), copy(r_out);
        max_sites=MAX_SITES, namespace=ns, metric=metric)
    println("    Greedy : ", [(String(t.site), round(t.cumulative_recovery, digits=3)) for t in traj])

    remaining, pruned = isempty(selected) ? (Symbol[], Symbol[]) :
        NeuroDSL.backward_prune!(g, :final_logits, selected,
            donor_cache, recv_cache, copy(d_out), copy(r_out);
            namespace=ns, tol=Float32(0.05), metric=metric)
    full_reset!(g, ns)
    println("    Après backward_prune! : ", remaining, "  (élagués : ", pruned, ")")

    push!(pair_results, (; c, deltas, ranked, rank_mlp1, selected, traj, remaining, pruned))
    flush(stdout)
end

# ══════════════════════════════════════════════════════════════════════════════
# Bilan : le filtre aurait-il pu retirer des candidats de `remaining` avant le
# round 1, SANS jamais exclure un site réellement choisi par greedy ?
# ══════════════════════════════════════════════════════════════════════════════
println("\n", "═"^70)
println("BILAN")

ever_selected = Set{Symbol}()
for pr in pair_results, t in pr.traj
    push!(ever_selected, t.site)
end
println("  Sites JAMAIS sélectionnés par aucun round d'aucune des 3 paires (",
        length(CANDIDATES) - length(ever_selected), "/", length(CANDIDATES), ") : ",
        sort(collect(setdiff(Set(CANDIDATES), ever_selected)); by=String))

# Pour chaque paire, seuil "sûr" = le plus petit Δ_j parmi les sites RÉELLEMENT
# sélectionnés dans CETTE paire -- tout candidat de Δ_j strictement inférieur
# aurait pu être retiré de `remaining` avant le round 1 SANS changer un seul
# choix de cette paire (test de faux-négatif direct, par construction).
total_baseline = 0
total_saved = 0
false_negative_found = false
for pr in pair_results
    selected_deltas = [pr.deltas[s] for s in pr.selected]
    safe_threshold = isempty(selected_deltas) ? -Inf : minimum(selected_deltas)
    skippable = [s for s in CANDIDATES if pr.deltas[s] < safe_threshold]
    n_rounds = length(pr.traj)
    n0 = length(CANDIDATES)
    baseline = sum((n0 - (r - 1)) for r in 1:n_rounds)
    saved = length(skippable) * n_rounds
    global total_baseline += baseline
    global total_saved += saved
    println("  Paire k=$(pr.c.k) : seuil sûr Δ >= ", round(safe_threshold, digits=3),
            " -> ", length(skippable), "/", length(CANDIDATES), " candidats filtrables sans risque, ",
            "économie = $saved/$baseline évaluations forward (round-par-round, phase greedy).")
    # Vérif faux-négatif : un site sélectionné a-t-il un Δ_j sous la médiane des 20 ?
    med = sort(collect(values(pr.deltas)))[cld(length(pr.deltas), 2)]
    for s in pr.selected
        if pr.deltas[s] < med
            global false_negative_found = true
            println("    ⚠ FAUX-NÉGATIF POSSIBLE : $s sélectionné avec Δ_j=$(round(pr.deltas[s],digits=3)) < médiane $(round(med,digits=3))")
        end
    end
end
println("\n  TOTAL (3 paires, phase greedy seule) : $total_saved / $total_baseline évaluations économisées ",
        "(", round(100*total_saved/total_baseline, digits=1), "%).")
println("  Faux-négatif détecté sur les données réelles : ", false_negative_found ? "OUI" : "NON")
println("═"^70)
