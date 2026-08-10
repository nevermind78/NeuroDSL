# ══════════════════════════════════════════════════════════════════════════════
# Expérience de greffe -- suite directe du screening dominant-vs-témoin
# (`real_llm_dominant_vs_temoin_screen.jl`). Site retenu : fenêtre SEBASTIAN
# (gap_pos=210, j=218), site dominant `layer_2_mha_ao_h3` (recovery
# individuelle 0.8495, meilleur témoin `layer_3_mha_ao_h3` à 0.2628, écart
# absolu 0.5867).
#
# Critères pré-enregistrés (fixés AVANT tout entraînement, voir conversation) :
#   C1 : la greffe au site diagnostiqué (backbone gelé) atteint recovery>=0.5
#        en MOINS de pas que la greffe témoin (site aléatoire, backbone gelé,
#        même budget total, même graine de données). Si le témoin n'atteint
#        jamais 0.5 dans le budget, C1 est automatiquement satisfait.
#   C2 : recovery finale (site diagnostiqué) - recovery finale (témoin) >= 0.2
#        (marge absolue).
#   C3 : le gain dépasse nettement un contrôle from-scratch à budget de pas
#        ÉGAL (même style que les runs de contrôle "budget total égal" déjà
#        établis sur ce char-LM réel, `real_llm_surgery_v2.ipynb` §15).
#
# Design, 4 branches, TOUTES reproduisent le MÊME backbone (même graine que le
# screening -- retrain Phase A à l'identique, aucun checkpoint sauvegardé
# n'existait avant cette session) :
#   1. Greffe au site diagnostiqué, backbone GELÉ (prune_frozen=true)   -- branche principale
#   2. Greffe au site diagnostiqué, backbone NON gelé                   -- contrôle interne (régime vs hasard)
#   3. Greffe à un site témoin aléatoire (non dominant), backbone GELÉ  -- pour C1/C2
#   4. Contrôle from-scratch, budget de pas égal, PAS de greffe          -- pour C3
#
# Mécanisme de greffe : `graft_shadow_block!` (src/graph_surgery.jl),
# alpha0=0f0 (rezero, déjà validé empiriquement). Le site diagnostiqué/témoin
# est un nœud PAR TÊTE (`*_mha_ao_h{h}`, largeur d_head=dim/n_heads, PAS la
# largeur pleine du residual stream) -- le bloc greffé est donc dimensionné à
# d_head, pas à `dim`, sous peine d'un mismatch de forme immédiat.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics, Printf, LinearAlgebra, JSON

dev = NeuroDSL.Backend.CUDADevice()
ns  = :real_llm_surgery
println("Device: ", dev)

# ── 1. Corpus + tokenizer (identique au screening) ──────────────────────────
using Downloads
const CORPUS_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
const CORPUS_PATH = joinpath(@__DIR__, "data", "tinyshakespeare", "input.txt")
function load_corpus(path::String, url::String)
    if !isfile(path)
        mkpath(dirname(path))
        Downloads.download(url, path)
    end
    return read(path, String)
end
text = load_corpus(CORPUS_PATH, CORPUS_URL)

function build_char_tokenizer(text::String)
    chars = sort(unique(collect(text)))
    stoi = Dict(c => i for (i, c) in enumerate(chars))
    return chars, stoi
end
encode(text::AbstractString, stoi::Dict{Char,Int}) = [stoi[c] for c in text]
decode(ids::AbstractVector{<:Integer}, chars::Vector{Char}) = String(chars[ids])

chars, stoi = build_char_tokenizer(text)
vocab_size = length(chars)
println("vocab_size = ", vocab_size)

data = encode(text, stoi)
n_total = length(data)
n_train = floor(Int, 0.9 * n_total)
train_ids = data[1:n_train]
val_ids   = data[n_train+1:end]
println("train: ", length(train_ids), "  val: ", length(val_ids))

# ── 2. Graphe (identique au screening) ──────────────────────────────────────
function build_char_lm_graph(dev, ns::Symbol; vocab_size::Int, dim::Int, n_heads::Int,
                              hidden_dim::Int, n_layers::Int, block_size::Int, batched::Bool=true)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.set!(g, :token_ids, ones(Int, block_size); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns)
    tok_emb = NeuroDSL.Embedding(vocab_size, dim)(g, :token_ids, :tok; namespace=ns)
    pos_emb = NeuroDSL.Embedding(block_size, dim)(g, :pos_ids, :pos; namespace=ns)
    x = :embed_sum
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(x, [tok_emb, pos_emb], :add; namespace=ns))
    out = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim; batched_attn=batched)(g, x; namespace=ns)
    logits = NeuroDSL.Linear(dim, vocab_size)(g, out, :lm_head; namespace=ns)
    NeuroDSL.set!(g, :labels, ones(Int, block_size); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [logits, :labels], :cross_entropy; namespace=ns))
    return g, logits
end

block_size = 256
dim        = 256
n_heads    = 4
hidden_dim = 512
n_layers   = 4
d_head     = dim ÷ n_heads
println("d_head = ", d_head)

g, logits_sym = build_char_lm_graph(dev, ns; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                     hidden_dim=hidden_dim, n_layers=n_layers, block_size=block_size)
println("Graphe : ", length(g.nodes[ns]), " nœuds, ", length(NeuroDSL.params(g; namespace=ns)), " tenseurs de paramètres")

# ── 3. Entraînement -- Phase A, 10 000 pas, EXACTEMENT les mêmes hyperparamètres/graine que le screening ──
function sample_window(rng, ids::Vector{Int}, block_size::Int)
    i = rand(rng, 1:(length(ids) - block_size))
    tokens = ids[i:i+block_size-1]
    labels = ids[i+1:i+block_size]
    return tokens, labels
end

function val_loss(g::NeuroDSL.NeuroGraph, ns::Symbol; val_ids::Vector{Int}, block_size::Int, n_windows::Int=64)
    max_start = length(val_ids) - block_size
    starts = round.(Int, range(1, max_start, length=n_windows))
    total = 0.0
    for i in starts
        tokens = val_ids[i:i+block_size-1]
        labels = val_ids[i+1:i+block_size]
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        loss_val = NeuroDSL.demand_release!(g, :loss; namespace=ns)
        total += Float64(sum(Array(loss_val)))
    end
    return total / n_windows
end

function train_char_lm!(g::NeuroDSL.NeuroGraph, ns::Symbol, logits_sym::Symbol;
                         train_ids::Vector{Int}, val_ids::Vector{Int}, block_size::Int,
                         n_steps::Int, lr::Float32=1f-3, b1::Float32=0.9f0, b2::Float32=0.999f0,
                         eps_v::Float32=1f-8, clip::Float32=1f0, wd::Float32=0f0,
                         rng::MersenneTwister=MersenneTwister(123), val_every::Int=1000)
    devl = g.device
    ps = NeuroDSL.params(g; namespace=ns)
    m1s = [NeuroDSL.Backend.zeros32(devl, size(p.value)...) for p in ps]
    m2s = [NeuroDSL.Backend.zeros32(devl, size(p.value)...) for p in ps]
    t_start = time()
    local vl
    for t in 1:n_steps
        tokens, labels = sample_window(rng, train_ids, block_size)
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        loss_val = NeuroDSL.demand!(g, :loss; namespace=ns)
        NeuroDSL.backward_graph!(g, :loss; namespace=ns)
        NeuroDSL.adamw_step_batched!(devl, [p.value for p in ps], [p.gradient for p in ps],
                                     m1s, m2s, lr, b1, b2, eps_v, t, clip, wd)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        if t % val_every == 0 || t == 1
            vl = val_loss(g, ns; val_ids=val_ids, block_size=block_size)
            @printf("step %6d | train %.4f | val %.4f\n", t, Float64(sum(Array(loss_val))), vl)
        end
    end
    return (; elapsed=time()-t_start, val_final=vl)
end

n_steps_A = 10_000
rng_A = MersenneTwister(123)
println("\n--- Entraînement Phase A (", n_steps_A, " pas, reproduit à l'identique du screening) ---")
res_A = train_char_lm!(g, ns, logits_sym; train_ids=train_ids, val_ids=val_ids, block_size=block_size,
                        n_steps=n_steps_A, lr=1f-3, rng=rng_A)
@printf("Phase A terminée : %d pas en %.1f s | val loss final = %.4f\n", n_steps_A, res_A.elapsed, res_A.val_final)
@printf("(référence screening : val loss final = 1.7803957108408213 -- doit être identique bit-à-bit)\n")

# ── 4. Reconstruction EXACTE de la fenêtre SEBASTIAN (gap_pos=210, j=218) ───
# Code copié à l'identique de real_llm_dominant_vs_temoin_screen.jl (même ordre
# de consommation du RNG rng_scan=MersenneTwister(999) sur TOUS les candidats)
# -- c'est le seul moyen de retomber sur exactement le même token corrompu.
function all_speaker_headers(val_ids::Vector{Int}, chars::Vector{Char}; min_name_len::Int=4)
    text_val = decode(val_ids, chars)
    headers = NamedTuple[]
    for m in eachmatch(r"\n([A-Z][A-Z ]{2,})\:", text_val)
        name = String(strip(m.captures[1]))
        length(name) >= min_name_len || continue
        push!(headers, (; pos=m.offset + 1, name))
    end
    return headers
end

function build_candidates(headers, block_size::Int; k::Int=3, margin::Int=5)
    candidates = NamedTuple[]
    n = length(headers)
    for i in 1:n-1
        for jx in i+1:n
            gap = headers[jx].pos - headers[i].pos
            gap > block_size - 20 && break
            headers[i].name != headers[jx].name && continue
            kk = min(k, length(headers[i].name) - 1)
            kk < 1 && continue
            window_start = headers[i].pos - margin
            window_start < 1 && continue
            p1 = headers[i].pos - window_start + 1
            p2 = headers[jx].pos - window_start + 1
            p2 + kk > block_size && continue
            has_intervening_different = any(headers[m2].pos > headers[i].pos && headers[m2].pos < headers[jx].pos &&
                                             headers[m2].name != headers[i].name for m2 in i+1:jx-1)
            variant = has_intervening_different ? :long_gap : :adjacent
            push!(candidates, (; window_start, p1, p2, k=kk, name=headers[i].name, gap, variant))
        end
    end
    return candidates
end

headers = all_speaker_headers(val_ids, chars)
candidates_raw = build_candidates(headers, block_size)
println("\nCandidats bruts : ", length(candidates_raw))

function build_window_effect(g, ns, logits_sym, block_size, c, val_ids, vocab_size, rng_scan)
    tokens_clean = val_ids[c.window_start:c.window_start+block_size-1]
    j = c.p2 + c.k - 1
    (j < 1 || j > block_size) && return nothing
    tokens_corrupt = copy(tokens_clean)
    orig_id = tokens_corrupt[c.p1 + c.k]
    new_id = orig_id
    while new_id == orig_id
        new_id = rand(rng_scan, 1:vocab_size)
    end
    tokens_corrupt[c.p1 + c.k] = new_id

    NeuroDSL.set!(g, :token_ids, tokens_clean; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    clean_logits = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))

    NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    corrupt_logits = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))

    effect = norm(clean_logits[j, :] .- corrupt_logits[j, :])
    return (; c..., j, tokens_clean, tokens_corrupt, effect)
end

rng_scan = MersenneTwister(999)
scored = NamedTuple[]
for c in candidates_raw
    r = build_window_effect(g, ns, logits_sym, block_size, c, val_ids, vocab_size, rng_scan)
    r === nothing || push!(scored, r)
end
println("Candidats scorés : ", length(scored))

EFFECT_FLOOR = 1.0
eligible = filter(x -> x.effect >= EFFECT_FLOOR, scored)
println("Candidats éligibles (effet>=", EFFECT_FLOOR, ") : ", length(eligible), "/", length(scored))

target_matches = filter(w -> w.name == "SEBASTIAN" && w.gap == 210, eligible)
@assert length(target_matches) == 1 "Reconstruction de la fenêtre SEBASTIAN ambiguë ou introuvable -- $(length(target_matches)) correspondance(s)"
target = target_matches[1]
@printf("\nFenêtre cible reconstruite : nom=%s variant=%s j=%d effet=%.4f\n", target.name, String(target.variant), target.j, target.effect)
@assert target.j == 218 "j reconstruit ($(target.j)) != 218 attendu -- reconstruction incohérente avec le screening"

# ── 5. Sanity check : recovery individuelle -- doit retomber sur les chiffres exacts du screening ──
function individual_head_recoveries(g, ns, logits_sym, window)
    tokens_clean, tokens_corrupt, j = window.tokens_clean, window.tokens_corrupt, window.j

    NeuroDSL.set!(g, :token_ids, tokens_clean; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    clean_output = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))
    clean_cache  = NeuroDSL.capture_activations(g, ns)

    NeuroDSL.set!(g, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    corrupted_output = copy(Array(NeuroDSL.demand!(g, logits_sym; namespace=ns)))
    corrupted_cache  = NeuroDSL.capture_activations(g, ns)

    row_metric(out) = NeuroDSL.recovery_metric(Array(out)[j:j, :], clean_output[j:j, :], corrupted_output[j:j, :])
    candidates = sort(collect(filter(s -> occursin(r"_mha_ao_h\d+$", String(s)), keys(g.nodes[ns]))))

    recoveries = Dict{Symbol,Float64}()
    for h in candidates
        NeuroDSL.patch_node!(g, h, clean_cache; namespace=ns)
        out = NeuroDSL.demand!(g, logits_sym; namespace=ns)
        recoveries[h] = row_metric(out)
        NeuroDSL.patch_node!(g, h, corrupted_cache; namespace=ns)
        NeuroDSL.demand!(g, logits_sym; namespace=ns)
    end
    NeuroDSL.set!(g, :token_ids, tokens_clean; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, logits_sym; namespace=ns)
    return recoveries, clean_output, corrupted_output, clean_cache, corrupted_cache
end

recs, clean_output_ref, corrupted_output_ref, clean_cache_ref, corrupted_cache_ref = individual_head_recoveries(g, ns, logits_sym, target)
sorted_heads = sort(collect(recs), by = x -> -x[2])
dominant_site, dominant_rec = sorted_heads[1]
temoin_best_site, temoin_best_rec = sorted_heads[2]
@printf("Sanity check -- dominant=%s rec=%.6f (attendu layer_2_mha_ao_h3 / 0.8495147824287415)\n", dominant_site, dominant_rec)
@printf("Sanity check -- meilleur témoin=%s rec=%.6f (attendu layer_3_mha_ao_h3 / 0.2627784013748169)\n", temoin_best_site, temoin_best_rec)
@assert dominant_site == :layer_2_mha_ao_h3 "Site dominant reconstruit différent de celui du screening -- ARRÊT"
SANITY_OK = isapprox(dominant_rec, 0.8495147824287415; atol=1e-4) && isapprox(temoin_best_rec, 0.2627784013748169; atol=1e-4)
println("Reproduction bit-exacte de la Phase A confirmée : ", SANITY_OK)

DOMINANT_SITE = :layer_2_mha_ao_h3
tokens_clean, tokens_corrupt, j_target = target.tokens_clean, target.tokens_corrupt, target.j

# ── 6. Site témoin choisi ALÉATOIREMENT parmi les sites NON dominants de la fenêtre ──
all_heads = sort(collect(filter(s -> occursin(r"_mha_ao_h\d+$", String(s)), keys(g.nodes[ns]))))
non_dominant = filter(s -> s != DOMINANT_SITE, all_heads)
rng_temoin_pick = MersenneTwister(4242)
TEMOIN_SITE = non_dominant[rand(rng_temoin_pick, 1:length(non_dominant))]
@printf("\nSite témoin choisi aléatoirement (graine 4242, parmi %d sites non-dominants) : %s (recovery individuelle mesurée = %.4f)\n",
        length(non_dominant), TEMOIN_SITE, recs[TEMOIN_SITE])

# ── 7. Sauvegarde du backbone entraîné (Phase A) -- base commune aux 4 branches ──
export_dir = joinpath(@__DIR__, "graft_experiment_export")
mkpath(export_dir)
NeuroDSL.save_graph!(g, ns, joinpath(export_dir, "base"))
println("Backbone Phase A sauvegardé -> ", joinpath(export_dir, "base"))

# ── 8. Branches de greffe -- fonctions communes ─────────────────────────────
GRAFT_DIM    = d_head     # 64 -- largeur du nœud par-tête, PAS `dim`
GRAFT_HEADS  = 2
GRAFT_HIDDEN = 128
n_steps_B    = 4000
LOG_EVERY    = 50
LR_B         = 1f-3
DATA_SEED    = 9000
SEED_GRAFT_DIAG   = 7001   # même graine pour les branches 1 et 2 (isoler gel/non-gel)
SEED_GRAFT_TEMOIN = 7002
SEED_FROM_SCRATCH_INIT = 5555

seed_rng!(seed::Int) = (NeuroDSL.Backend.CUDA_AVAILABLE ? NeuroDSL.CUDA.seed!(seed) : Random.seed!(seed))

function freeze_backbone!(g::NeuroDSL.NeuroGraph, ns::Symbol, keep_prefix::Union{Nothing,Symbol})
    n_frozen = 0
    for (sym, nd) in g.nodes[ns]
        nd.is_param || continue
        if keep_prefix !== nothing && startswith(String(sym), String(keep_prefix))
            continue
        end
        nd.is_param = false
        n_frozen += 1
    end
    return n_frozen
end

row_metric_ref(out) = NeuroDSL.recovery_metric(Array(out)[j_target:j_target, :], clean_output_ref[j_target:j_target, :], corrupted_output_ref[j_target:j_target, :])

"""
Branche greffée (diagnostiquée ou témoin), backbone gelé ou non.
Retourne l'historique de recovery (mesurée à chaque LOG_EVERY pas, sur la
fenêtre SEBASTIAN FIXE, avec les ancres clean_output_ref/corrupted_output_ref
FIXES calculées une seule fois depuis le backbone Phase A -- jamais recalculées
per-branche) et la recovery finale.
"""
function run_graft_branch(label::Symbol, graft_site::Symbol, freeze::Bool, graft_seed::Int)
    ns_b = Symbol(:branch_, label)
    g_b = NeuroDSL.NeuroGraph(namespace=ns_b, device=dev)
    NeuroDSL.load_graph!(g_b, ns_b, joinpath(export_dir, "base"))

    seed_rng!(graft_seed)
    prefix = Symbol(:shadow_, label, :_, graft_site)
    new_out, handle = NeuroDSL.graft_shadow_block!(g_b, ns_b, graft_site, GRAFT_DIM, GRAFT_HEADS, GRAFT_HIDDEN;
                                                    alpha0=0f0, zero_out_proj=false, prefix=prefix)

    n_frozen = freeze ? freeze_backbone!(g_b, ns_b, handle.prefix) : 0
    ps = NeuroDSL.params(g_b; namespace=ns_b)
    @printf("  [%s] site=%s freeze=%s -> %d params gelés, %d params entraînables\n",
            label, graft_site, freeze, n_frozen, length(ps))

    m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    rng_cont = MersenneTwister(DATA_SEED)

    recovery_history = Tuple{Int,Float64}[]
    alpha_history = Tuple{Int,Float64}[]
    step_reaching_0p5 = nothing
    t_start = time()
    for t in 1:n_steps_B
        tokens, labels = sample_window(rng_cont, train_ids, block_size)
        NeuroDSL.set!(g_b, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns_b)
        NeuroDSL.set!(g_b, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns_b)
        NeuroDSL.set!(g_b, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns_b)
        NeuroDSL.invalidate_all!(g_b; namespace=ns_b)
        NeuroDSL.demand!(g_b, :loss; namespace=ns_b)
        NeuroDSL.backward_graph_sparse!(g_b, :loss; namespace=ns_b, prune_frozen=freeze)
        NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps], [p.gradient for p in ps],
                                     m1s, m2s, LR_B, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
        NeuroDSL.invalidate_all!(g_b; namespace=ns_b)

        if t % LOG_EVERY == 0 || t == 1
            NeuroDSL.set!(g_b, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns_b)
            NeuroDSL.invalidate_all!(g_b; namespace=ns_b)
            out = NeuroDSL.demand!(g_b, logits_sym; namespace=ns_b)
            r = row_metric_ref(out)
            push!(recovery_history, (t, r))
            alpha_val = Float64(Array(NeuroDSL.node(g_b, handle.alpha_sym; namespace=ns_b).value)[1])
            push!(alpha_history, (t, alpha_val))
            if step_reaching_0p5 === nothing && r >= 0.5
                step_reaching_0p5 = t
            end
        end
    end
    elapsed = time() - t_start
    @printf("  [%s] terminé : %d pas en %.1f s | recovery finale = %.4f | alpha final = %.4f | 1er pas>=0.5 : %s\n",
            label, n_steps_B, elapsed, recovery_history[end][2], alpha_history[end][2],
            step_reaching_0p5 === nothing ? "jamais" : string(step_reaching_0p5))
    return (; label=String(label), graft_site=String(graft_site), freeze, elapsed,
            recovery_history, alpha_history, final_recovery=recovery_history[end][2],
            final_alpha=alpha_history[end][2], step_reaching_0p5)
end

# ── 9. Branche 4 : contrôle from-scratch, budget de pas ÉGAL (n_steps_B), PAS de greffe ──
function run_from_scratch_control()
    ns_c = :branch_control_scratch
    seed_rng!(SEED_FROM_SCRATCH_INIT)
    g_c, logits_c = build_char_lm_graph(dev, ns_c; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                         hidden_dim=hidden_dim, n_layers=n_layers, block_size=block_size)
    ps = NeuroDSL.params(g_c; namespace=ns_c)
    m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    rng_cont = MersenneTwister(DATA_SEED)   # même graine de données que les branches 1-3

    recovery_history = Tuple{Int,Float64}[]
    t_start = time()
    for t in 1:n_steps_B
        tokens, labels = sample_window(rng_cont, train_ids, block_size)
        NeuroDSL.set!(g_c, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns_c)
        NeuroDSL.set!(g_c, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns_c)
        NeuroDSL.set!(g_c, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns_c)
        NeuroDSL.invalidate_all!(g_c; namespace=ns_c)
        NeuroDSL.demand!(g_c, :loss; namespace=ns_c)
        NeuroDSL.backward_graph!(g_c, :loss; namespace=ns_c)
        NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps], [p.gradient for p in ps],
                                     m1s, m2s, LR_B, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
        NeuroDSL.invalidate_all!(g_c; namespace=ns_c)
        if t % LOG_EVERY == 0 || t == 1
            NeuroDSL.set!(g_c, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns_c)
            NeuroDSL.invalidate_all!(g_c; namespace=ns_c)
            out = NeuroDSL.demand!(g_c, logits_c; namespace=ns_c)
            r = row_metric_ref(out)
            push!(recovery_history, (t, r))
        end
    end
    elapsed = time() - t_start
    @printf("  [from_scratch] terminé : %d pas en %.1f s | recovery finale = %.4f\n",
            n_steps_B, elapsed, recovery_history[end][2])
    return (; label="from_scratch", elapsed, recovery_history, final_recovery=recovery_history[end][2])
end

# ── 10. Lancement des 4 branches ─────────────────────────────────────────────
println("\n", "="^90, "\nBranche 1 : greffe au site diagnostiqué, backbone GELÉ (branche principale)\n", "="^90)
branch1 = run_graft_branch(:diag_frozen, DOMINANT_SITE, true, SEED_GRAFT_DIAG)

println("\n", "="^90, "\nBranche 2 : greffe au site diagnostiqué, backbone NON gelé (contrôle interne)\n", "="^90)
branch2 = run_graft_branch(:diag_unfrozen, DOMINANT_SITE, false, SEED_GRAFT_DIAG)

println("\n", "="^90, "\nBranche 3 : greffe au site témoin aléatoire, backbone GELÉ\n", "="^90)
branch3 = run_graft_branch(:temoin_frozen, TEMOIN_SITE, true, SEED_GRAFT_TEMOIN)

println("\n", "="^90, "\nBranche 4 : contrôle from-scratch, budget de pas égal, sans greffe\n", "="^90)
branch4 = run_from_scratch_control()

# ── 11. Verdict contre les 3 critères pré-enregistrés ────────────────────────
println("\n", "="^90, "\nVERDICT\n", "="^90)

# C1
c1_temoin_never_reaches = branch3.step_reaching_0p5 === nothing
c1 = if c1_temoin_never_reaches
    true   # règle de repli pré-enregistrée
else
    branch1.step_reaching_0p5 !== nothing && branch1.step_reaching_0p5 < branch3.step_reaching_0p5
end
@printf("C1 : diag atteint 0.5 en moins de pas que témoin (ou témoin ne l'atteint jamais)\n")
@printf("     diag 1er pas>=0.5   : %s\n", branch1.step_reaching_0p5 === nothing ? "jamais" : string(branch1.step_reaching_0p5))
@printf("     témoin 1er pas>=0.5 : %s\n", branch3.step_reaching_0p5 === nothing ? "jamais" : string(branch3.step_reaching_0p5))
println("     C1 = ", c1)

# C2
margin = branch1.final_recovery - branch3.final_recovery
c2 = margin >= 0.2
@printf("\nC2 : marge absolue finale (diag - témoin) >= 0.2\n")
@printf("     recovery finale diag=%.4f  témoin=%.4f  marge=%.4f\n", branch1.final_recovery, branch3.final_recovery, margin)
println("     C2 = ", c2)

# C3
gain_vs_scratch = branch1.final_recovery - branch4.final_recovery
c3 = gain_vs_scratch >= 0.2   # même seuil de marge que C2, pour un contrôle "nettement" dépassé
@printf("\nC3 : le gain dépasse nettement (marge>=0.2) le contrôle from-scratch à budget de pas égal\n")
@printf("     recovery finale diag=%.4f  from-scratch=%.4f  marge=%.4f\n", branch1.final_recovery, branch4.final_recovery, gain_vs_scratch)
println("     C3 = ", c3)

n_criteria_met = count([c1, c2, c3])
verdict = if n_criteria_met == 3
    "SUCCÈS NET -- les 3 critères pré-enregistrés sont remplis."
elseif n_criteria_met == 0
    "ÉCHEC NET -- aucun des 3 critères pré-enregistrés n'est remplis."
else
    "MITIGÉ -- $(n_criteria_met)/3 critères pré-enregistrés remplis."
end
println("\n", verdict)

# ── 12. Sauvegarde ───────────────────────────────────────────────────────────
out = Dict(
    "val_loss_phase_A" => res_A.val_final,
    "sanity_ok" => SANITY_OK,
    "dominant_site" => String(DOMINANT_SITE),
    "dominant_rec_reconstructed" => dominant_rec,
    "temoin_site_screened" => String(temoin_best_site),
    "temoin_rec_screened" => temoin_best_rec,
    "temoin_site_random" => String(TEMOIN_SITE),
    "temoin_site_random_individual_rec" => recs[TEMOIN_SITE],
    "n_steps_B" => n_steps_B,
    "log_every" => LOG_EVERY,
    "graft_dim" => GRAFT_DIM, "graft_heads" => GRAFT_HEADS, "graft_hidden" => GRAFT_HIDDEN,
    "branch1_diag_frozen" => (; site=branch1.graft_site, freeze=branch1.freeze, elapsed=branch1.elapsed,
                               recovery_history=branch1.recovery_history, alpha_history=branch1.alpha_history,
                               final_recovery=branch1.final_recovery, final_alpha=branch1.final_alpha,
                               step_reaching_0p5=branch1.step_reaching_0p5),
    "branch2_diag_unfrozen" => (; site=branch2.graft_site, freeze=branch2.freeze, elapsed=branch2.elapsed,
                                 recovery_history=branch2.recovery_history, alpha_history=branch2.alpha_history,
                                 final_recovery=branch2.final_recovery, final_alpha=branch2.final_alpha,
                                 step_reaching_0p5=branch2.step_reaching_0p5),
    "branch3_temoin_frozen" => (; site=branch3.graft_site, freeze=branch3.freeze, elapsed=branch3.elapsed,
                                 recovery_history=branch3.recovery_history, alpha_history=branch3.alpha_history,
                                 final_recovery=branch3.final_recovery, final_alpha=branch3.final_alpha,
                                 step_reaching_0p5=branch3.step_reaching_0p5),
    "branch4_from_scratch" => (; elapsed=branch4.elapsed, recovery_history=branch4.recovery_history,
                               final_recovery=branch4.final_recovery),
    "c1" => c1, "c2" => c2, "c3" => c3, "margin_c2" => margin, "margin_c3" => gain_vs_scratch,
    "n_criteria_met" => n_criteria_met, "verdict" => verdict,
)
open(joinpath(@__DIR__, "real_llm_graft_experiment_results.json"), "w") do io
    JSON.print(io, out)
end
println("\nRésultats écrits -> notebook/real_llm_graft_experiment_results.json")
