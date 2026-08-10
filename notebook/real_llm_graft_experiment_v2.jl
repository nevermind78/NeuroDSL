# ══════════════════════════════════════════════════════════════════════════════
# Suite directe -- run UNIQUE, process UNIQUE (pas de reproduction bit-exacte
# entre process séparés : le non-déterminisme GPU rend ça non fiable, cf.
# diagnostic accepté). Tout se passe ICI :
#   1. Entraînement Phase A (10 000 pas).
#   2. Screening dominant-vs-témoin sur les candidats éligibles de CE run
#      (mêmes deux critères pré-enregistrés que le screening original :
#      (a) écart absolu >= 0.3, (b) recovery dominant < 0.85).
#   3. Sélection du site cible : parmi les candidats qui cumulent (a) ET (b),
#      celui avec le plus grand écart absolu. Si aucun ne cumule les deux,
#      ARRÊT (pas de site forcé) -- il faudra une nouvelle graine.
#   4. Sauvegarde du backbone (save_graph!) par sécurité.
#   5. Les 4 branches de greffe (diag gelé / diag non-gelé / témoin aléatoire
#      gelé / from-scratch budget égal), critères C1/C2/C3 déjà fixés.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics, Printf, LinearAlgebra, JSON

dev = NeuroDSL.Backend.CUDADevice()
ns  = :real_llm_surgery
println("Device: ", dev)

# ── 1. Corpus + tokenizer ────────────────────────────────────────────────────
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

# ── 2. Graphe ────────────────────────────────────────────────────────────────
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

# ── 3. Entraînement -- Phase A, 10 000 pas (ce run, cette graine, pas de contrainte de reproduction) ──
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
println("\n--- Entraînement Phase A (", n_steps_A, " pas, CE run/process uniquement) ---")
res_A = train_char_lm!(g, ns, logits_sym; train_ids=train_ids, val_ids=val_ids, block_size=block_size,
                        n_steps=n_steps_A, lr=1f-3, rng=rng_A)
@printf("Phase A terminée : %d pas en %.1f s | val loss final = %.4f\n", n_steps_A, res_A.elapsed, res_A.val_final)

# ── 4. Construction de TOUS les candidats + screening dominant-vs-témoin, DANS ce run ──
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

GAP_THRESHOLD     = 0.3
CEILING_THRESHOLD = 0.85

println("\n--- Screening dominant-vs-témoin sur les ", length(eligible), " candidats éligibles de CE run ---")
screen_results = NamedTuple[]
for w in eligible
    recs, clean_out, corrupt_out, clean_cache, corrupt_cache = individual_head_recoveries(g, ns, logits_sym, w)
    sorted_heads = sort(collect(recs), by = x -> -x[2])
    dominant_site, dominant_rec = sorted_heads[1]
    temoin_site, temoin_rec     = sorted_heads[2]
    gap = dominant_rec - temoin_rec
    crit_a = gap >= GAP_THRESHOLD
    crit_b = dominant_rec < CEILING_THRESHOLD
    push!(screen_results, (; name=w.name, variant=String(w.variant), gap_pos=w.gap, j=w.j, effect=w.effect,
                            dominant_site, dominant_rec, temoin_site, temoin_rec, abs_gap=gap,
                            crit_a, crit_b, both=(crit_a && crit_b),
                            window=w, recs, clean_out, corrupt_out, clean_cache, corrupt_cache))
    @printf("nom=%-10s var=%-9s j=%-4d effet=%5.2f | dominant=%-24s rec=%.4f | témoin=%-24s rec=%.4f | écart_abs=%.4f%s\n",
            w.name, String(w.variant), w.j, w.effect, dominant_site, dominant_rec, temoin_site, temoin_rec, gap,
            (crit_a && crit_b) ? "  <=== (a) ET (b)" : (crit_a ? "  (a seul)" : (crit_b ? "  (b seul)" : "")))
end

qualifying = filter(r -> r.both, screen_results)
println("\nCandidats cumulant (a) écart_abs>=", GAP_THRESHOLD, " ET (b) dominant<", CEILING_THRESHOLD, " : ",
        length(qualifying), "/", length(screen_results))

if isempty(qualifying)
    println("\n❌ AUCUN des ", length(screen_results), " candidats éligibles de ce run ne cumule (a) ET (b).")
    println("Conformément au protocole : ARRÊT, pas de site forcé. Il faut relancer Phase A avec une nouvelle graine.")
    out_fail = Dict(
        "val_loss_phase_A" => res_A.val_final,
        "n_eligible" => length(eligible),
        "n_qualifying" => 0,
        "screen_results" => [(; name=r.name, variant=r.variant, gap_pos=r.gap_pos, j=r.j, effect=r.effect,
                               dominant_site=String(r.dominant_site), dominant_rec=r.dominant_rec,
                               temoin_site=String(r.temoin_site), temoin_rec=r.temoin_rec,
                               abs_gap=r.abs_gap, crit_a=r.crit_a, crit_b=r.crit_b) for r in screen_results],
    )
    open(joinpath(@__DIR__, "real_llm_graft_experiment_v2_NOSITE_results.json"), "w") do io
        JSON.print(io, out_fail)
    end
    error("Aucun site qualifiant dans ce run -- voir real_llm_graft_experiment_v2_NOSITE_results.json")
end

best = sort(qualifying, by = x -> -x.abs_gap)[1]
@printf("\n✅ Site cible retenu (plus grand écart absolu parmi les qualifiants de CE run) :\n")
@printf("   fenêtre = %s (variant=%s, gap_pos=%d, j=%d, effet=%.4f)\n", best.name, best.variant, best.gap_pos, best.j, best.effect)
@printf("   site dominant = %s  recovery = %.4f\n", best.dominant_site, best.dominant_rec)
@printf("   meilleur témoin = %s  recovery = %.4f\n", best.temoin_site, best.temoin_rec)
@printf("   écart absolu = %.4f\n", best.abs_gap)

DOMINANT_SITE = best.dominant_site
target        = best.window
recs          = best.recs
clean_output_ref    = best.clean_out
corrupted_output_ref = best.corrupt_out
tokens_clean, tokens_corrupt, j_target = target.tokens_clean, target.tokens_corrupt, target.j

# ── 5. Site témoin choisi ALÉATOIREMENT parmi les sites NON dominants de LA MÊME FENÊTRE ──
all_heads = sort(collect(filter(s -> occursin(r"_mha_ao_h\d+$", String(s)), keys(g.nodes[ns]))))
non_dominant = filter(s -> s != DOMINANT_SITE, all_heads)
rng_temoin_pick = MersenneTwister(4242)
TEMOIN_SITE = non_dominant[rand(rng_temoin_pick, 1:length(non_dominant))]
@printf("\nSite témoin choisi aléatoirement (graine 4242, parmi %d sites non-dominants de la fenêtre %s) : %s (recovery individuelle mesurée = %.4f)\n",
        length(non_dominant), best.name, TEMOIN_SITE, recs[TEMOIN_SITE])

# ── 6. Sauvegarde du backbone entraîné (Phase A), par sécurité ──────────────
export_dir = joinpath(@__DIR__, "graft_experiment_v2_export")
mkpath(export_dir)
NeuroDSL.save_graph!(g, ns, joinpath(export_dir, "base"))
println("Backbone Phase A sauvegardé -> ", joinpath(export_dir, "base"))

# ── 7. Branches de greffe -- fonctions communes ─────────────────────────────
GRAFT_DIM    = d_head     # largeur du nœud par-tête, PAS `dim`
GRAFT_HEADS  = 2
GRAFT_HIDDEN = 128
n_steps_B    = 4000
LOG_EVERY    = 50
LR_B         = 1f-3
# ── Correctif 1 (post-diagnostic) : theta du greffon (Q/K/V/W_O + MLP, tout
# sauf alpha) dérive sans frein sous AdamW malgré alpha petit -- confirmé par
# diag_graft_v2_quick.jl (||R(x;theta)|| ×26, recovery s'effondre en <150 pas).
# Corrigé en donnant à theta un LR nettement plus petit qu'alpha (÷500) ET une
# weight decay non nulle (0.1). alpha garde son LR/wd actuels (inchangé).
# Vérifié : avec ces réglages, ||R(x;theta)|| reste borné (0.3237→0.3261 sur
# 300 pas) et recovery ne s'effondre plus (reste positive dès le pas ~10,
# finit à 0.1983 au pas 300) -- cf. notebook/diag_graft_v2_quick_fixB.log.
LR_THETA_DIV = 500f0
WD_THETA     = 1f-1
LR_THETA     = LR_B / LR_THETA_DIV
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

    # Partition alpha / theta(greffon) / backbone(reste, non gelé) pour appliquer
    # des LR/wd différents (correctif 1).
    alpha_idx = findall(p -> p.name == handle.alpha_sym, ps)
    theta_idx = findall(p -> p.name != handle.alpha_sym && startswith(String(p.name), String(handle.prefix)), ps)
    backbone_idx = setdiff(1:length(ps), alpha_idx ∪ theta_idx)
    @printf("  [%s] partition optim : %d alpha (lr=%.1e,wd=0) | %d theta-greffon (lr=%.1e,wd=%.2f) | %d backbone (lr=%.1e,wd=0)\n",
            label, length(alpha_idx), LR_B, length(theta_idx), LR_THETA, WD_THETA, length(backbone_idx), LR_B)

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
        if !isempty(alpha_idx)
            NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps[alpha_idx]], [p.gradient for p in ps[alpha_idx]],
                                         m1s[alpha_idx], m2s[alpha_idx], LR_B, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
        end
        if !isempty(theta_idx)
            NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps[theta_idx]], [p.gradient for p in ps[theta_idx]],
                                         m1s[theta_idx], m2s[theta_idx], LR_THETA, 0.9f0, 0.999f0, 1f-8, t, 1f0, WD_THETA)
        end
        if !isempty(backbone_idx)
            NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps[backbone_idx]], [p.gradient for p in ps[backbone_idx]],
                                         m1s[backbone_idx], m2s[backbone_idx], LR_B, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
        end
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

function run_from_scratch_control()
    ns_c = :branch_control_scratch
    seed_rng!(SEED_FROM_SCRATCH_INIT)
    g_c, logits_c = build_char_lm_graph(dev, ns_c; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                         hidden_dim=hidden_dim, n_layers=n_layers, block_size=block_size)
    ps = NeuroDSL.params(g_c; namespace=ns_c)
    m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
    rng_cont = MersenneTwister(DATA_SEED)

    # ── Correctif 2 : clean_output/corrupted_output DOIVENT venir du forward de
    # g_c lui-même (from-scratch), pas être réutilisés de g/g_b (backbone Phase A)
    # -- sinon la métrique de recovery compare les logits d'un modèle non entraîné
    # à la référence d'un modèle entraîné 10000 pas : nombres non comparables,
    # C3 ne teste jamais rien d'utile. On recapture clean/corrupted_output_c à
    # CHAQUE point de log, sur CE modèle g_c, dans son état d'entraînement courant
    # (même convention que row_metric_ref pour les autres branches : référence
    # fixée une fois pour route de comparaison -- ici on la fixe à l'init de g_c,
    # avant tout pas de gradient, pour rester cohérent avec le protocole des
    # branches 1-3 où clean/corrupted_output_ref est capturé sur le backbone
    # AVANT la greffe/l'entraînement de la branche).
    NeuroDSL.set!(g_c, :token_ids, tokens_clean; atom_type=NeuroDSL.Datom, namespace=ns_c)
    NeuroDSL.set!(g_c, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns_c)
    NeuroDSL.invalidate_all!(g_c; namespace=ns_c)
    clean_output_c = copy(Array(NeuroDSL.demand!(g_c, logits_c; namespace=ns_c)))

    NeuroDSL.set!(g_c, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns_c)
    NeuroDSL.invalidate_all!(g_c; namespace=ns_c)
    corrupted_output_c = copy(Array(NeuroDSL.demand!(g_c, logits_c; namespace=ns_c)))

    row_metric_c(out) = NeuroDSL.recovery_metric(Array(out)[j_target:j_target, :],
                                                  clean_output_c[j_target:j_target, :],
                                                  corrupted_output_c[j_target:j_target, :])

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
            r = row_metric_c(out)
            push!(recovery_history, (t, r))
        end
    end
    elapsed = time() - t_start
    @printf("  [from_scratch] terminé : %d pas en %.1f s | recovery finale = %.4f\n",
            n_steps_B, elapsed, recovery_history[end][2])
    return (; label="from_scratch", elapsed, recovery_history, final_recovery=recovery_history[end][2])
end

# ── 8. Lancement des 4 branches ──────────────────────────────────────────────
println("\n", "="^90, "\nBranche 1 : greffe au site diagnostiqué, backbone GELÉ (branche principale)\n", "="^90)
branch1 = run_graft_branch(:diag_frozen, DOMINANT_SITE, true, SEED_GRAFT_DIAG)

println("\n", "="^90, "\nBranche 2 : greffe au site diagnostiqué, backbone NON gelé (contrôle interne)\n", "="^90)
branch2 = run_graft_branch(:diag_unfrozen, DOMINANT_SITE, false, SEED_GRAFT_DIAG)

println("\n", "="^90, "\nBranche 3 : greffe au site témoin aléatoire, backbone GELÉ\n", "="^90)
branch3 = run_graft_branch(:temoin_frozen, TEMOIN_SITE, true, SEED_GRAFT_TEMOIN)

println("\n", "="^90, "\nBranche 4 : contrôle from-scratch, budget de pas égal, sans greffe\n", "="^90)
branch4 = run_from_scratch_control()

# ── 9. Verdict contre les 3 critères pré-enregistrés ─────────────────────────
println("\n", "="^90, "\nVERDICT\n", "="^90)

c1_temoin_never_reaches = branch3.step_reaching_0p5 === nothing
c1 = if c1_temoin_never_reaches
    true
else
    branch1.step_reaching_0p5 !== nothing && branch1.step_reaching_0p5 < branch3.step_reaching_0p5
end
@printf("C1 : diag atteint 0.5 en moins de pas que témoin (ou témoin ne l'atteint jamais)\n")
@printf("     diag 1er pas>=0.5   : %s\n", branch1.step_reaching_0p5 === nothing ? "jamais" : string(branch1.step_reaching_0p5))
@printf("     témoin 1er pas>=0.5 : %s\n", branch3.step_reaching_0p5 === nothing ? "jamais" : string(branch3.step_reaching_0p5))
println("     C1 = ", c1)

margin = branch1.final_recovery - branch3.final_recovery
c2 = margin >= 0.2
@printf("\nC2 : marge absolue finale (diag - témoin) >= 0.2\n")
@printf("     recovery finale diag=%.4f  témoin=%.4f  marge=%.4f\n", branch1.final_recovery, branch3.final_recovery, margin)
println("     C2 = ", c2)

gain_vs_scratch = branch1.final_recovery - branch4.final_recovery
c3 = gain_vs_scratch >= 0.2
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

# ── 10. Sauvegarde ────────────────────────────────────────────────────────────
out = Dict(
    "val_loss_phase_A" => res_A.val_final,
    "n_eligible" => length(eligible),
    "n_qualifying" => length(qualifying),
    "target_name" => best.name, "target_variant" => best.variant, "target_gap_pos" => best.gap_pos,
    "target_j" => best.j, "target_effect" => best.effect,
    "dominant_site" => String(DOMINANT_SITE), "dominant_rec" => best.dominant_rec,
    "temoin_site_screened" => String(best.temoin_site), "temoin_rec_screened" => best.temoin_rec,
    "abs_gap" => best.abs_gap,
    "temoin_site_random" => String(TEMOIN_SITE), "temoin_site_random_individual_rec" => recs[TEMOIN_SITE],
    "n_steps_B" => n_steps_B, "log_every" => LOG_EVERY,
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
    "screen_results_summary" => [(; name=r.name, variant=r.variant, gap_pos=r.gap_pos, j=r.j, effect=r.effect,
                                   dominant_site=String(r.dominant_site), dominant_rec=r.dominant_rec,
                                   temoin_site=String(r.temoin_site), temoin_rec=r.temoin_rec,
                                   abs_gap=r.abs_gap, crit_a=r.crit_a, crit_b=r.crit_b) for r in screen_results],
)
open(joinpath(@__DIR__, "real_llm_graft_experiment_v2_results.json"), "w") do io
    JSON.print(io, out)
end
println("\nRésultats écrits -> notebook/real_llm_graft_experiment_v2_results.json")
