# ══════════════════════════════════════════════════════════════════════════════
# Screening dominant-vs-témoin (avant tout entraînement de greffon)
#
# Question posée : existe-t-il, sur le char-LM RÉEL déjà entraîné/validé
# (TinyShakespeare, dim=256/4 têtes/4 couches, patron exact de
# `real_llm_surgery_v2.ipynb` / `real_llm_wide_screen.ipynb`), au moins une
# fenêtre où le site diagnostiqué (dominant = tête individuelle à la plus
# haute recovery) cumule :
#   (a) écart ABSOLU >= 0.3 avec le meilleur témoin (= la 2e meilleure tête
#       individuelle, le témoin le plus fort possible -- test conservateur)
#   (b) recovery du site dominant < 0.85 (marge réelle, pas déjà saturé)
#
# Différence avec `real_llm_wide_screen.ipynb` (déjà exécuté, 8 candidats) :
# ce screening-ci mesure la recovery INDIVIDUELLE de chaque tête (un seul
# site patché depuis l'état corrompu, comme le fait déjà la cellule "Sweep
# individuel des têtes greffées" de `real_llm_surgery_v2.ipynb`), pas la
# recovery du CIRCUIT COMPLET après recherche gloutonne + élagage (jusqu'à 8
# sites). Le wide-screen avait déjà établi qu'aucun candidat éligible (8/8)
# n'a une recovery de circuit complet < 0.85 -- mais un circuit complet peut
# saturer à ~1.0 même si aucune tête individuelle n'en explique la majorité
# seule (répartition diffuse). Cette question -- distincte -- n'avait jamais
# été posée sur ce modèle : c'est elle qui décide si un site unique est un
# candidat de greffe crédible.
#
# Aucun nouveau mécanisme de patching : réutilise tel quel
# `capture_activations`, `patch_node!`, `recovery_metric` (src/patching.jl),
# et la fonction `find_circuit!`/le protocole de sélection de fenêtres de
# `real_llm_surgery_v2.ipynb` (repris ici sans modification de logique).
# Reproduit exactement la Phase A (10 000 pas, même graine/hyperparamètres)
# pour retomber sur le même état de modèle déjà utilisé par les deux
# notebooks existants -- aucun checkpoint sauvegardé n'existe à ce stade,
# donc le seul moyen d'obtenir cet état exact est de le ré-entraîner.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics, Printf, LinearAlgebra, JSON

dev = NeuroDSL.Backend.CUDADevice()
ns  = :real_llm_surgery
println("Device: ", dev)

# ── 1. Corpus + tokenizer (identique à real_llm_surgery_v2.ipynb) ──────────
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

# ── 2. Graphe (identique) ───────────────────────────────────────────────────
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

g, logits_sym = build_char_lm_graph(dev, ns; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                     hidden_dim=hidden_dim, n_layers=n_layers, block_size=block_size)
println("Graphe : ", length(g.nodes[ns]), " nœuds, ", length(NeuroDSL.params(g; namespace=ns)), " tenseurs de paramètres")

# ── 3. Entraînement -- Phase A, 10 000 pas, mêmes hyperparamètres/graine ────
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
println("\n--- Entraînement Phase A (", n_steps_A, " pas, reproduit à l'identique des notebooks existants) ---")
res_A = train_char_lm!(g, ns, logits_sym; train_ids=train_ids, val_ids=val_ids, block_size=block_size,
                        n_steps=n_steps_A, lr=1f-3, rng=rng_A)
@printf("Phase A terminée : %d pas en %.1f s | val loss final = %.4f\n", n_steps_A, res_A.elapsed, res_A.val_final)

# ── 4. Construction de TOUS les candidats (paires même-nom), pas seulement 6/8 ──
using LinearAlgebra
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
println("\nCandidats bruts (paires même-nom dans une fenêtre) : ", length(candidates_raw))

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
println("Candidats scorés (effet mesuré) : ", length(scored))

effect_vals = [x.effect for x in scored]
println("Distribution de l'effet : min=", round(minimum(effect_vals),digits=3),
        " médiane=", round(median(effect_vals),digits=3),
        " max=", round(maximum(effect_vals),digits=3))

# Seuil FIXE (pas relatif au max -- élargit délibérément le pool par rapport
# au wide-screen existant, qui utilisait max(1.0, 0.25*effect_max)=2.5 et ne
# gardait que 8 candidats). 1.0 exclut seulement les fenêtres où la
# corruption n'a quasi aucun effet mesurable (bruit, pas un vrai signal
# d'induction) -- toujours un filtre honnête, pas une sélection de résultat.
EFFECT_FLOOR = 1.0
eligible = filter(x -> x.effect >= EFFECT_FLOOR, scored)
println("Candidats au-dessus du seuil d'effet fixe (", EFFECT_FLOOR, ") : ", length(eligible), "/", length(scored))

# ── 5. Screening dominant-vs-témoin sur TOUS les candidats éligibles ────────
# Recovery INDIVIDUELLE de chaque tête (patch d'UN site depuis l'état
# corrompu, mesuré, puis restauration) -- réutilise patch_node!/recovery_metric
# tels quels (src/patching.jl), même patron que la cellule "sweep individuel
# des têtes greffées" de real_llm_surgery_v2.ipynb.
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

    # Remise en état propre.
    NeuroDSL.set!(g, :token_ids, tokens_clean; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, logits_sym; namespace=ns)

    return recoveries
end

GAP_THRESHOLD = 0.3
CEILING_THRESHOLD = 0.85

screen_results = NamedTuple[]
t_screen_start = time()
for (i, w) in enumerate(eligible)
    recs = individual_head_recoveries(g, ns, logits_sym, w)
    sorted_heads = sort(collect(recs), by = x -> -x[2])   # décroissant
    dominant_site, dominant_rec = sorted_heads[1]
    temoin_site, temoin_rec     = sorted_heads[2]
    gap = dominant_rec - temoin_rec
    crit_a = gap >= GAP_THRESHOLD
    crit_b = dominant_rec < CEILING_THRESHOLD
    push!(screen_results, (; name=w.name, variant=String(w.variant), gap_pos=w.gap, j=w.j, effect=w.effect,
                            dominant_site=String(dominant_site), dominant_rec,
                            temoin_site=String(temoin_site), temoin_rec,
                            abs_gap=gap, crit_a, crit_b, both=(crit_a && crit_b)))
    if i % 20 == 0
        @printf("  ... %d/%d fenêtres criblées (%.1f s écoulées)\n", i, length(eligible), time()-t_screen_start)
    end
end
t_screen_elapsed = time() - t_screen_start
@printf("\nCriblage terminé : %d fenêtres en %.1f s (%.3f s/fenêtre)\n",
        length(eligible), t_screen_elapsed, t_screen_elapsed/length(eligible))

# ── 6. Rapport complet, par fenêtre ─────────────────────────────────────────
println("\n", "="^100)
println("RAPPORT COMPLET -- ", length(screen_results), " fenêtres criblées")
println("="^100)
sorted_report = sort(screen_results, by = x -> -x.abs_gap)
for r in sorted_report
    marker = r.both ? "  <=== (a) ET (b) SATISFAITS" : (r.crit_a ? "  (a seul)" : (r.crit_b ? "  (b seul)" : ""))
    @printf("nom=%-10s var=%-9s j=%-4d effet=%5.2f | dominant=%-24s rec=%.4f | témoin=%-24s rec=%.4f | écart_abs=%.4f%s\n",
            r.name, r.variant, r.j, r.effect, r.dominant_site, r.dominant_rec, r.temoin_site, r.temoin_rec, r.abs_gap, marker)
end

n_a_only = count(r -> r.crit_a && !r.crit_b, screen_results)
n_b_only = count(r -> r.crit_b && !r.crit_a, screen_results)
n_both   = count(r -> r.both, screen_results)
n_neither = count(r -> !r.crit_a && !r.crit_b, screen_results)

println("\n", "="^100)
println("SYNTHÈSE")
println("="^100)
println("Fenêtres criblées                          : ", length(screen_results), " (sur ", length(candidates_raw), " candidats bruts, ", length(scored), " scorés, seuil d'effet=", EFFECT_FLOOR, ")")
println("(a) seul (écart>=0.3, dominant>=0.85)      : ", n_a_only)
println("(b) seul (dominant<0.85, écart<0.3)        : ", n_b_only)
println("(a) ET (b) simultanément                   : ", n_both)
println("Ni (a) ni (b)                               : ", n_neither)

if n_both > 0
    best = sort(filter(r -> r.both, screen_results), by = x -> -x.abs_gap)[1]
    println("\n✅ AU MOINS UN SITE SATISFAIT (a) ET (b) SIMULTANÉMENT.")
    println("Meilleur candidat : fenêtre '", best.name, "' (", best.variant, "), site dominant = ", best.dominant_site)
    @printf("  recovery dominant = %.4f  |  meilleur témoin = %s (recovery = %.4f)  |  écart absolu = %.4f\n",
            best.dominant_rec, best.temoin_site, best.temoin_rec, best.abs_gap)
else
    println("\n❌ AUCUN SITE NE SATISFAIT (a) ET (b) SIMULTANÉMENT, sur ", length(screen_results), " fenêtres criblées.")
end

# ── 7. Sauvegarde ───────────────────────────────────────────────────────────
out = Dict(
    "n_candidates_raw" => length(candidates_raw),
    "n_scored" => length(scored),
    "effect_floor" => EFFECT_FLOOR,
    "n_eligible" => length(eligible),
    "gap_threshold" => GAP_THRESHOLD,
    "ceiling_threshold" => CEILING_THRESHOLD,
    "elapsed_screen_s" => t_screen_elapsed,
    "val_loss_phase_A" => res_A.val_final,
    "n_steps_A" => n_steps_A,
    "n_a_only" => n_a_only,
    "n_b_only" => n_b_only,
    "n_both" => n_both,
    "n_neither" => n_neither,
    "screen_results" => screen_results,
)
open(joinpath(@__DIR__, "real_llm_dominant_vs_temoin_screen_results.json"), "w") do io
    JSON.print(io, out)
end
println("\nRésultats écrits -> notebook/real_llm_dominant_vs_temoin_screen_results.json")
