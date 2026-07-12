# ══════════════════════════════════════════════════════════════════════════════
# P1-bis — Attention batchée vs non-batchée : cône de patch et coût, sur un
# modèle de la même forme que real_llm_surgery_v2 (5 couches après greffe à
# :layer_1_out). Conçu avec Fable le 2026-07-11.
#
# Question : patcher un site EN AMONT du tenseur groupé `ao3` (pr_h/sk_h/sc_h/
# q_h/k_h/v_h) sous batched_attn=true élargit-il le cône de patch aux têtes
# SŒURS de la même couche (puisque `:batched_pv`/`:batched_qk` lisent TOUTES
# les têtes en une fois) -- alors qu'en mode non-batché, patcher une tête ne
# touche jamais ses sœurs ? Prédiction chiffrée (Fable, avant mesure) :
#   - sites pr_h/sk_h/sc_h/q_h/k_h/v_h : delta de cône = +4 nœuds EXACTEMENT
#     (le tenseur groupé ao3 + les 3 vues sœurs ao_h qui en dépendent).
#   - site ao_h lui-même : delta de cône = 0 (le patch court-circuite la vue,
#     ao3 n'est jamais en aval de ao_h).
# Le mécanisme ne dépend PAS des valeurs des poids (structure du graphe
# seulement) -- un modèle à poids aléatoires (non entraîné) est donc
# méthodologiquement suffisant, pas besoin de réentraîner.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics, Printf, LinearAlgebra, JSON

dev = NeuroDSL.Backend.CUDADevice()

vocab_size, block_size, dim, n_heads, hidden_dim, n_layers = 65, 256, 256, 4, 512, 4

function build_graph(dev, ns::Symbol; batched::Bool)
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

Random.seed!(42)
ns_b = :p1bis_batched
g_b, logits_b = build_graph(dev, ns_b; batched=true)
new_out_b = NeuroDSL.insert_block!(g_b, ns_b, :layer_1_out, dim, n_heads, hidden_dim; batched_attn=true)

ns_nb = :p1bis_nonbatched
g_nb, logits_nb = build_graph(dev, ns_nb; batched=false)
new_out_nb = NeuroDSL.insert_block!(g_nb, ns_nb, :layer_1_out, dim, n_heads, hidden_dim; batched_attn=false)

# Copie des poids du graphe batché vers le non-batché, par NOM -- les deux
# graphes ont exactement les mêmes symboles de paramètres (seule la règle des
# nœuds intermédiaires d'attention diffère), donc une copie directe suffit.
function copy_params!(dst_g, dst_ns, src_g, src_ns)
    for (sym, nd) in src_g.nodes[src_ns]
        nd.is_param && NeuroDSL.set!(dst_g, sym, Array(nd.value); is_param=true, namespace=dst_ns)
    end
    NeuroDSL.invalidate_all!(dst_g; namespace=dst_ns)
end
copy_params!(g_nb, ns_nb, g_b, ns_b)

tokens = rand(MersenneTwister(1), 1:vocab_size, block_size)
for (g, ns, logits) in ((g_b, ns_b, logits_b), (g_nb, ns_nb, logits_nb))
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
end

# ═══ 1) Parité -- bloquant avant toute mesure de coût ═══
logits_val_b  = Array(NeuroDSL.demand!(g_b, logits_b; namespace=ns_b))
logits_val_nb = Array(NeuroDSL.demand!(g_nb, logits_nb; namespace=ns_nb))
parity_max_abs_diff = maximum(abs.(logits_val_b .- logits_val_nb))
println("Parité batché vs non-batché (max abs diff des logits) : ", parity_max_abs_diff)
@assert parity_max_abs_diff < 1f-2 "Parité insuffisante -- les deux graphes ne calculent pas la même fonction, arrêt avant toute mesure de coût"

# ═══ 2) Grille de sites × couches, cône + coût ═══
# Préfixes testés directement (pas d'arithmétique d'indice fragile) :
# couche 1 (originale, précoce), la greffe elle-même (position fonctionnelle
# 2, entre layer_1 et layer_2), et layer_3 (originale, tardive) -- échantillon
# qui couvre précoce/greffé/tardif sans supposer un mapping d'indices.
layer_prefixes = ["layer_1", "surgery_layer_1_out", "layer_3"]
sites_kind = ["pr_h", "sc_h", "ao_h", "q_h", "k_h"]

function sync()
    NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.synchronize()
end

function timed_patch!(g, ns, site::Symbol, logits_sym, cache; reps::Int=30, warmup::Int=3)
    for _ in 1:warmup
        NeuroDSL.patch_node!(g, site, cache; namespace=ns)
        out = NeuroDSL.demand!(g, logits_sym; namespace=ns)
        Array(out); sync()
    end
    times = Float64[]
    for _ in 1:reps
        t0 = time_ns()
        NeuroDSL.patch_node!(g, site, cache; namespace=ns)
        out = NeuroDSL.demand!(g, logits_sym; namespace=ns)
        Array(out)   # force la lecture -> synchronisation implicite avant l'arrêt du chrono
        sync()
        push!(times, (time_ns() - t0) / 1e6)
    end
    s = sort(times)
    n = length(s)
    return (; median=s[n÷2+1], q25=s[max(1,n÷4)], q75=s[min(n,3n÷4)])
end

rows = NamedTuple[]
for h in [2]   # une seule tête suffit pour la mesure de cône/coût (le mécanisme ne dépend pas de l'indice de tête)
    for lp in layer_prefixes
        for kind in sites_kind
            for (g, ns, logits, mode) in ((g_b, ns_b, logits_b, "batched"), (g_nb, ns_nb, logits_nb, "non_batched"))
                site = Symbol(lp, "_mha_", kind, h)
                haskey(g.nodes[ns], site) || continue

                cache = NeuroDSL.capture_activations(g, ns)
                cone = NeuroDSL._downstream_nodes(g, site, ns)
                n_sibling_ao = count(s -> occursin(Regex(lp * "_mha_ao_h\\d+\$"), String(s)), cone)

                out0 = Array(NeuroDSL.demand!(g, logits; namespace=ns))
                timing = timed_patch!(g, ns, site, logits, cache)
                NeuroDSL.patch_node!(g, site, cache; namespace=ns)   # remise en état (valeur identique -- no-op sémantique)
                NeuroDSL.demand!(g, logits; namespace=ns)

                push!(rows, (; site=String(site), layer=lp, head=h, site_class=kind, mode,
                              cone_size=length(cone), n_sibling_ao, med_ms=timing.median, q25=timing.q25, q75=timing.q75))
                @printf("  %-42s mode=%-11s cône=%-4d  sœurs_ao=%d  médiane=%.4f ms\n",
                        String(site), mode, length(cone), n_sibling_ao, timing.median)
            end
        end
    end
end

println("\n" * "="^70)
println("Vérification du théorème corrigé (notebook/conjecture_surcout_batched.md) : Δ = c(s) + 2D")
println("="^70)
# D = nombre de couches d'attention en aval, fixé par la structure du modèle
# (4 couches originales + 1 greffe entre layer_1 et layer_2) -- déjà établi
# lors de la vérification initiale de P1-bis.
D_of_layer = Dict("layer_1" => 4, "surgery_layer_1_out" => 3, "layer_3" => 1)
c_of_kind(kind) = kind == "ao_h" ? 0 : (kind in ("q_h", "k_h") ? 4n_heads - 2 : n_heads)
for kind in sites_kind
    r_b  = filter(r -> r.site_class == kind && r.mode == "batched", rows)
    r_nb = filter(r -> r.site_class == kind && r.mode == "non_batched", rows)
    for (rb, rnb) in zip(r_b, r_nb)
        delta = rb.cone_size - rnb.cone_size
        predicted = c_of_kind(kind) + 2 * D_of_layer[rb.layer]
        marker = delta == predicted ? "✅" : "❌ ÉCART"
        @printf("  %-8s %-22s : cône batché=%d non-batché=%d  delta=%+d  (théorème : %+d)  %s\n",
                kind, rb.layer, rb.cone_size, rnb.cone_size, delta, predicted, marker)
    end
end

p1bis_results = Dict("parity_max_abs_diff" => parity_max_abs_diff, "rows" => rows)
open(joinpath(@__DIR__, "p1bis_results.json"), "w") do io
    JSON.print(io, p1bis_results)
end
println("\nRésultats écrits -> notebook/p1bis_results.json")
