# ══════════════════════════════════════════════════════════════════════════════
# code4_step2_ioi_patching.jl — ÉTAPE 2 : circuit causal réel (Qwen2.5-1.5B,
# famille IOI retenue à l'étape 1) + trois sondes empiriques honnêtes des
# théorèmes de code4.tex (collapse conditionnel, dissociation patch/ablation,
# interaction cross-layer) SUR ce circuit réel.
#
# GRAPHE : recalcul complet uniquement (namespace :qwen2_probe2), comme
# qwen2_parity_check.jl / code4_real_llm_step1_matched_pair_search.jl -- PAS
# le graphe de cache KV (ns_chat), consigne explicite (aucune garantie de
# capture_activations!/patch_node! sous état de cache KV, voir garde dans
# src/patching.jl elle-même).
#
# OUTILLAGE : uniquement des fonctions déjà écrites et testées dans
# src/patching.jl (capture_activations, patch_node!, patch_nodes!,
# restore_from_cache!, recovery_metric, greedy_patch_search!, backward_prune!)
# + les fonctions internes déjà utilisées PAR greedy_patch_search! lui-même
# (_downstream_nodes, _adaptive_restore!) -- ré-orchestrées ICI, dans ce
# script, pour obtenir une visibilité candidat-par-candidat que
# greedy_patch_search! n'expose pas (il ne retourne qu'à la fin). Aucun
# nouveau mécanisme de graphe : la séquence patch_nodes!->demand!->
# _adaptive_restore!->re-patch_nodes! est EXACTEMENT celle documentée dans
# greedy_patch_search! (src/patching.jl:512-561), on y ajoute seulement des
# println/flush.
#
# CANDIDATS : sortie de chaque tête d'attention (:layer_i_mha_ao_h<h>, vue
# zero-copy déjà individuellement adressable même en attention batchée -- voir
# note du 2026-07-10 dans src/layers.jl) + sortie de chaque MLP
# (:layer_i_mlp_out), pour les 28 couches x 12 têtes = 336 + 28 = 364 sites.
#
# SÉLECTEUR : s(x) = logit(cand1) - logit(cand2) à la DERNIÈRE position (même
# convention que l'étape 1). `metric` de greedy_patch_search!/backward_prune!
# restreint la récupération à ce scalaire (pas la sortie vocab entière) --
# extension déjà prévue par leur signature (voir docstring, "sert par exemple
# à restreindre la mesure à une seule ligne/position").
#
# TROIS SONDES EMPIRIQUES, honnêtes, PAS des preuves formelles :
#   1. COLLAPSE (Thm 1) : ablation à zéro du circuit minimal trouvé
#      (backward_prune!) sur x_A ET x_B séparément -- si s_ablated(x_A) ~=
#      s_ablated(x_B), c'est un collapse conditionnel réel observé.
#   2. DISSOCIATION (Thm 2) : compare l'effet de PATCHER (corrompu->propre,
#      déjà mesuré par la recherche gloutonne) à l'effet d'ABLATER (propre->
#      zéro) sur le même ensemble de sites -- si ces deux effets ne sont pas
#      de simples miroirs l'un de l'autre, c'est la dissociation prédite.
#   3. INTERACTION (Thm 3) : sur deux sites du circuit minimal (une tête +
#      un MLP, si les deux sont présents), effet d'ablation individuel vs
#      joint -- le terme d'interaction = Δ(joint) - Δ(H) - Δ(M).
#
# Résultat honnête quel qu'il soit -- confirmé, infirmé, ou non concluant --
# écrit progressivement dans ce log ET dans code4_step2_results.json
# (checkpoint après CHAQUE instance, pas seulement à la fin).
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, LinearAlgebra

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=1))s] ", msg); flush(stdout))

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6
const MAX_SITES = 6          # profondeur max de la recherche gloutonne
const RESULTS_PATH = joinpath(MODEL_DIR, "code4_step2_results.json")

_log("══ ÉTAPE 2 : circuit causal IOI + sondes des théorèmes de code4.tex (Qwen2.5-1.5B) ══")

# ── Graphe : recalcul complet uniquement (namespace dédié, distinct de tout
# noyau Jupyter déjà vivant sur la machine -- ce script est un process julia
# autonome, séparé) ────────────────────────────────────────────────────────
dev = NeuroDSL.Backend.CUDADevice()
ns = :qwen2_probe2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
out_sym = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                               batched_attn=true, n_kv_heads=N_KV_HEADS,
                               qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out_sym, :final_norm; namespace=ns)
logits_sym = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)
_log("Graphe construit. Chargement des poids réels (qwen2_neurodsl.json/.bin)...")
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
_log("Poids chargés.")

function run_forward!(g, ns, tokens1idx::Vector{Int})
    NeuroDSL.set!(g, :token_ids, tokens1idx; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))
end

# ── Tokenizer : serveur persistant, action "encode" BRUTE (pas de ChatML) ──
const PYTHON_ENV = raw"C:\Users\Nevermind\anaconda3\envs\neurodsl_llm_check\python.exe"
const TOKENIZER_HELPER = joinpath(@__DIR__, "qwen2_tokenizer_helper.py")
_log("Démarrage du serveur tokenizer persistant...")
const TOK_PROC = open(`$PYTHON_ENV -u $TOKENIZER_HELPER --serve`, "r+")
JSON.parse(readline(TOK_PROC))
function tok_call(req::Dict)
    println(TOK_PROC, JSON.json(req))
    flush(TOK_PROC)
    return JSON.parse(readline(TOK_PROC))
end
encode_raw(text::String) = Int.(tok_call(Dict("action"=>"encode", "text"=>text))["ids"])
_log("Serveur tokenizer prêt.")

function first_token_id(w::String)
    ids = encode_raw(w)
    return ids[1], length(ids)
end

# ── Familles/instances IOI -- IDENTIQUES à l'étape 1 (résultat retenu) ─────
struct Instance
    label::String
    prompt_A::String
    prompt_B::String
    cand1::String
    cand2::String
end

const IOI_INSTANCES = [
    Instance("Mary/John",  "When Mary and John went to the store, John gave the bag to",
                            "When Mary and John went to the store, Mary gave the bag to", " Mary", " John"),
    Instance("Alice/Bob",  "When Alice and Bob went to the park, Bob gave the ball to",
                            "When Alice and Bob went to the park, Alice gave the ball to", " Alice", " Bob"),
    Instance("Sarah/Tom",  "When Sarah and Tom went to the market, Tom gave the flowers to",
                            "When Sarah and Tom went to the market, Sarah gave the flowers to", " Sarah", " Tom"),
    Instance("Anna/Mike",  "When Anna and Mike went to the cafe, Mike gave the book to",
                            "When Anna and Mike went to the cafe, Anna gave the book to", " Anna", " Mike"),
    Instance("Lucy/Sam",   "When Lucy and Sam went to the beach, Sam gave the towel to",
                            "When Lucy and Sam went to the beach, Lucy gave the towel to", " Lucy", " Sam"),
]

# ── Espace de candidats : sortie par tête (28x12) + sortie MLP par couche (28) ──
candidate_sites = Symbol[]
site_kind = Dict{Symbol,String}()
for i in 1:N_LAYERS
    for h in 1:N_HEADS
        sym = Symbol(:layer_, i, :_mha_ao_h, h)
        push!(candidate_sites, sym); site_kind[sym] = "head(layer=$i,head=$h)"
    end
    mlp_sym = Symbol(:layer_, i, :_mlp_out)
    push!(candidate_sites, mlp_sym); site_kind[mlp_sym] = "mlp(layer=$i)"
end
_log("Espace de candidats : $(length(candidate_sites)) sites ($(N_LAYERS)x$(N_HEADS) têtes + $(N_LAYERS) MLP).")

# ── Recherche gloutonne instrumentée (candidat-par-candidat) ────────────────
# Réplique EXACTEMENT la séquence de greedy_patch_search! (src/patching.jl),
# en appelant les mêmes fonctions publiques/internes, avec logging ajouté.
function instrumented_greedy_search!(g, output_sym, candidates, clean_cache, corrupted_cache,
                                      clean_s, corrupted_s, metric_fn; max_sites::Int, ns::Symbol, log_prefix::String)
    selected = Symbol[]
    remaining = collect(candidates)
    trajectory = NamedTuple[]
    best_so_far = 0.0
    selected_cone_union = Set{Symbol}()
    n_tested = 0

    for rnd in 1:max_sites
        best_site = nothing
        best_recovery = best_so_far
        t_round0 = time()
        for cand in remaining
            cand_cone = NeuroDSL._downstream_nodes(g, cand, ns)
            NeuroDSL.patch_nodes!(g, vcat(selected, [cand]), clean_cache; namespace=ns)
            out = NeuroDSL.demand!(g, output_sym; namespace=ns)
            r = metric_fn(out)
            n_tested += 1
            if n_tested % 20 == 0 || r > best_recovery
                _log("$log_prefix   [round $rnd] candidat testé #$n_tested : $(site_kind[cand]) -> recovery=$(round(r,digits=4)) ($(round(time()-t_round0,digits=1))s dans ce round)")
            end

            NeuroDSL._adaptive_restore!(g, ns, cand, cand_cone, selected_cone_union, corrupted_cache, output_sym)
            isempty(selected) || NeuroDSL.patch_nodes!(g, selected, clean_cache; namespace=ns)

            if r > best_recovery
                best_recovery = r
                best_site = cand
            end
            n_tested % 60 == 0 && GC.gc(true)
        end
        best_site === nothing && (_log("$log_prefix   round $rnd : aucune amélioration -- arrêt de la recherche gloutonne."); break)

        best_cone = NeuroDSL._downstream_nodes(g, best_site, ns)
        push!(selected, best_site)
        NeuroDSL.patch_nodes!(g, selected, clean_cache; namespace=ns)
        NeuroDSL.demand!(g, output_sym; namespace=ns)

        filter!(s -> s != best_site, remaining)
        union!(selected_cone_union, best_cone)
        best_so_far = best_recovery
        push!(trajectory, (; site=best_site, cumulative_recovery=best_recovery))
        _log("$log_prefix ✅ SITE RETENU (round $rnd) : $(site_kind[best_site]) -- recovery cumulée = $(round(best_recovery,digits=4)) [$(round(time()-t_round0,digits=1))s pour ce round, $(length(remaining)) candidats restants]")
    end
    return selected, trajectory
end

# ── Boucle principale : 5 instances IOI ─────────────────────────────────────
all_results = Dict{String,Any}[]

function save_checkpoint!()
    open(RESULTS_PATH, "w") do io
        JSON.print(io, Dict("instances"=>all_results, "max_sites"=>MAX_SITES,
                             "n_candidates"=>length(candidate_sites)))
    end
    _log("── checkpoint écrit -> $RESULTS_PATH ($(length(all_results))/$(length(IOI_INSTANCES)) instances) ──")
end

for (idx, inst) in enumerate(IOI_INSTANCES)
    _log("")
    _log("═"^78)
    _log("INSTANCE $idx/$(length(IOI_INSTANCES)) : $(inst.label)")
    _log("═"^78)
    try
        ids0_A = encode_raw(inst.prompt_A); ids0_B = encode_raw(inst.prompt_B)
        if length(ids0_A) != length(ids0_B)
            _log("⚠ longueurs de séquence différentes (A=$(length(ids0_A)), B=$(length(ids0_B))) -- patch_node! exige des formes identiques. INSTANCE IGNORÉE, rapporté honnêtement.")
            push!(all_results, Dict("label"=>inst.label, "skipped"=>true, "reason"=>"length_mismatch"))
            save_checkpoint!()
            continue
        end
        c1_id, c1_n = first_token_id(inst.cand1)
        c2_id, c2_n = first_token_id(inst.cand2)

        ids_A = ids0_A .+ 1; ids_B = ids0_B .+ 1

        outA = run_forward!(g, ns, ids_A)
        clean_output = copy(outA)
        clean_cache = NeuroDSL.capture_activations(g, ns)
        s_clean = Float64(clean_output[end, c1_id+1] - clean_output[end, c2_id+1])

        outB = run_forward!(g, ns, ids_B)
        corrupted_output = copy(outB)
        corrupted_cache = NeuroDSL.capture_activations(g, ns)
        s_corrupted = Float64(corrupted_output[end, c1_id+1] - corrupted_output[end, c2_id+1])

        denom = s_clean - s_corrupted
        _log("s_clean(x_A)=$(round(s_clean,digits=3))  s_corrupted(x_B)=$(round(s_corrupted,digits=3))  denom=$(round(denom,digits=3))")
        if abs(denom) < 1e-6
            _log("⚠ denom ~ 0 -- paire non appariée sur ce run (inattendu vu étape 1). INSTANCE IGNORÉE.")
            push!(all_results, Dict("label"=>inst.label, "skipped"=>true, "reason"=>"denom_zero"))
            save_checkpoint!()
            continue
        end

        metric_fn = out -> begin
            oa = Array(out)
            s = Float64(oa[end, c1_id+1] - oa[end, c2_id+1])
            1.0 - abs(s - s_clean) / abs(denom)
        end
        s_of(out) = Float64(Array(out)[end, c1_id+1] - Array(out)[end, c2_id+1])

        # ── Recherche gloutonne (corrompu=x_B -> propre=x_A) ────────────────
        _log("── Recherche gloutonne (candidat-par-candidat) sur $(length(candidate_sites)) sites, max_sites=$MAX_SITES ──")
        selected, trajectory = instrumented_greedy_search!(g, logits_sym, candidate_sites,
            clean_cache, corrupted_cache, s_clean, s_corrupted, metric_fn;
            max_sites=MAX_SITES, ns=ns, log_prefix="[$(inst.label)]")

        _log("Circuit gloutonne (avant élagage) : $(join([site_kind[s] for s in selected], ", "))")

        # ── Élagage arrière (fonction officielle, non ré-instrumentée -- peu
        # de sites, coût négligeable, aucune raison de dupliquer) ───────────
        remaining, pruned = NeuroDSL.backward_prune!(g, logits_sym, selected,
            clean_cache, corrupted_cache, clean_output, corrupted_output;
            namespace=ns, metric=metric_fn)
        out_after_prune = NeuroDSL.demand!(g, logits_sym; namespace=ns)
        s_patched_corrupted = s_of(out_after_prune)
        recovery_final = metric_fn(out_after_prune)
        _log("Circuit minimal (après backward_prune!) : $(join([site_kind[s] for s in remaining], ", "))")
        _log("Élagués (redondants) : $(isempty(pruned) ? "aucun" : join([site_kind[s] for s in pruned], ", "))")
        _log("Recovery finale (circuit minimal, patché sur x_B corrompu) = $(round(recovery_final,digits=4)), s_patched=$(round(s_patched_corrupted,digits=3))")

        # ── SONDE 1 : COLLAPSE (Thm 1) -- ablation à zéro de `remaining` sur
        # x_A et x_B séparément, chacun reparti d'un forward NATUREL propre
        # (pas du run corrompu de la recherche) ─────────────────────────────
        zero_cache = Dict{Symbol,Any}(sym => (t = copy(clean_cache[sym]); t .= 0f0; t) for sym in remaining)

        function ablate_and_measure(ids::Vector{Int}, sites::Vector{Symbol})
            run_forward!(g, ns, ids)
            isempty(sites) && return s_of(NeuroDSL.demand!(g, logits_sym; namespace=ns))
            NeuroDSL.patch_nodes!(g, sites, zero_cache; namespace=ns)
            out = NeuroDSL.demand!(g, logits_sym; namespace=ns)
            return s_of(out)
        end

        s_ablated_A = ablate_and_measure(ids_A, remaining)
        s_ablated_B = ablate_and_measure(ids_B, remaining)
        gap_before = abs(s_clean - s_corrupted)
        gap_after = abs(s_ablated_A - s_ablated_B)
        collapse_ratio = gap_before == 0 ? NaN : gap_after / gap_before
        _log("SONDE 1 (collapse) : s_ablated(x_A)=$(round(s_ablated_A,digits=3))  s_ablated(x_B)=$(round(s_ablated_B,digits=3))" *
             "  |écart avant ablation|=$(round(gap_before,digits=3))  |écart après ablation|=$(round(gap_after,digits=3))" *
             "  ratio=$(round(collapse_ratio,digits=4)) ($(collapse_ratio < 0.3 ? "collapse net" : collapse_ratio < 0.7 ? "collapse partiel" : "PAS de collapse observé"))")

        # ── SONDE 2 : DISSOCIATION (Thm 2) -- effet de PATCHER (déjà mesuré,
        # recovery_final sur l'échelle standard) vs effet d'ABLATER (destruction
        # relative de l'écart propre) ───────────────────────────────────────
        ablation_effect = gap_before == 0 ? NaN : abs(s_clean - s_ablated_A) / gap_before
        patching_effect = recovery_final
        dissociation_gap = abs(patching_effect - ablation_effect)
        _log("SONDE 2 (dissociation) : effet_patch=$(round(patching_effect,digits=4))  effet_ablation=$(round(ablation_effect,digits=4))" *
             "  écart(dissociation)=$(round(dissociation_gap,digits=4)) ($(dissociation_gap > 0.15 ? "dissociation nette" : "effets proches, pas de dissociation nette ici"))")

        # ── SONDE 3 : INTERACTION (Thm 3) -- une tête + un MLP du circuit
        # minimal, si les deux types sont présents ──────────────────────────
        heads_in_circuit = [s for s in remaining if occursin("head", site_kind[s])]
        mlps_in_circuit  = [s for s in remaining if occursin("mlp", site_kind[s])]
        interaction_probe = nothing
        if !isempty(heads_in_circuit) && !isempty(mlps_in_circuit)
            H = heads_in_circuit[1]; M = mlps_in_circuit[1]
            s_H = ablate_and_measure(ids_A, [H])
            s_M = ablate_and_measure(ids_A, [M])
            s_HM = ablate_and_measure(ids_A, [H, M])
            dH = s_clean - s_H
            dM = s_clean - s_M
            dHM = s_clean - s_HM
            interaction = dHM - dH - dM
            rel = (abs(dH) + abs(dM)) == 0 ? NaN : abs(interaction) / (abs(dH) + abs(dM))
            _log("SONDE 3 (interaction) : H=$(site_kind[H]) M=$(site_kind[M])  ΔH=$(round(dH,digits=3)) ΔM=$(round(dM,digits=3)) ΔHM=$(round(dHM,digits=3))" *
                 "  interaction=ΔHM-ΔH-ΔM=$(round(interaction,digits=4))  |interaction|/(|ΔH|+|ΔM|)=$(round(rel,digits=4))" *
                 " ($(rel > 0.2 ? "interaction non-négligeable, cohérent avec Thm 3" : "interaction faible, modèle additif approximativement valide ici"))")
            interaction_probe = Dict("H"=>site_kind[H], "M"=>site_kind[M], "delta_H"=>dH, "delta_M"=>dM,
                                      "delta_HM"=>dHM, "interaction"=>interaction, "relative_interaction"=>rel)
        else
            _log("SONDE 3 (interaction) : circuit minimal ne contient pas à la fois une tête ET un MLP ($(length(heads_in_circuit)) têtes, $(length(mlps_in_circuit)) MLP) -- sonde non applicable, rapporté tel quel.")
        end

        push!(all_results, Dict(
            "label"=>inst.label, "skipped"=>false,
            "s_clean"=>s_clean, "s_corrupted"=>s_corrupted,
            "greedy_trajectory"=>[Dict("site"=>site_kind[t.site], "cumulative_recovery"=>t.cumulative_recovery) for t in trajectory],
            "circuit_before_prune"=>[site_kind[s] for s in selected],
            "circuit_minimal"=>[site_kind[s] for s in remaining],
            "pruned"=>[site_kind[s] for s in pruned],
            "recovery_final"=>recovery_final,
            "probe1_collapse"=>Dict("s_ablated_A"=>s_ablated_A, "s_ablated_B"=>s_ablated_B,
                                     "gap_before"=>gap_before, "gap_after"=>gap_after, "ratio"=>collapse_ratio),
            "probe2_dissociation"=>Dict("patching_effect"=>patching_effect, "ablation_effect"=>ablation_effect,
                                         "gap"=>dissociation_gap),
            "probe3_interaction"=>interaction_probe,
        ))
        save_checkpoint!()
    catch e
        _log("❌ ERREUR sur l'instance $(inst.label) : $(sprint(showerror, e))")
        push!(all_results, Dict("label"=>inst.label, "skipped"=>true, "reason"=>"error: $(sprint(showerror, e))"))
        save_checkpoint!()
    end
end

close(TOK_PROC)

# ── Verdict agrégé, honnête ─────────────────────────────────────────────────
_log("")
_log("═"^78)
_log("VERDICT ÉTAPE 2")
_log("═"^78)
n_ok = count(r -> !get(r, "skipped", false), all_results)
_log("Instances traitées avec succès : $n_ok/$(length(IOI_INSTANCES))")
for r in all_results
    if get(r, "skipped", false)
        _log("  [$(r["label"])] IGNORÉE ($(r["reason"]))")
    else
        p1 = r["probe1_collapse"]; p2 = r["probe2_dissociation"]
        _log("  [$(r["label"])] circuit_minimal=$(join(r["circuit_minimal"], "+"))  recovery=$(round(r["recovery_final"],digits=3))" *
             "  collapse_ratio=$(round(p1["ratio"],digits=3))  dissociation_gap=$(round(p2["gap"],digits=3))")
    end
end
flush(stdout)
save_checkpoint!()
_log("Terminé -> $RESULTS_PATH")
