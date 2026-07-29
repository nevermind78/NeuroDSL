# ══════════════════════════════════════════════════════════════════════════════
# kv_cache_qwen_gate.jl — Porte de parité #2 (Qwen2.5-1.5B-Instruct, poids
# réels) : logits du décodage incrémental avec cache KV
# (`build_cached_decode_graph!`, src/layers.jl) VS logits du chemin de
# recalcul complet DÉJÀ VALIDÉ contre HuggingFace (`qwen2_parity_check.jl`,
# écart abs max 3.7e-5 contre la référence HF réelle). Ce script NE compare
# PAS à HuggingFace directement -- il compare le chemin caché au chemin
# recalcul complet, tous deux sur le MÊME graphe NeuroDSL.
#
# RÉVISION 2026-07-29 (deux changements, voir aussi kv_cache_toy_gate.jl) :
#   1. Namespaces SÉPARÉS pour le chemin caché et le chemin de référence,
#      poids partagés par VALEUR via `copy_params_to_namespace!`
#      (src/graph_api.jl) -- PAS le même namespace comme dans la version
#      précédente, qui a fait planter ce script (`demand!` peut ré-exécuter
#      un nœud à état d'un AUTRE chemin comme effet de bord s'il est invalide
#      et ordonné avant la cible demandée dans le namespace partagé -- voir
#      le commentaire détaillé de `copy_params_to_namespace!` pour le
#      mécanisme exact). `:kv_cache_append` s'est retrouvé ré-exécuté avec un
#      `cur_step` périmé, d'où l'erreur "historique... taille incohérente".
#   2. Progression imprimée ET flush à CHAQUE étape susceptible d'être lente
#      (chargement des poids, compilation CUDA du premier passage avant,
#      chaque pas de chaque prompt) -- exigence explicite du coordinateur
#      après 34 minutes de silence total sur un processus qui s'est avéré
#      être le serveur de langage VS Code, pas ce script -- mais la
#      confusion elle-même montre qu'un script qui ne dit rien pendant
#      plusieurs minutes est indiscernable d'un script bloqué de l'extérieur.
#
# Étend le test au-delà du prompt : après avoir consommé le prompt token par
# token, continue avec les tokens de la continuation gloutonne DÉJÀ CONNUE
# (reference_logits.json, colonne greedy_continuation_ids) -- traverse donc
# la frontière prompt/génération, le cas réellement utilisé par le futur
# notebook de chat multi-tours.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON

const T0 = time()
_elapsed() = round(time() - T0, digits=1)
_log(msg) = (println("[t+$(_elapsed())s] ", msg); flush(stdout))

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6

_log("── Porte de parité Qwen2.5-1.5B : cache KV vs recalcul complet (NeuroDSL vs NeuroDSL) ──")
_log("Démarrage. Étapes lentes attendues : chargement du checkpoint (~7 Go), " *
     "et compilation CUDA au tout premier passage avant -- chacune annoncée ci-dessous.")

dev = NeuroDSL.Backend.CUDADevice()
ns_full   = :qwen2_full
ns_cached = :qwen2_cached
g = NeuroDSL.NeuroGraph(namespace=ns_full, device=dev)
const SEQ_LEN_PLACEHOLDER = 8
NeuroDSL.set!(g, :token_ids, ones(Int, SEQ_LEN_PLACEHOLDER); atom_type=NeuroDSL.Datom, namespace=ns_full)

_log("Construction du graphe recalcul complet (namespace :$ns_full)...")
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns_full)
out = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                           batched_attn=true, n_kv_heads=N_KV_HEADS,
                           qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns_full)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out, :final_norm; namespace=ns_full)
logits_full = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns_full)
_log("Graphe recalcul complet construit ($(N_LAYERS) couches, $(N_HEADS)/$(N_KV_HEADS) têtes Q/KV).")

_log("Chargement des poids réels depuis qwen2_neurodsl.json/.bin (~7 Go -- peut prendre 1-2 minutes)...")
NeuroDSL.load_graph!(g, ns_full, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
_log("Poids chargés.")

_log("Copie des poids vers le namespace caché (:$ns_cached), PAR VALEUR (copy_params_to_namespace!)...")
n_copied = NeuroDSL.copy_params_to_namespace!(g, ns_full, ns_cached)
_log("  $n_copied paramètres copiés.")

_log("Construction du graphe caché (namespace :$ns_cached, batched_attn=false -- " *
     "le cache KV n'a pas été câblé sur le chemin batché)...")
logits_cached = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns_cached)
_log("Graphe caché construit.")

function run_forward!(g, ns, tokens::Vector{Int})
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, logits_full; namespace=ns))
end

ref = JSON.parsefile(joinpath(MODEL_DIR, "reference_logits.json"))
_log("$(length(ref)) prompts chargés depuis reference_logits.json.")
_log("Pas 1 de chaque genre (référence ET cache) inclut la compilation CUDA à froid des " *
     "kernels -- attendre un délai plus long sur les tout premiers pas seulement.")

global_max_abs_err = 0.0
global_all_argmax_match = true
global_all_top5_match = true
results = Dict{String,Any}[]

for (pi, r) in enumerate(ref)
    prompt = r["prompt"]
    tokens0 = Int.(r["token_ids"])                       # 0-indexé (HF)
    gen0    = Int.(r["greedy_continuation_ids"])          # 0-indexé (HF), continuation déjà connue
    full_seq_0idx = vcat(tokens0, gen0)                   # traverse prompt -> génération
    full_seq = full_seq_0idx .+ 1                         # 1-indexé (NeuroDSL::Embedding)
    n = length(full_seq)

    println("═"^74); flush(stdout)
    _log("Prompt $pi/$(length(ref)): $(repr(prompt))  ($(length(tokens0)) tokens prompt + " *
         "$(length(gen0)) tokens continuation = $n au total)")

    prompt_max_abs_err = 0.0
    prompt_all_argmax_match = true
    prompt_all_top5_match = true

    for t in 1:n
        t_step0 = time()
        # -- référence : recalcul complet du préfixe [1..t], namespace ns_full --
        prefix = full_seq[1:t]
        fwd_out = run_forward!(g, ns_full, prefix)
        ref_row = Float64.(fwd_out[t, :])

        # -- chemin caché : un pas de décodage incrémental, namespace ns_cached --
        NeuroDSL.set!(g, :dec_token_id, [full_seq[t]]; atom_type=NeuroDSL.Datom, namespace=ns_cached)
        NeuroDSL.set!(g, :dec_cur_step, Float32[t]; namespace=ns_cached)
        NeuroDSL.set!(g, :dec_pos, Float32[t-1]; namespace=ns_cached)
        NeuroDSL.invalidate_all!(g; namespace=ns_cached)
        cached_row = Float64.(Array(NeuroDSL.demand!(g, logits_cached; namespace=ns_cached))[1, :])
        dt_step = round(time() - t_step0, digits=2)

        abs_err = maximum(abs.(ref_row .- cached_row))
        am_ref = argmax(ref_row); am_cached = argmax(cached_row)
        top5_ref = Set(sortperm(ref_row; rev=true)[1:5])
        top5_cached = Set(sortperm(cached_row; rev=true)[1:5])
        argmax_match = am_ref == am_cached
        top5_match = top5_ref == top5_cached

        # PAS de `global` ici : `prompt_max_abs_err`/`prompt_all_argmax_match`/
        # `prompt_all_top5_match` sont déjà des locales de la boucle EXTÉRIEURE
        # (`for (pi,r) in enumerate(ref)`, portée A) -- cette boucle intérieure
        # (`for t in 1:n`, portée B, imbriquée DANS A) les voit et les modifie
        # par portée souple normale. `global` ici entrerait en conflit avec le
        # fait que le nom est déjà lié localement dans A (erreur de syntaxe
        # obtenue au tout premier essai de ce script).
        prompt_max_abs_err = max(prompt_max_abs_err, abs_err)
        prompt_all_argmax_match &= argmax_match
        prompt_all_top5_match &= top5_match

        role = t <= length(tokens0) ? "prompt" : "génération"
        _log("  pas $t/$n [$role] ($(dt_step)s) : écart abs max=$(round(abs_err,sigdigits=4))  " *
             "argmax_ref=$(am_ref-1)  argmax_cache=$(am_cached-1)  match=$argmax_match  top5_match=$top5_match")
    end

    _log("  -- résumé prompt $pi : écart abs max=$(prompt_max_abs_err)  " *
         "argmax identique sur TOUS les pas=$(prompt_all_argmax_match)  " *
         "top5 identique sur TOUS les pas=$(prompt_all_top5_match)")

    global global_max_abs_err = max(global_max_abs_err, prompt_max_abs_err)
    global global_all_argmax_match &= prompt_all_argmax_match
    global global_all_top5_match &= prompt_all_top5_match
    push!(results, Dict("prompt"=>prompt, "max_abs_err"=>prompt_max_abs_err,
                         "all_argmax_match"=>prompt_all_argmax_match, "all_top5_match"=>prompt_all_top5_match))
end

println("\n", "═"^74)
_log("VERDICT GLOBAL -- PORTE DE PARITÉ QWEN2.5-1.5B (cache KV vs recalcul complet)")
println("  écart absolu max sur TOUS les prompts, TOUS les pas : ", global_max_abs_err)
println("  argmax identique partout : ", global_all_argmax_match)
println("  top5 (ensemble) identique partout : ", global_all_top5_match)
if global_max_abs_err < 1e-2 && global_all_argmax_match
    println("  VERDICT : PASS -- le chemin caché reproduit le recalcul complet sur Qwen2.5-1.5B réel.")
else
    println("  VERDICT : FAIL -- divergence réelle, à NE PAS utiliser dans le notebook de chat.")
end
flush(stdout)

open(joinpath(MODEL_DIR, "kv_cache_gate_results.json"), "w") do io
    JSON.print(io, Dict("global_max_abs_err"=>global_max_abs_err,
                         "global_all_argmax_match"=>global_all_argmax_match,
                         "global_all_top5_match"=>global_all_top5_match,
                         "per_prompt"=>results))
end
_log("Écrit -> $(joinpath(MODEL_DIR, "kv_cache_gate_results.json"))")
