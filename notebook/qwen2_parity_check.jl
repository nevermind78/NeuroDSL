# ══════════════════════════════════════════════════════════════════════════════
# qwen2_parity_check.jl — PORTE DE VÉRIFICATION (étape 3 du plan LLM réel).
#
# Charge le graphe NeuroDSL sauvegardé par load_qwen2.jl, relit
# reference_logits.json (produit indépendamment par qwen2_reference_logits.py
# via HuggingFace transformers, environnement conda isolé neurodsl_llm_check),
# et pour CHAQUE prompt :
#   (a) fait passer la MÊME séquence de token IDs par le graphe NeuroDSL,
#       compare les logits à la position finale à ceux de la référence
#       (écart absolu, écart relatif, accord du rang top-1/top-5) ;
#   (b) reproduit la continuation gloutonne token par token (recalcul complet
#       à chaque pas, comme generate_text de real_llm.ipynb -- PAS de cache
#       KV, cohérent avec le reste de cette session) et compare la séquence
#       de tokens générés, un par un, à celle de la référence.
#
# AUCUNE tolérance déguisée : les écarts sont rapportés tels quels, pas
# arrondis pour "passer" un seuil choisi après coup.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, LinearAlgebra

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6

dev = NeuroDSL.Backend.CUDADevice()
ns = :qwen2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
const SEQ_LEN_PLACEHOLDER = 8
NeuroDSL.set!(g, :token_ids, ones(Int, SEQ_LEN_PLACEHOLDER); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN_PLACEHOLDER); atom_type=NeuroDSL.Datom, namespace=ns)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
out = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                           batched_attn=true, n_kv_heads=N_KV_HEADS,
                           qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out, :final_norm; namespace=ns)
logits_sym = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)

println("Graphe (re)construit -- chargement des poids sauvegardés (qwen2_neurodsl.json/.bin)...")
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
println("Poids chargés.")

function run_forward!(g, ns, tokens::Vector{Int})
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    t0 = time()
    out = Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))
    dt = time() - t0
    return out, dt  # (seqlen, vocab), temps du passage avant
end

ref = JSON.parsefile(joinpath(MODEL_DIR, "reference_logits.json"))
println("\n$(length(ref)) prompts de référence chargés.\n")

results = Dict{String,Any}[]
for r in ref
    prompt = r["prompt"]
    tokens = Int.(r["token_ids"]) .+ 1   # HF est 0-indexé, NeuroDSL::Embedding attend des indices 1-indexés
    ref_logits = Float64.(r["logits_last_position"])
    ref_gen    = Int.(r["greedy_continuation_ids"])

    fwd_out, dt_first = run_forward!(g, ns, tokens)  # nom distinct du `out` (Symbol) construit plus haut -- évite le piège de soft-scope top-niveau
    my_logits = Float64.(fwd_out[end, :])

    abs_err = abs.(my_logits .- ref_logits)
    rel_err = abs_err ./ max.(abs.(ref_logits), 1e-6)
    top1_ref = argmax(ref_logits); top1_mine = argmax(my_logits)
    top5_ref = sortperm(ref_logits; rev=true)[1:5]
    top5_mine = sortperm(my_logits; rev=true)[1:5]

    println("═"^74)
    println("Prompt: ", repr(prompt), "  (", length(tokens), " tokens)")
    println("  logit dernière position -- écart abs max=", maximum(abs_err),
            "  moyen=", sum(abs_err)/length(abs_err),
            "  écart relatif max=", maximum(rel_err))
    println("  top1 référence(0-idx)=", top1_ref-1, "  top1 NeuroDSL(0-idx)=", top1_mine-1,
            "  identiques=", top1_ref == top1_mine)
    println("  top5 référence(0-idx)=", top5_ref .- 1, "  top5 NeuroDSL(0-idx)=", top5_mine .- 1,
            "  ensembles identiques=", Set(top5_ref) == Set(top5_mine))
    println("  temps du 1er passage avant (compilation incluse) : ", round(dt_first, digits=3), " s")

    # continuation gloutonne, recalcul complet à chaque pas (pas de cache KV)
    cur = copy(tokens)
    my_gen = Int[]
    gen_times = Float64[]
    for _ in 1:length(ref_gen)
        o, dt = run_forward!(g, ns, cur)
        nxt = argmax(o[end, :]) - 1  # retour en indices 0-idx façon HF
        push!(my_gen, nxt)
        push!(cur, nxt + 1)
        push!(gen_times, dt)
    end
    match = my_gen == ref_gen
    println("  continuation référence (0-idx) = ", ref_gen)
    println("  continuation NeuroDSL  (0-idx) = ", my_gen)
    println("  IDENTIQUE token pour token : ", match)
    println("  temps par token générés (s) : ", round.(gen_times, digits=3))
    flush(stdout)

    push!(results, Dict("prompt"=>prompt, "abs_err_max"=>maximum(abs_err), "abs_err_mean"=>sum(abs_err)/length(abs_err),
                         "rel_err_max"=>maximum(rel_err), "top1_match"=>(top1_ref==top1_mine),
                         "top5_set_match"=>(Set(top5_ref)==Set(top5_mine)),
                         "greedy_match"=>match, "gen_times"=>gen_times, "first_pass_time"=>dt_first))
end

println("\n", "═"^74)
println("VERDICT GLOBAL")
println("═"^74)
println("  écart absolu max sur tous les prompts : ", maximum(r["abs_err_max"] for r in results))
println("  top1 identique partout : ", all(r["top1_match"] for r in results))
println("  top5 (ensemble) identique partout : ", all(r["top5_set_match"] for r in results))
println("  continuation gloutonne identique partout : ", all(r["greedy_match"] for r in results))

open(joinpath(MODEL_DIR, "parity_check_results.json"), "w") do io
    JSON.print(io, results)
end
println("\nÉcrit -> ", joinpath(MODEL_DIR, "parity_check_results.json"))
