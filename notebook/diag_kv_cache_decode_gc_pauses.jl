# ══════════════════════════════════════════════════════════════════════════════
# diag_kv_cache_decode_gc_pauses.jl — la boucle de décodage (`cached_step!`)
# alloue un tampon FRAIS pour chaque nœud non mis en cache à CHAQUE pas
# (`execute_rule!` standard -- `kv_cache_append!` lui-même alloue aussi un
# tampon frais par pas, voir kv_cache.jl) SANS AUCUN nettoyage périodique
# (contrairement à la construction du graphe, corrigée cette session dans
# `LlamaModel`, src/layers.jl). Hypothèse testée ici (soupçonnée après
# `diag_kv_cache_growth_and_profile.jl`, qui a mesuré des pics isolés
# jusqu'à 2.166s au milieu d'une moyenne stable ~0.48-0.51s/tok, ET après
# avoir observé un run entier 2x plus lent qu'un run identique juste après,
# `diag_kv_cache_final_correctness_vram_run.log` vs `_run2.log`) : ce sont
# des PAUSES GC HÔTE occasionnelles (pas une croissance liée à la position),
# dues à l'accumulation de milliers de petits `CuArray` orphelins entre deux
# passages du GC Julia (dont le déclenchement est basé sur la pression du
# TAS HÔTE, pas sur la VRAM -- gotcha CUDA.jl documenté ; déjà rencontré et
# corrigé UNE FOIS dans ce dépôt via un paramètre `gc_every=`, voir
# `project_neurodsl_capture_activations_gc_fix_2026-07-12` dans la mémoire).
#
# `julia diag_kv_cache_decode_gc_pauses.jl <reclaim_every>` -- 0 désactive le
# nettoyage périodique (référence, comportement actuel) ; >0 nettoie
# (`GC.gc(); Backend.reclaim!`) toutes les N étapes.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, CUDA

reclaim_every = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=2))s] ", msg); flush(stdout))
_log("reclaim_every = $reclaim_every")

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6

dev = NeuroDSL.Backend.CUDADevice()
ns = :qwen2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
out_sym = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                               batched_attn=true, n_kv_heads=N_KV_HEADS,
                               qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out_sym, :final_norm; namespace=ns)
logits_load = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
NeuroDSL.alias_tied_param!(g, ns, :tok_E, :lm_head_W)
dec_logits = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns)
_log("Graphe prêt.")

const PYTHON_ENV = raw"C:\Users\Nevermind\anaconda3\envs\neurodsl_llm_check\python.exe"
const TOKENIZER_HELPER = joinpath(@__DIR__, "qwen2_tokenizer_helper.py")
function _call_helper(req::Dict)
    input_json = JSON.json(req)
    out = IOBuffer()
    run(pipeline(`$PYTHON_ENV $TOKENIZER_HELPER`, stdin=IOBuffer(input_json), stdout=out))
    return JSON.parse(String(take!(out)))
end
encode_chat(messages) = Int.(_call_helper(Dict("action"=>"encode_chat", "messages"=>messages))["ids"])

function batched_prefix_pass!(prefix1idx::Vector{Int})
    NeuroDSL.set!(g, :token_ids, prefix1idx; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    out = Array(NeuroDSL.demand!(g, logits_load; namespace=ns))
    NeuroDSL.prime_kv_cache_from_prefix!(g; src_ns=ns, dst_ns=ns,
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, use_rope=true)
    return Float32.(out[end, :])
end
function cached_step!(tok0::Int, cur_step::Int)
    NeuroDSL.set!(g, :dec_token_id, [tok0 + 1]; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :dec_cur_step, Float32[cur_step]; namespace=ns)
    NeuroDSL.set!(g, :dec_pos, Float32[cur_step-1]; namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, dec_logits; namespace=ns))[1, :]
end

const PROMPT = "Explain in one short sentence why the sky is blue."
history = Dict{String,Any}[Dict("role"=>"user", "content"=>PROMPT)]
prefix = encode_chat(history) .+ 1
logits_row = batched_prefix_pass!(prefix)
cur_step = length(prefix)

const N_GEN = 180
gen_times = Float64[]
for step in 1:N_GEN
    nxt0 = argmax(logits_row) - 1
    global cur_step += 1
    ts = time()
    global logits_row = cached_step!(nxt0, cur_step)
    push!(gen_times, time() - ts)
    if reclaim_every > 0 && step % reclaim_every == 0
        GC.gc(); NeuroDSL.Backend.reclaim!(dev)
    end
end

sorted_t = sort(gen_times)
med = sorted_t[length(sorted_t) ÷ 2]
outliers = [t for t in gen_times if t > 1.5 * med]
println("\n", "═"^78)
println("RÉSUMÉ (reclaim_every=$reclaim_every)")
println("═"^78)
println("  N=$(length(gen_times))  médiane=$(round(med,digits=4))s  moyenne=$(round(sum(gen_times)/length(gen_times),digits=4))s  max=$(round(maximum(gen_times),digits=4))s  min=$(round(minimum(gen_times),digits=4))s")
println("  Nb pas > 1.5x médiane : $(length(outliers)) / $(length(gen_times))  (temps cumulé de ces pics : $(round(sum(outliers),digits=3))s)")
println("  Pics (>1.5x médiane) : ", round.(outliers, digits=3))
println("  Temps total génération ($(N_GEN) pas) : $(round(sum(gen_times),digits=2))s")

open(joinpath(@__DIR__, "diag_kv_cache_decode_gc_pauses_reclaim$(reclaim_every)_results.json"), "w") do io
    JSON.print(io, Dict("reclaim_every"=>reclaim_every, "gen_times"=>gen_times,
        "median"=>med, "mean"=>sum(gen_times)/length(gen_times), "max"=>maximum(gen_times),
        "n_outliers"=>length(outliers), "outlier_sum"=>sum(outliers), "total_time"=>sum(gen_times)), 2)
end
_log("Écrit.")
flush(stdout)
