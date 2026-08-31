# ══════════════════════════════════════════════════════════════════════════════
# diag_decode_op_breakdown.jl — décompose FINEMENT le coût du bloc attention
# d'UN pas de décodage (au lieu de "attention vs MLP" comme
# diag_kv_cache_growth_and_profile.jl, qui a déjà établi attn≈8x MLP) : on
# veut savoir QUELLE PARTIE de l'attention coûte cher --
#   (a) tranchage Q/K/V + RoPE (16+14=30 nœuds/couche)
#   (b) matmul Q·Kᵀ par tête (12 nœuds/couche)
#   (c) scale_no_mask + softmax (24 nœuds/couche)
#   (d) matmul P·V par tête (12 nœuds/couche)
#   (e) kv_cache_append (4 nœuds/couche)
#   (f) hcat_heads + projection de sortie (2 nœuds/couche)
# -- pour trancher entre deux explications concurrentes de "attention
# ≈8x MLP, kernel-launch-count driven" :
#   (H1) les matmul Q·Kᵀ/P·V par tête dominent -> batcher ces deux ops
#        précisément (approche déjà utilisée en training, src/kernels.jl)
#        est le bon levier.
#   (H2) le reste (tranchage/RoPE/scale/softmax/cache-append, PAS touchés
#        par un simple batching QK/PV) domine -> batcher QK/PV seuls ne
#        réglerait qu'une fraction du problème.
#
# Méthode : checkpoints `demand!` PROGRESSIFS à l'intérieur du bloc
# attention de la couche 1 (mêmes limites méthodologiques documentées dans
# diag_kv_cache_growth_and_profile.jl : CUDA.@sync ajouté à chaque
# checkpoint, absent du chemin chaud réel -- décompose la même somme de
# travail, le total recombiné est comparé au total bout-en-bout pour
# vérifier l'absence de biais grossier). Couches 2-28 mesurées en UN SEUL
# bloc (grain couche, pas besoin de le refaire 28 fois -- diag précédent a
# déjà montré la couche 1 est représentative, croissance ratio≈0.81 = PLAT).
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, CUDA

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=2))s] ", msg); flush(stdout))

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
_log("Graphe construit (poids aléatoires).")

NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
NeuroDSL.alias_tied_param!(g, ns, :tok_E, :lm_head_W)
dec_logits = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns)
_log("Poids réels chargés, cache KV construit.")

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
_log("Préfixe traité ($(length(prefix)) tok). Avance de 100 pas pour atteindre un régime établi (comme le diag précédent)...")

for step in 1:100
    nxt0 = argmax(logits_row) - 1
    global cur_step += 1
    global logits_row = cached_step!(nxt0, cur_step)
end
_log("Régime établi atteint (cur_step=$cur_step).")

# ── Pas profilé : checkpoints fins DANS la couche 1, grossiers pour 2-28 ───
function step_setup!(tok0::Int, cs::Int)
    NeuroDSL.set!(g, :dec_token_id, [tok0 + 1]; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :dec_cur_step, Float32[cs]; namespace=ns)
    NeuroDSL.set!(g, :dec_pos, Float32[cs-1]; namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
end

# 1) Référence bout-en-bout (SANS checkpoints intermédiaires) sur le pas cur_step+1
nxt0 = argmax(logits_row) - 1
global cur_step += 1
step_setup!(nxt0, cur_step)
CUDA.synchronize(); t0 = time()
logits_ref = Array(NeuroDSL.demand!(g, dec_logits; namespace=ns))
CUDA.synchronize()
t_total_baseline = time() - t0
_log("Référence bout-en-bout (pas cur_step=$cur_step) : $(round(t_total_baseline,digits=4))s")

# 2) Pas segmentés finement, sur PLUSIEURS pas consécutifs (régime établi) --
#    avec un `GC.gc()` explicite juste avant chaque, pour vérifier si le
#    premier essai était contaminé par une pause GC tombant sur le 1er
#    checkpoint (hypothèse concurrente à "l'op embedding est réellement
#    lente") -- comparaison directe, pas une supposition.
mha1 = :layer_1_mha
checkpoints = Symbol[]
labels = String[]
push!(checkpoints, :dec_tok_emb);                                    push!(labels, "embedding")
push!(checkpoints, :layer_1_dec_norm1);                              push!(labels, "norm1 (couche 1)")
push!(checkpoints, Symbol(mha1,:_dec_v));                            push!(labels, "Q/K/V proj (couche 1)")
push!(checkpoints, Symbol(mha1,:_dec_q_h,N_HEADS,:_rope));           push!(labels, "tranchage+RoPE Q, 12 têtes (couche 1)")
push!(checkpoints, Symbol(mha1,:_vcache_h,N_KV_HEADS));              push!(labels, "tranchage+RoPE K/V+cache-append, 2 têtes KV (couche 1)")
push!(checkpoints, Symbol(mha1,:_dec_sc_h,N_HEADS));                 push!(labels, "matmul Q·Kᵀ, 12 têtes (couche 1)")
push!(checkpoints, Symbol(mha1,:_dec_pr_h,N_HEADS));                 push!(labels, "scale_no_mask+softmax, 12 têtes (couche 1)")
push!(checkpoints, Symbol(mha1,:_dec_ao_h,N_HEADS));                 push!(labels, "matmul P·V, 12 têtes (couche 1)")
push!(checkpoints, Symbol(mha1,:_dec_output));                       push!(labels, "hcat_heads+projection sortie (couche 1)")
push!(checkpoints, :layer_1_dec_out);                                push!(labels, "residual+norm2+MLP (couche 1)")
push!(checkpoints, Symbol(:layer_,N_LAYERS,:_dec_out));              push!(labels, "couches 2-$(N_LAYERS) (bloc complet, grain grossier)")
push!(checkpoints, dec_logits);                                      push!(labels, "norme finale + lm_head")

function run_checkpoints!(g, ns, checkpoints)
    seg_times = Float64[]
    CUDA.synchronize()
    tprev = time()
    for cp in checkpoints
        NeuroDSL.demand!(g, cp; namespace=ns)
        CUDA.synchronize()
        tnow = time()
        push!(seg_times, tnow - tprev)
        tprev = tnow
    end
    return seg_times
end

# ── Sonde supplémentaire : le coût "embedding=180-250ms" vient-il du DFS
#    `_ancestors_of!` (première fois que :dec_tok_emb est CIBLE directe d'un
#    `demand!`, jamais seulement visité comme ancêtre de :dec_logits) ou de
#    `execute_rule!` (calcul réel) ? Mesurés SÉPARÉMENT, hors de la boucle
#    de checkpoints, pour ne pas se fier à une hypothèse non vérifiée. ─────
_log("Sonde : _ancestors_of!(:dec_tok_emb) vs execute_rule! réel, séparément...")
t_anc = @elapsed anc_emb = NeuroDSL._ancestors_of!(g, ns, :dec_tok_emb)
_log("  _ancestors_of!(:dec_tok_emb) : $(round(1000*t_anc,digits=3))ms, longueur=$(length(anc_emb)) -- contenu=$anc_emb")
NeuroDSL.set!(g, :dec_token_id, [7]; atom_type=NeuroDSL.Datom, namespace=ns)  # force invalidation réelle
CUDA.synchronize()
t_ancB = @elapsed anc_embB = NeuroDSL._ancestors_of!(g, ns, :dec_tok_emb)  # devrait être un HIT de cache (même cible)
_log("  _ancestors_of!(:dec_tok_emb) après re-set! (cache DEVRAIT être un hit) : $(round(1000*t_ancB,digits=3))ms")
CUDA.synchronize()
t_demand_only = @elapsed NeuroDSL.demand!(g, :dec_tok_emb; namespace=ns)
CUDA.synchronize()
_log("  demand!(:dec_tok_emb) mur (ancêtres déjà en cache + execute_rule! réel) : $(round(1000*t_demand_only,digits=3))ms")

const N_TRIALS = 4
all_trials = Vector{Float64}[]
for trial in 1:N_TRIALS
    nxtT = argmax(logits_ref[1,:]) - 1
    global cur_step += 1
    step_setup!(nxtT, cur_step)
    GC.gc(); NeuroDSL.Backend.reclaim!(g.device)   # élimine la pause GC comme facteur de confusion
    CUDA.synchronize()
    seg_times = run_checkpoints!(g, ns, checkpoints)
    push!(all_trials, seg_times)
    global logits_ref = reshape(Array(NeuroDSL.demand!(g, dec_logits; namespace=ns)), 1, :)
    _log("  Essai $trial/$N_TRIALS (cur_step=$cur_step) : total segmenté = $(round(sum(seg_times),digits=4))s, checkpoint 'embedding' = $(round(1000*seg_times[1],digits=2))ms")
end

println("\n", "═"^86)
println("DÉCOMPOSITION FINE D'UN PAS DE DÉCODAGE -- $N_TRIALS essais indépendants, GC.gc() forcé avant chacun")
println("═"^86)
println("Référence bout-en-bout (1 seul demand!, sans checkpoints, essai unique) : $(round(t_total_baseline,digits=4))s")
for (t_idx, seg_times) in enumerate(all_trials)
    seg_total = sum(seg_times)
    println("\n-- Essai $t_idx -- somme segmentée = $(round(seg_total,digits=4))s --")
    for (lbl, t) in zip(labels, seg_times)
        println("  $(rpad(lbl,58)) $(round(1000*t,digits=2))ms  ($(round(100*t/seg_total,digits=1))%)")
    end
end

# Médiane par checkpoint à travers les essais -- robuste à UNE pause GC/OS
# isolée sur un essai donné (contrairement à la moyenne).
using Statistics
med_by_cp = [median([all_trials[t][i] for t in 1:N_TRIALS]) for i in 1:length(checkpoints)]
med_total = sum(med_by_cp)
println("\n", "-"^86)
println("MÉDIANE par checkpoint sur les $N_TRIALS essais (robuste aux pauses GC ponctuelles) :")
for (lbl, t) in zip(labels, med_by_cp)
    println("  $(rpad(lbl,58)) $(round(1000*t,digits=2))ms  ($(round(100*t/med_total,digits=1))%)")
end

layer1_attn_ops = sum(med_by_cp[4:9])   # slice+rope Q, slice+rope+cache K/V, QK, scale+softmax, PV, hcat+outproj
layer1_total = sum(med_by_cp[3:10])
println("-"^86)
println("Couche 1 -- Q/K/V proj + attention (têtes), médiane : $(round(1000*layer1_attn_ops,digits=2))ms")
println("Couche 1 -- bloc complet (QKV..MLP), médiane         : $(round(1000*layer1_total,digits=2))ms")
qk_pv_only = med_by_cp[6] + med_by_cp[8]
rest_of_attn = layer1_attn_ops - qk_pv_only
println("  dont matmul QK+PV seuls (candidats au batching gemm_strided_batched) : $(round(1000*qk_pv_only,digits=2))ms")
println("  dont le RESTE (slice/RoPE/scale/softmax/cache-append/hcat/outproj)   : $(round(1000*rest_of_attn,digits=2))ms")
println("  -> le reste représente $(round(100*rest_of_attn/layer1_attn_ops,digits=1))% du temps d'attention de la couche 1")

open(joinpath(@__DIR__, "diag_decode_op_breakdown_results.json"), "w") do io
    JSON.print(io, Dict(
        "cur_step_final"=>cur_step, "t_total_baseline"=>t_total_baseline,
        "labels"=>labels, "n_trials"=>N_TRIALS,
        "all_trials_ms"=>[1000 .* st for st in all_trials],
        "median_by_checkpoint_ms"=>1000 .* med_by_cp,
        "layer1_qk_pv_median_ms"=>1000*qk_pv_only, "layer1_rest_median_ms"=>1000*rest_of_attn,
        "layer1_attn_total_median_ms"=>1000*layer1_attn_ops,
    ), 2)
end
_log("Écrit -> diag_decode_op_breakdown_results.json")
