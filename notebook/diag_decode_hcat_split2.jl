# ══════════════════════════════════════════════════════════════════════════════
# diag_decode_hcat_split2.jl — CORRECTIF de méthodologie découvert en
# instrumentant `hcat_heads_fwd_batched!` (src/kernels.jl, prototype de fusion
# du 2026-08-31) : la "sonde bis" a montré que `demand!(concat_sym)` coûtait
# encore 6.8ms MÊME avec le kernel fusionné actif et son cache déjà chaud
# (`stale=false` confirmé par instrumentation) -- alors qu'un micro-benchmark
# isolé du même kernel, mêmes tampons, cache chaud, ne coûte que ~0.02ms.
#
# Cause trouvée : `diag_decode_op_breakdown.jl` définit CHAQUE checkpoint de
# "tête" avec `Symbol(mha1,:_dec_q_h,N_HEADS,:_rope)` etc. -- càd le symbole
# de la DERNIÈRE tête (h=N_HEADS=12) SEULEMENT, pas "toutes les têtes". Or
# `demand!` (src/dispatch.jl:806) ne calcule QUE les ancêtres RÉELS de la
# cible demandée (correctif O(cône) du 2026-07-29) -- demander q_h12_rope ne
# force PAS le calcul de q_h1_rope..q_h11_rope (branches parallèles
# indépendantes, pas des ancêtres les unes des autres). Donc chaque checkpoint
# "N têtes" du script original ne mesurait en réalité QUE le coût de LA
# DERNIÈRE tête -- et le checkpoint `hcat_heads` (qui dépend de TOUTES les
# 12 têtes) se retrouvait à calculer pour la première fois les 11 chaînes
# complètes (tranche+RoPE+QK+scale+softmax+PV) des têtes 1-11, jamais
# calculées par un checkpoint précédent -- tout ce travail (11 têtes × ~6
# opérations = 66 lancements de kernel) étant FAUSSEMENT attribué au seul
# nœud `hcat_heads`.
#
# Ce script corrige la méthodologie : à chaque étape, on `demand!` le symbole
# de TOUTES les têtes concernées (boucle 1:n_heads ou 1:n_kv_heads), pas
# seulement la dernière -- pour obtenir la vraie décomposition.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, CUDA, Statistics

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
_log("Préfixe traité ($(length(prefix)) tok). Avance de 100 pas pour atteindre un régime établi...")

for step in 1:100
    nxt0 = argmax(logits_row) - 1
    global cur_step += 1
    global logits_row = cached_step!(nxt0, cur_step)
end
_log("Régime établi atteint (cur_step=$cur_step).")

function step_setup!(tok0::Int, cs::Int)
    NeuroDSL.set!(g, :dec_token_id, [tok0 + 1]; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :dec_cur_step, Float32[cs]; namespace=ns)
    NeuroDSL.set!(g, :dec_pos, Float32[cs-1]; namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
end

mha1 = :layer_1_mha

# ── Checkpoints CORRIGÉS : chaque étape "têtes" demande TOUTES les têtes,
#    pas seulement la dernière (boucle explicite + CUDA.synchronize() APRÈS
#    la boucle complète, pas par tête -- même granularité que l'original). ──
function run_checkpoints_fixed!(g, ns)
    seg_times = Float64[]; labels = String[]
    CUDA.synchronize(); tprev = time()
    push!(seg_times, 0.0)  # placeholder, remplacé ci-dessous par embedding
    empty!(seg_times)

    # embedding
    NeuroDSL.demand!(g, :dec_tok_emb; namespace=ns); CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "embedding"); tprev = tnow

    # norm1
    NeuroDSL.demand!(g, :layer_1_dec_norm1; namespace=ns); CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "norm1"); tprev = tnow

    # Q/K/V proj
    NeuroDSL.demand!(g, Symbol(mha1,:_dec_v); namespace=ns); CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "QKV proj"); tprev = tnow

    # Q slice+RoPE, TOUTES les 12 têtes
    for h in 1:N_HEADS
        NeuroDSL.demand!(g, Symbol(mha1,:_dec_q_h,h,:_rope); namespace=ns)
    end
    CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "Q slice+RoPE (12 têtes, CORRIGÉ)"); tprev = tnow

    # K/V slice+RoPE+cache-append, TOUTES les n_kv_heads têtes
    for h in 1:N_KV_HEADS
        NeuroDSL.demand!(g, Symbol(mha1,:_kcache_h,h); namespace=ns)
        NeuroDSL.demand!(g, Symbol(mha1,:_vcache_h,h); namespace=ns)
    end
    CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "K/V slice+RoPE+cache (2 têtes KV, CORRIGÉ)"); tprev = tnow

    # matmul QK, TOUTES les 12 têtes
    for h in 1:N_HEADS
        NeuroDSL.demand!(g, Symbol(mha1,:_dec_sc_h,h); namespace=ns)
    end
    CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "matmul QK (12 têtes, CORRIGÉ)"); tprev = tnow

    # scale+softmax, TOUTES les 12 têtes
    for h in 1:N_HEADS
        NeuroDSL.demand!(g, Symbol(mha1,:_dec_pr_h,h); namespace=ns)
    end
    CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "scale+softmax (12 têtes, CORRIGÉ)"); tprev = tnow

    # matmul PV, TOUTES les 12 têtes
    for h in 1:N_HEADS
        NeuroDSL.demand!(g, Symbol(mha1,:_dec_ao_h,h); namespace=ns)
    end
    CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "matmul PV (12 têtes, CORRIGÉ)"); tprev = tnow

    # hcat_heads SEUL -- maintenant que TOUTES les têtes sont valides, ce
    # checkpoint isole VRAIMENT le coût de la concaténation.
    NeuroDSL.demand!(g, Symbol(mha1,:_dec_concat); namespace=ns); CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "hcat_heads SEUL (CORRIGÉ)"); tprev = tnow

    # projection sortie seule
    NeuroDSL.demand!(g, Symbol(mha1,:_dec_output); namespace=ns); CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "projection sortie SEULE"); tprev = tnow

    # reste (MLP couche 1)
    NeuroDSL.demand!(g, :layer_1_dec_out; namespace=ns); CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "residual+norm2+MLP"); tprev = tnow

    # couches 2-28
    NeuroDSL.demand!(g, Symbol(:layer_,N_LAYERS,:_dec_out); namespace=ns); CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "couches 2-28"); tprev = tnow

    # final
    NeuroDSL.demand!(g, dec_logits; namespace=ns); CUDA.synchronize()
    tnow = time(); push!(seg_times, tnow - tprev); push!(labels, "norme finale + lm_head"); tprev = tnow

    return seg_times, labels
end

const N_TRIALS = 4
all_trials = Vector{Float64}[]
local labels_ref = String[]
for trial in 1:N_TRIALS
    nxtT = argmax(logits_row) - 1
    global cur_step += 1
    step_setup!(nxtT, cur_step)
    GC.gc(); NeuroDSL.Backend.reclaim!(g.device)
    CUDA.synchronize()
    seg_times, labels = run_checkpoints_fixed!(g, ns)
    global labels_ref = labels
    push!(all_trials, seg_times)
    global logits_row = reshape(Array(NeuroDSL.demand!(g, dec_logits; namespace=ns)), 1, :)[1,:]
    _log("  Essai $trial/$N_TRIALS (cur_step=$cur_step) : total = $(round(1000*sum(seg_times),digits=2))ms")
end

med_by_cp = [median([all_trials[t][i] for t in 1:N_TRIALS]) for i in 1:length(labels_ref)]
med_total = sum(med_by_cp)
println("\n", "-"^86)
println("MÉDIANE par checkpoint CORRIGÉ (toutes les têtes demandées à chaque étape) :")
for (lbl, t) in zip(labels_ref, med_by_cp)
    println("  $(rpad(lbl,58)) $(round(1000*t,digits=3))ms  ($(round(100*t/med_total,digits=1))%)")
end
println("-"^86)
println("TOTAL couche 1 (QKV..sortie attn) : $(round(1000*sum(med_by_cp[3:9]),digits=2))ms")

open(joinpath(@__DIR__, "diag_decode_hcat_split2_results.json"), "w") do io
    JSON.print(io, Dict("labels"=>labels_ref, "median_by_checkpoint_ms"=>1000 .* med_by_cp,
                         "all_trials_ms"=>[1000 .* st for st in all_trials], "n_trials"=>N_TRIALS), 2)
end
_log("Écrit -> diag_decode_hcat_split2_results.json")
