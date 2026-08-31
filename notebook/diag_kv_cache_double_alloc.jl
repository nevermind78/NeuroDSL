# ══════════════════════════════════════════════════════════════════════════════
# diag_kv_cache_double_alloc.jl — investigation de l'observation utilisateur :
# VRAM ~13-14 Go pour un chat Qwen2.5-1.5B-Instruct alors que le budget
# "poids seuls" attendu (Float32, ~1.5e9 paramètres) est ~6.2 Go.
#
# HYPOTHÈSE TESTÉE (celle de l'utilisateur : "double allocation quelque part") :
# le chemin de PRODUCTION réel pour "charger les poids + discuter"
# (`notebook/qwen2.ipynb`, et son clone de mesure
# `kv_cache_chat_speed_mem_probe2.jl`) construit DEUX namespaces --
# `ns_load` (recalcul complet, sert aussi de passage avant batché du préfixe)
# et `ns_chat` (décodage incrémental avec cache KV) -- et copie TOUS les
# poids par VALEUR de l'un vers l'autre via `copy_params_to_namespace!`
# (`src/graph_api.jl:360`, `copy(nd.value)`, jamais un alias). Pour un modèle
# à ~1.5 Go de paramètres, ça double le budget poids : ~6.2 Go -> ~12.4 Go,
# rien qu'en tenseurs de poids, avant même le cache KV ou les activations.
#
# POURQUOI CETTE DUPLICATION EXISTAIT (raison documentée, `copy_params_to_namespace!`
# et `kv_cache.jl`, 2026-07-29) : `demand!` (`src/dispatch.jl`) parcourait
# ALORS tout le PRÉFIXE topologique du namespace jusqu'à la cible -- pas
# seulement les ancêtres réels -- donc partager un seul namespace entre le
# graphe "recalcul complet" et le graphe "décodage caché" pouvait faire
# ré-exécuter un nœud `:kv_cache_append` à l'insu de l'appelant, avec un
# `cur_step`/`pos` périmé, corrompant `aux_data[:history]`. Deux namespaces +
# copie complète des poids était le correctif robuste À L'ÉPOQUE.
#
# CE QUI A CHANGÉ DEPUIS (vérifié dans le code actuel, `src/dispatch.jl:773-805`,
# `src/graph_api.jl:44-88`) : `demand!` restreint DÉSORMAIS son parcours au
# cône des ANCÊTRES RÉELS de la cible (`_ancestors_of!`), pas au préfixe
# topologique complet -- correctif écrit EXPLICITEMENT pour éliminer cette
# classe de bug (voir la docstring de `_ancestors_of!`, qui cite ce bug KV-cache
# du 2026-07-29 comme cause directe corrigée). `CachedMultiHeadAttention`/
# `CachedLlamaBlock`/`CachedLlamaModel` (`src/layers.jl:273-492`) sont eux-mêmes
# CONÇUS pour partager littéralement les mêmes nœuds de poids qu'un
# `LlamaModel` déjà construit DANS LE MÊME NAMESPACE (aucun `set!(is_param=true)`
# dans ces trois structs -- seulement une convention de nommage qui retrouve
# les nœuds déjà créés). Rien n'a jamais profité de cette capacité en
# production : `qwen2.ipynb` garde toujours les deux namespaces séparés,
# avec un commentaire qui cite encore l'ancien comportement de `demand!`
# (dette technique non nettoyée après le correctif ancêtres-seuls).
#
# CE SCRIPT : reproduit le chemin de production actuel (Phase A, DEUX
# namespaces + copie), mesure la VRAM à chaque étape, PUIS construit le
# même modèle dans un unique namespace partagé (Phase B, le correctif
# proposé), mesure la VRAM aux mêmes étapes, et vérifie que les logits/tokens
# générés sont NUMÉRIQUEMENT IDENTIQUES entre A et B sur les mêmes prompts
# (même garde-fou de correction que `kv_cache_qwen_gate.jl`).
#
# ATTENTION -- LA COMPARAISON VRAM A/B DE CE SCRIPT PRÉCIS EST FAUSSÉE, GARDÉE
# TELLE QUELLE POUR LA TRACE (résultats dans diag_kv_cache_double_alloc_results.json,
# "Économie" y apparaît à TORT négative) : Phase A et Phase B tournent dans LE
# MÊME process Julia, l'une après l'autre -- le graphe `gA` de la phase A
# (~13.8 Go) n'est jamais libéré avant que la phase B ne construise le sien,
# donc la VRAM de B s'empile sur celle de A au lieu de partir d'une base
# propre. Comparaison VRAM propre (deux process séparés, chacun parti de VRAM
# quasi nulle) : voir `diag_kv_cache_double_alloc_phaseA.jl` +
# `diag_kv_cache_double_alloc_phaseB.jl` -- résultat correct : 14420 Mo (A)
# -> 10324 Mo (B), -4.1 Go (-28.4%). La VÉRIFICATION DE CORRECTION de CE
# script (tokens/logits A vs B, calculée AVANT toute contamination mémoire)
# reste, elle, pleinement valide : écart abs max = 0.0, 34/34 tokens identiques.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, CUDA

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=2))s] ", msg); flush(stdout))

# ── Mesure mémoire : DEUX instruments, comme le reste de la session ─────────
# (a) watermark du pool CUDA.jl (précis pour des deltas AVANT/APRÈS un
#     changement de code précis -- même méthodologie que
#     `article_benchmark_vram_probe.jl`/`kv_cache_chat_speed_mem_probe2.jl`).
# (b) `nvidia-smi` (mémoire GPU RÉELLE, contexte CUDA/cuBLAS/fragmentation
#     du pool inclus) -- c'est CE nombre que l'utilisateur regarde ("13-14 Go"),
#     pas le watermark interne du pool CUDA.jl, qui peut être plus bas.
const _POOL = CUDA.CUDACore.pool_create(CUDA.CUDACore.active_state().device)
pool_current_mb() = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_CURRENT)) / 1024^2
pool_high_mb()     = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH)) / 1024^2
reset_high!()      = CUDA.attribute!(_POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
quiesce()          = (CUDA.synchronize(); GC.gc(true); GC.gc(true); CUDA.synchronize())

function nvidia_smi_used_mb()
    out = try
        read(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`, String)
    catch
        return NaN
    end
    parse(Float64, strip(split(strip(out), '\n')[1]))
end

function checkpoint(label::String)
    quiesce()
    smi = nvidia_smi_used_mb()
    cur = pool_current_mb()
    hi  = pool_high_mb()
    _log("  [checkpoint] $label : nvidia-smi=$(round(smi,digits=1))MB  pool_current=$(round(cur,digits=1))MB  pool_high_watermark=$(round(hi,digits=1))MB")
    return (label=label, smi_mb=smi, pool_current_mb=cur, pool_high_mb=hi)
end

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6

n_params_expected = VOCAB_SIZE*DIM +  # embedding (tied lm_head, comptée une fois)
    N_LAYERS*(DIM*DIM + DIM*(N_KV_HEADS*(DIM÷N_HEADS))*2 + DIM*DIM +      # q,k,v,o
              DIM + N_KV_HEADS*(DIM÷N_HEADS)*2 +                          # biais qkv
              3*DIM*HIDDEN_DIM +                                          # mlp w1,w2,w3
              2*DIM) +                                                    # 2 normes
    DIM  # norme finale
_log("Paramètres attendus (calcul indépendant depuis config) : $(n_params_expected) " *
     "-> Float32 = $(round(n_params_expected*4/1024^3, digits=3)) Go (poids SEULS, une copie).")

results = Any[]

# ══════════════════════════════════════════════════════════════════════════
# PHASE A -- chemin de PRODUCTION ACTUEL (qwen2.ipynb) : deux namespaces,
# copy_params_to_namespace! (duplication complète des poids).
# ══════════════════════════════════════════════════════════════════════════
_log("═"^78); _log("PHASE A -- chemin de production actuel (2 namespaces, poids dupliqués)")
push!(results, checkpoint("A0_avant_tout"))

devA = NeuroDSL.Backend.CUDADevice()
ns_load = :A_load
ns_chat = :A_chat
gA = NeuroDSL.NeuroGraph(namespace=ns_load, device=devA)
NeuroDSL.set!(gA, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns_load)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(gA, :token_ids, :tok; namespace=ns_load)
out_sym = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                               batched_attn=true, n_kv_heads=N_KV_HEADS,
                               qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(gA, tok_emb; namespace=ns_load)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(gA, out_sym, :final_norm; namespace=ns_load)
logits_load = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(gA, final_norm, :lm_head; namespace=ns_load)
push!(results, checkpoint("A1_graphe_construit_poids_aleatoires"))

NeuroDSL.load_graph!(gA, ns_load, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
push!(results, checkpoint("A2_apres_chargement_poids_reels_1_copie"))

n_copied = NeuroDSL.copy_params_to_namespace!(gA, ns_load, ns_chat)
_log("  $n_copied paramètres copiés vers :$ns_chat (2e copie complète).")
push!(results, checkpoint("A3_apres_copy_params_to_namespace_2e_copie"))

dec_logits_A = NeuroDSL.build_cached_decode_graph!(gA;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns_chat)
push!(results, checkpoint("A4_apres_construction_graphe_cache"))

const PYTHON_ENV = raw"C:\Users\Nevermind\anaconda3\envs\neurodsl_llm_check\python.exe"
const TOKENIZER_HELPER = joinpath(@__DIR__, "qwen2_tokenizer_helper.py")
function _call_helper(req::Dict)
    input_json = JSON.json(req)
    out = IOBuffer()
    run(pipeline(`$PYTHON_ENV $TOKENIZER_HELPER`, stdin=IOBuffer(input_json), stdout=out))
    return JSON.parse(String(take!(out)))
end
encode_chat(messages) = Int.(_call_helper(Dict("action"=>"encode_chat", "messages"=>messages))["ids"])
decode_ids(ids::Vector{Int}) = _call_helper(Dict("action"=>"decode", "ids"=>ids))["text"]
const EOS_ID = _call_helper(Dict("action"=>"eos_id"))["id"]
_log("EOS_ID = $EOS_ID.")

function batched_prefix_pass_A!(prefix1idx::Vector{Int})
    NeuroDSL.set!(gA, :token_ids, prefix1idx; atom_type=NeuroDSL.Datom, namespace=ns_load)
    NeuroDSL.invalidate_all!(gA; namespace=ns_load)
    out = Array(NeuroDSL.demand!(gA, logits_load; namespace=ns_load))
    NeuroDSL.prime_kv_cache_from_prefix!(gA; src_ns=ns_load, dst_ns=ns_chat,
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, use_rope=true)
    return Float32.(out[end, :])
end
function cached_step_A!(tok0::Int, cur_step::Int)
    NeuroDSL.set!(gA, :dec_token_id, [tok0 + 1]; atom_type=NeuroDSL.Datom, namespace=ns_chat)
    NeuroDSL.set!(gA, :dec_cur_step, Float32[cur_step]; namespace=ns_chat)
    NeuroDSL.set!(gA, :dec_pos, Float32[cur_step-1]; namespace=ns_chat)
    NeuroDSL.invalidate_all!(gA; namespace=ns_chat)
    return Array(NeuroDSL.demand!(gA, dec_logits_A; namespace=ns_chat))[1, :]
end

const PROMPT = "Explain in one short sentence why the sky is blue."
const N_GEN = 40
history = Dict{String,Any}[Dict("role"=>"user", "content"=>PROMPT)]
prefix = encode_chat(history) .+ 1

logits_row = batched_prefix_pass_A!(prefix)
push!(results, checkpoint("A5_apres_passage_prefixe"))
cur_step = length(prefix)
gen_A = Int[]
logit_trace_A = Vector{Float32}[]
mem_at_token = Dict{Int,Any}()
for step in 1:N_GEN
    nxt0 = argmax(logits_row) - 1
    nxt0 == EOS_ID && break
    push!(gen_A, nxt0)
    push!(logit_trace_A, copy(logits_row))
    global cur_step += 1
    global logits_row = cached_step_A!(nxt0, cur_step)
    if step in (1, 10, 40)
        mem_at_token[step] = checkpoint("A6_apres_$(step)_tokens_generes")
    end
end
reply_A = decode_ids(gen_A)
_log("Phase A -- réponse générée ($(length(gen_A)) tokens) : $(repr(reply_A))")
push!(results, checkpoint("A7_fin_phase_A"))

# ══════════════════════════════════════════════════════════════════════════
# PHASE B -- CORRECTIF PROPOSÉ : un seul namespace partagé, AUCUNE copie de
# poids. Valide seulement parce que `demand!` restreint désormais son
# parcours aux ancêtres réels (voir en-tête) -- sinon dangereux comme
# documenté dans `copy_params_to_namespace!`/`kv_cache.jl`.
# ══════════════════════════════════════════════════════════════════════════
_log("═"^78); _log("PHASE B -- correctif proposé (1 namespace, poids partagés, ZÉRO copie)")

devB = NeuroDSL.Backend.CUDADevice()
ns_single = :B_single
gB = NeuroDSL.NeuroGraph(namespace=ns_single, device=devB)
NeuroDSL.set!(gB, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns_single)
tok_embB = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(gB, :token_ids, :tok; namespace=ns_single)
out_symB = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                                batched_attn=true, n_kv_heads=N_KV_HEADS,
                                qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(gB, tok_embB; namespace=ns_single)
final_normB = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(gB, out_symB, :final_norm; namespace=ns_single)
logits_loadB = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(gB, final_normB, :lm_head; namespace=ns_single)
push!(results, checkpoint("B1_graphe_construit_poids_aleatoires"))

NeuroDSL.load_graph!(gB, ns_single, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
push!(results, checkpoint("B2_apres_chargement_poids_reels_1_seule_copie"))

# PAS de copy_params_to_namespace! ici -- build_cached_decode_graph! dans LE
# MÊME namespace retrouve les nœuds de poids déjà créés par LlamaModel
# ci-dessus, par convention de nommage (voir src/layers.jl:288-313).
dec_logits_B = NeuroDSL.build_cached_decode_graph!(gB;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns_single)
push!(results, checkpoint("B3_apres_construction_graphe_cache_MEME_namespace"))

function batched_prefix_pass_B!(prefix1idx::Vector{Int})
    NeuroDSL.set!(gB, :token_ids, prefix1idx; atom_type=NeuroDSL.Datom, namespace=ns_single)
    NeuroDSL.invalidate_all!(gB; namespace=ns_single)
    out = Array(NeuroDSL.demand!(gB, logits_loadB; namespace=ns_single))
    NeuroDSL.prime_kv_cache_from_prefix!(gB; src_ns=ns_single, dst_ns=ns_single,
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, use_rope=true)
    return Float32.(out[end, :])
end
function cached_step_B!(tok0::Int, cur_step::Int)
    NeuroDSL.set!(gB, :dec_token_id, [tok0 + 1]; atom_type=NeuroDSL.Datom, namespace=ns_single)
    NeuroDSL.set!(gB, :dec_cur_step, Float32[cur_step]; namespace=ns_single)
    NeuroDSL.set!(gB, :dec_pos, Float32[cur_step-1]; namespace=ns_single)
    NeuroDSL.invalidate_all!(gB; namespace=ns_single)
    return Array(NeuroDSL.demand!(gB, dec_logits_B; namespace=ns_single))[1, :]
end

logits_rowB = batched_prefix_pass_B!(prefix)
push!(results, checkpoint("B4_apres_passage_prefixe"))
cur_stepB = length(prefix)
gen_B = Int[]
logit_trace_B = Vector{Float32}[]
for step in 1:N_GEN
    nxt0 = argmax(logits_rowB) - 1
    nxt0 == EOS_ID && break
    push!(gen_B, nxt0)
    push!(logit_trace_B, copy(logits_rowB))
    global cur_stepB += 1
    global logits_rowB = cached_step_B!(nxt0, cur_stepB)
    if step in (1, 10, 40)
        mem_at_token[1000+step] = checkpoint("B5_apres_$(step)_tokens_generes")
    end
end
reply_B = decode_ids(gen_B)
_log("Phase B -- réponse générée ($(length(gen_B)) tokens) : $(repr(reply_B))")
push!(results, checkpoint("B6_fin_phase_B"))

# ══════════════════════════════════════════════════════════════════════════
# VÉRIFICATION DE CORRECTION -- A et B doivent produire des tokens/logits
# NUMÉRIQUEMENT IDENTIQUES (même modèle, mêmes poids réels, même prompt).
# ══════════════════════════════════════════════════════════════════════════
_log("═"^78); _log("VÉRIFICATION DE CORRECTION (A vs B)")
same_tokens = gen_A == gen_B
n_cmp = min(length(logit_trace_A), length(logit_trace_B))
max_abs_err = n_cmp > 0 ? maximum(maximum(abs.(logit_trace_A[i] .- logit_trace_B[i])) for i in 1:n_cmp) : NaN
_log("  tokens générés identiques : $same_tokens  (A: $(length(gen_A)) tok, B: $(length(gen_B)) tok)")
_log("  écart absolu max sur les logits comparés ($n_cmp pas) : $max_abs_err")
_log("  réponse A : $(repr(reply_A))")
_log("  réponse B : $(repr(reply_B))")

# ══════════════════════════════════════════════════════════════════════════
# RÉSUMÉ MÉMOIRE
# ══════════════════════════════════════════════════════════════════════════
println("\n", "═"^78)
println("RÉSUMÉ -- VRAM à chaque étape (nvidia-smi, Mo)")
println("═"^78)
for r in results
    println("  $(rpad(r.label,45))  smi=$(round(r.smi_mb,digits=0))MB  pool_high=$(round(r.pool_high_mb,digits=0))MB")
end
println()
for (k,r) in sort(collect(mem_at_token); by=first)
    println("  token-checkpoint $k -> $(r.label)  smi=$(round(r.smi_mb,digits=0))MB")
end

peak_A = maximum(r.smi_mb for r in results if startswith(r.label, "A"))
peak_B = maximum(r.smi_mb for r in results if startswith(r.label, "B"))
println("\nPIC VRAM Phase A (production actuelle, 2 namespaces) : $(round(peak_A,digits=0)) MB")
println("PIC VRAM Phase B (correctif, 1 namespace)             : $(round(peak_B,digits=0)) MB")
println("Économie                                              : $(round(peak_A-peak_B,digits=0)) MB " *
        "($(round(100*(peak_A-peak_B)/peak_A,digits=1))%)")

open(joinpath(@__DIR__, "diag_kv_cache_double_alloc_results.json"), "w") do io
    JSON.print(io, Dict(
        "n_params_expected"=>n_params_expected,
        "expected_weight_gb_one_copy"=>n_params_expected*4/1024^3,
        "results"=>[Dict("label"=>r.label, "smi_mb"=>r.smi_mb, "pool_current_mb"=>r.pool_current_mb, "pool_high_mb"=>r.pool_high_mb) for r in results],
        "token_checkpoints"=>Dict(string(k)=>Dict("label"=>r.label,"smi_mb"=>r.smi_mb) for (k,r) in mem_at_token),
        "same_tokens"=>same_tokens, "max_abs_logit_err"=>max_abs_err,
        "gen_A"=>gen_A, "gen_B"=>gen_B,
        "reply_A"=>reply_A, "reply_B"=>reply_B,
        "peak_vram_mb_A"=>peak_A, "peak_vram_mb_B"=>peak_B,
    ), 2)
end
_log("Écrit -> diag_kv_cache_double_alloc_results.json")
flush(stdout)
