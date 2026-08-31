# ══════════════════════════════════════════════════════════════════════════════
# diag_multiturn_vram_growth.jl — reproduit EXACTEMENT le scénario signalé par
# l'utilisateur ce soir : une VRAIE conversation multi-tours dans
# notebook/qwen2.ipynb (4 tours, dans l'ordre : "hi qwen" / "how are you" /
# "integral x*exp(x)" / "integral x*exp(-x)"), avec les tours 3-4 produisant
# des réponses longues (LaTeX, intégration par parties). L'utilisateur a
# observé la VRAM passer de ~8.5 Go à ~13.3 Go autour/après le tour 4 -- un
# saut de ~4.8 Go, bien plus que la croissance attendue du cache KV
# (quelques centaines de Ko/token) mesurée aujourd'hui plus tôt sur un test
# à UN SEUL tour (~34 tokens).
#
# Utilise le chemin de production EXACT tel qu'il est dans notebook/qwen2.ipynb
# APRÈS tous les correctifs de ce matin (namespace unique, alias_tied_param!,
# construction sans pic transitoire) -- reproduit ici en process isolé (pas
# dans le notebook lui-même) pour une mesure VRAM propre (nvidia-smi), comme
# toute la méthodologie établie aujourd'hui (diag_kv_cache_*.jl).
#
# Mesure la VRAM (nvidia-smi + pool CUDA current/high-watermark) IMMÉDIATEMENT
# APRÈS CHAQUE TOUR (pas seulement à la fin) -- c'est la seule façon de voir
# QUEL tour cause le saut, pas seulement le total.
#
# Contrôle séparé (option --single-long-turn) : UNE SEULE conversation à UN
# tour avec max_new_tokens=250 (~même longueur que les tours 3-4), pour
# isoler l'effet "réponse longue" de l'effet "historique qui grandit sur
# plusieurs tours" -- les deux hypothèses candidates du rapport.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, CUDA

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=2))s] ", msg); flush(stdout))

const _POOL = CUDA.CUDACore.pool_create(CUDA.CUDACore.active_state().device)
pool_current_mb() = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_CURRENT)) / 1024^2
pool_high_mb()     = Int(CUDA.attribute(UInt64, _POOL, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH)) / 1024^2
quiesce()          = (CUDA.synchronize(); GC.gc(true); GC.gc(true); CUDA.synchronize())

function nvidia_smi_used_mb()
    out = try
        read(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`, String)
    catch
        return NaN
    end
    parse(Float64, strip(split(strip(out), '\n')[1]))
end

results = Any[]
function checkpoint(label::String)
    quiesce()
    smi = nvidia_smi_used_mb()
    cur = pool_current_mb()
    hi  = pool_high_mb()
    _log("  [checkpoint] $label : nvidia-smi=$(round(smi,digits=1))MB  pool_current=$(round(cur,digits=1))MB  pool_high_watermark=$(round(hi,digits=1))MB")
    r = (label=label, smi_mb=smi, pool_current_mb=cur, pool_high_mb=hi)
    push!(results, r)
    return r
end

# ── Mode : "quiesce" avant checkpoint force un GC.gc()+sync, ce qui rendrait
# la fuite invisible si elle n'existait QUE parce que le GC ne tourne jamais
# assez souvent -- mais nvidia-smi mesure la mémoire remise au DRIVER
# (`pool_high_watermark`/CUDA.reclaim() n'est PAS appelé ici), donc un pic
# transitoire qui a forcé le pool CUDA à grossir reste visible dans
# `nvidia-smi` même après un GC.gc() complet -- exactement ce qu'on veut
# mesurer (le high-water mark réel, pas juste l'état actuel après ménage).
# On mesure aussi `pool_current_mb` SANS GC pour capter l'état brut juste
# après le tour (voir `checkpoint_raw` ci-dessous), pour distinguer un vrai
# leak (survit même au GC.gc() de `checkpoint`) d'un simple retard de GC
# (disparaît avec `quiesce()` mais gonflait `nvidia-smi` avant).
function checkpoint_raw(label::String)
    smi = nvidia_smi_used_mb()
    cur = pool_current_mb()
    hi  = pool_high_mb()
    _log("  [checkpoint_RAW, pas de GC] $label : nvidia-smi=$(round(smi,digits=1))MB  pool_current=$(round(cur,digits=1))MB  pool_high_watermark=$(round(hi,digits=1))MB")
    return (label=label, smi_mb=smi, pool_current_mb=cur, pool_high_mb=hi)
end

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6

_log("Construction du graphe (chemin de production, namespace unique)...")
push!(results, checkpoint("T0_avant_tout"))

dev = NeuroDSL.Backend.CUDADevice()
const ns = :qwen2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
out_sym = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                               batched_attn=true, n_kv_heads=N_KV_HEADS,
                               qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out_sym, :final_norm; namespace=ns)
logits_load = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)
push!(results, checkpoint("T1_graphe_construit"))

NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
push!(results, checkpoint("T2_apres_load_graph"))

NeuroDSL.alias_tied_param!(g, ns, :tok_E, :lm_head_W)
push!(results, checkpoint("T3_apres_alias_tied_param"))

dec_logits = NeuroDSL.build_cached_decode_graph!(g;
    n_layers=N_LAYERS, dim=DIM, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
    n_kv_heads=N_KV_HEADS, qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA, namespace=ns)
push!(results, checkpoint("T4_apres_construction_graphe_cache"))

# ── Tokenizer, mode UN-COUP (comme diag_kv_cache_final_correctness_vram.jl --
# pas le serveur persistant, pour un script autonome simple à lancer) ──
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

# ── chat(), copie EXACTE de la logique de notebook/qwen2.ipynb (cell 6) ──
const HISTORY = Dict{String,Any}[]

function cached_step!(tok0::Int, cur_step::Int)
    NeuroDSL.set!(g, :dec_token_id, [tok0 + 1]; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :dec_cur_step, Float32[cur_step]; namespace=ns)
    NeuroDSL.set!(g, :dec_pos, Float32[cur_step-1]; namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, dec_logits; namespace=ns))[1, :]
end

function chat_turn(prompt::AbstractString; max_new_tokens::Int=200)
    push!(HISTORY, Dict("role"=>"user", "content"=>String(prompt)))
    ids0 = encode_chat(HISTORY)
    prefix = ids0 .+ 1

    gen0 = Int[]
    stopped_on_eos = false
    t0 = time()

    NeuroDSL.set!(g, :token_ids, prefix; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    prefix_out = Array(NeuroDSL.demand!(g, logits_load; namespace=ns))
    logits_row = Float32.(prefix_out[end, :])
    NeuroDSL.prime_kv_cache_from_prefix!(g; src_ns=ns, dst_ns=ns,
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, use_rope=true)
    cur_step = length(prefix)

    for step in 1:max_new_tokens
        nxt0 = argmax(logits_row) - 1
        if nxt0 == EOS_ID
            stopped_on_eos = true
            break
        end
        push!(gen0, nxt0)
        cur_step += 1
        logits_row = cached_step!(nxt0, cur_step)
    end
    dt = time() - t0
    reply = decode_ids(gen0)
    push!(HISTORY, Dict("role"=>"assistant", "content"=>reply))
    return (reply=reply, n_tokens=length(gen0), prefix_len=length(prefix), dt=dt, stopped_on_eos=stopped_on_eos)
end

# ══════════════════════════════════════════════════════════════════════════
# SCÉNARIO PRINCIPAL : la conversation réelle à 4 tours signalée par l'utilisateur
# ══════════════════════════════════════════════════════════════════════════
const MODE = length(ARGS) >= 1 ? ARGS[1] : "multiturn"

turn_records = Any[]
if MODE == "multiturn"
    prompts = [
        "hi qwen",
        "how are you",
        "integral x*exp(x)",
        "integral x*exp(-x)",
    ]
    for (i, p) in enumerate(prompts)
        _log("═"^70)
        _log("TOUR $i : \"$p\"")
        raw_before = checkpoint_raw("turn$(i)_RAW_avant")
        res = chat_turn(p; max_new_tokens=200)
        _log("  -> $(res.n_tokens) tokens générés, préfixe=$(res.prefix_len) tok, $(round(res.dt,digits=1))s, " *
             (res.stopped_on_eos ? "arrêt sur EOS" : "tronqué à max_new_tokens"))
        _log("  Réponse : $(res.reply[1:min(end,300)])" * (length(res.reply) > 300 ? "[...]" : ""))
        raw_after = checkpoint_raw("turn$(i)_RAW_apres_AVANT_GC")
        cpt = checkpoint("turn$(i)_apres_GC")
        push!(turn_records, Dict(
            "turn"=>i, "prompt"=>p, "reply"=>res.reply, "n_tokens"=>res.n_tokens,
            "prefix_len"=>res.prefix_len, "dt"=>res.dt, "stopped_on_eos"=>res.stopped_on_eos,
            "raw_before"=>Dict("smi_mb"=>raw_before.smi_mb, "pool_current_mb"=>raw_before.pool_current_mb, "pool_high_mb"=>raw_before.pool_high_mb),
            "raw_after"=>Dict("smi_mb"=>raw_after.smi_mb, "pool_current_mb"=>raw_after.pool_current_mb, "pool_high_mb"=>raw_after.pool_high_mb),
            "after_gc"=>Dict("smi_mb"=>cpt.smi_mb, "pool_current_mb"=>cpt.pool_current_mb, "pool_high_mb"=>cpt.pool_high_mb),
        ))
    end
elseif MODE == "single-long-turn"
    _log("═"^70)
    _log("CONTRÔLE : UN SEUL tour, max_new_tokens=250 (isole l'effet 'réponse longue' de l'effet 'historique multi-tours')")
    raw_before = checkpoint_raw("single_RAW_avant")
    res = chat_turn("integral x*exp(x), full worked solution with steps"; max_new_tokens=250)
    _log("  -> $(res.n_tokens) tokens générés, préfixe=$(res.prefix_len) tok, $(round(res.dt,digits=1))s, " *
         (res.stopped_on_eos ? "arrêt sur EOS" : "tronqué à max_new_tokens"))
    _log("  Réponse : $(res.reply[1:min(end,300)])" * (length(res.reply) > 300 ? "[...]" : ""))
    raw_after = checkpoint_raw("single_RAW_apres_AVANT_GC")
    cpt = checkpoint("single_apres_GC")
    push!(turn_records, Dict(
        "turn"=>1, "prompt"=>"integral x*exp(x), full worked solution with steps", "reply"=>res.reply, "n_tokens"=>res.n_tokens,
        "prefix_len"=>res.prefix_len, "dt"=>res.dt, "stopped_on_eos"=>res.stopped_on_eos,
        "raw_before"=>Dict("smi_mb"=>raw_before.smi_mb, "pool_current_mb"=>raw_before.pool_current_mb, "pool_high_mb"=>raw_before.pool_high_mb),
        "raw_after"=>Dict("smi_mb"=>raw_after.smi_mb, "pool_current_mb"=>raw_after.pool_current_mb, "pool_high_mb"=>raw_after.pool_high_mb),
        "after_gc"=>Dict("smi_mb"=>cpt.smi_mb, "pool_current_mb"=>cpt.pool_current_mb, "pool_high_mb"=>cpt.pool_high_mb),
    ))
else
    error("MODE inconnu : $MODE (attendu \"multiturn\" ou \"single-long-turn\")")
end

peak = maximum(r.smi_mb for r in results)
println("\n", "═"^78)
println("RÉSUMÉ FINAL (mode=$MODE)")
println("═"^78)
for r in results
    println("  $(rpad(r.label,40))  smi=$(round(r.smi_mb,digits=0))MB  pool_current=$(round(r.pool_current_mb,digits=0))MB  pool_high=$(round(r.pool_high_mb,digits=0))MB")
end
println("PIC VRAM (nvidia-smi) : $(round(peak,digits=0)) MB")

out_suffix = MODE == "multiturn" ? "" : "_$(MODE)"
outfile = joinpath(@__DIR__, "diag_multiturn_vram_growth$(out_suffix)_results.json")
open(outfile, "w") do io
    JSON.print(io, Dict(
        "mode"=>MODE,
        "checkpoints"=>[Dict("label"=>r.label, "smi_mb"=>r.smi_mb, "pool_current_mb"=>r.pool_current_mb, "pool_high_mb"=>r.pool_high_mb) for r in results],
        "turns"=>turn_records,
        "peak_vram_mb"=>peak,
    ), 2)
end
_log("Écrit -> $outfile")
flush(stdout)
