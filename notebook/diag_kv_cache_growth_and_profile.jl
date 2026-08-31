# ══════════════════════════════════════════════════════════════════════════════
# diag_kv_cache_growth_and_profile.jl — DEUX vérifications demandées pour la
# revue algorithmique du cache KV, sur le chemin de production FINAL (namespace
# unique + alias_tied_param! + load_graph!/LlamaModel corrigés, même graphe
# que diag_kv_cache_final_correctness_vram.jl) :
#
# (A) "Fusil fumant" O(n)/O(n²) : `kv_cache_append!` (src/kv_cache.jl) copie
#     l'historique COMPLET (`output_buffer[1:cur_step-1,:] .= hist`) dans un
#     tampon FRAÎCHEMENT alloué à CHAQUE pas (design documenté explicitement
#     dans kv_cache.jl comme "correct d'abord, zero-copy plus tard si mesuré
#     nécessaire") -- génère 220 tokens (ignore l'arrêt sur EOS, seule la
#     MÉCANIQUE de position compte ici, pas la cohérence du texte) et compare
#     le temps/token en début vs. milieu vs. fin de séquence.
#
# (B) Profil du pas de décodage : décompose le temps d'UN pas en régime établi
#     (~position 100) en attention/MLP/reste, en appelant `demand!`
#     PROGRESSIVEMENT sur des nœuds de plus en plus profonds du graphe déjà
#     construit (checkpoints par couche : sortie attention `layer_i_dec_res1`,
#     sortie bloc complet `layer_i_dec_out`), chacun entouré de `CUDA.@sync`
#     pour une mesure fiable (ajoute des points de synchronisation qui
#     n'existent PAS dans le chemin chaud normal -- décompose la même somme de
#     travail GPU en segments observables, au prix d'un peu de synchronisation
#     supplémentaire ; le total recombiné est comparé à la mesure bout-en-bout
#     normale pour vérifier qu'il n'introduit pas de biais grossier).
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
_log("Préfixe traité ($(length(prefix)) tok), début génération longue (220 pas, EOS IGNORÉ -- seule la mécanique du cache compte).")

# ── (A) Fusil fumant : temps/token vs POSITION ──────────────────────────────
const N_GEN = 220
gen_times = Float64[]
for step in 1:N_GEN
    nxt0 = argmax(logits_row) - 1   # ignore EOS volontairement
    global cur_step += 1
    ts = time()
    global logits_row = cached_step!(nxt0, cur_step)
    push!(gen_times, time() - ts)
    if step % 50 == 0
        _log("  ... pas $step / $N_GEN (cur_step=$cur_step) : $(round(gen_times[end],digits=4))s")
    end
end

function bin_stats(times, lo, hi)
    seg = times[lo:hi]
    (mean=sum(seg)/length(seg), min=minimum(seg), max=maximum(seg))
end
bins = [(1,10,"début (pas 1-10, positions $(length(prefix)+1)-$(length(prefix)+10))"),
        (51,60,"début-milieu (pas 51-60)"),
        (101,110,"milieu (pas 101-110)"),
        (151,160,"milieu-fin (pas 151-160)"),
        (211,220,"fin (pas 211-220, position ~$(length(prefix)+220))")]
println("\n", "═"^78)
println("(A) TEMPS/TOKEN vs POSITION (cache KV, 220 pas, préfixe=$(length(prefix)) tok)")
println("═"^78)
bin_results = Dict{String,Any}[]
for (lo,hi,label) in bins
    st = bin_stats(gen_times, lo, hi)
    println("  $(rpad(label,55)) moy=$(round(st.mean,digits=4))s  min=$(round(st.min,digits=4))s  max=$(round(st.max,digits=4))s")
    push!(bin_results, Dict("label"=>label, "lo"=>lo, "hi"=>hi, "mean"=>st.mean, "min"=>st.min, "max"=>st.max))
end
first_bin_mean = bin_results[1]["mean"]
last_bin_mean = bin_results[end]["mean"]
growth_ratio = last_bin_mean / first_bin_mean
println("\nRatio (fin/début) = $(round(growth_ratio,digits=3))x -- ",
        growth_ratio > 1.15 ? "CROISSANCE mesurable avec la position (fusil fumant confirmé)." :
        "PLAT -- pas de croissance mesurable avec la position (le design O(n) par pas documenté dans kv_cache.jl n'a PAS d'impact mesurable à cette échelle de n_kv_heads/d_head/longueur).")

# ── (B) Profil d'UN pas de décodage en régime établi ────────────────────────
# IMPORTANT -- `:kv_cache_append` est un nœud À ÉTAT (`aux_data[:history]`,
# voir src/kv_cache.jl) : on NE PEUT PAS rejouer deux fois le MÊME cur_step
# (un second `invalidate_all!`+`demand!` sur le même pas trouverait
# `size(hist,1) == cur_step` au lieu de `cur_step-1` et lèverait une erreur
# explicite -- comportement voulu, garde-fou contre un cache corrompu). La
# mesure de référence bout-en-bout et la mesure segmentée portent donc
# chacune sur un pas RÉEL DIFFÉRENT (cur_step et cur_step+1, positions
# adjacentes en régime établi -- coût attendu quasi identique).
_log("\nProfilage d'un pas de décodage (position actuelle cur_step=$cur_step, régime bien établi)...")
nxt0 = argmax(logits_row) - 1
global cur_step += 1
NeuroDSL.set!(g, :dec_token_id, [nxt0 + 1]; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :dec_cur_step, Float32[cur_step]; namespace=ns)
NeuroDSL.set!(g, :dec_pos, Float32[cur_step-1]; namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)

CUDA.synchronize()
t_total0 = time()
logits_ref = Array(NeuroDSL.demand!(g, dec_logits; namespace=ns))
CUDA.synchronize()
t_total_baseline = time() - t_total0
_log("  Mesure de référence (1 seul demand! bout-en-bout, comme le chemin normal, pas cur_step=$cur_step) : $(round(t_total_baseline,digits=4))s")

# Pas SUIVANT (cur_step+1) pour la mesure segmentée -- même entrée logique
# (continuation gloutonne), position adjacente.
nxt1 = argmax(logits_ref[1, :]) - 1
global cur_step += 1
NeuroDSL.set!(g, :dec_token_id, [nxt1 + 1]; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :dec_cur_step, Float32[cur_step]; namespace=ns)
NeuroDSL.set!(g, :dec_pos, Float32[cur_step-1]; namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
_log("  Mesure segmentée : pas cur_step=$cur_step (position suivante).")
function run_segmented_profile!(g, ns, dec_logits, n_layers)
    CUDA.synchronize()
    t0 = time()
    NeuroDSL.demand!(g, :dec_tok_emb; namespace=ns)
    CUDA.synchronize()
    embed = time() - t0
    attn = 0.0; mlp = 0.0
    tprev = time()
    for i in 1:n_layers
        r1 = Symbol(:layer_, i, :_dec_res1)
        os = Symbol(:layer_, i, :_dec_out)
        NeuroDSL.demand!(g, r1; namespace=ns)
        CUDA.synchronize()
        tnow = time(); attn += (tnow - tprev); tprev = tnow
        NeuroDSL.demand!(g, os; namespace=ns)
        CUDA.synchronize()
        tnow = time(); mlp += (tnow - tprev); tprev = tnow
    end
    NeuroDSL.demand!(g, dec_logits; namespace=ns)
    CUDA.synchronize()
    norm_logits = time() - tprev
    return (embed=embed, attn=attn, mlp=mlp, norm_logits=norm_logits)
end
prof = run_segmented_profile!(g, ns, dec_logits, N_LAYERS)
seg_embed, seg_attn, seg_mlp, seg_norm_logits = prof.embed, prof.attn, prof.mlp, prof.norm_logits

seg_total = seg_embed + seg_attn + seg_mlp + seg_norm_logits
println("\n", "═"^78)
println("(B) PROFIL D'UN PAS DE DÉCODAGE (position cur_step=$cur_step, checkpoints par couche + CUDA.@sync)")
println("═"^78)
println("  Référence bout-en-bout (1 seul demand!, sans checkpoints intermédiaires) : $(round(t_total_baseline,digits=4))s")
println("  Somme segmentée (avec sync supplémentaire par couche)                    : $(round(seg_total,digits=4))s")
println("  Embedding (:dec_tok_emb)                    : $(round(seg_embed,digits=4))s  ($(round(100*seg_embed/seg_total,digits=1))%)")
println("  Attention (28 couches cumulées, :_dec_res1)  : $(round(seg_attn,digits=4))s  ($(round(100*seg_attn/seg_total,digits=1))%)")
println("  MLP       (28 couches cumulées, :_dec_out)   : $(round(seg_mlp,digits=4))s  ($(round(100*seg_mlp/seg_total,digits=1))%)")
println("  Norme finale + lm_head (:dec_logits)         : $(round(seg_norm_logits,digits=4))s  ($(round(100*seg_norm_logits/seg_total,digits=1))%)")

open(joinpath(@__DIR__, "diag_kv_cache_growth_and_profile_results.json"), "w") do io
    JSON.print(io, Dict(
        "gen_times"=>gen_times, "bins"=>bin_results, "growth_ratio_last_over_first"=>growth_ratio,
        "profile"=>Dict(
            "t_total_baseline"=>t_total_baseline, "seg_total"=>seg_total,
            "seg_embed"=>seg_embed, "seg_attn"=>seg_attn, "seg_mlp"=>seg_mlp, "seg_norm_logits"=>seg_norm_logits,
        ),
    ), 2)
end
_log("Écrit -> diag_kv_cache_growth_and_profile_results.json")
flush(stdout)
