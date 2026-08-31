# ══════════════════════════════════════════════════════════════════════════════
# diag_decode_batched_attn_microbench.jl — isole la question "le batching
# gemm_strided_batched (déjà utilisé en training, src/kernels.jl) aiderait-il
# au DÉCODAGE (batch=1, 1 requête Q contre N clés K en cache) ?" AU NIVEAU
# CUDA BRUT, sans passer par le moteur de graphe NeuroDSL (donc sans le
# surcoût de dispatch Julia/Dict/Symbol de `demand!`) -- pour séparer deux
# hypothèses concurrentes :
#   (H1) le coût vient du LANCEMENT DE KERNEL CUDA lui-même (WDDM sur
#        Windows ajoute un surcoût de soumission par lancement) -> batcher
#        aiderait même sans changer le moteur de graphe.
#   (H2) le coût vient du DISPATCH du moteur de graphe NeuroDSL (Dict lookup,
#        appel de fonction générique par Symbol, allocation de GraphNode) ->
#        batcher les gemm CUDA n'aiderait quasi rien tant que le nombre de
#        NŒUDS DE GRAPHE ne change pas dans les mêmes proportions.
#
# Forme EXACTE du décodage (PAS celle du training) : Q a 1 SEULE ligne
# (nouveau token), K/V ont `cur_step` lignes (historique caché) -- gemm
# ASYMÉTRIQUE (M=1, N=cur_step), contrairement au self-attention training où
# Q et K partagent le même seq. `batched_qk_fwd!`/`batched_pv_fwd!` existants
# (src/kernels.jl:937-946, :971-980) supposent `size(q_heads[1],1) ==
# size(k_heads[1],1)` (un seul `seq` réutilisé pour reshaper Q3 ET K3) --
# INAPPLICABLES tels quels au décodage, il faudrait une variante M≠N. Ce
# script mesure d'abord SI ça vaudrait le coût de l'écrire.
#
# Additionnellement : les buffers K/V du cache (`kv_cache_append!`,
# src/kv_cache.jl) sont documentés comme alloués FRAIS À CHAQUE PAS (pas des
# vues zero-copy sur un tampon parent partagé) -- donc `_sibling_view_parent`
# ne reconnaîtra JAMAIS ces têtes comme tranches contiguës, et le chemin
# batché retomberait TOUJOURS sur `_gather3` (copie réelle) pour le
# décodage. Ce script mesure aussi le coût de ce gather.
#
# MÉTHODE DE MESURE (révisée après un 1er run où `time()`+sync PAR ITÉRATION
# quantifiait tout à ~1ms -- le coût du `CUDA.synchronize()` HÔTE lui-même
# dominait le signal) : `CUDA.@elapsed` sur un lot de N_REPS itérations
# consécutives SANS synchronisation intermédiaire (empile les lancements sur
# le stream, un seul `CUDA.@elapsed` mesure le tout via des événements CUDA
# -- résolution largement sub-milliseconde, pas de surcoût de sync hôte
# répété qui écraserait le signal qu'on essaie de mesurer).
# ══════════════════════════════════════════════════════════════════════════════

using CUDA, LinearAlgebra, Statistics, JSON

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=2))s] ", msg); flush(stdout))

CUDA.functional() || error("CUDA non fonctionnel -- ce microbench nécessite un vrai GPU.")
dev = CUDA.device()
_log("GPU: $(CUDA.name(dev))")

const N_HEADS = 12
const D_HEAD = 128
const CUR_STEP = 250   # position typique "régime établi" (voir diag_kv_cache_growth_and_profile.jl)
const N_LAYERS = 28
const N_BATCH_ITERS = 500   # lancements empilés sur le stream par mesure CUDA.@elapsed
const N_TRIALS = 8          # répétitions indépendantes de la mesure batchée (pour un IC)

q_heads = [CUDA.rand(Float32, 1, D_HEAD) for _ in 1:N_HEADS]
k_heads = [CUDA.rand(Float32, CUR_STEP, D_HEAD) for _ in 1:N_HEADS]
v_heads = [CUDA.rand(Float32, CUR_STEP, D_HEAD) for _ in 1:N_HEADS]
pr_heads = [CUDA.rand(Float32, 1, CUR_STEP) for _ in 1:N_HEADS]

sc_out_per_head = [CUDA.zeros(Float32, 1, CUR_STEP) for _ in 1:N_HEADS]
ao_out_per_head = [CUDA.zeros(Float32, 1, D_HEAD) for _ in 1:N_HEADS]

function per_head_qk!(out, qh, kh)
    for h in eachindex(qh)
        LinearAlgebra.mul!(out[h], qh[h], kh[h]')
    end
end
function per_head_pv!(out, prh, vh)
    for h in eachindex(prh)
        LinearAlgebra.mul!(out[h], prh[h], vh[h])
    end
end

function gather3(heads::Vector{<:CuArray}, a::Int, b::Int, H::Int, buf)
    for h in 1:H
        buf[:, :, h] .= heads[h]
    end
    return buf
end
Q3buf = CUDA.zeros(Float32, 1, D_HEAD, N_HEADS)
K3buf = CUDA.zeros(Float32, CUR_STEP, D_HEAD, N_HEADS)
P3buf = CUDA.zeros(Float32, 1, CUR_STEP, N_HEADS)
V3buf = CUDA.zeros(Float32, CUR_STEP, D_HEAD, N_HEADS)
qk3_buf = CUDA.zeros(Float32, 1, CUR_STEP, N_HEADS)
pv3_buf = CUDA.zeros(Float32, 1, D_HEAD, N_HEADS)

function batched_qk_decode!(q_heads, k_heads, out3)
    gather3(q_heads, 1, D_HEAD, N_HEADS, Q3buf)
    gather3(k_heads, CUR_STEP, D_HEAD, N_HEADS, K3buf)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'T', 1f0, Q3buf, K3buf, 0f0, out3)
    return out3
end
function batched_pv_decode!(pr_heads, v_heads, out3)
    gather3(pr_heads, 1, CUR_STEP, N_HEADS, P3buf)
    gather3(v_heads, CUR_STEP, D_HEAD, N_HEADS, V3buf)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', 1f0, P3buf, V3buf, 0f0, out3)
    return out3
end

# ── Mesure batchée robuste : CUDA.@elapsed sur N_BATCH_ITERS lancements
#    empilés sur le stream, répété N_TRIALS fois. ──────────────────────────
function bench_stacked(f::Function, n_iters::Int, n_trials::Int)
    CUDA.synchronize()
    f(); CUDA.synchronize()  # warm-up supplémentaire
    trial_times = Float64[]
    for _ in 1:n_trials
        t = CUDA.@elapsed begin
            for _ in 1:n_iters
                f()
            end
        end
        push!(trial_times, t / n_iters)
    end
    return trial_times   # secondes par itération, n_trials valeurs
end

_log("Warm-up (compile CUBLAS strided batched paths, mul! kernels, etc.)...")
per_head_qk!(sc_out_per_head, q_heads, k_heads)
per_head_pv!(ao_out_per_head, pr_heads, v_heads)
batched_qk_decode!(q_heads, k_heads, qk3_buf)
batched_pv_decode!(pr_heads, v_heads, pv3_buf)
CUDA.synchronize()

_log("Mesure QK (1 couche, H=$N_HEADS, cur_step=$CUR_STEP) -- $N_BATCH_ITERS itérations empilées x $N_TRIALS essais")
t_per_head_qk = bench_stacked(() -> per_head_qk!(sc_out_per_head, q_heads, k_heads), N_BATCH_ITERS, N_TRIALS)
t_batched_qk  = bench_stacked(() -> batched_qk_decode!(q_heads, k_heads, qk3_buf), N_BATCH_ITERS, N_TRIALS)

_log("Mesure PV (1 couche)")
t_per_head_pv = bench_stacked(() -> per_head_pv!(ao_out_per_head, pr_heads, v_heads), N_BATCH_ITERS, N_TRIALS)
t_batched_pv  = bench_stacked(() -> batched_pv_decode!(pr_heads, v_heads, pv3_buf), N_BATCH_ITERS, N_TRIALS)

# ── Mesure additionnelle : le gather SEUL (sans le gemm) -- pour voir
#    combien du coût "batché" est le gather (copie réelle) vs le gemm. ─────
function gather_only_qk!(q_heads, k_heads)
    gather3(q_heads, 1, D_HEAD, N_HEADS, Q3buf)
    gather3(k_heads, CUR_STEP, D_HEAD, N_HEADS, K3buf)
end
t_gather_only = bench_stacked(() -> gather_only_qk!(q_heads, k_heads), N_BATCH_ITERS, N_TRIALS)

fmt(x) = "$(round(1000*mean(x),digits=5))ms ± $(round(1000*std(x),digits=5))ms"
println("\n", "═"^78)
println("MICROBENCH -- 1 COUCHE (H=$N_HEADS têtes, d_head=$D_HEAD, cur_step=$CUR_STEP)")
println("(moyenne ± écart-type sur $N_TRIALS essais de $N_BATCH_ITERS itérations empilées)")
println("═"^78)
println("QK par-tête (boucle, comportement ACTUEL du décodage)  : $(fmt(t_per_head_qk))")
println("QK batché (gather+gemm_strided_batched)                : $(fmt(t_batched_qk))")
println("  dont gather seul (Q+K)                                : $(fmt(t_gather_only))")
println("PV par-tête (boucle, comportement ACTUEL du décodage)  : $(fmt(t_per_head_pv))")
println("PV batché (gather+gemm_strided_batched)                : $(fmt(t_batched_pv))")

per_head_total = mean(t_per_head_qk) + mean(t_per_head_pv)
batched_total  = mean(t_batched_qk) + mean(t_batched_pv)
speedup_per_layer = per_head_total / batched_total
println("\nTotal QK+PV par-tête (1 couche) : $(round(1000*per_head_total,digits=5))ms")
println("Total QK+PV batché   (1 couche) : $(round(1000*batched_total,digits=5))ms")
println("Speedup par couche               : $(round(speedup_per_layer,digits=3))x")
println("Extrapolé sur $N_LAYERS couches   : gain net = $(round(1000*N_LAYERS*(per_head_total-batched_total),digits=3))ms/token")

open(joinpath(@__DIR__, "diag_decode_batched_attn_microbench_results.json"), "w") do io
    JSON.print(io, Dict(
        "n_heads"=>N_HEADS, "d_head"=>D_HEAD, "cur_step"=>CUR_STEP, "n_layers"=>N_LAYERS,
        "n_batch_iters"=>N_BATCH_ITERS, "n_trials"=>N_TRIALS,
        "t_per_head_qk_ms"=>1000 .* t_per_head_qk, "t_batched_qk_ms"=>1000 .* t_batched_qk,
        "t_per_head_pv_ms"=>1000 .* t_per_head_pv, "t_batched_pv_ms"=>1000 .* t_batched_pv,
        "t_gather_only_ms"=>1000 .* t_gather_only,
        "speedup_per_layer"=>speedup_per_layer,
        "extrapolated_gain_ms_per_token"=>1000*N_LAYERS*(per_head_total-batched_total),
    ), 2)
end
_log("Écrit -> diag_decode_batched_attn_microbench_results.json")
