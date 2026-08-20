# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION run for notebook/quantization_signature_qwen_preregistration.md,
# BEFORE any part of the real per-position-loss data collection. Measures real
# forward-pass wall-clock time (FORWARD ONLY -- no backward pass is needed for
# this experiment, since we only need logits -> per-position loss, never a
# gradient) at window lengths 256 / 512 / 1024 tokens on Qwen2.5-1.5B-Instruct,
# on THIS machine/GPU, to size the real run's window length and token budget
# instead of assuming linear scaling from the ~16-token / 336ms figure in
# bench_eps_vocab_projection_profile_results.txt (attention cost grows faster
# than linear in window length, so that number cannot be extrapolated safely).
#
# COORDINATION NOTE: another real background job (a corpus-size training sweep)
# is running on this same GPU. This script checks free VRAM via `nvidia-smi`
# immediately before loading the Qwen checkpoint, and again after, and prints
# both readings so the calibration log itself documents the headroom at the
# time it ran.
#
# USAGE : julia --project=. notebook/bench_quantization_signature_qwen_calibration.jl
# ÉCRIT : notebook/bench_quantization_signature_qwen_calibration_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, Printf, Statistics
using CUDA

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const TOKENS_F  = joinpath(@__DIR__, "quantization_signature_qwen_corpus_tokens.json")
const OUT       = joinpath(@__DIR__, "bench_quantization_signature_qwen_calibration_results.txt")

const WINDOW_LENGTHS = [256, 512, 1024]
const N_WARMUP = 2
const N_TIMING = 5

vram_line() = strip(read(`nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader`, String))

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))

    emit("CALIBRATION -- forward-pass ms vs window length, Qwen2.5-1.5B-Instruct, real GPU.")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("VRAM before checkpoint load : " * vram_line())

    tok_data = JSON.parsefile(TOKENS_F)
    all_ids = Int.(tok_data["token_ids"]) .+ 1   # 0-indexed BPE ids -> 1-indexed NeuroDSL token_ids
    emit(@sprintf("Held-out corpus tokens available : %d", length(all_ids)))
    @assert length(all_ids) >= maximum(WINDOW_LENGTHS) + 1 "corpus too short for largest calibration window"

    dev = NeuroDSL.Backend.CUDADevice()
    ns = :qwen2
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    t0 = time()
    NeuroDSL.load_graph!(g, ns, CKPT)
    emit(@sprintf("Checkpoint loaded in %.2f s", time() - t0))
    GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && CUDA.reclaim()
    emit("VRAM after checkpoint load : " * vram_line())

    logits_sym = :lm_head_out   # confirmed by inspecting g.rules[ns] below

    @assert haskey(g.rules[ns], logits_sym) "expected :lm_head_out rule not found in loaded graph"
    emit(@sprintf("Confirmed logits symbol :%s present in loaded graph.", logits_sym))

    function set_input!(ids::Vector{Int})
        n = length(ids)
        NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
    end

    function time_forward_window!(W::Int; warmup=N_WARMUP, n=N_TIMING)
        ids = all_ids[1:W]
        set_input!(ids)
        for _ in 1:warmup
            NeuroDSL.invalidate_all!(g; namespace=ns)
            NeuroDSL.demand!(g, logits_sym; namespace=ns)
            CUDA.synchronize()
        end
        ts = Float64[]
        for _ in 1:n
            NeuroDSL.invalidate_all!(g; namespace=ns)
            t0 = time_ns()
            NeuroDSL.demand!(g, logits_sym; namespace=ns)
            CUDA.synchronize()
            push!(ts, (time_ns() - t0) / 1e6)
        end
        return median(ts), ts
    end

    emit("\n" * "="^80)
    emit(@sprintf("FORWARD-ONLY timing (median of %d, +%d warmup), by window length", N_TIMING, N_WARMUP))
    results = Dict{Int,Float64}()
    for W in WINDOW_LENGTHS
        vram_pre = vram_line()
        med, ts = time_forward_window!(W)
        results[W] = med
        emit(@sprintf("  W=%4d tokens : median = %8.2f ms   samples(ms) = %s   [VRAM: %s]",
                      W, med, string(round.(ts, digits=1)), vram_pre))
    end

    emit("\n" * "="^80)
    emit("Scaling check (ms per token, and ratio vs linear-from-W=256 prediction):")
    base_W, base_ms = WINDOW_LENGTHS[1], results[WINDOW_LENGTHS[1]]
    for W in WINDOW_LENGTHS
        ms = results[W]
        linear_pred = base_ms * (W / base_W)
        emit(@sprintf("  W=%4d : %8.2f ms  (%.4f ms/token)   linear-from-W=%d prediction = %8.2f ms   actual/linear = %.3fx",
                      W, ms, ms / W, base_W, linear_pred, ms / linear_pred))
    end

    emit("\nVRAM after calibration : " * vram_line())
    emit("\nDone. See notebook/quantization_signature_qwen_preregistration.md for how these numbers")
    emit("are used to size the real run's window length and token budget.")
end
println("\nÉcrit : ", OUT)
