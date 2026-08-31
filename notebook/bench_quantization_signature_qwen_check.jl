# ══════════════════════════════════════════════════════════════════════════════
# REAL-SCALE REPLICATION of the PRIMARY test from
# notebook/quantization_signature_preregistration.md /
# notebook/bench_quantization_signature_check.jl on Qwen2.5-1.5B-Instruct (real
# trained weights, real natural-language held-out corpus), per the
# pre-registration written BEFORE this run:
# notebook/quantization_signature_qwen_preregistration.md
#
# The toy character-level version of this test (L in {1,4,16}, TinyShakespeare)
# returned a clean REFUTATION: the 1-vs-2-component Gaussian-mixture BIC test
# never fired at any depth (Ashman's D topped out at 1.845 vs the required
# D>2). This script asks whether that refutation holds at real production scale
# on real, held-out (freshly written, non-memorized) natural language, or was a
# toy-model/tiny-dataset artifact.
#
# Only the PRIMARY single-checkpoint test applies here (there is only one Qwen
# checkpoint in this repo, not a depth sweep, so the original's SECONDARY
# depth-graduation test does not apply and is not attempted).
#
# src/kernels.jl is NOT modified and its cross_entropy_loss is not even called:
# only raw logits (:lm_head_out) are needed. The exact softmax/-log/clip
# formula is replicated externally, without the final mean reduction, exactly
# as in the toy experiment's per_position_losses().
#
# No backward pass anywhere in this script -- forward-only, per the
# pre-registration (cheaper and all that's needed for per-position loss).
#
# COORDINATION: another real background job (a corpus-size training sweep) may
# be running on this same GPU. This script checks free VRAM before the
# checkpoint load and periodically during the run, and calls GC.gc()+
# CUDA.reclaim() between windows to keep its own footprint close to its real
# working set (repeated invalidate_all!+demand! cycles otherwise accumulate
# orphaned GPU buffers that Julia's CPU-driven GC does not collect promptly --
# confirmed during this experiment's own calibration run).
#
# USAGE : julia --project=. notebook/bench_quantization_signature_qwen_check.jl
# ÉCRIT : notebook/bench_quantization_signature_qwen_check_results.txt
#         notebook/bench_quantization_signature_qwen_check_results.json
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, Printf, Statistics, Random, LinearAlgebra
using CUDA

const MODEL_DIR   = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT        = joinpath(MODEL_DIR, "qwen2_neurodsl")
const CORPUS_TOK_F = joinpath(@__DIR__, "quantization_signature_qwen_corpus_tokens.json")
const FREQREF_F    = joinpath(@__DIR__, "quantization_signature_qwen_freqref_tokens.json")
const VOCAB_F       = joinpath(MODEL_DIR, "vocab.json")
const OUT_TXT      = joinpath(@__DIR__, "bench_quantization_signature_qwen_check_results.txt")
const OUT_JSON     = joinpath(@__DIR__, "bench_quantization_signature_qwen_check_results.json")

# ── Window design (set from notebook/bench_quantization_signature_qwen_calibration_results.txt,
#    see notebook/quantization_signature_qwen_preregistration.md §4c) ─────────
const WINDOW_LEN   = 512   # chosen from calibration (see notebook/bench_quantization_signature_qwen_calibration_results.txt):
                            # W=256 -> 2.15s (8.41ms/token), W=512 -> 2.47s (4.81ms/token, SUB-linear,
                            # more efficient per token than W=256), W=1024 -> 330.9s (323.1ms/token,
                            # a 38x-vs-linear / 134x-vs-W=512 blowup, VRAM down to 445 MiB free
                            # afterward). W=1024 is rejected outright: both far too slow for this
                            # token budget and dangerously close to VRAM exhaustion, a real coordination
                            # risk with the other job sharing this GPU. W=512 is the sub-linearly
                            # efficient sweet spot actually measured on this hardware.
const GC_EVERY      = 4     # reclaim() every this many windows (keeps footprint bounded; the
                            # calibration run showed measurable VRAM growth even at fixed shape when
                            # forward passes are repeated without intervening reclaim())
const RNG_ANALYSIS_SEED = 20260820

vram_line() = strip(read(`nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader`, String))
free_vram_mib() = parse(Int, strip(read(`nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`, String)))

const MIN_FREE_VRAM_MIB = 3000   # back off / wait if headroom for the other job gets this tight
function ensure_vram_headroom!(io; wait_s=15, max_wait_s=600)
    waited = 0
    while free_vram_mib() < MIN_FREE_VRAM_MIB
        if waited == 0
            println(io, "  [VRAM] headroom below $(MIN_FREE_VRAM_MIB) MiB -- backing off, waiting for the other job...")
            println("  [VRAM] headroom below $(MIN_FREE_VRAM_MIB) MiB -- backing off, waiting for the other job...")
            flush(io)
        end
        sleep(wait_s); waited += wait_s
        waited >= max_wait_s && error("VRAM headroom did not recover after $(max_wait_s)s -- aborting rather than risk contention")
    end
    waited > 0 && (println(io, "  [VRAM] headroom recovered after $(waited)s, proceeding."); println("  [VRAM] headroom recovered after $(waited)s, proceeding."); flush(io))
end

# ══════════════════════════════════════════════════════════════════════════════
# Statistical machinery -- COPIED VERBATIM from
# notebook/bench_quantization_signature_check.jl (gmm_em_2comp, gmm_1comp_ll,
# multimodality_test), so the primary test is byte-for-byte the same procedure
# run on the toy character-level checkpoints, per §5 of the pre-registration.
# ══════════════════════════════════════════════════════════════════════════════
function gmm_em_2comp(x::Vector{Float64}; n_restarts=20, max_iter=400, tol=1e-7,
                       rng=MersenneTwister(0))
    n = length(x)
    vtot = var(x)
    var_floor = max(1e-4 * vtot, 1e-10)
    best = nothing
    for _ in 1:n_restarts
        qs = sort(0.15 .+ 0.7 .* rand(rng, 2))
        mus = Float64[quantile(x, qs[1]), quantile(x, qs[2])]
        if abs(mus[1] - mus[2]) < 1e-6
            mus[2] += 1e-3
        end
        sigma2s = fill(vtot / 2, 2)
        weights = [0.5, 0.5]
        prev_ll = -Inf
        for _iter in 1:max_iter
            lp1 = @. -0.5 * log(2π * sigma2s[1]) - (x - mus[1])^2 / (2 * sigma2s[1])
            lp2 = @. -0.5 * log(2π * sigma2s[2]) - (x - mus[2])^2 / (2 * sigma2s[2])
            l1 = lp1 .+ log(weights[1] + 1e-300)
            l2 = lp2 .+ log(weights[2] + 1e-300)
            m = max.(l1, l2)
            s = exp.(l1 .- m) .+ exp.(l2 .- m)
            ll = sum(m .+ log.(s))
            r1 = exp.(l1 .- m) ./ s
            N1 = max(sum(r1), 1e-8); N2 = max(n - N1, 1e-8)
            r2 = 1.0 .- r1
            weights = [N1 / n, N2 / n]
            mus = [sum(r1 .* x) / N1, sum(r2 .* x) / N2]
            sigma2s = [max(sum(r1 .* (x .- mus[1]) .^ 2) / N1, var_floor),
                       max(sum(r2 .* (x .- mus[2]) .^ 2) / N2, var_floor)]
            if abs(ll - prev_ll) < tol * (abs(prev_ll) + 1e-10)
                prev_ll = ll
                break
            end
            prev_ll = ll
        end
        if best === nothing || prev_ll > best.ll
            best = (weights=weights, mus=mus, sigma2s=sigma2s, ll=prev_ll)
        end
    end
    return best
end

gmm_1comp_ll(x::Vector{Float64}) = begin
    mu = mean(x); s2 = max(var(x), 1e-10); n = length(x)
    sum(@. -0.5 * log(2π * s2) - (x - mu)^2 / (2 * s2))
end

function multimodality_test(x::Vector{Float64}; rng)
    n = length(x)
    ll1 = gmm_1comp_ll(x)
    fit2 = gmm_em_2comp(x; rng=rng)
    bic1 = -2 * ll1 + 2 * log(n)
    bic2 = -2 * fit2.ll + 5 * log(n)
    dbic = bic2 - bic1
    mu1, mu2 = fit2.mus
    s1, s2 = sqrt.(fit2.sigma2s)
    D = abs(mu2 - mu1) / sqrt((s1^2 + s2^2) / 2)
    w1, w2 = fit2.weights
    nondeg = (0.05 <= w1 <= 0.95) && (0.05 <= w2 <= 0.95)
    fires = (dbic < -10) && (D > 2) && nondeg
    return (; n, ll1, ll2=fit2.ll, bic1, bic2, dbic, mu1, mu2, s1, s2, w1, w2, D, nondeg, fires, fit2)
end

# ── Per-position loss -- EXACT replica of src/kernels.jl:760 cross_entropy_loss's
#    softmax/-log/clip formula, WITHOUT the final mean reduction. src/kernels.jl
#    is not touched. ────────────────────────────────────────────────────────
function per_position_losses(logits_cpu::AbstractMatrix{Float32}, labels::AbstractVector{Int})
    n = size(logits_cpu, 1)
    losses = Vector{Float32}(undef, n)
    @inbounds for i in 1:n
        row = @view logits_cpu[i, :]
        m = maximum(row)
        e = exp.(row .- m)
        p = e ./ sum(e)
        losses[i] = -log(max(p[labels[i]], 1f-10))
    end
    return losses
end

# ── BPE confound control (§6 of the pre-registration): primary regressors are
#    log(token_id+1) [vocab-rank frequency proxy], token_byte_length, and
#    log(position_in_window). Secondary/descriptive: TinyShakespeare-based
#    empirical frequency (Laplace-smoothed), reported but not decisive. ───────
function residualize(losses::Vector{Float64}, X::Matrix{Float64})
    coef = X \ losses
    resid = losses .- X * coef
    return resid, coef
end

# ══════════════════════════════════════════════════════════════════════════════
# Execution
# ══════════════════════════════════════════════════════════════════════════════
open(OUT_TXT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))

    emit("REAL-SCALE quantization-signature test, Qwen2.5-1.5B-Instruct (real trained weights).")
    emit("Pre-registration: notebook/quantization_signature_qwen_preregistration.md")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("VRAM before checkpoint load : " * vram_line())
    ensure_vram_headroom!(io)

    # ── Load corpus tokens + metadata (0-indexed BPE ids -> 1-indexed) ────────
    tok_data = JSON.parsefile(CORPUS_TOK_F)
    all_ids0 = Int.(tok_data["token_ids"])              # 0-indexed, as produced by BPE encoder
    all_ids  = all_ids0 .+ 1                             # 1-indexed, for NeuroDSL token_ids
    all_nbytes = Int.(tok_data["token_nbytes"])
    n_corpus = length(all_ids)
    emit(@sprintf("Held-out corpus: %d BPE tokens (notebook/quantization_signature_qwen_corpus_full.txt)", n_corpus))

    freqref = JSON.parsefile(FREQREF_F)
    freqref_counts = Dict{Int,Int}(parse(Int, k) => v for (k, v) in freqref["token_counts"])
    freqref_total = freqref["n_tokens"]
    freqref_vocab_size_seen = length(freqref_counts)
    # Laplace-smoothed neg-log-frequency over the union of ids actually seen in
    # the held-out corpus (avoids building a 151,936-entry table for ids never used).
    laplace_neglogfreq(id0::Int) = -log((get(freqref_counts, id0, 0) + 1) / (freqref_total + freqref_vocab_size_seen))

    # ── Load Qwen checkpoint ──────────────────────────────────────────────────
    dev = NeuroDSL.Backend.CUDADevice()
    ns = :qwen2
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    t0 = time()
    NeuroDSL.load_graph!(g, ns, CKPT)
    dt_load = time() - t0
    GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && CUDA.reclaim()
    emit(@sprintf("Checkpoint loaded in %.2f s", dt_load))
    emit("VRAM after checkpoint load : " * vram_line())
    logits_sym = :lm_head_out
    @assert haskey(g.rules[ns], logits_sym) "expected :lm_head_out rule not found"

    function set_input!(ids::Vector{Int})
        n = length(ids)
        NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
    end

    # ── Non-overlapping windows tiling the corpus, TRUE next-token labels
    #    (requires one token beyond each window's own extent, from the corpus's
    #    own natural continuation -- never a fabricated label). Last incomplete
    #    tail dropped. ────────────────────────────────────────────────────────
    n_windows = (n_corpus - 1) ÷ WINDOW_LEN
    emit(@sprintf("Window length : %d tokens -> %d non-overlapping windows -> %d per-position loss samples",
                  WINDOW_LEN, n_windows, n_windows * WINDOW_LEN))
    @assert n_windows >= 1 "corpus too short for WINDOW_LEN=$WINDOW_LEN"

    all_losses = Float32[]; all_targets0 = Int[]; all_posidx = Int[]; all_nbytes_tgt = Int[]
    t_eval0 = time()
    for w in 1:n_windows
        ensure_vram_headroom!(io)
        start = (w - 1) * WINDOW_LEN + 1
        tokens = all_ids[start : start + WINDOW_LEN - 1]
        labels1 = all_ids[start + 1 : start + WINDOW_LEN]     # 1-indexed labels for per_position_losses
        set_input!(tokens)
        logits_val = NeuroDSL.demand_release!(g, logits_sym; namespace=ns)
        logits_cpu = Array{Float32}(Array(logits_val))
        append!(all_losses, per_position_losses(logits_cpu, labels1))
        append!(all_targets0, labels1 .- 1)                    # store 0-indexed BPE id for freq/nbytes lookup
        append!(all_posidx, collect(1:WINDOW_LEN))
        append!(all_nbytes_tgt, all_nbytes[start + 1 : start + WINDOW_LEN])
        if w % GC_EVERY == 0
            GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && CUDA.reclaim()
        end
        if w == 1 || w % max(1, n_windows ÷ 10) == 0 || w == n_windows
            emit(@sprintf("  window %d/%d done   (VRAM: %s)", w, n_windows, vram_line()))
        end
    end
    dt_eval = time() - t_eval0
    GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && CUDA.reclaim()
    n_pos = length(all_losses)
    losses = Float64.(all_losses)
    emit(@sprintf("\nExtraction complete: %.1fs, %d positions, mean=%.4f std=%.4f", dt_eval, n_pos, mean(losses), std(losses)))

    # ── §5 primary test on RAW loss ───────────────────────────────────────────
    rng_a = MersenneTwister(RNG_ANALYSIS_SEED)
    raw_test = multimodality_test(losses; rng=rng_a)
    emit(@sprintf("\n[RAW] n=%d  ΔBIC=%.3f  D=%.4f  w=(%.4f,%.4f)  mu=(%.4f,%.4f)  s=(%.4f,%.4f)  fires=%s",
                  raw_test.n, raw_test.dbic, raw_test.D, raw_test.w1, raw_test.w2,
                  raw_test.mu1, raw_test.mu2, raw_test.s1, raw_test.s2, raw_test.fires))

    # ── §6 BPE confound control -- PRIMARY: log(token_id+1) + byte_length + log(position) ──
    log_id = log.(Float64.(all_targets0) .+ 1.0)
    nbytes_f = Float64.(all_nbytes_tgt)
    log_pos = log.(Float64.(all_posidx))
    X_primary = hcat(ones(n_pos), log_id, nbytes_f, log_pos)
    resid_primary, coef_primary = residualize(losses, X_primary)
    rng_b = MersenneTwister(RNG_ANALYSIS_SEED + 1)
    resid_test = multimodality_test(resid_primary; rng=rng_b)
    emit(@sprintf("[RESIDUAL-primary] ΔBIC=%.3f  D=%.4f  w=(%.4f,%.4f)  fires=%s   coef(intercept,log_id,nbytes,log_pos)=%s",
                  resid_test.dbic, resid_test.D, resid_test.w1, resid_test.w2, resid_test.fires,
                  string(round.(coef_primary, digits=5))))

    # ── §6 secondary/descriptive: TinyShakespeare-based frequency control ─────
    neglogfreq_shk = [laplace_neglogfreq(id0) for id0 in all_targets0]
    X_secondary = hcat(ones(n_pos), neglogfreq_shk, nbytes_f, log_pos)
    resid_secondary, coef_secondary = residualize(losses, X_secondary)
    rng_c = MersenneTwister(RNG_ANALYSIS_SEED + 2)
    resid_test_secondary = multimodality_test(resid_secondary; rng=rng_c)
    emit(@sprintf("[RESIDUAL-secondary/TinyShakespeare-freq, descriptive only] ΔBIC=%.3f  D=%.4f  w=(%.4f,%.4f)  fires=%s   coef=%s",
                  resid_test_secondary.dbic, resid_test_secondary.D, resid_test_secondary.w1, resid_test_secondary.w2,
                  resid_test_secondary.fires, string(round.(coef_secondary, digits=5))))

    # ── §7 verdict ─────────────────────────────────────────────────────────────
    emit("\n" * "="^70 * "\nVERDICT (§7 of quantization_signature_qwen_preregistration.md)\n" * "="^70)
    verdict = if !raw_test.fires
        "REPLICATES THE REFUTATION : no multimodal signature on raw per-token loss (ΔBIC=$(round(raw_test.dbic,digits=2)), D=$(round(raw_test.D,digits=3))), same qualitative result as the toy character-level test."
    elseif raw_test.fires && resid_test.fires
        "REVERSES THE REFUTATION (supports quantization at real scale) : raw signature fires AND survives the primary BPE confound control."
    else
        "INCONCLUSIVE : raw signature fires but is explained away by the primary BPE confound control (residual test does not fire) -- consistent with ordinary token-granularity/frequency/position difficulty, not evidence of quantization."
    end
    emit("\n>>> " * verdict)

    results = Dict(
        "n_corpus_tokens" => n_corpus, "window_len" => WINDOW_LEN, "n_windows" => n_windows,
        "n_positions" => n_pos, "checkpoint_load_s" => dt_load, "eval_elapsed_s" => dt_eval,
        "mean_loss" => mean(losses), "std_loss" => std(losses),
        "raw_test" => Dict("dbic"=>raw_test.dbic, "D"=>raw_test.D, "w1"=>raw_test.w1, "w2"=>raw_test.w2,
                            "mu1"=>raw_test.mu1, "mu2"=>raw_test.mu2, "s1"=>raw_test.s1, "s2"=>raw_test.s2,
                            "bic1"=>raw_test.bic1, "bic2"=>raw_test.bic2, "fires"=>raw_test.fires),
        "residual_primary_test" => Dict("dbic"=>resid_test.dbic, "D"=>resid_test.D, "w1"=>resid_test.w1,
                                         "w2"=>resid_test.w2, "fires"=>resid_test.fires, "coef"=>coef_primary),
        "residual_secondary_test" => Dict("dbic"=>resid_test_secondary.dbic, "D"=>resid_test_secondary.D,
                                           "w1"=>resid_test_secondary.w1, "w2"=>resid_test_secondary.w2,
                                           "fires"=>resid_test_secondary.fires, "coef"=>coef_secondary),
        "verdict" => verdict,
    )
    open(OUT_JSON, "w") do jio
        JSON.print(jio, results)
    end
    emit("\nÉcrit : " * OUT_JSON)
end
println("\nÉcrit : ", OUT_TXT)
