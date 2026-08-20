# Pre-registration: per-character loss distribution as a single-checkpoint test of the quantization model

Written before running any part of `notebook/bench_quantization_signature_check.jl`. This is the
untried alternative flagged in `artilce/article3.tex` (around line 639): the `-1/3`
Euler-Maclaurin constraint on `a(L)` was found STRUCTURALLY UNTESTABLE (see
`notebook/growth_constraint_test_results.json` / `notebook/analyze_growth_constraint.jl`) because
it lives entirely in a temporal-extrapolation regime where the model's own asymptotic expansion is
either inaccurate (small `L`) or below the achievable noise floor (large `L`). This experiment
sidesteps that failure mode entirely: it is a single forward pass on a single fixed checkpoint, no
temporal extrapolation, no curve-fitting of `a(L)`/`b(L)` at all.

## 1. What is being tested

The quantization model (Michaud et al. 2023, `arXiv:2303.13506`, as used in `article3.tex`
§"An Exact Closed Form and a Falsifiable Constraint") claims a network of depth `L` has learned
the top `K(L) = μL` of `K_max` frequency-ranked competence "quanta", each with Zipf weight
`p_k ∝ 1/k`. The literal, structural reading of this claim is that **at a fixed checkpoint**,
individual predictions split into two populations: those touching an already-acquired quantum
(near-floor loss) and those touching a not-yet-acquired quantum (high loss) — a discrete mixture,
not a smoothly-varying continuum of difficulty. This is a claim about the *shape* of the
per-position loss distribution at one checkpoint, never checked anywhere in the existing growth
campaigns (all of which fit only the *mean* validation loss `a(L)`).

The central risk this pre-registration must guard against, named explicitly in the task: **any**
trained character-LM will show a heavy-tailed, non-Gaussian per-character loss distribution, for
entirely boring reasons unrelated to quantization — some characters are simply rarer (higher
surprisal under any model) and some context positions have less causal context (position 1 of a
window has only itself to go on). A bare "is this multimodal?" test would almost certainly return
yes on any network, trained or not, real quantization or not. Sections 4-5 below build the
controls this design needs to be informative rather than a foregone conclusion.

## 2. Architecture, task, depths (reusing the existing campaign for comparability)

Identical protocol to `notebook/growth_theory_constraint_test.jl` / `notebook/growth_jalon0.jl`,
so results sit on the same footing as `growth_constraint_test_results.json` and the 16-depth fit
in `notebook/growth_theory_README.md`:

- Task: character-level LM on TinyShakespeare (65-char vocab), `BLOCK_SIZE=256`.
- Architecture: `dim=256`, `n_heads=4`, `hidden_dim=512`, `LlamaModel(n_layers, ...; batched_attn=true)`.
- Optimizer: AdamW, `lr=1e-3`, `adamw_step_batched!`, batch size 1 (one window per step), exactly
  as `train_growth_arm!` in `growth_jalon0.jl`.
- Depths: **L ∈ {1, 4, 16}** — shallow / mid / deep, all three already characterized in both the
  original 16-depth fit and the 9-depth constraint campaign, chosen for direct comparability and
  to span the depth range where `a(L)` is known to differ substantially (`a(1)≈1.57`,
  `a(4)≈1.44`, `a(16)≈1.28` in the retained log fit).
- Steps per depth: matching the constraint-campaign convention — 20,000 steps for L∈{1,4},
  10,000 for L=16 (single-stage schedule `[L]`, i.e. plain fixed-depth training via
  `train_growth_arm!([L], n_steps*L; ...)`, no `insert_block!` growth event).
  Confirmed by direct check: **no trained checkpoints (weights) exist anywhere in the repo** —
  `growth_constraint_test_results.json` stores only scalar `val_history` (mean loss vs step), not
  model weights — so all 6 runs below are trained from scratch for this experiment.
- Seeds: **2 per depth** (6 training runs total), a deliberate reduction from the 4-seed campaign
  convention to fit this session's compute budget. The primary statistical test (per-position
  mixture fit) operates on ~10^5 positions from a single checkpoint and does not need multiple
  seeds for its own power; 2 seeds exist only to check the depth-comparison result (§5) is not a
  one-off training accident, not to average away noise on the primary test.

## 3. Held-out measurement and the per-position loss computation

- Held-out data: `val_ids`, the last 10% of TinyShakespeare, exactly as in `growth_jalon0.jl`
  (`val_loss` function) — never seen during training of any run in this project.
- Windows: **non-overlapping** `BLOCK_SIZE=256` chunks tiling `val_ids` (`floor(length(val_ids)/256)`
  windows), rather than the 64 randomly-spaced windows `val_loss` uses for a scalar validation
  metric. Non-overlapping tiling gives each held-out character exactly one loss sample (no
  duplicate measurement of the same character at multiple context lengths), producing a real
  empirical distribution over ~10^5 independent-ish positions per checkpoint.
- Per-position loss: `src/kernels.jl:760` (`cross_entropy_loss`) is **not modified**. Its formula
  is replicated externally in the new script only: fetch the `:logits`-equivalent symbol via
  `NeuroDSL.demand_release!(g, logits_sym; namespace=ns)`, then for each row `i` (position) with
  label `y_i`,
  ```
  p_i = softmax(logits_i)
  loss_i = -log(max(p_i[y_i], 1f-10))
  ```
  — the exact same softmax/clip formula as the kernel, just without the final `mean` reduction.
  This produces one loss value per held-out character position, per checkpoint.

## 4. Primary statistical test (pre-registered, not eyeballed)

**Test: 1-component vs 2-component univariate Gaussian-mixture BIC comparison**, fit by
expectation-maximization implemented from scratch in the new script using only packages already in
`Project.toml` (`Statistics`, `LinearAlgebra`, `Random`) — no `Project.toml` change, no
`Distributions.jl`/`HartiganDipTest.jl` dependency. Hartigan's dip test was considered but has no
available implementation in this project's dependency set and was ruled out rather than adding a
package.

For each checkpoint's per-position loss vector `x` (raw nats, not log-transformed):
- Fit `k=1` (closed form: sample mean/variance) and `k=2` (EM, ≥20 random restarts, variance floor
  `1e-4 * var(x)` to block degenerate single-point components) univariate Gaussian mixtures.
- `BIC(k) = -2·logL(k) + (3k-1)·ln(n)`.
- **A checkpoint shows a "multimodal signature"** iff *all three* hold:
  1. `ΔBIC = BIC(2) - BIC(1) < -10` (Kass–Raftery "very strong" evidence, not just nominally
     favorable — this project's own `-1/3` campaign was burned once by a nominally-favorable
     `ΔAIC=-10.1` that collapsed under robustness checks, so the bar here is deliberately strict
     and applied with no post-hoc softening).
  2. The two fitted components are well-separated: Ashman's D,
     `D = |μ2-μ1| / sqrt((σ1²+σ2²)/2) > 2` (standard cleanly-bimodal threshold — rules out a
     "2-component" fit that is really one blob plus a sliver).
  3. Non-degenerate mixing weights: both components have weight in `[0.05, 0.95]` (rules out
     a fit that is essentially 1-component with 1% of mass parked in an outlier bucket).

## 5. Confound control — the part that decides whether this means anything

Two "boring difficulty" explanations must be ruled out *before* any multimodal signature is
attributed to quantization:

**(a) Target-character frequency.** Rare characters are intrinsically harder to predict for any
model, any capacity, quantized or not. Control: compute each character's unigram log-frequency
from `train_ids` (never `val_ids`), `f(c) = -ln(count(c)/total)`. This is what a context-free
unigram model's loss on `c` would be — the "boring" null.

**(b) Context length.** Position 1 of every non-overlapping window has only itself as context;
position 256 has the full window. This is a training/eval-protocol artifact, not a model property.
Control: `position_in_window` (1..256).

**Procedure:** OLS-regress `loss_i ~ 1 + f(target_char_i) + ln(position_in_window_i)` on the full
per-position vector, take residuals, and **re-run the exact §4 test (same three criteria, same
thresholds) on the residuals**. This directly operationalizes "is there discrete structure beyond
ordinary per-character difficulty and context-length effects" rather than testing a strawman.

## 6. Depth-comparison signature (secondary, tests the *acquisition* story specifically)

The quantization model doesn't just predict *a* mixture — it predicts the high-loss ("not yet
acquired") component's weight should **shrink monotonically with depth** (more capacity → more
quanta learned), and that the *same* positions should graduate from the high-loss bucket to the
low-loss bucket as depth grows (an ordering/nesting property), not that the bucket membership
churns randomly.

Using GMM responsibility > 0.5 to assign each of the (same, depth-independent) held-out positions
to "high-loss" or "low-loss" component at each depth:
- Report the high-loss-component weight at L=1, 4, 16 and check monotonic decrease.
- Report the Jaccard overlap of the high-loss position-sets between (L=1,L=4), (L=4,L=16),
  (L=1,L=16), compared against the overlap expected under independent random membership at the
  observed weights (`w_i·w_j / (w_i+w_j-w_i·w_j)`). Overlap clearly above this baseline is
  consistent with "graduation" (nested acquisition); overlap at or below baseline indicates the
  high-loss bucket reshuffles with depth rather than shrinking around a stable core, i.e. no
  acquisition ordering.
- This check is run once per depth per seed (2 seeds) to flag whether it is training-seed noise or
  a reproducible property of the architecture at that depth; it is not itself subject to the
  strict §4 significance bar (there is no established test for it) and is reported descriptively.

## 7. Verdict criteria, fixed before running

- **SUPPORTS the quantization picture** if: the primary test (§4) fires (multimodal signature) at
  ≥2 of the 3 depths, on RAW loss, **and** the signature survives the frequency+position control
  (§5) — i.e. the residual test also fires at those same depths — **and** the high-loss weight
  decreases from L=1 to L=16 in both seeds **and** the graduation overlap (§6) exceeds the random
  baseline for at least the (L=1,L=16) pair in both seeds.
- **REFUTES / negative** if: the primary test (§4) does not fire (no `ΔBIC<-10` + separated +
  non-degenerate 2-component fit) at *any* of the 3 depths on raw loss — i.e. the per-position
  loss distribution is statistically indistinguishable from a single (possibly skewed, but
  unimodal-fit-adequate) component at every depth tested. This would mean no detectable discrete
  structure at all, at the scale/architecture tested.
- **INCONCLUSIVE** if any of: (i) the primary test fires on raw loss but is fully explained away
  by the frequency+position control (residual test does not fire) — indistinguishable from
  ordinary per-character/context difficulty, not evidence of quantization; (ii) the primary test
  fires and survives the control, but the depth-monotonicity or graduation-overlap checks (§6) are
  inconsistent across seeds or depths (mixture exists, but the specific "acquisition" story the
  model requires is not corroborated); (iii) EM fails to converge to a stable, restart-robust
  fit at one or more depths.

## 8. Explicit acknowledgment of what this test can and cannot show

Even a clean "SUPPORTS" verdict under §7 would **not** establish that the discrete structure is
Zipf-exponent-1, or that `μ` (quanta/layer) is identified, or validate the specific `a1a3/a2²=-1/3`
functional form — none of that is tested here. It would only establish that per-position loss at a
fixed checkpoint is better described by a small number of discrete components than by one smooth
distribution, *net of* frequency and context-length confounds, and that component membership
shifts with depth in the direction and pattern the model predicts. That is a weaker, but real and
previously entirely untested, structural claim of the quantization picture — exactly the trade the
`article3.tex` §"An Exact Closed Form..." passage describes ("trade `-1/3` itself for a more
measurable invariant of the same underlying model").

Results, raw numbers, and code: `notebook/bench_quantization_signature_check.jl` and
`notebook/bench_quantization_signature_check_results.txt`.
