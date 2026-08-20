# Pre-registration: corpus-size sweep as a direct measurement of the Zipf exponent epsilon

Written and committed BEFORE any training run of this campaign. Follows article3.tex's own
untried suggestion (line 641-643): "a corpus-size sweep at fixed depth measures the Zipf
exponent epsilon directly on the data axis the quantization model actually predicts it for,
rather than through a(L)'s curvature."

## 1. The model, derived explicitly (not assumed)

article3.tex (line 423-435) derives, from the quantization model of neural scaling
(Michaud, Liu, Girit, Tegmark, "The Quantization Model of Neural Scaling", NeurIPS 2023,
arXiv:2303.13506), that a network of depth `L` has learned the top `K(L) = mu*L` of `K_max`
frequency-ranked competence quanta, with Zipf weights `p_k ∝ 1/k^{1+epsilon}` (epsilon=0 in
the paper's main derivation), giving

    a(L) = a_inf + c * sum_{k > K(L)} 1/k^{1+epsilon}
         = a_inf + c * [zeta(1+epsilon, K(L)+1) - zeta(1+epsilon, K_max+1)]     (article3, line 587)

which at epsilon=0 telescopes exactly to the digamma closed form and, in leading order, to the
retained log fit `a(L) = A - c*ln(L)`.

**Generalizing to the data axis.** Michaud et al.'s own paper derives the analogous result for
dataset size directly (confirmed by reading the paper: they define quantum-use frequency
`p_k ∝ k^{-(alpha+1)}` -- identical notation to article3's `epsilon`, i.e. `alpha_Michaud =
epsilon_article3` -- and give a threshold argument: a quantum k is "learned" once the training
set has presented it at least tau times. With D training samples, quantum k is presented
`~D*p_k` times, so it becomes learnable once `D*p_k >= tau`, i.e. once

    k <= K(D) = (D / (tau * zeta(1+epsilon)))^{1/(1+epsilon)}  =: mu_D * D^{1/(1+epsilon)}

This is the exact data-axis analogue of `K(L) = mu*L`: same threshold logic, same Zipf weights,
different resource variable, and -- critically -- a different EXPONENT relating the resource to
the number of learnable quanta (`D^{1/(1+epsilon)}` vs `L^1`). Plugging into the same residual-loss
sum gives

    a(D) = a_inf + c * [zeta(1+epsilon, K(D)+1) - zeta(1+epsilon, K_max+1)],  K(D) = mu_D * D^{1/(1+epsilon)}

**Two regimes, both derived here, not guessed:**

- **epsilon = 0 (pure Zipf, singular case, needs a cutoff exactly as in article3 line 588):**
  `1/(1+epsilon) = 1`, so `K(D) = mu_D * D` -- linear in D, exactly as `K(L) = mu*L` is linear
  in L. The harmonic telescoping is IDENTICAL to the depth case and gives, in leading (large-K)
  order, `a(D) = A_D - c*ln(D)` -- **a log-in-corpus-size law with the SAME slope magnitude c**
  as the depth-axis log-slope, because the leading digamma/harmonic asymptotic `psi(K+1) ~ ln(K)`
  only cares about K, and K is linear in the resource on both axes. Only the intercept differs
  (it absorbs `ln(mu_D)` vs `ln(mu)`).

- **epsilon > 0:** the tail sum has a standard closed asymptotic, `zeta(1+epsilon,K+1) ~
  K^{-epsilon}/epsilon` for `K>>1`, giving

      a(D) - a_inf ~ (c/epsilon) * (mu_D)^{-epsilon} * D^{-epsilon/(1+epsilon)}

  which is EXACTLY Michaud et al.'s own published result: predicted data-scaling exponent
  `alpha_D = alpha/(1+alpha)` with `alpha = epsilon`. This independent literature derivation
  matches the one worked out here from article3's own recurrence, which cross-validates both.

**Taylor-expanding the epsilon>0 form around epsilon=0** (same technique article3 line 589-593
already uses to explain the log/power-law near-degeneracy on the depth axis) gives the
practically fittable form used below:

    a(D) = A_0 + A_1*ln(D) + A_2*(ln D)^2 + O(epsilon^2),   A_1 = -c,   A_2 = c*epsilon/2

`A_1` is the SAME robust leading-order object as article3's `a_1 = -c` on the depth axis (does
not require knowing `a_inf`, `K_max`, or `mu_D` -- exactly the reason `c` was identifiable from
the depth fit while `mu` was not). `A_2` is the curvature term that carries `epsilon`, and it is
of the same structural kind as the depth axis's `a_2/L, a_3/L^2` terms: small, second-order, and
-- per the diagnosis below -- likely underpowered at achievable scale.

## 2. Design decisions, with justification

**(a) Fixed depth: L=6.** Chosen from the existing 9-depth, 4-seed, 20,000-step "constraint
test" campaign (`growth_constraint_test_results.json`, `growth_constraint_fit_results.json`),
which already gives a well-anchored `a(6) = 1.4239 +/- 0.0047` (naive pooled SE; see the honest
uncertainty note in (b) below) at exactly this step budget -- a free, independent cross-check
point at corpus fraction f=1 for the new campaign. L=6 is deliberately chosen PAST the
training-dynamics anomaly article3 documents at L=4-5 (line 646-658: latest plateau-escape time,
peak seed-to-seed variance) -- sweeping corpus size at an anomalous depth would confound "corpus
size effect" with "depth-localized optimization pathology," exactly the kind of hidden shared
confound this campaign must avoid (see the independence discussion in the final report).

**(b) Seeds and run length: 4 seeds x 20,000 steps per corpus fraction, val_every=250, same
`a(D) + b(D)/(n+n0)` (n0=3000) within-run extrapolation fit already validated in
`analyze_growth_constraint.jl`.** This deliberately repeats the depth axis's *signal-bearing*
convention (not the original 4000-step convention whose truncation bias article3 measured at
+0.055 to +0.114 nats -- larger than the effects being sought here). This is the "do not
repeat the short-run mistake" requirement: the mu-reconfirmation section of article3
(line 463-477) found that MORE SEEDS at a SHORT budget shrinks variance without widening the
n-range the fit sees, which is precisely what causes a/b trade-off contamination -- so run length
is matched to the existing long-run convention, not shortened for convenience.

Per-point uncertainty is NOT the naive pooled-OLS SE from `fit_ab`-style pooling (article3's own
diagnosis: pooling seed-level points as i.i.d. understates the true seed-cluster uncertainty by
up to 4x, `growth_constraint_fit_results.json` vs `growth_joint_fit_results.json`'s profile vs
cluster-bootstrap intervals). All uncertainty reported below uses a seed-cluster bootstrap
(resample which of the 4 seeds contribute, refit), never naive pooling.

**(c) Corpus grid.** Six log2-spaced fractions of the 90%-train split of TinyShakespeare
(`notebook/data/tinyshakespeare/input.txt`, confirmed 1,115,394 characters, char-level
tokenization, `stoi`/`encode` exactly as in `growth_jalon0.jl`):

| fraction | train chars | ln(D)  |
|----------|------------:|-------:|
| 1/32     |      31,370 | 10.354 |
| 1/16     |      62,740 | 11.047 |
| 1/8      |     125,481 | 11.740 |
| 1/4      |     250,963 | 12.433 |
| 1/2      |     501,927 | 13.126 |
| 1        |   1,003,854 | 13.819 |

Total log-range `ln(32) = 3.466`, wider than the depth axis's `ln(16) = 2.773` -- deliberately
chosen to maximize curvature-detection power per the argument in article3 line 590-593
(`epsilon*ln(range)` is the relevant small parameter; a wider range gives more leverage for the
same epsilon).

**Subsetting method.** For fraction f<1, at each of the 4 seeds, draw a uniformly random
contiguous span of the target length from the training region (offset drawn from a seed-derived
RNG, deterministic and resumable). This is deliberate: taking a fixed prefix (e.g. always the
first N characters) would confound "corpus size" with "which play/scene happens to be first in
the file" -- a real, well-known risk in TinyShakespeare, whose plays are far from
style-homogeneous. Randomizing the span per seed converts this into ordinary seed-level noise
that the bootstrap already accounts for, rather than a silent fixed bias shared by every corpus
size. At f=1 there is only one possible span (the whole training region), so this seed is
deterministic there -- expected, and a useful zero-randomness anchor.

**Held-out set.** Identical for every run: the last 10% of the corpus (`val_ids`, exactly as in
`growth_jalon0.jl`), never subsetted. This is what makes cross-corpus-size loss comparisons valid
-- everyone is scored on the same text.

**(d) Fitting procedure.**

1. Per corpus point, per seed: fit `a(D) + b(D)/(n+n0)` (n0=3000) by OLS on that seed's
   val_history, exactly `fit_ab`'s method in `analyze_growth_constraint.jl` but not yet pooling
   seeds.
2. Per corpus point: pool the 4 seeds' points for a point estimate `a_hat(D)` (same OLS-pooled
   method as the depth study, for a direct apples-to-apples number), AND separately compute a
   seed-cluster bootstrap SE (resample which seeds contribute, 1000 draws) -- report both, use
   ONLY the bootstrap SE for inference.
3. **Primary fit (well-powered, pre-registered as the main deliverable):** weighted linear fit
   `a_hat(D) = A_0 + A_1*ln(D)` across the 6 corpus points, weights `1/bootstrap_SE^2`. Report
   `c_D_hat = -A_1_hat` with its own bootstrap CI (resample seeds within each corpus point,
   refit, 1000 draws).
4. **Secondary fit (exploratory, pre-registered as likely underpowered -- see power calc below):**
   weighted quadratic fit `a_hat(D) = A_0 + A_1*ln(D) + A_2*(ln D)^2`. Report `Delta AIC` vs the
   linear model, leave-one-out stability (drop each of the 6 points), and a cluster bootstrap
   (same seed-resampling as above) fraction selecting the quadratic model and agreeing on
   `sign(A_2)>0`. If selected, extract `epsilon_hat = 2*A_2_hat / c_D_hat` (using the linear
   term's OWN slope from step 3 as the plug-in for c -- self-contained, does not require `a_inf`
   or `K_max`, unlike a full nonlinear Hurwitz-zeta fit, which this pre-registration explicitly
   rules out below as non-identifiable with 6 points, mirroring the depth axis's `mu` fiasco).

## 3. Power / detectability calculation, done BEFORE running

**Per-point noise floor.** The depth-axis campaign's own bootstrap-vs-naive comparison
establishes the pooled-OLS SE understates true SE by up to 4x; naive `ases` at the matched
20,000-step convention range 0.0047-0.0093 nats, so a realistic per-point bootstrap SE is
estimated at ~0.02-0.03 nats. Take 0.025 as the working number.

**Primary test power (c_D).** For 6 log2-spaced points, `sum((ln D_i - mean)^2) = 8.41`, so
`SE(A_1_hat) = SE_point / sqrt(8.41) = 0.025/2.90 ~= 0.0086` nats, i.e. `c_D_hat` should be
determined to roughly `+/-0.017` (2 SE). The target comparison band `[0.08, 0.13]` has width
0.05 -- comfortably larger than this precision. **Verdict: well-powered to detect a
qualitatively different c_D (e.g. near 0 or above 0.2), underpowered to finely discriminate
0.08 from 0.13 within the band.** This is registered as the expected outcome shape before
running: a useful coarse consistency check, not a precision measurement.

**Secondary test power (epsilon via A_2).** Using the depth study's own weak point estimate
`epsilon ~= 0.076` (article3 line 593) and `c ~= 0.10` as illustrative (not fitted) plug-ins:
`A_2 = c*epsilon/2 ~= 0.10*0.076/2 = 0.0038` per `(ln D)^2`. The component of this quadratic
term orthogonal to the `{1, ln D}` basis over these 6 points -- the only part a nested-model
comparison can see -- peaks near the range extremes at roughly `A_2 * (half-range)^2 ~= 0.0038 *
1.733^2 ~= 0.011` nats. This is the SAME order of magnitude as article3's own depth-curvature
orthogonal signal (0.005-0.011 nats, line 627-628), against a comparable per-point noise floor
(~0.02-0.03 nats here vs their per-depth SEs of similar size). **Verdict, registered explicitly
before running: the epsilon-via-curvature route on the corpus axis is expected to be similarly
underpowered to the already-failed depth-curvature route, for a structurally similar reason (the
signal is second-order in a small epsilon regardless of which axis carries the sweep). This
campaign is NOT expected to pin down epsilon precisely. It is expected to give a well-powered
`c_D` cross-check and, at best, a suggestive (not decisive) sign/magnitude read on epsilon.**
Reporting this honestly either way is the point.

## 4. Verdict criteria, fixed before fitting

**Primary (c_D vs c universality across resource axes):**
- **Consistent**: bootstrap 95% CI of `c_D_hat` overlaps `[0.08, 0.13]` (article3's own quoted
  plausible range across its three depth-based estimates).
- **Contradicts**: CI of `c_D_hat` excludes the generously doubled band `[0.04, 0.20]` entirely.
- **Inconclusive**: anything between (CI overlaps the doubled band but not the tight one, or is
  simply very wide).

**Secondary (epsilon via curvature) -- three criteria, ALL required for "confirmed", mirroring
`analyze_growth_constraint.jl`'s own three-criterion structure for direct comparability:**
1. `Delta AIC(quadratic - linear) < -6`, stable under every leave-one-out removal, selected in
   `>=90%` of bootstrap resamples.
2. `sign(A_2_hat) > 0` in `>=95%` of bootstrap resamples (predicted sign for epsilon>0).
3. Bootstrap 95% CI of `epsilon_hat` overlaps `[0, 0.15]` (a generous band around the depth
   study's own uncertain ~0.076 point value, doubled since that value itself never had a
   rigorous CI).
- **Confirmed**: all three pass.
- **Clean refutation**: criterion 1 passes (curvature is robustly real) but criterion 3 fails
  (wrong magnitude/sign) -- an informative, real finding.
- **Inconclusive**: criterion 1 fails (expected outcome per the power calculation above) --
  curvature is not robustly distinguishable from noise, so `A_2_hat`/`epsilon_hat` are not
  reported as a measurement, exactly as article3 itself does for its own nominal `R` estimate
  when its criteria fail (line 623-624).

## 5bis. AMENDMENT (2026-08-20, mid-campaign, before any fitting on real data)

After the first corpus point (f=1/32, all 4 seeds, full 20,000 steps) completed, inspection of
the raw val_history revealed a finding that invalidates the extrapolation method registered in
section 2, item (d)-1: at this smallest corpus size (31,370 characters), val loss reaches a
minimum around step 2,500-3,500 (2.44-2.54 nats, consistent across all 4 seeds) and then
DIVERGES -- monotonically WORSENING to 6.2-7.0 nats by step 20,000 (worse than the ~4.17-nat
uniform-random baseline for this vocabulary). This is severe overfitting: at 20,000 steps x 256
tokens/step, the corpus is seen roughly 160+ times, with a constant lr=1e-3 and no weight decay
(wd=0, the existing convention's own choice, inherited unchanged here). This is a genuinely
different regime from the depth axis's capacity-limited floor, where more steps only help (the
existing 16/9-depth campaigns never show this at full corpus, confirming it is a data-scarcity
effect, not a general training instability).

**Consequence:** the `a(D) + b(D)/(n+n0)` fit (design for a monotonically-decaying-to-floor
curve) is not meaningful at small D -- fitting it to a U-shaped curve returns whatever the linear
extrapolation of a mostly-increasing segment happens to produce, not a floor.

**Fix, decided now, before fitting anything else:** replace the per-run floor estimator with the
minimum of a smoothed val curve (5-point moving average over val_history, then take the argmin of
the smoothed sequence) -- well-defined and robust whether the true curve is monotone-decaying
(large D: the smoothed minimum sits at or near the final value, no change to results) or U-shaped
(small D: correctly recovers the pre-divergence floor instead of the meaningless late-training
extrapolation). The smoothing exists specifically to reduce the "pick the luckiest of ~80 noisy
points" selection bias a raw minimum would carry.

**What is NOT changed:** the step budget (20,000), val_every (250), seed count (4), corpus
fractions, and random-span subsetting are all left exactly as pre-registered, to preserve
comparability and because the already-completed f=1/32 runs remain valid raw data under the new
estimator (no re-run needed). Only the EXTRACTION method for `a(D)` from an already-collected
curve changes -- decided from data at only one corpus point, before any fit or verdict was
computed on the full grid, in the same spirit as this project's own history of catching and
documenting analysis-method problems (truncation bias, plateau escape) rather than silently
absorbing them into a final number.

**A substantive, pre-registrable prediction this amendment lets us test cheaply:** if the
quantization model is right that a(D) is a genuine architecture x data capacity floor, the
smoothed-minimum estimator should agree closely with the (valid, monotone) `a+b/(n+n0)`
extrapolation at the LARGE corpus fractions (where both methods should agree, since there is no
divergence to bias the naive fit) -- this is checked directly in the analysis and reported as a
sanity check on the new estimator, not assumed.

## 5. Independence from the depth-based estimate -- flagged in advance

This is registered here, before seeing results, as a structural concern to check honestly in the
final report: corpus size and depth both interact with the SAME training dynamics this project
has already shown distort short/finite-budget floor estimates (plateau-escape timing,
transient-vs-floor conflation). A very small corpus at 20,000 steps means many passes over the
same ~30K characters -- a memorization/overfitting regime that is mechanically different from the
depth axis's "network too shallow to represent all quanta" regime, even though the quantization
model predicts both should bottleneck through the same `K(resource)` object. If the smallest
corpus fractions show anomalously high seed-to-seed variance or non-monotone deviations (the same
symptom article3 found at L=4-5), that would be evidence the two axes are NOT cleanly independent
probes of the same underlying constant, and this will be reported as such rather than
silently averaged over.
