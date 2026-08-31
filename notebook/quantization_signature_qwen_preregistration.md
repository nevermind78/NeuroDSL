# Pre-registration: single-checkpoint per-position loss bimodality test on Qwen2.5-1.5B-Instruct

Written before running `notebook/bench_quantization_signature_qwen_check.jl` (the real data-collection
and statistical-test run). This replicates, at real production scale on real natural language, the
PRIMARY test from `notebook/quantization_signature_preregistration.md` /
`notebook/bench_quantization_signature_check.jl`, which ran on tiny character-level models
(L in {1,4,16}, TinyShakespeare) and returned a clean REFUTATION: the 1-vs-2-component Gaussian-mixture
BIC test never fired at any depth, with Ashman's D topping out at 1.845 against a required D>2.

Only the PRIMARY single-checkpoint test is replicated here. The prior experiment's SECONDARY
depth-graduation test (§6 of the original pre-registration) does not apply: there is only one Qwen
checkpoint available in this repository (`notebook/qwen2.5-1.5b-instruct/qwen2_neurodsl`), a single
trained model at a single fixed depth, not a depth sweep.

## 1. What is being tested (unchanged from the original)

The quantization model (Michaud et al. 2023) implies that, at a fixed checkpoint, individual
predictions split into a discrete mixture: positions touching an already-acquired competence "quantum"
(near-floor loss) versus positions touching a not-yet-acquired quantum (high loss) -- not a smoothly
varying continuum of difficulty. The original test built this exact hypothesis test on per-CHARACTER
loss from tiny from-scratch character-LMs. This experiment asks whether the same structural claim
shows up in per-TOKEN loss on a real 1.5B-parameter instruction-tuned model that has actually learned
natural language, rather than in a toy setting where the refutation could plausibly be attributed to
insufficient scale or insufficient training data.

## 2. Model and loading (already available, not re-derived)

- Model: Qwen2.5-1.5B-Instruct, real trained weights (never trained by NeuroDSL), already converted to
  NeuroDSL's native checkpoint format at `notebook/qwen2.5-1.5b-instruct/qwen2_neurodsl` (produced by
  `notebook/load_qwen2.jl`). Loaded via `NeuroDSL.load_graph!`, the same pattern used throughout this
  project's prior Qwen experiments (`bench_eps_vocab_projection_profile.jl`,
  `bench_eps_layer22_anomaly_profile.jl`).
- `src/kernels.jl`'s `cross_entropy_loss` (line 760) is NOT modified and not even called: this
  experiment only needs the raw logits (`:lm_head_out`), never the mean-reduced loss, since per-position
  loss is required, not a scalar. The exact softmax/-log/clip formula from `cross_entropy_loss` is
  replicated externally, without the final `mean` reduction:
  ```
  p_i = softmax(logits_i)
  loss_i = -log(max(p_i[y_i], 1f-10))
  ```
- No backward pass is used anywhere in this experiment. Only forward passes to `:lm_head_out` are
  needed to obtain per-position loss, which is both simpler and considerably cheaper than the
  backward-pass-heavy protocols used in this project's ablation-cost benchmarks.

## 3. Held-out corpus

**Choice: freshly written, original text, composed for this experiment, spanning 26 short sections
across markedly different natural-language registers** (narrative fiction x2, hard-science explainers
x2, casual chat/dialogue transcript, a recipe, sports commentary, a historical essay, a tech product
review, a philosophical essay, a travel diary, a mock news article, a business memo, a nature
observation, an instructional how-to, a technical tutorial, a memoir essay, a weather forecast,
furniture assembly instructions, an interview transcript, a restaurant review, an op-ed, gardening
advice, HOA meeting minutes, a fantasy short story, a personal letter, and a plain-language legal
explainer). Files: `notebook/quantization_signature_qwen_corpus.txt` (sections 1-13) and
`notebook/quantization_signature_qwen_corpus_part2.txt` (sections 14-26), concatenated into
`notebook/quantization_signature_qwen_corpus_full.txt`.

**Why fresh text rather than an existing public-domain corpus.** No local text corpus in this repo
other than TinyShakespeare (character-level, already used for the toy version of this test and heavily
memorized by large instruction-tuned LLMs, which would risk contaminating per-token loss with
memorization artifacts rather than measuring ordinary predictive difficulty) was available. Classic
public-domain sources (Project Gutenberg texts, famous historical documents, well-known speeches) are
exactly the kind of "extremely famous/memorized passage" the task explicitly warns against: Qwen2.5 was
trained on a huge, largely undocumented web-scale corpus, and canonical, widely-reproduced English texts
are disproportionately likely to appear verbatim or near-verbatim in that pretraining data. Memorized
text produces a distinctive risk for this specific test: long verbatim-memorized stretches would show
near-zero loss essentially independent of ordinary difficulty, while non-memorized text would show the
usual smooth difficulty gradient -- exactly the kind of artificial two-population split (memorized vs.
not) that could produce a spurious "bimodal" signature having nothing to do with quantized capability
acquisition. Writing fresh text sidesteps this risk entirely: this text did not exist before this
experiment was run and cannot be memorized by a model with training data that predates it.

**Honest limitation.** "Fresh" is not the same claim as "matched in difficulty/style to Qwen's
pretraining distribution" -- this text was written by Claude (the assistant conducting this experiment)
in a deliberately plain, varied-register style, not sampled from the wild. It is a reasonable
approximation of ordinary natural language across many registers, and multiplicity of register/topic is
itself a deliberate design choice (a single register, e.g. all narrative fiction, would confound
register-specific predictability with any putative quantization signature). But it is not a random
sample of real-world text, and a skeptical reader should weight this result as "does the effect show up
on real-scale natural language reasonably distinct in kind from the toy Shakespeare corpus," not as "on
a rigorously IID sample of the true pretraining distribution." This limitation is symmetric with respect
to the SUPPORTS/REFUTES outcome -- it does not bias the test toward either conclusion, since nothing
about freshly written, varied-register prose is expected a priori to be more or less "quantized" in its
predictability than any other natural text.

**Tokenization.** No `transformers`/`tokenizers` Python package is installed in this environment (both
`import transformers` and `import tokenizers` fail). Reused the hand-built, gate-validated byte-level BPE
encoder from `notebook/qwen_tokenize_prompts.py` (validated against the 3 reference tokenizations in
`adhoc_prompts.json`, produced by the real HF tokenizer, before trusting any output) in a new script,
`notebook/qwen_tokenize_quantization_corpus.py`. The `regex` package (providing genuine Unicode
`\p{L}`/`\p{N}` support, not the ASCII fallback) is installed and used, so no approximation is in play
beyond what the original, already-validated encoder used for prior Qwen experiments in this repo.

**Realized size:** 70,309 characters -> **14,977 BPE tokens** (`notebook/quantization_signature_qwen_corpus_tokens.json`).

## 4. Sample size / window design

### 4a. Calibration (measured, not assumed)

Measured forward-pass-only wall-clock time (no backward pass needed anywhere in this experiment) at
window lengths 256 / 512 / 1024 tokens, on this machine's GPU, via
`notebook/bench_quantization_signature_qwen_calibration.jl` / `_results.txt`. Checkpoint load: 5.45s
(matches the 5.7s figure already measured for this exact checkpoint in
`bench_eps_layer22_anomaly_profile_results.txt`). Median of 5 timed forward passes (+2 warmup),
`invalidate_all!` before each:

| W (tokens) | median ms | ms/token | vs. linear-from-W=256 | VRAM used before block |
|---:|---:|---:|---:|---:|
| 256  | 2,152.00   | 8.41   | 1.00x (baseline)      | 8,218 MiB |
| 512  | 2,465.11   | 4.81   | 0.57x (SUB-linear)    | 10,192 MiB |
| 1024 | 330,889.24 | 323.13 | **38.44x** (severe superlinear blowup) | 13,180 MiB (445 MiB free after) |

**This directly validates the task's warning against assuming linear scaling, and then some.** The
jump from W=512 to W=1024 is not merely superlinear in the O(n^2)-attention sense (which alone would
predict roughly a 4x cost increase for a 2x length increase, i.e. ~10s, not 331s) -- it is a roughly
134x wall-clock cliff for a 2x increase in tokens, coinciding with VRAM headroom collapsing to 445 MiB
free by the end of that block. The most likely explanation is that W=1024 pushes this engine's
per-forward-pass memory footprint close enough to the 16 GB physical limit that some part of the
computation (allocation churn, or a fallback path once the fast-path memory budget is exceeded) starts
paying a drastically higher cost, rather than a smooth continuation of the same scaling regime seen at
256/512. Whatever the precise mechanism, the practical conclusion is unambiguous: **W=1024 is
categorically unsuitable for this run**, both because ~331s per window would make even a modest window
count take hours, and because operating with only ~445 MiB of GPU headroom is precisely the kind of
resource contention this experiment must avoid given another real background job (a corpus-size
training sweep) shares this GPU.

**Chosen window length: W=512.** It is not just acceptable but the empirically BETTER choice within the
safe range: sub-linearly efficient per token (4.81 ms/token vs. 8.41 ms/token at W=256, i.e. fewer
total forward passes needed for the same token count AND a lower per-token marginal cost), and its
peak measured VRAM usage (10,192 MiB, comfortably under half the 16 GB budget) leaves ample headroom for
the concurrent job.

### 4b. Token budget reasoning

The original toy test used ~111K per-character positions from 6 checkpoints (3 depths x 2 seeds), each
checkpoint contributing ~37K positions from non-overlapping 256-token windows over TinyShakespeare's
validation split. That scale was cheap to obtain because each toy forward pass was small (`dim=256`,
4-layer-max character LM). Qwen2.5-1.5B forward passes are dramatically more expensive per token
(measured above), and producing a genuinely fresh, non-memorized, varied-register corpus by hand does
not scale to hundreds of thousands of tokens the way sampling from an existing large corpus would.

**Whether 111K is the right target here: no, for a specific, checkable reason, not just a budget
excuse.** The primary test's binding constraint in the original experiment was never sample size --
Ashman's D topped out at 1.845 against a required D>2 even with ~37K positions per single checkpoint,
and the BIC criterion is the less restrictive of the two thresholds at any sample size in this range.
Concretely, `ΔBIC = -2(ll2-ll1) + 3 ln(n)`, so firing the `ΔBIC < -10` criterion requires
`ll2 - ll1 > 5 + 1.5 ln(n)`. At n=15,000, that is `ll2-ll1 > 19.4` nats total, i.e. an average
per-position log-likelihood gain from the second component of only ~0.0013 nats -- a trivially small
bar to clear if genuine, well-separated bimodal structure (the kind the D>2 criterion is built to
detect) is present at all. In other words: if Qwen's per-token loss really were governed by a
discrete-quanta mixture with Ashman's D>2, a sample in the low tens of thousands would detect it
comfortably; the toy experiment's null result was a statement about the ABSENCE of sufficient
separation (D), not about insufficient sample size, and there is no reason to expect that diagnosis to
reverse simply by adding more of the same kind of data. Where a larger n WOULD matter is in resolving a
minority-component weight very close to the [0.05, 0.95] boundary with precision, or in detecting a
subtle effect close to the D=2 threshold -- neither of which changes the qualitative REFUTES/SUPPORTS
verdict at the margins this test cares about.

**Target and realized size.** Given this reasoning, a token budget in the low tens of thousands is
judged adequately powered for this test, rather than a blind match to 111K. The realized corpus
(14,977 tokens) is used in full, non-overlapping, as described below.

### 4c. Window and sampling design

**W = 512 tokens, non-overlapping, tiling the full 14,977-token corpus in document order.**
`n_windows = (14,977 - 1) ÷ 512 = 29` (the trailing 320 tokens too short for one more full window are
dropped, per the design in §4c below, rather than padded or given a fabricated label) ->
**29 x 512 = 14,848 per-position loss samples**, the realized primary-test sample size. Projected
wall-clock for the data-collection loop: `29 x ~2.47s ≈ 72s`, plus the ~5.5s checkpoint load, plus
negligible CPU time for the GMM/BIC fitting (pure numpy-scale Julia arithmetic on <15K scalars) --
low single-digit minutes end to end, comfortably inside a safe, short-lived window on shared hardware.

Windows tile the corpus non-overlapping, in original document order (`all_ids[1:W]`,
`all_ids[W+1:2W]`, ...), exactly as `val_windows_nonoverlapping` did in the original experiment, so
that each held-out token position is measured exactly once and no position is scored at more than one
context length. As in the original experiment, each window's per-position labels are the TRUE next
tokens (`labels = all_ids[start+1 : start+W]`, requiring one token beyond the window's own extent, drawn
from the corpus's own natural continuation, i.e. the following window or corpus tail -- never a
fabricated/repeated label), and the very last incomplete tail of the corpus (fewer than W+1 tokens
remaining) is dropped rather than padded or given a fake label.

## 5. Primary statistical test (identical procedure to the original experiment, for direct comparability)

**Test: 1-component vs 2-component univariate Gaussian-mixture BIC comparison**, fit by
expectation-maximization, implemented exactly as in `notebook/bench_quantization_signature_check.jl`
(`gmm_em_2comp`, `gmm_1comp_ll`, `multimodality_test` -- copied verbatim into the new script, not
reimplemented or altered). No `Project.toml` change; only `Statistics`/`LinearAlgebra`/`Random` used for
the fit, same as before.

For the per-position loss vector `x` (raw nats):
- Fit `k=1` (closed form) and `k=2` (EM, >=20 random restarts, variance floor `1e-4*var(x)`) univariate
  Gaussian mixtures.
- `BIC(k) = -2*logL(k) + (3k-1)*ln(n)`.
- **Fires** (multimodal signature) iff all three hold:
  1. `ΔBIC = BIC(2) - BIC(1) < -10` (Kass-Raftery "very strong" evidence).
  2. Ashman's D, `D = |mu2-mu1| / sqrt((s1^2+s2^2)/2) > 2`.
  3. Both mixture weights in `[0.05, 0.95]`.

Same three criteria, same thresholds, same code, run once on this single Qwen checkpoint's per-token
loss vector.

## 6. BPE-specific confound control

Unlike per-character loss, per-BPE-token loss varies with token identity for reasons that have nothing
to do with any model property at all: a single BPE token can represent anywhere from one raw byte to an
entire common word, so token-level surprisal is entangled with token granularity in a way character-level
loss never was. A rare, multi-byte, specialized token is intrinsically harder to predict than a common
single-byte or short, high-frequency merge, independent of whether the model has "acquired" anything in
particular. This must be controlled for before any bimodal signature is attributed to quantization,
exactly as character frequency and context position were controlled for in the original experiment.

**Primary control regressors (BPE-adapted analogue of the original unigram-frequency + position
control):**

1. **`log(token_id + 1)`** -- a frequency-rank proxy specific to byte-level BPE vocabularies. Verified
   directly against `notebook/qwen2.5-1.5b-instruct/vocab.json`: ids 0-255ish are the raw single-byte
   tokens (enumeration order, not frequency-ordered); ids from ~256 upward are merged tokens, added
   during BPE training in the order pairs were merged, which is frequency order (most frequent pair
   merged first, hence lowest merge id) -- spot-checked directly (id 300 = `"as"`, id 1000 = `"atus"`,
   id 5000 = `"Ġdifficult"`, id 100000 = a rare CJK compound, id 150000 = rare Devanagari, id 151642 =
   near the very end of the vocabulary, a rare symbol). Lower token id is therefore a real, principled
   proxy for how common a token was in Qwen's own actual (very large, general-domain) pretraining
   corpus -- a more domain-general and theoretically grounded proxy here than any frequency table this
   experiment could compute from a small, unrelated held-out reference corpus of its own.
2. **`token_byte_length`** -- the number of raw UTF-8 bytes the target token represents (computed exactly:
   the GPT-2/Qwen byte-to-unicode map is a bijection from bytes to single printable characters, so the
   character length of a vocab piece string equals its raw byte count, no decoding ambiguity). Longer
   pieces tend to be rarer, more specific tokens; this is a second, more direct granularity control
   distinct from vocabulary rank.
3. **`log(position_in_window)`** -- identical role to the original experiment's context-length control:
   position 1 of every non-overlapping window has minimal context, position `W` has the most.

**Procedure:** OLS-regress `loss_i ~ 1 + log(token_id_i + 1) + token_byte_length_i +
log(position_in_window_i)` on the full per-position vector, take residuals, and re-run the exact §5 test
(same three criteria, same thresholds) on the residuals.

**Secondary, descriptive-only robustness check.** An empirical token-frequency table was also computed
by tokenizing TinyShakespeare (`notebook/data/tinyshakespeare/input.txt`, 301,829 BPE tokens, 12,111
distinct token ids) with the same BPE encoder, Laplace-smoothed, giving `-log(freq)` per token as an
alternative frequency proxy. This is reported alongside the primary control but is NOT decisive for the
verdict: TinyShakespeare's Early-Modern-English register is a poor domain match for either the held-out
corpus (modern varied-register prose) or Qwen's actual pretraining distribution, and many modern/technical
tokens in the held-out corpus (e.g. "warehouse", "spreadsheet", numerals used in specific contexts)
simply do not occur in TinyShakespeare at all, requiring smoothing that makes this proxy noisier and less
principled than the vocabulary-rank control. It is included only as a cross-check that the primary
result is not an artifact of the specific frequency-proxy choice.

## 7. Verdict criteria, fixed before running

Because only one checkpoint exists, this is a single-arm result (no depth-graduation secondary test, no
monotonicity/graduation-overlap criteria -- those require multiple depths, which are not available for
Qwen in this repository).

- **REPLICATES THE REFUTATION** if: the primary test (§5) does not fire (no `ΔBIC<-10` AND `D>2` AND
  both weights in `[0.05,0.95]`, jointly) on raw per-token loss. This would mean Qwen's per-position loss
  distribution is statistically indistinguishable from a single (possibly skewed) component, same
  qualitative conclusion as the toy-model result, now at real production scale on real natural language.
- **REVERSES THE REFUTATION (supports the quantization picture)** if: the primary test fires on raw loss
  AND survives the BPE confound control (§6 primary regressors) -- i.e. the residual test also fires. A
  single checkpoint cannot corroborate the acquisition/graduation story (§6 of the original
  pre-registration), so this would establish only the weaker structural claim: per-position loss is
  better described by a small number of discrete components than by one smooth distribution, net of the
  granularity/frequency and context-length confounds -- notably stronger than anything shown by the toy
  experiment, since it would mean the effect appears at real scale despite not appearing at toy scale.
- **INCONCLUSIVE** if: (i) the primary test fires on raw loss but is fully explained away by the BPE
  confound control (residual test does not fire) -- indistinguishable from ordinary token-granularity/
  frequency/position difficulty, not evidence of quantization; or (ii) EM fails to converge to a stable,
  restart-robust fit.

## 8. What this test can and cannot show

A "REVERSES" verdict would not establish the Zipf-exponent-1 claim, the specific `mu` (quanta/layer)
parameter, or any functional form beyond "more than one discrete component, net of confounds" -- exactly
the same scope limitation acknowledged in the original pre-registration. A "REPLICATES" verdict does not
prove quantization is false as a description of learning dynamics DURING training (this test, like the
original, only ever looks at a single frozen checkpoint, never the trajectory) -- only that the specific,
falsifiable, single-checkpoint structural prediction examined here does not hold, now checked at both toy
scale and real production scale.

Results, raw numbers, and code: `notebook/bench_quantization_signature_qwen_check.jl` and
`notebook/bench_quantization_signature_qwen_check_results.txt` /
`notebook/bench_quantization_signature_qwen_check_results.json`.
