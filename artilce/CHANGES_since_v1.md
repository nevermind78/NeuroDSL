# NeuroDSL (article 1) — advances and corrections since the held submission

Internal record, and a basis for an arXiv *replace* comment. Written 2026-08-08.

## What this document can and cannot claim

**There is no line-level diff.** `artilce/*.tex` is excluded by `.gitignore:71` and no `.tex`
file has ever been tracked (`git ls-files artilce/` returns PDFs and notebooks only), so the
v1 source is unrecoverable. The nearest artifact,
`artilce/NeuroDSL A Dynamic Computational Graph Framework.pdf`, is dated 2026-07-07 —
*after* the 2026-06-25 submission — so it is not v1 either.

Every comparison below is therefore against a claim **whose former wording is documented**,
either because the current paper quotes it in a retraction or because it was established
during the sessions that produced these results. Where a figure has no retained artifact,
that is said so explicitly rather than papered over. The marker **[no artifact]** was reserved
for findings this document could not substantiate; as of the final revision no figure below
carries it — every number cited resolves to a file in the repository.

## 1. Advances — established now, absent before

| Result | Figure | Artifact |
|---|---|---|
| **Invalidation locality**, structural | Traversal count invariant while \|V\| grows 601→1802→4204 (7×): 493 / 452 / 247 / 1 nodes at increasing site depth, not changing by a single unit | `notebook/bench_invalidation_locality{.jl,_driver.sh,_results.txt}` |
| — independent oracle | 16/16, BFS written from `rule.inputs`, calling neither `_consumers_index!` nor the function under test | same |
| — negative control | Editing `:input` saturates the invalidatable set at 100.00% (493/493) | same |
| **Retrieval locality** | Ancestor cone invariant (1/51/301/601) while the old topological prefix grows to 592/642/892/4105; 12.59× and 2.96× at shallow and mid-depth targets | `notebook/bench_retrieval_locality{.jl,_driver.sh,_results.txt}` |
| — first independent verification of `_ancestors_of!` | 16/16; the function had zero occurrences in `test/` | same |
| **Fusion exactness** | Max abs. difference fused vs. unfused = `0.000e+00` on **all 36 archived launches**, across both engine states | `notebook/bench_fusion_results.txt`, `notebook/bench_fusion_prefix_results.txt` |
| **Fusion memory** | 17→9 activation nodes, −47.1% activations invariant in width; −9.9% total resident at dim 512 | `notebook/bench_fusion_memory{.jl,_driver.sh,_results.txt}` |
| **Peak VRAM by episode** | 0.70× PyTorch on `val_window` and `gen_token`; 1.25× on `train_step`, 1.09× with opt-in `release_values` | `notebook/real_llm_vram_probe_results.txt` |

**Retrieval locality is new to this body of work.** `artilce/` holds 11 `.tex` files; the
string `retrieval` appears **zero times in the ten that are not this paper** (`article2`,
`article3`, `code4`, `conditional_collapse_theory`, `crosslayer_interaction_qwen`, `math`,
`math1`–`math4`), and 21 times in `NeuroDSL.tex` — all of them added in this revision. The
four `ancestor` hits in `article2.tex` and `math2.tex` concern patch recomputation and layer
reachability, not retrieval cost. The published bound covers invalidation only, and does not imply the
retrieval property in either direction.

## 2. Positive corrections — two distinct kinds

**(a) The number was right; the support was not.** These were not false, they were unsourced.

- `prune_frozen` **86.4% → 85.19%**: the mechanism was real, the figure was not measured.
  It is now a node count (27→4), deterministic, identical across 5/5 launches, with a
  bounded control arm (33.33% of nodes pruned yielding ≤4.19% of time) and a mechanism-level
  statement replacing the constant — *the work removed is the part of the graph strictly
  upstream of the shallowest trainable parameter*. `notebook/bench_prune_frozen_results.txt`
- **Gradient agreement** moved from "within 1e-6, asserted by the suite" to *measured
  exactly `0.000e+00`* on 10/10 launches — a stronger result than was previously claimed,
  and now claimed at that strength for the topologies measured. Same artifact.

**(b) The claim was wrong and is gone.**

- **2.40×–7.97× peak memory** — retracted. It compared a `MemTrack` episode delta against
  PyTorch's absolute post-construction peak, two incommensurable quantities; `MemTrack` also
  missed allocations made directly by the CUDA path of `rmsnorm_bwd!`. Replaced by
  absolute-peak-vs-absolute-peak via the CUDA pool watermark: **1.16×–3.71×**, narrowing
  monotonically with width. `notebook/article_benchmark_vram_probe.jl`
- **37% fusion speedup** — retracted. A point measurement in the one regime where the effect
  is largest. Replaced by a width sweep on both devices: +11.9%/+2.2%/−0.3% (CPU) and
  +9.7%/+2.9%/+1.7% (GPU), negligible at realistic width, because node fusion saves a
  dispatch and a buffer and *never a FLOP*.
- **53.8% GPU fusion speedup** — retracted, no reproducible source; the history of the two
  intervening regressions (4.2× then 1.9–2.4× slower) is retained in the paper.
- **Throughput parity with PyTorch** — gone. Measured 2.23× slower per step at matched
  validation loss (1.623 vs. 1.620). `notebook/real_llm_{neurodsl,pytorch}_results.json`
- **Buffer-count optimality** — not false, **vacuous as written**, and now scoped. Interval
  colouring is optimal in slot count, but with a backward pass to follow no two activation
  lifetimes are disjoint, so the optimum is the node count and **no slot was ever shared**:
  measured 25/25, 61/61, 121/121 and 201/201 nodes. A forward-only plan shares (201→40), yet
  peak VRAM is **unchanged at 46.96 MB**; `demand_release!` is the mechanism that lowers it,
  at 16.96 MB (2.77×). The paper's phrasing "assigns tensors with disjoint lifetimes to a
  shared buffer slot" invited the wrong inference and has been rewritten; the title's
  *Inspectable* Memory Planning is retained because inspectability is what the planner
  actually delivers. `bench_liveness_slots_results.txt`, `bench_planned_exec_results.txt`,
  `bench_planned_vram_results.txt`

- **A stale val-loss figure (1.6339)** was caught during this session and replaced with the
  archived 1.6229; the number appears nowhere in the repository.

## 3. Engine improvements, not paper edits

- **`@noinline` function barrier `_relu_into!` (`src/dispatch.jl`).** Removed a CPU fused-path
  pessimization of **−7.9% to −40.3%** (medians at dim 32/128/512). Pre- and post-fix
  per-launch ranges are **disjoint at every CPU width** (e.g. [−55.9, −39.6] vs. [+11.8,
  +12.0] at dim 32). Root cause: the *conjunction* of an aliased broadcast and a branchy
  non-inlinable scalar function (`Base.max` → `signbit`/`ifelse`); neither factor alone is
  expensive. GPU is unaffected, as the mechanism predicts.
  `notebook/bench_fusion_prefix_results.txt` vs. `notebook/bench_fusion_results.txt`
- **Three defects fixed in the memory-planning path** (`src/liveness.jl`, `demand_planned!`),
  suite green after each. (i) `compute_liveness` extended `last_use` to the end of the
  schedule for every backpropable activation unconditionally — now behind `for_backward=true`,
  default unchanged. (ii) `demand_planned!` guessed output shape as the first input's, so each
  freshly acquired pooled buffer was freed and reallocated: **0% pool reuse** (0 hits / 164
  allocations); now uses `_infer_output_shape`, giving **92.1%** (151 hits / 13 allocations) at
  40 slots. (iii) It released raw input leaves, which no rule can recompute, so the path worked
  exactly **once** and failed on the second call — invisible to any test invoking it once. All
  arms bit-exact against `demand!` after the fixes.
  `bench_planned_exec_results.txt`, `bench_liveness_slots_results.txt`

- **`_ancestors_of!` in `demand!`.** Retrieval was O(target's position in the whole graph);
  it is now O(ancestor cone). This also fixed a real defect: a `demand!` on an unrelated node
  could re-execute a stateful node merely because it sat earlier in the topological order.
- **Peak memory halved since the submission-time engine — now measured.** Same probe, same
  session, both engine states; the June state from a worktree at `b666428` (2026-06-24, the
  last commit touching `src/` before submission) with dependencies pinned so only `src/`
  varied. Single-block workload, absolute peak both sides:

  | engine | dim 256 | dim 512 | dim 1024 |
  |---|---|---|---|
  | as submitted (`b666428`) | 10.81 MB → 1.88× less | 35.44 MB → **1.13× MORE** | 126.71 MB → **1.70× MORE** |
  | current (2026-08-08) | 5.47 MB → 3.71× less | 18.07 MB → 1.74× less | 64.02 MB → 1.16× less |
  | reduction | −49.4% | −49.0% | −49.5% |

  PyTorch reference 20.32 / 31.38 / 74.51 MB, unchanged at both dates. The probe asserts peak
  equality across 5 trials, so an unstable figure fails rather than being reported.
  `notebook/bench_article_vram_engine_comparison_results.txt`

## 4. Infrastructure that did not exist

Seven archived results files where there were none; five script-driven benchmarks with
drivers; **one arm per process** so no arm inherits another's memory-pool state; multi-launch
aggregation reported as min/median/**max** (relabelled from "p90", which at n≤5 is the
maximum by construction); **independently written oracles** that call neither the function
under test nor its dependencies; and **pre-registered criteria** fixed before first launch —
including one that failed as written (the invalidation control at ≥95% of |V|, measured
82.03%) and is reported as mis-specified, with both denominators shown, rather than silently
relaxed.

## 5. What got weaker — stated plainly

- Peak-memory advantage: **2.40×–7.97× → 1.16×–3.71×**, and narrowing with width; at dim
  1024 it is 16%, essentially parity. Weaker than what was *printed* — though, per §3,
  stronger than what the submitted *engine* could actually have claimed, which at dim 512 and
  1024 was nothing. Both facts belong in the record; neither cancels the other.
- **Training-step memory is 1.25× worse** than PyTorch. The framework cannot be described as
  using less memory; it trades residency for locality.
- **Throughput claim removed entirely**: 2.23× slower per step.
- **Fusion's speed benefit is negligible** at realistic width, and its 47.1% memory figure is
  a reduction in *activation nodes*, worth 9.9% of the actual footprint at dim 512.
- The **Flux.jl comparison is demoted** to historical context — it is the one measurement
  with no retained per-run log — removed from the summary table, the abstract, and every
  claim the argument rests on.
- **The memory planner does not reduce memory**, and the paper now says so flatly. Zero slots
  shared on any training graph; peak VRAM identical to consulting no plan even when slots are
  shared, because `release!` returns buffers to a pool bucket rather than to the device.
  Making it a memory mechanism would need a pool that can return memory to the device and a
  byte-aware colouring; neither exists. `bench_planned_vram_results.txt`

- The `O(|V_θ|)` bound is now stated for `_invalidate_downstream!` only. The gradient sweep
  `_invalidate_upstream!` JUMPs to consumers then CLIMBs to their inputs, so it visits a
  **strict superset** of V_θ and is not covered.

## 6. Net position

This version replaces a paper whose headline quantitative claims were an artifact of
mismatched instrumentation with one whose central claims are **counts**: deterministic,
timer-free, clock-independent, reproduced across independent process launches, checked
against independently written oracles, and each with a stated negative control and a retained
artifact. The claims that survive are narrower than what was printed — the memory range is roughly a
third of the published one and the throughput advantage is gone — but they are also, on the
memory axis, *stronger than the submitted engine could support*: it lost to PyTorch at two of
three widths, and the peak has since halved at every width. They are the claims the
framework's design actually supports, and two of them (invalidation locality measured against a growing graph,
and retrieval locality at all) did not exist before. The paper now argues mutability, locality
and exactness, and every figure supporting that argument can be regenerated by a reader from
a script in the repository. If the choice is between a held submission that a reviewer could
dismantle by re-running one benchmark and this version, this version is the one to file.

## 7. Known gaps in the current text

- ~~The paper explains the retracted 2.40×–7.97× purely as an instrumentation artifact.~~
  **Resolved.** The `b666428` measurement is archived and the paper now carries both causes as
  distinct, in `\Cref{tab:memory_engine}` and the paragraph "Two causes, and neither subsumes
  the other": the published figure was inflated by comparing summed intercepted allocations
  against an absolute peak, an error that would have existed at any engine state; *and* the
  present range is a genuine improvement over an engine the corrected instrument would have
  scored below parity at dim 512 and 1024. Scoped in the text to the single-block workload —
  the training loop remains 1.25× worse — and noted there that the June figures were never
  published, so this establishes a baseline rather than catching a second false claim.
- The `prune_frozen` wall-clock result rests on one topology (8-layer chain, one GPU, pinned
  clock); depth- and topology-robustness are not shown.
- Both locality counts are from one architecture (12-layer `LlamaModel`, dim 32, CPU) and one
  padding construction (disjoint chains). Padding reachable from the model is untested.

---

# Triage of this version's results

Four categories, deliberately **not a partition**. *Positive* and *solid* are independent axes
and the divergences are the informative part. Every entry cites its artifact; nothing is listed
as positive on a number that cannot be sourced. Artifact paths are relative to `notebook/`.

## A. Solid — ranked by strength of evidence, favourable or not

Solidity mechanisms: **D** deterministic/no timer · **O** independent oracle · **C** negative
control · **I** invariance under a deliberately varied confounder · **M** multi-launch
agreement · **X** bit-exactness.

| # | Result | Mechanisms | Artifact |
|---|---|---|---|
| 1 | **Invalidation locality** — 493/452/247/1 nodes, unchanged as \|V\| grows 601→4204; and under *reachable* padding the control site grows by exactly 600/unit (493→1093→2293) while deep sites stay bit-identical | **D O C I M** + a **positive** control, i.e. the measure is shown to respond as well as to ignore | `bench_invalidation_locality_results.txt`, `bench_invalidation_locality_positive_control_results.txt` |
| 2 | **Retrieval locality** — cone 1/51/301/601 invariant while the prefix grows to 4105 | **D O C I M**, and it can return a *null* (1.00×), which few measures here can | `bench_retrieval_locality_results.txt` |
| 3 | **Fusion exactness** — `max_err` = 0 on 36/36 launches, both engine states | **X M I** (6 configs × 2 engines) | `bench_fusion_results.txt`, `bench_fusion_prefix_results.txt` |
| 4 | **Gradient agreement under pruning** — exactly 0 on 10/10 launches | **X M**, plus suite coverage on branched topologies | `bench_prune_frozen_results.txt` |
| 5 | **`prune_frozen` structural** — 27→4 nodes (85.19%), 27→18 (33.33%) | **D C M** (the control arm is the informative one) | same |
| 6 | **Engine comparison, June vs. current** — peak halved at all three widths | **M I** (only `src/` varied), same instrument both sides | `bench_article_vram_engine_comparison_results.txt` |
| 7 | **Fusion memory, structural** — 17→9 activation nodes, −47.1% at every width | **D I** | `bench_fusion_memory_results.txt` |
| 8 | **Episode peak VRAM** — 0.70× / 0.70× / 1.25× / 1.09× | **M** (5-trial equality assert), same instrument both sides | `real_llm_vram_probe_results.txt` |
| 9 | **CPU fusion pessimization removed** — pre/post ranges disjoint at every CPU width | **M I** | prefix vs. current fusion files |

## B. Positive — favourable, whatever the evidence

**Solid and positive:** A1, A2, A3, A4, A5 (favourable arm), A6, A7, and the forward-only half
of A8 — **0.70× PyTorch**, i.e. 30% less peak VRAM on `val_window` and `gen_token`.

**Positive, adequate but narrower evidence:** 1.16×–3.71× less single-block peak memory (5
trials per size, `article_benchmark_vram_probe.jl`); 69.9% median backward-time gain in the
favourable `prune_frozen` arm (5 launches, pinned clock, single topology).

**Positive but thin — flagged as such:**

- **Fusion speed at mid and high width.** Three launches per point, and at CPU dim 512 they
  straddle zero (−0.98, −0.28, +1.92). The paper already reports ≈0% there, which is right.
  *Correction to how this case was put to me:* the post-fix GPU dim-32 point is **not** thin —
  its launches are 9.21 / 9.72 / 10.65, a 1.4-point spread. The wide [0.57, 15.55] range is the
  **pre-fix** engine, which the paper does not claim as favourable. The genuinely thin
  favourable points are mid-width (GPU dim 128: 2.06 / 2.95 / 6.47), not dim 32.
- **GPT-2-shaped forward-incremental, 166.8 → 0.6 ms.** Two isolated timings, no node count, no
  control. The most flattering number in the paper and among the least supported.
  (`GPT2.ipynb`)
- **Flux.jl 0.45–0.94×.** No retained per-run log; already demoted out of the summary table.

## C. Solid *and* unfavourable — the divergence worth naming

As well evidenced as anything in section A, and bad news. Burying these would repeat the
original paper's failure in a new costume.

- **Training-step peak VRAM 1.25× worse** than PyTorch (91.78 vs. 73.62 MB). Same instrument
  both sides, 5-trial assert, no leak over 100 further steps. `real_llm_vram_probe_results.txt`
- **2.23× slower per step** end to end at matched validation loss (25.29 vs. 11.35 ms).
  `real_llm_neurodsl_results.json`, `real_llm_pytorch_results.json`
- **Fusion buys ≈0% speed at realistic width** (−0.3% CPU, +1.7% GPU at dim 512): it saves a
  dispatch, never a FLOP. `bench_fusion_results.txt`
- **`prune_frozen`'s control arm — 33.33% of nodes pruned yields ≤4.19% of time.** The
  structural figure is an upper bound on time, and a loose one.
- **`backward_graph_sparse!` is 80% slower** on the GPT-2-shaped model while allocating 37%
  less. (`GPT2.ipynb`)
- **The `O(|V_θ|)` bound does not cover the gradient sweep.** `_invalidate_upstream!` JUMPs to
  consumers then CLIMBs to their inputs, visiting a strict superset of V_θ.

## D. To analyse — measured, not explained

- **Why the June→current peak reduction is ~49.5% at *all three* widths.** A constant factor
  across widths points to one mechanism — a duplicated buffer class, not a width-dependent term
  — and we have not identified which. We can state the improvement but not say what was fixed.
  `bench_article_vram_engine_comparison_results.txt`
- **Why the GPU fusion ranges are disjoint at dim 512 in the *wrong* direction** ([2.6, 3.2]
  pre-fix vs. [−0.6, 1.8] post-fix, the fixed engine marginally slower). A sub-effect at three
  launches, but the disjointness is real and its cause is unknown.
- **How often retrieval locality's precondition holds.** The gain requires subgraphs the target
  does not depend on, occupying earlier topological positions. The condition is known exactly;
  its frequency in real use is not.
- **Whether the two localities compose.** They are measured separately. An edit-then-demand
  cycle exercises both, and the joint cost has never been counted.
- **`backward_graph_sparse!`'s 80% slowdown** is attributed to added indirection. That
  attribution is asserted, not isolated.

## E. To improve — with what specifically to measure

*Evidence-thin — the outcome may be fine, the support is not:*

1. **Locality on a second architecture.** Both benchmarks on a branched/residual topology at
   larger width: 4 sites × 3 paddings, same four-argument `_invalidate_downstream!` call, same
   oracles. Converts "one architecture" into "generalises".
2. ~~**A positive control for invalidation**~~ — **done, and it found something.** With
   `padmode=reachable` the count at `:input` grows by exactly $n_{	ext{model}}-1 = 600$ per
   padding unit (493→1093→2293) while the three deeper sites are bit-identical to the disjoint
   case; oracle 8/8. The two controls now bracket the measurement: a count constant by
   instrumentation error fails the positive control, and a count really measuring the whole
   graph fails the invariance. *Byproduct:* the first attempt disagreed with the oracle
   (493 vs. 1093) because `_invalidate_downstream!` propagates only to **currently valid**
   consumers, so `|V_θ|` is an **upper bound the engine can beat** — strictly fewer nodes are
   visited when part of the cone is already invalid. Both are now in the paper
   (`\Cref{tab:invalidation_positive_control}` and §invalidation_proof).
   `bench_invalidation_locality_positive_control_results.txt`
3. **Fusion speed at n ≥ 10 launches**, dim 128 and 512, to settle whether the small GPU deltas
   and the wrong-direction dim-512 disjointness are real.
4. **GPT-2 forward-incremental: add node counts and a control.** Converts the paper's most
   flattering orphan timings into a structural result, cheaply.
5. **`prune_frozen` at other depths and on a branched topology** — structural counts only, no
   timer, so nearly free.

*Outcome-poor — the evidence is fine, the result is not:*

6. **Training-step footprint (1.25×).** Measure gradient-buffer lifetime overlap during the
   reverse pass, to find whether residency can be kept for locality while gradient buffers are
   released. `release_values=true` currently gives up both.
7. **Throughput (2.23×).** Before building a compiled-backend path, measure the fraction of
   step time spent in symbol-keyed dispatch — that bounds the achievable gain.
8. **Fusion.** A cuBLASLt-style epilogue applied from registers would cut memory traffic, which
   node fusion structurally cannot. Measure traffic, not time.
9. **Retire the Flux comparison** rather than re-measure it (see below).

## Two direct answers

**Strongest result: invalidation locality.** Not because it is favourable — that is incidental
— but because it is the only result carrying all five solidity mechanisms at once. It is a count
with no timer in it, so no clock and no machine can move it. It reproduces identically across
independent processes. It is invariant under a sevenfold increase in the exact confounder the
claim denies matters, which is the strongest available form of this test. It agrees on 16 of 16
points with an oracle built from a different data structure, calling neither the function under
test nor that function's dependency. And its negative control saturates its ceiling at 100.00%,
which demonstrates the measure can tell local from non-local rather than merely reporting small
numbers. It is also the paper's title claim, which previously rested on two wall-clock timings
with no node counts, no control and no artifact.

**Weakest: the Flux.jl comparison** (`tab:transformer` and the dense-MLP counterexample). It is
the only measurement in the paper with no retained per-run output, and it is off-thesis:
throughput, which the paper explicitly declines as a contribution. Two remedies exist.
Re-measure it to the standard of everything else — rejected, because it would manufacture
strong evidence on the one axis we decline to be judged on. Or **retire it** — the
recommendation. It is currently demoted to historical context and excluded from the summary
table, the abstract and every load-bearing claim, which is defensible for this revision; but a
table a reviewer can attack while it supports no conclusion is decoration, and the next revision
should delete it.
