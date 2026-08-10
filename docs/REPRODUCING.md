# Reproducing the experiments

Every numeric claim in the NeuroDSL papers is produced by a script in this
repository, and the output of each run is archived next to it. This document
tells you which script produces which claim, how to run it, and what result
should come back.

If a number in a paper does not appear in this guide, treat that as a defect and
open an issue — it means the claim has no traceable artifact.

---

## Before you start

```bash
git clone https://github.com/nevermind78/NeuroDSL.git
cd NeuroDSL
julia --project -e 'import Pkg; Pkg.instantiate()'
```

Requirements: Julia 1.10+. A CUDA GPU is required only for the arms marked
**GPU** below; every *structural* measurement runs on CPU in minutes.

### Two kinds of measurement, and why the distinction matters

**Structural measurements** count nodes, buffers or allocations. They are
deterministic, involve no timer, and reproduce bit-for-bit on any machine. Most
of the load-bearing claims are of this kind, deliberately: a node count cannot be
explained away by clock noise.

**Wall-clock measurements** need a pinned GPU clock, or the variance swamps the
effect being measured. On this hardware an unpinned clock produced swings that
inverted a 22-point result. Before any timed arm:

```powershell
nvidia-smi -lgc 1402,1402      # Administrator PowerShell
```

and afterwards:

```powershell
nvidia-smi -rgc
```

Timed arms in this guide are marked **clock**. Structural arms are not affected.

### Protocol conventions used throughout

These are not decoration; each one exists because its absence produced a wrong
number at some point in this project.

- **One arm per process.** Sharing a process between two arms let the second
  inherit the first's memory state. That produced +1.0% and +22.8% for the same
  configuration on two runs. Every driver launches a fresh process per arm.
- **Several independent launches**, reported as `min / median / max` across
  launches — never a single figure. Ranges are reported as `max`, not as a
  percentile: at five launches the 90th-percentile index *is* the last order
  statistic, so calling it `p90` would be a maximum under a misleading name.
- **A negative control** in every experiment: a configuration where the
  mechanism must *not* help. A measurement that cannot return "no effect" is not
  a measurement.
- **A positive control** where one is available: a configuration where the
  measured quantity must *change* by a predicted amount. A count that is
  constant by instrumentation error passes every negative control.
- **Independent oracles.** Locality counts are checked against a BFS written
  from scratch in the benchmark file, which calls neither the function under
  test nor the engine's own consumer index. Otherwise you only verify that code
  agrees with itself.
- **Correctness before speed.** Every arm that changes an execution path first
  compares its output to the unmodified path, bit-for-bit. An arm that is faster
  but different is disqualified, not optimised.

### Reading a results file

Each `notebook/*_results.txt` begins with a commented protocol header, then one
`RESULT` line per measurement with `key=value` fields. Files are appended to by
per-arm processes and truncated by their driver at the start of a series.

---

## Paper index

| Paper | arXiv | Guide |
|---|---|---|
| NeuroDSL: A Mutable Computational Graph Framework with Local Invalidation and Inspectable Memory Planning | submission on hold | [§1](#1-neurodsl-the-framework-paper) |
| Exact Network Surgery: Functional Invariance and Gradient Plasticity | [2607.16568](https://arxiv.org/abs/2607.16568) | [§2](#2-exact-network-surgery) |
| Cost Accounting for Reactive Computational Graphs | [2607.18323](https://arxiv.org/abs/2607.18323) | [§3](#3-cost-accounting) |
| A Theory of Conditional Collapse under Low-Rank Weight-Space Ablations: I. The Single-Block Theory and Synthetic Validation | [2608.03620](https://arxiv.org/abs/2608.03620) | [§4](#4-conditional-collapse-and-crosslayer-interaction) |
| Cross-Layer Interaction under Weight-Space Ablation: A Closed-Form Attention Jacobian Bound and a Test on a Real Pretrained Model | [2608.03629](https://arxiv.org/abs/2608.03629) | [§4](#4-conditional-collapse-and-crosslayer-interaction) |

---

## 1. NeuroDSL, the framework paper

Every experiment in this section was built and measured on 2026-08-08, and the
commands below are the ones that produced the figures in the paper.

### 1.1 Invalidation locality — the paper's central claim

**Claim.** The number of nodes an edit traverses is a property of the edited
site, not of the graph around it: inflate the graph sevenfold with subgraphs the
edit cannot reach and the count does not move by one unit.

```bash
bash notebook/bench_invalidation_locality_driver.sh
```

Structural, CPU, no clock needed, about 8 minutes (16 isolated Julia launches).
Artifacts: `notebook/bench_invalidation_locality_results.txt`,
`notebook/bench_invalidation_locality_positive_control_results.txt`.

Expected, on a 12-layer Transformer graph at three padding levels
(|V| = 601 → 1802 → 4204):

| edited site | count |
|---|---|
| `:input` (negative control) | 493 / 493 / 493 |
| `layer_1_out` | 452 / 452 / 452 |
| `layer_6_out` | 247 / 247 / 247 |
| `layer_12_out` | 1 / 1 / 1 |

Pre-registered criteria, all four met: counts identical across five independent
launches; count **exactly** equal under 4× padding; agreement with the
independent BFS oracle on 16 of 16 points; negative control reaching 100% of the
invalidatable set.

One note on the negative control, because the criterion was corrected *before*
the result was seen: it was first written as ">= 95% of |V|" and measures 82.03%
against that denominator, because parameter nodes are leaves and can never be
downstream of anything. Against the invalidatable set — the only reachable
ceiling — it is exactly 100%. Both numbers are in the output.

### 1.2 Invalidation positive control — making the measure respond

A count that were constant *by error* would pass every negative control. This
arm makes the padding reachable from `:input`, so the count must grow by exactly
`(n_model − 1)` per padding unit while deeper sites stay bit-identical.

```bash
julia --project=. notebook/bench_invalidation_locality.jl 2 input reachable
julia --project=. notebook/bench_invalidation_locality.jl 4 input reachable
julia --project=. notebook/bench_invalidation_locality.jl 4 layer6 reachable
```

Expected: `:input` gives 1093 then 2293 (493 + 600 and 493 + 1800); `layer6`
stays at 247. Oracle agreement on all eight archived points.

The default `padmode` is `disjoint`, which reproduces §1.1 unchanged.

### 1.3 Retrieval locality — a result not reported elsewhere

**Claim.** Evaluation used to walk the cached topological order from its start,
so its cost followed the target's *position*. Restricting it to the target's real
ancestor cone makes retrieval invariant in total graph size.

```bash
bash notebook/bench_retrieval_locality_driver.sh
```

Structural, CPU, about 8 minutes. Artifact:
`notebook/bench_retrieval_locality_results.txt`.

Expected (ancestor cone / old prefix walk):

| target | \|V\|=601 | \|V\|=1802 | \|V\|=4204 |
|---|---|---|---|
| `layer_1_out` | 51 / 51 | 51 / 52 | **51 / 642** |
| `layer_6_out` | 301 / 301 | 301 / 302 | **301 / 892** |
| model output | 601 / 601 | 601 / 1789 | 601 / 4105 |

**Read the first column before the third.** Every entry there is exactly 1.00×.
On a graph holding one model the fix buys nothing; the gain comes entirely from
excluding subgraphs the target does not depend on. The padding is also built
*before* the model so it can occupy earlier topological positions — the
situation the old prefix walk penalised. Both are stated conditions of the
measurement, not caveats discovered afterwards.

This run is also the first test coverage `_ancestors_of!` has ever had.

### 1.4 Frozen-prefix backward pruning

**Claim.** With a frozen prefix contiguous with the graph input, the backward
pass skips the part of the graph strictly upstream of the shallowest trainable
parameter.

```powershell
nvidia-smi -lgc 1402,1402     # Administrator
```
```bash
bash notebook/bench_prune_frozen_driver.sh 5
```

**GPU**, **clock**, about 8 minutes. Artifact:
`notebook/bench_prune_frozen_results.txt`.

Expected — favourable arm (`W1..W7` frozen, `W8` trainable):

- work pruned **85.19%** (27 nodes receiving backward treatment → 4), identical
  in all five launches, no timer involved;
- gradient difference with pruning on versus off: **exactly 0.000e+00**;
- wall-clock gain median **69.9%**, min 68.3%, max 76.4%.

Negative control (`W1` trainable, `W2..W8` frozen): 33.33% of nodes pruned but
wall-clock gain bounded at **4.19%**. The two arms do not overlap across ten
launches. The control explains itself: its nine pruned nodes are all leaves, and
pruning a leaf removes no computation.

State the claim as a mechanism rather than a constant: *the work removed is the
fraction of the graph lying strictly upstream of the shallowest trainable
parameter.* Compute it for your own model instead of transferring 85.19%.

### 1.5 Node fusion — speed

**Claim.** Fusion calls the vendor GEMM and applies the elementwise epilogue
separately. It saves a dispatch and a buffer, never a FLOP, so the gain shrinks
as tensors grow.

```bash
bash notebook/bench_fusion_driver.sh 3
```

**GPU + CPU**, **clock**, about 12 minutes. Artifacts:
`notebook/bench_fusion_results.txt` (current engine),
`notebook/bench_fusion_prefix_results.txt` (before the `_relu_into!` fix).

Expected, median gain: CPU +11.9% / +2.2% / −0.3% and GPU +9.7% / +2.9% / +1.7%
at dim 32 / 128 / 512. Output bit-identical to unfused at all six
configurations (`max_err` exactly 0).

The pre-fix file documents an engine defect worth knowing about: the CPU fused
path used to be a **pessimization** of −7.9% to −40.3%. Cause was the
conjunction of an aliased broadcast and a branchy non-inlinable function; the
fix is a `@noinline` function barrier in `src/dispatch.jl`. The `@noinline` is
load-bearing — annotating it `@inline` restores the defect.

### 1.6 Node fusion — memory, the axis that pays

```bash
bash notebook/bench_fusion_memory_driver.sh
```

**GPU** for the watermark arms, no clock needed, about 8 minutes. Artifact:
`notebook/bench_fusion_memory_results.txt`.

Expected: resident activations **−47.1%** at every width (17 activation nodes →
9, a node count and therefore width-invariant); total resident VRAM −37.8% /
−24.2% / **−9.9%** at dim 32 / 128 / 512.

Do not quote 47.1% as "memory saved". It is the activation share; the total
figure shrinks with width because parameters dominate the footprint and fusion
does not touch them. At dim 512 the eight weight matrices are 8.39 MB against
2.125 MB of activations, so removing 1 MB is 9.9% of the whole.

### 1.7 The memory planner — what it actually delivers

```bash
julia --project=. notebook/bench_liveness_slots.jl      # structural, CPU, ~2 min
julia --project=. notebook/bench_planned_exec.jl        # correctness + pool, CPU
julia --project=. notebook/bench_planned_vram.jl demand
julia --project=. notebook/bench_planned_vram.jl release
julia --project=. notebook/bench_planned_vram.jl planned    # GPU
```

Artifacts: `bench_liveness_slots_results.txt`, `bench_planned_exec_results.txt`,
`bench_planned_vram_results.txt`.

Expected, and this is the honest part:

- With the shipped default, `n_slots` **equals the node count** on every graph
  tested — 25/25, 61/61, 121/121, 201/201. Interval colouring is optimal in slot
  count for a fixed order, but a graph expecting a backward pass has no two
  disjoint activation lifetimes, so that optimum is the node count and no slot is
  ever shared. Optimality without disjointness buys nothing.
- With `for_backward=false` (forward-only graphs) slots fall 201 → 40, pool
  reuse reaches 92.1%, and output is bit-exact.
- **Peak VRAM is unchanged**: 46.96 MB for planned execution, identical to plain
  `demand!`, because the buffer pool retains what it reclaims. The mechanism that
  does lower the forward-only peak is eager release, at **16.96 MB** — 2.77×
  better.

Three defects were fixed in this path on 2026-08-08 and are documented in the
source: an unconditional lifetime extension, a guessed output shape that made
the executor destroy its own pooled buffers, and the release of raw input leaves,
which made `demand_planned!` work exactly once.

### 1.8 End-to-end run — the memory/throughput trade-off

```bash
julia --project=. notebook/real_llm_vram_probe.jl
```

**GPU**, no clock needed (memory watermark), about 5 minutes. Artifact:
`notebook/real_llm_vram_probe_results.txt`. PyTorch reference:
`notebook/real_llm_pytorch_results.json` (mirror script `notebook/real_llm_py.py`,
notebook `notebook/notebook_py.ipynb`).

Expected, against PyTorch's 73.62 MB:

| episode | NeuroDSL | ratio |
|---|---|---|
| `train_step`, default | 91.78 MB | 1.25× worse |
| `train_step`, `release_values=true` | 80.27 MB | 1.09× worse |
| `val_window` | 51.44 MB | **0.70×** |
| `gen_token` | 51.80 MB | **0.70×** |

Baseline stable at 78.03 MB after 100 cumulative steps — no leak. Throughput:
25.29 ms/step against 11.35, i.e. about 2.2× slower.

This is the trade-off, not a defeat and not to be softened: the persistent graph
keeps activations resident, which is the mechanism that makes §1.1 and §1.3
possible and also what makes the training step heavier.

### 1.9 Engine comparison — June submission versus now

The paper reports that the retracted "2–8× less memory" claim had two
independent causes: an instrument error, *and* a real engine improvement since
submission. Reproducing the second requires a worktree at the pre-submission
commit with today's dependencies pinned, so that only `src/` varies.

```bash
git worktree add --detach /tmp/engine_june b666428
cp -r src/. /tmp/engine_june/src/
cp Project.toml Manifest.toml /tmp/engine_june/
mkdir -p /tmp/engine_june/notebook
cp notebook/article_benchmark_vram_probe.jl /tmp/engine_june/notebook/
# then, in each tree:
julia --project=. notebook/article_benchmark_vram_probe.jl
```

Careful: copy today's `src/` in and then let git's checkout of `b666428` provide
the old files, or the comparison is confounded by dependency drift. Verify with
`diff -rq` that only the files you intend differ.

Artifact: `notebook/bench_article_vram_engine_comparison_results.txt`.

Expected: at the June commit, 10.81 / 35.44 / 126.71 MB against PyTorch's
20.32 / 31.38 / 74.51 — that is 1.88× *less* at dim 256 but **1.13× and 1.70×
more** at dim 512 and 1024. Today: 5.47 / 18.07 / 64.02 MB, winning at all
three. Peak halved at every width.

---

## 2. Exact Network Surgery

**arXiv:2607.16568.** The graft primitive, its exactness proofs, and
post-insertion gate dynamics.

Entry points:

- `notebook/experiments_surgery.ipynb` and
  `notebook/experiments_surgery_gpu.ipynb` — the E1/E2 falsifiability checks
  (bit-exactness of the graft, escape from the zero gate, and the degenerate
  zero-gate-plus-zero-projection saddle).
- `notebook/graph_surgery.ipynb` — the surgery primitive itself
  (`insert_block!`, `graft_shadow_block!`, `src/graph_surgery.jl`).
- `notebook/real_llm_surgery.ipynb`, `notebook/real_llm_surgery_v2.ipynb` — the
  character-level language-model grafting experiments.
- `notebook/graft_porte_sortie_E.jl` — the learning-rate ladder behind the
  appendix table on gate magnitude versus branch norm.

The mechanism-level claims (F1–F4) are also covered by the test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

`test/test_surgery.jl` asserts bit-exact identity at insertion, invalidation
locality of the graft, continuity across a real process restart, and that the
grafted model still learns.

**Caveat, stated rather than hidden:** the mapping from each numbered experiment
in that paper to a specific cell has not been re-verified end to end in this
guide. The entry points above are correct; if you need the exact cell for a
specific figure, open an issue and it will be pinned down.

---

## 3. Cost Accounting

**arXiv:2607.18323.** Exact combinatorial cost identities for exhaustive sweeps,
sequential mutation, and the backward-locality gap.

- **E9, cost geometry at GPT-2 scale** — the one arm reproduced here directly:

  ```bash
  julia --project=. notebook/jalon0_cost_geometry_gpt2_scale.jl
  julia --project=. notebook/jalon0b_cost_geometry_gpt2_medium.jl
  ```

  Artifacts `notebook/jalon0_results.json`, `notebook/jalon0b_results.json`.
  Expected aggregate ratios 2.12× at (L=12, H=12) and 2.10× at (L=24, H=16),
  against a purely combinatorial prediction of 2.145 and 2.073 that assumes zero
  per-call overhead. The closed-form cone size
  `cone_head(i) = 10 + (L−i)(7H+15)` reproduces all 564 measured cones with zero
  residual.

  Note what these files also show: the per-site spread is three orders of
  magnitude. `layer_12_mlp_out` re-forwards in 0.089 ms against a 23.04 ms full
  forward — 259× — while `layer_1_mha_ao_h8` is 1.04×. The 2× aggregate is the
  average, and it is a proven ceiling for a depth-uniform sweep.

- **E4, E5, E7** — the sweep-ratio, training-collapse and sequential-cost
  identities. These were measured on synthetic layered graphs built inside the
  paper's own harness; the scripts are not isolated in this repository as
  standalone files. Open an issue if you need them extracted.

- **Sparse-backward rows** quoted in that paper and in the framework paper come
  from `notebook/GPT2.ipynb` (cells 16–17) and
  `notebook/neurodsl_benchmarks.ipynb` (cell 13). They concern
  `backward_graph_sparse!`, a different mechanism from `prune_frozen`.

---

## 4. Conditional collapse and crosslayer interaction

**arXiv:2608.03620** and **arXiv:2608.03629.**

- `notebook/colab_conditional_collapse_theory.ipynb` — the conditional-collapse
  theory experiments.
- `notebook/colab_crosslayer_interaction_qwen.ipynb` — the Qwen crosslayer
  interaction experiments. A pre-bootstrap-fix backup is kept alongside it for
  comparison.

Both notebooks were written to run on Colab and download their own model
weights; they are not wired to the `notebook/*_results.txt` convention used by
the framework paper's benchmarks.

---

## What is deliberately not in this repository

- **Model weights and checkpoints** (`*.safetensors`, `*.bin`, `notebook/*_ckpt/`,
  `notebook/qwen2.5-1.5b-instruct/`). Regenerable by download or by training.
- **Generated viewer exports** (`docs/*.html`, `notebook/*.html`). One of these
  was 68 MB. Regenerate with `save_interactive_graph`.
- **LaTeX sources of the papers** (`artilce/*.tex`). The compiled PDFs of the
  published papers are tracked; the sources are kept in a separate local
  repository by the author's choice.
- **Intermediate figures** produced by experiment scripts (`notebook/*.pdf`).
  The paper's final figures live in `figures/` and are tracked.

---

## If a number does not reproduce

Report it. The most common causes, in order of how often they bit this project:

1. **An unpinned GPU clock** on a timed arm. Check with
   `nvidia-smi --query-gpu=clocks.gr --format=csv` — it should sit at 1402 MHz,
   not fall back to idle.
2. **Two arms sharing a process.** If you run a benchmark by hand rather than
   through its driver, the second arm inherits the first's memory state.
3. **Comparing across sessions.** Absolute timings drift with thermal state by
   more than 10% on this hardware. Compare ratios measured in the same series,
   which is what every driver reports.
4. **A different GPU.** Structural numbers are hardware-independent by
   construction; wall-clock numbers are not, and no figure in these papers is
   claimed to transfer across machines.
