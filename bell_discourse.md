


# Mutable Computational Graphs in Pure Julia: Computing Bell Numbers with Automatic Memoization

I've been building **NeuroDSL**, a Julia package for declarative, persistent computation
graphs — you write a recurrence as if it were an ordinary recursive function, and the package
turns it into a graph of named nodes, memoizing every distinct sub-call automatically. No
`@memoize`d function, no manual dictionary cache: the recursion structure itself *is* the graph,
and you get to inspect and visualize it afterward.

If you saw my earlier post on [performing "hot surgery" on a running LLaMA transformer](https://discourse.julialang.org/t/mutable-computational-graphs-in-pure-julia-performing-hot-surgery-on-a-llama-transformer/138018)
— inserting a residual block into a live, mid-training model without recompiling the graph or
losing optimizer state — this is the exact same underlying graph engine (persistent nodes,
targeted invalidation, no rebuild-from-scratch), just pointed at a much smaller, pure-math problem
instead of a neural network. If the LLaMA post showed *what the engine is for*, this one is a
better place to actually see *how the DSL itself is written*, one line at a time, without a
transformer's worth of incidental complexity in the way.

To show what that looks like end to end, here's a small, self-contained example: computing the
6th [Bell number](https://en.wikipedia.org/wiki/Bell_number) via the recurrence for
[Stirling numbers of the second kind](https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind),
then exporting the *entire computation* as an interactive HTML graph you can click through node
by node.

## The math

Stirling numbers of the second kind $S(n,k)$ count the number of ways to partition an
$n$-element set into exactly $k$ non-empty subsets. They satisfy:

$$
S(n,k) =
\begin{cases}
1 & \text{if } n = 0,\ k = 0,\\
0 & \text{if } k = 0 \text{ or } k > n,\\
k \cdot S(n-1,k) + S(n-1,k-1) & \text{otherwise}.
\end{cases}
$$

The Bell number $B_n = \sum_{k=1}^{n} S(n,k)$ then counts *all* partitions of an $n$-element set,
regardless of how many parts they have. For $n=6$, $B_6 = 203$.

This recurrence is a good demo case precisely because it's *not* tail-recursive and has
overlapping sub-calls all over the place — `stirling(6,3)` and `stirling(6,2)` both eventually
call `stirling(4,2)`, `stirling(3,2)`, and so on. Naively recursive code re-derives those shared
sub-results every time it hits them; a memoized version (whether via `@memoize`, a graph, or a
hand-rolled `Dict`) computes each one once.

## Step 1 — install and import

```julia
using NeuroDSL, Printf
```

## Step 2 — define the custom operators with `@defop`

NeuroDSL ships a small set of built-in ops, but the interesting part of any DSL is defining your
own. Every op ultimately becomes a plain Julia function with a fixed 7-argument signature
`(dev, out, inputs, attrs, out_sym, nd, ctx)`:

- `dev` — the device the op runs on (`Backend.CPUDevice()`/`CUDADevice()`), for dispatch.
- `out` — a pre-allocated output buffer; the op writes its result *into* it rather than returning
  a value (this is what lets NeuroDSL reuse buffers across recomputation).
- `inputs` — a `Vector` of the op's input arrays, in the order the `GraphRule` records them.
- `attrs` — a `Dict{Symbol,Any}` of extra per-call keyword arguments that aren't graph nodes
  themselves (e.g. the scalar factor `k` below).
- `out_sym`, `nd`, `ctx` — the output node's symbol, its full `GraphNode`, and a shared
  `CtxStore` for ops that need cross-call state (RNG for dropout, etc.) — unused by the three ops
  here.

`@defop` gives you three ways to write that function, from terse to explicit:

```julia
@defop scale_add out = attrs[:factor] * inputs[1] + inputs[2]
@defop identity  out = inputs[1]
@defop nsum (dev, out, inputs, attrs, out_sym, nd, ctx) -> begin
    copyto!(out, inputs[1])
    for i in 2:length(inputs)
        out .+= inputs[i]
    end
end
```

- `scale_add`: the one-liner form. `@defop op_sym out = <rhs>` expands to a full 7-argument
  lambda whose body is `@. out = <rhs>` (broadcasted in place) — so this line alone becomes
  `(dev,out,inputs,attrs,out_sym,nd,ctx) -> @. out = attrs[:factor] * inputs[1] + inputs[2]`,
  then registers it under the name `:scale_add` via `NeuroDSL.register_op!`. It computes
  `factor * a + b` — exactly the recurrence's right-hand side, `k * S(n-1,k) + S(n-1,k-1)`, with
  `k` passed as the per-call attribute `factor` rather than as a third graph input.
- `identity`: same one-liner form, just copies its single input through — used internally
  whenever a `@node`/`@rule` result needs an alias under a different name (see Step 3).
- `nsum`: the explicit-lambda form, needed here because summing a *variable* number of inputs
  (as many as `n`) isn't a single broadcast expression. `copyto!(out, inputs[1])` seeds the
  accumulator with the first term, then the loop adds every remaining term in place.

## Step 3 — build the recursive graph with `@neuro`

```julia
function build_stirling_graph(g, n)
    builder = @neuro g ns=:stirling operators=[:scale_add] begin
        @node one  = 1.0
        @node zero = 0.0

        @rule stirling(n::Int, k::Int) = begin
            if n == 0 && k == 0
                :one
            elseif k == 0 || k > n
                :zero
            else
                scale_add(factor=k, stirling(n-1, k), stirling(n-1, k-1))
            end
        end

        # Compute all S(n,k) for k=1..n
        all_terms = [stirling(n, i) for i in 1:n]
        @node bell = nsum(all_terms...)
    end
    return builder
end

n = 6
g = NeuroGraph(namespace=:stirling, device=Backend.CPUDevice())
builder = build_stirling_graph(g, n)
```

Line by line:

- `@neuro g ns=:stirling operators=[:scale_add] begin ... end` — `g` is the `NeuroGraph` every
  node gets added to; `ns=:stirling` is a *namespace*, a label that partitions nodes so unrelated
  graphs can share one `NeuroGraph` without name clashes; `operators=[:scale_add]` tells the
  macro which extra op names (beyond the built-in `nsum`/`identity`/`wsum`/`add`) should be
  recognized as op calls rather than as ordinary Julia function calls when it scans the block.
  `@neuro` expands, at macro-expansion time, into code that builds a `GraphBuilder` (holding the
  graph, the namespace, and a `memo::Dict` used for memoization below) and rewrites every
  `@node`/`@rule` line inside the block into calls against that builder.
- `@node one = 1.0` and `@node zero = 0.0` — each expands directly to
  `set!(g, :one, Float32[1.0]; namespace=:stirling)` (and similarly for `:zero`): a *leaf* node
  with no rule attached, holding a fixed 1-element array.
- `@rule stirling(n::Int, k::Int) = begin ... end` — this does **not** run the recurrence. It
  stores the `if/elseif/else` body as an ordinary Julia closure in `builder.rules[:stirling]` and
  returns immediately. Nothing about $S(n,k)$ has been computed or even referenced yet at this
  point — this line only teaches the builder *how* to compute `stirling(n,k)` the first time it's
  asked for.
- `all_terms = [stirling(n, i) for i in 1:n]` — **this is where the recursion actually fires.**
  Inside a `@neuro` block, a call to a name declared via `@rule` (like `stirling(...)`) is
  rewritten to `call_rule(builder, :stirling, n, i)`. `call_rule` computes the key
  `(:stirling, n, i)`, and:
  - if that exact key is already in `builder.memo` (because some earlier call already reached the
    same `(n,k)` pair), it returns the existing node's symbol immediately — **no recomputation,
    no new node** — this one `Dict` lookup is the entire memoization mechanism;
  - otherwise it calls the stored closure, which runs the recurrence's `if/elseif/else` *right
    now*, recursively calling `call_rule` again for `stirling(n-1,k)` and `stirling(n-1,k-1)` --
    each such nested call is memoized the same way -- until every branch bottoms out at `:one` or
    `:zero`, wraps the result in a freshly named node (e.g. `stirling_7`), records it in the memo
    `Dict`, and returns its symbol.

  So by the time this one list comprehension finishes, every distinct `(n,k)` pair the recurrence
  ever actually visits for $n=6$ has become exactly one graph node — 46 of them in total for this
  example, not the far larger count a non-memoized recursive `stirling(6,3)` would otherwise
  retrigger on every shared sub-call.
- `@node bell = nsum(all_terms...)` — `nsum` is a registered op name (Step 2), so this expands
  into an `addrule!` call wiring a new `nsum`-rule node to all of `all_terms`, then aliases it to
  the name `:bell` via a small `:identity` rule if the two symbols differ.
- `n = 6; g = NeuroGraph(namespace=:stirling, device=Backend.CPUDevice())` — `NeuroGraph(...)`
  constructs an empty graph (empty `nodes`/`rules` dictionaries) tagged with a default namespace
  and a *device* — here plain CPU arrays; swapping in `CUDADevice()` is the only change needed to
  run the same graph on GPU, since every op above is written against `dev`/`out`/`inputs` rather
  than concrete array types.
- `builder = build_stirling_graph(g, n)` — calling the function actually **executes** everything
  described above, synchronously, in ordinary Julia call-stack order. By the time this line
  returns, `g` already contains all 46 nodes and every `GraphRule` connecting them. What has
  *not* happened yet is any arithmetic: no op's function body (`scale_add`'s `@.`, `nsum`'s loop)
  has been invoked. The graph's **shape** is fully built; its **values** are not.

## Step 4 — run it

```julia
log = ExecutionLog()
ctx = CtxStore()
NeuroDSL.demand!(g, :bell; ctx_store=ctx, namespace=:stirling, log=log)

bell_val = Array(NeuroDSL.node(g, :bell; namespace=:stirling).value)[]
@printf "Bell number B_%d = %d\n" n Int(bell_val)
```

```
Bell number B_6 = 203
```

- `log = ExecutionLog()` — just a growable list of event records (`node`, `phase`, `status`,
  `value`); passing it to `demand!` makes every rule execution below append one entry, which is
  the raw material `save_interactive_graph` turns into a step-through timeline.
- `ctx = CtxStore()` — a shared context object threaded through to every op's `ctx` argument, for
  ops that need cross-call state. None of `scale_add`/`identity`/`nsum` use it, but `demand!`
  requires one regardless.
- `NeuroDSL.demand!(g, :bell; ...)` — the only call that does arithmetic. Internally it computes
  the graph's topological order once (nodes before whatever depends on them), then walks that
  order and, for every node that doesn't already have a valid value, calls its op's function
  (writing into that node's `out` buffer) — skipping nodes already computed, and stopping as soon
  as `:bell` itself has been produced. Every one of the 46 nodes gets computed exactly once, in
  dependency order, bottom-up from `:one`/`:zero`.
- `NeuroDSL.node(g, :bell; namespace=:stirling).value` — every node's `.value` field is always an
  `AbstractArray` (never a bare scalar, even for a 1-element result), so `Array(...)` copies it
  off-device into a plain CPU `Array` (a no-op here since we're already on CPU, but the same code
  works unchanged if `device=CUDADevice()` had been used instead), and the trailing `[]` pulls out
  the lone element, since every node in this graph holds a 1-element array all the way through.

## Step 5 — export the interactive trace

```julia
save_interactive_graph(g, log, "stirling_bell_6.html"; title = "Stirling & Bell numbers (n=$n)")
```

This writes a single self-contained HTML file: an interactive, pannable/zoomable view of the
whole graph, with the recursion's actual unfolding order recorded from the `ExecutionLog`, so you
can step through *which* node got computed *when*, hover any node to see its formula
(e.g. `scale_add_16 = 2·stirling(2, 2) + stirling(2, 1)`), and see at a glance how the recursion's
shared sub-calls converge onto shared nodes instead of fanning out into separate branches.



## Links

- NeuroDSL on GitHub: <https://github.com/nevermind78/NeuroDSL>
- Previous post: [Mutable Computational Graphs in Pure Julia: Performing "Hot Surgery" on a LLaMA Transformer](https://discourse.julialang.org/t/mutable-computational-graphs-in-pure-julia-performing-hot-surgery-on-a-llama-transformer/138018) — the same engine, inserting a residual block into a live, mid-training transformer instead of computing Stirling/Bell numbers.
- [Stirling numbers of the second kind](https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind), [Bell numbers](https://en.wikipedia.org/wiki/Bell_number)
