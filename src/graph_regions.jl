# ══════════════════════════════════════════════════════════════════════════════
# graph_regions.jl — Partitioning a compute graph into regions by CUTTING.
#
# Sibling of `graph_surgery.jl` (which mutates an architecture in place). Here we
# do not mutate the computed function at all — we DECOMPOSE one declared graph
# into region-tagged pieces along boundaries the caller chooses, so a downstream
# host (NeuroSlate) can run each region on its own worker.
#
# The algebra is deliberately small and compositional:
#
#   • A `RegionPlan` holds a built graph plus a (partial) node → region colouring.
#   • A `GraphPiece` is a subset of the graph's nodes tied to a plan. The whole
#     graph is a piece; `cut(piece, boundary)` splits ONE piece into TWO —
#       – upstream:   the boundary nodes and everything they depend on (inside the
#                     piece), i.e. the closure of the seam;
#       – downstream: the rest of the piece.
#     Either resulting piece can be cut again — a graph is carved with as many
#     cuts as its shape needs.
#   • `assign!(piece, region)` colours a whole piece; region assignment is
#     ORTHOGONAL to cutting — colour at cut time (`cut!`'s `above`/`below`, either
#     may be left `nothing`) or later on any piece.
#
# The cross-region transfers are then DERIVED from the colouring, never wired by
# hand: an edge `u → v` where `region(u) ≠ region(v)` and `u` is a COMPUTED node
# means "materialise `u` in its region, ship it, inject it as the leaf `u` in
# `v`'s region." Pure leaves (constants / params) are replicated locally in every
# region that reads them rather than shipped. `restrict_to_region!` prunes a fresh
# build down to one region: its own nodes keep their rules, incoming boundary
# sources become injectable leaves, and everything else is dropped — so a region
# never recomputes another region's cone.
# ══════════════════════════════════════════════════════════════════════════════

# ── Resolving named nodes from a rule row ────────────────────────────────────────
#
# Rule-call nodes are auto-named (`stirling_12`, …), but each is tagged with its
# structured call `(rule, args...)` in `aux_data[:call]` (set in `call_rule`), so a
# boundary can be resolved programmatically — e.g. row n-1 of a Stirling triangle is
# `surface(g, :stirling, n-1)` — instead of hard-coding generated symbols.

"""
    surface(g, rule, prefix...; ns) -> Vector{Symbol}

The nodes produced by rule-call `rule` whose leading arguments match `prefix`, in
argument order. `surface(g, :stirling, 5)` returns the row-5 nodes `stirling(5,·)`
(constant results like `:zero`/`:one` are never tagged, so they are excluded). Use
the result as a cut boundary.
"""
function surface(g::NeuroGraph, rule::Symbol, prefix...; ns::Symbol = g.active_ns)
    hits = Tuple{Tuple, Symbol}[]
    for (sym, nd) in g.nodes[ns]
        call = get(nd.aux_data, :call, nothing)
        call === nothing && continue
        call[1] === rule || continue
        length(call) >= 1 + length(prefix) || continue
        all(call[1 + i] == prefix[i] for i in eachindex(prefix)) || continue
        push!(hits, (call, sym))
    end
    return [sym for (_, sym) in sort(hits; by = first)]
end

# ── The plan and its pieces ──────────────────────────────────────────────────────

"""
    RegionPlan(g; ns) -> RegionPlan

A node → region colouring over a built graph `g` (namespace `ns`). Start empty; grow
it with [`cut!`](@ref) / [`assign!`](@ref). The plan references a built instance only
for its TOPOLOGY — the colouring is a set of node labels, so the same builder rebuilt
on a worker yields the same symbols and the plan applies there verbatim.
"""
mutable struct RegionPlan
    graph     :: NeuroGraph
    ns        :: Symbol
    region_of :: Dict{Symbol, Symbol}
end
RegionPlan(g::NeuroGraph; ns::Symbol = g.active_ns) =
    (_ensure_namespace!(g, ns); RegionPlan(g, ns, Dict{Symbol, Symbol}()))

"""
    GraphPiece(plan, nodes)

A subset of a [`RegionPlan`](@ref)'s graph. Produced by [`cut`](@ref) / [`wholepiece`](@ref);
consumed by [`cut`](@ref) (to split further) and [`assign!`](@ref) (to colour).
"""
struct GraphPiece
    plan  :: RegionPlan
    nodes :: Set{Symbol}
end

"""
    wholepiece(plan) -> GraphPiece

The piece containing every node of the plan's graph — the starting point for the first cut.
"""
wholepiece(p::RegionPlan) = GraphPiece(p, Set(keys(p.graph.nodes[p.ns])))

"""
    piece(plan, nodes) -> GraphPiece

An arbitrary sub-piece from an explicit node set — the closure-free counterpart to [`cut`](@ref).
`cut` carves by dependency; `piece` names any subset directly, which is how an automatic placement
pass (reading per-node resource profiles + a mesh model) will express a colouring: `assign!(piece(
plan, gpu_nodes), :gpu_box)`. A cut is an ARBITRARY slice, so the colouring — not the closure shape —
is what [`crossings`](@ref) and [`restrict_to_region!`](@ref) act on.
"""
piece(p::RegionPlan, nodes) = GraphPiece(p, Set(Symbol.(collect(nodes))))

# A RegionPlan stands in for its whole-graph piece anywhere a piece is accepted.
_piece(p::GraphPiece)  = p
_piece(p::RegionPlan)  = wholepiece(p)
_plan(p::GraphPiece)   = p.plan
_plan(p::RegionPlan)   = p

Base.show(io::IO, p::RegionPlan) = print(io, "RegionPlan(", length(p.graph.nodes[p.ns]),
    " nodes · ", length(unique(values(p.region_of))), " regions · ",
    length(p.region_of), " coloured)")
Base.show(io::IO, p::GraphPiece) = print(io, "GraphPiece(", length(p.nodes), " nodes)")

# Boundary spec → concrete node symbols (a single node, or any iterable of them).
_boundary_syms(b::Symbol)  = Symbol[b]
_boundary_syms(b)          = collect(Symbol, b)

# ── Graph traversal within a piece ───────────────────────────────────────────────

# The seam and everything it (transitively) depends on, restricted to `within`.
function _upstream_closure(g::NeuroGraph, ns::Symbol, seeds, within::Set{Symbol})
    seen = Set{Symbol}()
    stack = collect(Symbol, seeds)
    while !isempty(stack)
        s = pop!(stack)
        (s in seen || !(s in within)) && continue
        push!(seen, s)
        rule = get(g.rules[ns], s, nothing)
        rule === nothing && continue
        for inp in rule.inputs
            inp in seen || push!(stack, inp)
        end
    end
    return seen
end

# ── Cutting ──────────────────────────────────────────────────────────────────────

"""
    cut(piece, boundary) -> (upstream::GraphPiece, downstream::GraphPiece)

Split `piece` along `boundary` (a node symbol, an iterable of them, or the result of
[`surface`](@ref)). `upstream` is the boundary plus everything it depends on inside the
piece; `downstream` is the rest of the piece. Neither piece is coloured — pass the result
to [`assign!`](@ref), or use [`cut!`](@ref) to colour in one step. Either piece can be cut
again. `piece` may be a [`RegionPlan`](@ref) (its whole graph) or a [`GraphPiece`](@ref).
"""
function cut(piece, boundary)
    pc = _piece(piece); plan = pc.plan; g = plan.graph; ns = plan.ns
    bsyms = _boundary_syms(boundary)
    for b in bsyms
        b in pc.nodes ||
            error("cut: boundary node :$b is not in this piece (ns :$ns)")
    end
    upstream = _upstream_closure(g, ns, bsyms, pc.nodes)
    downstream = setdiff(pc.nodes, upstream)
    return GraphPiece(plan, upstream), GraphPiece(plan, downstream)
end

"""
    assign!(piece, region) -> GraphPiece

Colour every node of `piece` with `region`. Explicit and last-write-wins, so re-cutting a
piece and reassigning a sub-piece REFINES the colouring. Colouring is independent of
cutting — a whole uncoloured piece can be assigned in its entirety. `piece` may be a
[`RegionPlan`](@ref) (colour the whole graph) or a [`GraphPiece`](@ref).
"""
function assign!(piece, region)
    pc = _piece(piece); reg = Symbol(region)
    for n in pc.nodes
        pc.plan.region_of[n] = reg
    end
    return pc
end

"""
    cut!(piece, boundary; above, below) -> (upstream, downstream)

Cut `piece` and colour the two results in one step: `above` → the upstream piece, `below` →
the downstream piece. Either may be omitted (`nothing`) to leave that side uncoloured — so a
`cut!` can peel one region, or just record a seam for later assignment. Returns the two pieces
for further cutting.
"""
function cut!(piece, boundary; above = nothing, below = nothing)
    u, d = cut(piece, boundary)
    above === nothing || assign!(u, above)
    below === nothing || assign!(d, below)
    return u, d
end

# ── Derived cross-region transfers ───────────────────────────────────────────────

"""
    crossings(plan) -> Vector{@NamedTuple{node::Symbol, src::Symbol, dst::Symbol}}

The transfers implied by the colouring: one entry per (computed node, destination region)
where a COMPUTED node feeds a consumer in another region. `node` is materialised in `src` and
injected as the leaf `node` in `dst`. Pure leaves (no rule — constants/params) are excluded:
they are replicated locally by [`restrict_to_region!`](@ref), not shipped.
"""
function crossings(plan::RegionPlan)
    g = plan.graph; ns = plan.ns; ro = plan.region_of
    seen = Set{Tuple{Symbol, Symbol}}()
    out = @NamedTuple{node::Symbol, src::Symbol, dst::Symbol}[]
    for (o, rule) in g.rules[ns]
        dst = get(ro, o, nothing)
        dst === nothing && continue
        for inp in rule.inputs
            haskey(g.rules[ns], inp) || continue          # leaf → replicated, not shipped
            src = get(ro, inp, nothing)
            (src === nothing || src === dst) && continue
            key = (inp, dst)
            key in seen && continue
            push!(seen, key)
            push!(out, (node = inp, src = src, dst = dst))
        end
    end
    return out
end

# ── Restricting a graph to one region ────────────────────────────────────────────

"""
    restrict_to_region!(g, region_of, region; ns) -> NeuroGraph

Prune a freshly built graph `g` in place down to one `region` under colouring `region_of`:

  • nodes coloured `region` keep their rules (computed locally);
  • a COMPUTED node in another region that feeds this region becomes an injectable LEAF
    (its rule dropped, value cleared) — the boundary tensor is set on it at run time;
  • pure leaves (constants/params) that this region reads are kept locally (replicated,
    never shipped);
  • every other node is removed.

The result computes exactly this region's slice — it never recomputes another region's cone.
Errors if a COMPUTED node feeding this region was left uncoloured (an incomplete plan).
"""
function restrict_to_region!(g::NeuroGraph, region_of::Dict{Symbol, Symbol}, region;
                             ns::Symbol = g.active_ns)
    reg = Symbol(region)
    keep = Set(n for (n, r) in region_of if r === reg && haskey(g.nodes[ns], n))
    inject = Set{Symbol}()   # computed boundary sources → leaves to inject
    replicate = Set{Symbol}()   # pure leaves read here → kept locally
    for n in keep
        rule = get(g.rules[ns], n, nothing)
        rule === nothing && continue
        for inp in rule.inputs
            inp in keep && continue
            if !haskey(g.rules[ns], inp)
                push!(replicate, inp)
            elseif get(region_of, inp, nothing) === nothing
                error("restrict_to_region!: computed node :$inp feeds region :$reg but is " *
                      "uncoloured — assign it to a region (cut!/assign!) before realising")
            else
                push!(inject, inp)
            end
        end
    end
    survive = union(keep, inject, replicate)
    for sym in collect(keys(g.nodes[ns]))
        if !(sym in survive)
            delete!(g.nodes[ns], sym)
            delete!(g.rules[ns], sym)
        end
    end
    for sym in inject                      # turn boundary sources into fresh injectable leaves
        delete!(g.rules[ns], sym)
        nd = g.nodes[ns][sym]
        nd.valid = false
        nd.value = nothing
    end
    g._topo_cache[ns] = nothing
    g._consumers_cache[ns] = nothing
    return g
end

restrict_to_region!(g::NeuroGraph, plan::RegionPlan, region) =
    restrict_to_region!(g, plan.region_of, region; ns = plan.ns)
