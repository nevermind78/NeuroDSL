t0 = time()
function logmsg(s)
    println(s, "  [+", round(time() - t0, digits=1), "s]")
    flush(stdout)
end
import Pkg
Pkg.activate(@__DIR__; io=devnull)
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra
using SymbolicUtils: node_count
logmsg("using done (STOCK, genuinely isolated: file-reverted copies of Symbolics diff.jl, linear_algebra.jl, MTKTearing reassemble.jl)")

# Single spherical-pendulum-shaped DAE with ONE holonomic dot(p,p)~L^2 constraint over an
# n-dimensional array p. This is the axis where the compact-derivative rule can plausibly
# pay off: compact keeps the constraint's first-round derivative as O(1) (`dot(p,D(p))`)
# regardless of n, while stock/scalarized differentiation produces an explicit O(n) sum.
# Extends the prior n=3..30 sweep (which found no measured difference, but -- discovered
# in this session -- ran "stock" against an environment that Julia's content-addressed
# package cache had silently aliased onto the SAME patched install, invalidating that
# comparison) further, to see if a difference emerges at larger n, and re-measures with a
# genuine, freshly-verified, file-level-isolated stock environment as the counterpart.
ns = [3, 10, 20, 30, 50, 75, 100, 150, 200, 300]
results = Tuple{Int, Float64, Bool, Int, Int}[]  # (n, seconds, succeeded, n_eqs, total_node_count)

for n in ns
    @parameters L = 1.0
    @variables p(t)[1:n] = ones(n) ./ sqrt(n)
    @variables v(t)[1:n] = zeros(n)
    @variables lambda(t) = 1.0
    pv = collect(p); vv = collect(v)
    F = zeros(n)
    eqs = [D.(pv) .~ vv; D.(vv) .~ -lambda .* pv .+ F; dot(p, p) ~ L^2]
    @named sys = ODESystem(eqs, t)
    tstart = time()
    ok = true
    nc = -1
    neq = -1
    try
        simplified = ModelingToolkit.mtkcompile(sys)
        eqs2 = equations(simplified)
        neq = length(eqs2)
        nc = sum(node_count(eq.lhs) + node_count(eq.rhs) for eq in eqs2)
        @assert neq > 0
    catch e
        ok = false
        logmsg("n=$n THREW: $(sprint(showerror, e))")
    end
    dt = time() - tstart
    push!(results, (n, dt, ok, neq, nc))
    logmsg("n=$n : mtkcompile took $(round(dt, digits=3))s, succeeded=$ok, n_eqs=$neq, total_node_count=$nc")
    dt > 120 && (logmsg("aborting sweep: n=$n already took >120s"); break)
end

open(joinpath(@__DIR__, "array_width_stock_results.tsv"), "w") do io
    println(io, "n\tseconds\tsucceeded\tn_eqs\ttotal_node_count")
    for (n, dt, ok, neq, nc) in results
        println(io, "$n\t$dt\t$ok\t$neq\t$nc")
    end
end
logmsg("done")
