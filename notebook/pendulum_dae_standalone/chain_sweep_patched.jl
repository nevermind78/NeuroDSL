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
logmsg("using done (PATCHED)")

# A genuine chain of N coupled 3D spherical pendulums: bob i is linked to bob i-1 (bob 0 is
# a fixed anchor at the origin) by a rigid massless link of length L_i, enforced by the
# holonomic constraint dot(r_i, r_i) ~ L_i^2 where r_i = p_i - p_{i-1}. Each bob's equation
# of motion feels its own link's tension AND (for i<N) a reaction from the next link down
# the chain -- a structurally real multi-body DAE shape, NOT a synthetic array-width
# rescaling of one equation. Each individual constraint is still an ordinary 3-vector (so
# this sweep is NOT expected to exercise the O(1)-vs-O(n) compact-derivative size win --
# that requires a WIDE array per constraint, see array_width_sweep_*.jl); it instead tests
# whether the fix costs more, or breaks in some new way, as the NUMBER of coupled
# reduction-shaped constraints and the coupling between them grows -- a distinct, more
# "real-multibody-DAE-shaped" complexity axis, and the one this session was specifically
# asked to probe for new bugs/limits at scale.
Ns = [1, 2, 4, 8, 16, 32, 48, 64, 96]
results = Tuple{Int, Float64, Bool, Int, Int}[]  # (N, seconds, succeeded, n_eqs, total_node_count)

for N in Ns
    @parameters g = 9.81
    Lvals = ones(N)
    @parameters L[1:N] = Lvals
    @variables p(t)[1:N, 1:3]   # row i = bob i's position, avoids matrix-slice getindex shapes
    @variables v(t)[1:N, 1:3]
    @variables lambda(t)[1:N]
    Lp = collect(L)
    lamv = collect(lambda)
    # Represent each bob's position/velocity as its OWN bare 3-vector (a Vector of
    # independent scalar variables per component), matching exactly the shape the fix was
    # developed and verified against (p(t)[1:3] as a bare leaf array), rather than slicing
    # one big N x 3 matrix (an untested, structurally different getindex shape).
    pvecs = Vector{Vector{Num}}(undef, N)
    vvecs = Vector{Vector{Num}}(undef, N)
    for i in 1:N
        pvecs[i] = [p[i, 1], p[i, 2], p[i, 3]]
        vvecs[i] = [v[i, 1], v[i, 2], v[i, 3]]
    end
    anchor = zeros(3)
    eqs = Equation[]
    for i in 1:N
        append!(eqs, D.(pvecs[i]) .~ vvecs[i])
    end
    for i in 1:N
        rprev = i == 1 ? anchor : pvecs[i - 1]
        r_i = pvecs[i] .- rprev
        reaction = i < N ? lamv[i + 1] .* (pvecs[i + 1] .- pvecs[i]) : zeros(3)
        append!(eqs, D.(vvecs[i]) .~ -lamv[i] .* r_i .+ reaction .+ [0, 0, -g])
    end
    for i in 1:N
        rprev = i == 1 ? anchor : pvecs[i - 1]
        r_i = pvecs[i] .- rprev
        push!(eqs, dot(r_i, r_i) ~ Lp[i]^2)
    end
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
        logmsg("N=$N THREW: $(sprint(showerror, e))")
    end
    dt = time() - tstart
    push!(results, (N, dt, ok, neq, nc))
    logmsg("N=$N (7N=$(7N) raw eqs) : mtkcompile took $(round(dt, digits=3))s, succeeded=$ok, n_eqs=$neq, total_node_count=$nc")
    dt > 150 && (logmsg("aborting sweep: N=$N already took >150s"); break)
end

open(joinpath(@__DIR__, "chain_patched_results.tsv"), "w") do io
    println(io, "N\tseconds\tsucceeded\tn_eqs\ttotal_node_count")
    for (N, dt, ok, neq, nc) in results
        println(io, "$N\t$dt\t$ok\t$neq\t$nc")
    end
end
logmsg("done")
