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
logmsg("using done (PATCHED Symbolics: compact dot derivative)")

# Generalized "spherical pendulum"-shaped DAE on n position/velocity components with a
# single holonomic dot(p,p) ~ L^2 constraint, for n = 3, 6, 10, 15, 20, 30. Same equation
# SHAPE as the real 3D pendulum (this is the exact shape that exercises the compact
# dot-derivative rule and the dummy-derivative substitution fix), just scaled up, purely
# to demonstrate mtkcompile cost scaling. Not meant to be a physically meaningful system
# for n > 3.
ns = [3, 6, 10, 15, 20, 30]
results = Tuple{Int, Float64, Bool}[]  # (n, seconds, succeeded)

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
    try
        simplified = ModelingToolkit.mtkcompile(sys)
        @assert length(equations(simplified)) > 0
    catch e
        ok = false
        logmsg("n=$n THREW: $(sprint(showerror, e))")
    end
    dt = time() - tstart
    push!(results, (n, dt, ok))
    logmsg("n=$n : mtkcompile took $(round(dt, digits=3))s, succeeded=$ok")
end

open(joinpath(@__DIR__, "scaling_patched_results.tsv"), "w") do io
    println(io, "n\tseconds\tsucceeded")
    for (n, dt, ok) in results
        println(io, "$n\t$dt\t$ok")
    end
end
logmsg("done")
