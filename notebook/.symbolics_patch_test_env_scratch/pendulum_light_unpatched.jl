t0 = time()
function logmsg(s)
    println(s, "  [+", round(time() - t0, digits=1), "s]")
    flush(stdout)
end

import Pkg
Pkg.activate(@__DIR__; io=devnull)
logmsg("Pkg.activate done")

using ModelingToolkit
logmsg("using ModelingToolkit done")

using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra
logmsg("using LinearAlgebra done")

@parameters L=1.0 g=9.81
@variables p(t)[1:3] = [1.0, 0.0, 0.0]
@variables v(t)[1:3] = [0.0, 0.0, 0.0]
@variables lambda(t) = 1.0
logmsg("variables declared")

pv = collect(p)
vv = collect(v)

eqs = [
    D.(pv) .~ vv;
    D.(vv) .~ -lambda .* pv .+ [0, 0, -g];
    dot(p, p) ~ L^2;
]
logmsg("equations built")

@named sys = ODESystem(eqs, t)
logmsg("ODESystem built")
logmsg("stored constraint equation: $(equations(sys)[end])")

logmsg("=== calling mtkcompile with UNCOLLECTED dot(p,p) ~ L^2 (UNPATCHED Symbolics, comparison run) ===")
try
    simplified = ModelingToolkit.mtkcompile(sys)
    logmsg("SUCCEEDED")
    logmsg("n_eqs = $(length(equations(simplified)))")
    logmsg("unknowns = $(unknowns(simplified))")
catch e
    logmsg("THREW: $(sprint(showerror, e))")
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        i > 25 && break
        logmsg("  $frame")
    end
end
