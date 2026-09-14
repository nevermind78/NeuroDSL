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
logmsg("using done")

@parameters L=1.0 g=9.81
@variables p(t)[1:3] = [1.0, 0.0, 0.0]
@variables v(t)[1:3] = [0.0, 0.0, 0.0]
@variables lambda(t) = 1.0

pv = collect(p)
vv = collect(v)

eqs = [
    D.(pv) .~ vv;
    D.(vv) .~ -lambda .* pv .+ [0, 0, -g];
    dot(p, p) ~ L^2;
]

@named sys = ODESystem(eqs, t)
logmsg("ODESystem built")

simplified = ModelingToolkit.mtkcompile(sys)
logmsg("mtkcompile SUCCEEDED")
logmsg("n_eqs = $(length(equations(simplified)))")
for (i, eq) in enumerate(equations(simplified))
    logmsg("  eq[$i]: $eq")
end
logmsg("unknowns = $(unknowns(simplified))")
logmsg("observed:")
for (i, eq) in enumerate(observed(simplified))
    logmsg("  obs[$i]: $eq")
end

logmsg("=== full_equations(simplified) ===")
try
    fe = ModelingToolkit.full_equations(simplified)
    for (i, eq) in enumerate(fe)
        logmsg("  full_eq[$i]: $eq")
    end
catch e
    logmsg("full_equations THREW: $(sprint(showerror, e))")
end

logmsg("=== attempting ODEProblem ===")
try
    u0 = [pv[1] => 0.6, pv[2] => 0.0, pv[3] => -0.8, vv[1] => 0.0, vv[2] => 0.9, vv[3] => 0.0]
    prob = ODEProblem(simplified, u0, (0.0, 1.0))
    logmsg("ODEProblem SUCCEEDED")
catch e
    logmsg("ODEProblem THREW: $(sprint(showerror, e))")
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        i > 40 && break
        logmsg("  $frame")
    end
end
