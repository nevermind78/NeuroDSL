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
logmsg("using done (GENUINELY ISOLATED stock Symbolics: reverted diff.jl/linear_algebra.jl, reverted MTKTearing reassemble.jl)")

@parameters L = 1.0 g = 9.81
@variables p(t)[1:3] = [0.6, 0.0, -0.8]
@variables v(t)[1:3] = [0.0, 0.9, 0.0]
@variables lambda(t) = 1.0
pv = collect(p); vv = collect(v)

eqs = [D.(pv) .~ vv; D.(vv) .~ -lambda .* pv .+ [0, 0, -g]; dot(p, p) ~ L^2]
@named sys = ODESystem(eqs, t)
logmsg("ODESystem built with $(length(eqs)) equations")

tstart = time()
try
    simplified = ModelingToolkit.mtkcompile(sys)
    logmsg("mtkcompile SUCCEEDED in $(round(time()-tstart, digits=3))s")
    logmsg("n_eqs = $(length(equations(simplified)))")
    for (i, eq) in enumerate(equations(simplified))
        logmsg("  eq[$i]: $eq")
    end
catch e
    logmsg("mtkcompile THREW after $(round(time()-tstart, digits=3))s: $(sprint(showerror, e))")
end
