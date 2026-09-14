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
logmsg("using done (stock, unpatched Symbolics)")

@parameters L=1.0 g=9.81
@variables p(t)[1:3] = [1.0, 0.0, 0.0]
@variables v(t)[1:3] = [0.0, 0.0, 0.0]
@variables lambda(t) = 1.0
pv = collect(p); vv = collect(v)
eqs = [D.(pv) .~ vv; D.(vv) .~ -lambda .* pv .+ [0, 0, -g]; dot(p, p) ~ L^2]
@named sys = ODESystem(eqs, t)
logmsg("ODESystem built; calling mtkcompile (bounded timing probe, stock Symbolics compact dot derivative -> scalarizes eagerly)")
simplified = ModelingToolkit.mtkcompile(sys)
logmsg("mtkcompile SUCCEEDED (unexpectedly fast for stock Symbolics on this shape)")
logmsg("n_eqs = $(length(equations(simplified)))")
