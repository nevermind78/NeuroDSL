import Pkg
Pkg.activate(@__DIR__; io=devnull)
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq, LinearAlgebra

@parameters L = 1.0 g = 9.81
@variables p(t)[1:3] = [0.6, 0.0, -0.8]
@variables v(t)[1:3] = [0.0, 0.9, 0.0]
@variables lambda(t) = 1.0
pv = collect(p); vv = collect(v)
eqs = [D.(pv) .~ vv; D.(vv) .~ -lambda .* pv .+ [0, 0, -g]; dot(p, p) ~ L^2]
@named sys = ODESystem(eqs, t)
simplified = ModelingToolkit.mtkcompile(sys)
lambda0 = (0.81 - 9.81 * (-0.8)) / 1.0
u0 = [pv[1] => 0.6, pv[2] => 0.0, pv[3] => -0.8, vv[1] => 0.0, vv[2] => 0.9, vv[3] => 0.0, lambda => lambda0]
prob = ODEProblem(simplified, u0, (0.0, 1.14))
sol = solve(prob, Rodas5P(); abstol=1e-9, reltol=1e-9, saveat=0.02, maxiters=10_000_000, dtmax=0.01)
println("retcode=", sol.retcode, " last t=", sol.t[end])
for ti in 0.8:0.02:min(1.14, sol.t[end])
    p1 = sol(ti; idxs=pv[1])
    println("t=", round(ti,digits=2), "  p1=", p1)
end
