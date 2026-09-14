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
prob = ODEProblem(simplified, u0, (0.0, 5.0))
f = prob.f
println("state order: ", unknowns(simplified))

utest = [0.4123456789, 0.9987654321, -0.6234567891, 0.7345678912, 3.696e-08, -1.025e-05, 9.984]
du = zeros(7)
f(du, utest, prob.p, 1.15)
println("PATCHED f(utest) = ")
for (i,x) in enumerate(du); println("  du[$i] = ", repr(x)); end
println("params: ", prob.p)
