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
println("=== PATCHED simplified equations ===")
for eq in equations(simplified)
    println("  ", eq)
end
println("unknowns: ", unknowns(simplified))
