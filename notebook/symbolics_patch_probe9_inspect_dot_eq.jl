import Pkg
Pkg.activate(joinpath(@__DIR__, "mtk_translator_env"); io=devnull)
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra, Symbolics

@parameters L=1.0 g=9.81
@variables p(t)[1:3] = [1.0, 0.0, 0.0]
@variables v(t)[1:3] = [0.0, 0.0, 0.0]
@variables lambda(t) = 1.0
pv = collect(p); vv = collect(v)
eqs = [D.(pv) .~ vv; D.(vv) .~ -lambda .* pv .+ [0,0,-g]; dot(pv,pv) ~ L^2]
@named sys = ODESystem(eqs, t)
constraint_eq = equations(sys)[end]
println("stored constraint equation: ", constraint_eq)
rhs_minus_lhs = Symbolics.unwrap(constraint_eq.rhs) - Symbolics.unwrap(constraint_eq.lhs)
println("rhs - lhs = ", rhs_minus_lhs)
println("operation(rhs-lhs) = ", operation(rhs_minus_lhs))
println("arguments(rhs-lhs) = ", arguments(rhs_minus_lhs))
