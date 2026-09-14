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
eqs = [D.(pv) .~ vv; D.(vv) .~ -lambda .* pv .+ [0,0,-g]; p'*p ~ L^2]
@named sys = ODESystem(eqs, t)
constraint_eq = equations(sys)[end]
println("stored constraint equation: ", constraint_eq)
rhs_minus_lhs = Symbolics.unwrap(constraint_eq.rhs) - Symbolics.unwrap(constraint_eq.lhs)
println("rhs - lhs = ", rhs_minus_lhs)
println("typeof = ", typeof(rhs_minus_lhs))

D_t = Differential(t)
println("\nDirectly calling Symbolics.derivative(rhs-lhs, t; throw_no_derivative=true):")
try
    r = Symbolics.derivative(rhs_minus_lhs, Symbolics.unwrap(t); throw_no_derivative=true)
    println("OK: ", r)
catch e
    println("THREW: ", sprint(showerror, e))
end
