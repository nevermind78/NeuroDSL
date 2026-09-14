import Pkg
Pkg.activate(joinpath(@__DIR__, "mtk_translator_env"))
using Symbolics
using LinearAlgebra

println("=== Regression: ordinary scalar differentiation unaffected ===")
@variables x y z k
Dx = Differential(x)

f1 = k*(abs(x-y)/y-z)^2
r1 = expand_derivatives(Dx(f1))
println("k*(abs(x-y)/y-z)^2 wrt x -> ", r1)

f2 = x^3*sin(y) + exp(x*z)
r2 = expand_derivatives(Dx(f2))
expected2 = 3*x^2*sin(y) + z*exp(x*z)
diffcheck = Symbolics.simplify(r2 - expected2)
println("x^3*sin(y)+exp(x*z) wrt x -> ", r2)
println("  diff from hand-derived (expect 0): ", diffcheck)

f3 = (x + y)/(x - y)
r3 = expand_derivatives(Dx(f3))
println("(x+y)/(x-y) wrt x -> ", r3)

@variables xx[1:3]
f4 = sum(xx .^ 2)
Dxx1 = Differential(xx[1])
r4 = expand_derivatives(Dxx1(f4))
println("sum(xx.^2) wrt xx[1] -> ", r4, "   (expect 2*xx[1], still scalarizes -- ordinary mapreduce, untouched by patch)")

println("\n=== Second-order ordinary scalar differentiation (unaffected) ===")
Dx2 = Differential(x)^2
r5 = expand_derivatives(Dx2(x^4 + x^2*y))
println("D^2(x^4+x^2*y) wrt x -> ", r5, " (expect 12x^2+2y)")

println("\n=== Bonus: Symbolics.jacobian silent-zero pre-scan check ===")
@variables t2
@variables p2(t2)[1:2]
eqs_jac = [dot(p2, p2) - 1]
vars_jac = Symbolics.scalarize(p2)
J = Symbolics.jacobian(eqs_jac, vars_jac)
println("jacobian([dot(p2,p2)-1], [p2[1],p2[2]]) = ", J)
println("  (BEFORE any pre-scan fix: this is expected to show whether search_variables misses p2[i] inside dot(...))")

println("\n=== Try running a slice of Symbolics' own derivatives.jl test file ===")
testfile = joinpath(dirname(Base.find_package("Symbolics")), "..", "test", "derivatives.jl")
println("looking for: ", testfile, "  exists=", isfile(testfile))
