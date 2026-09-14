import Pkg
Pkg.activate(joinpath(@__DIR__, "mtk_translator_env"); io=devnull)
using Symbolics, LinearAlgebra

@variables t
@variables p(t)[1:3]
@variables L
D = Differential(t)

println("=== D(dot(p,p) - L^2)  [minus on the RIGHT] ===")
try
    r = Symbolics.derivative(dot(p,p) - L^2, Symbolics.unwrap(t); throw_no_derivative=true)
    println("OK: ", r)
catch e
    println("THREW: ", sprint(showerror, e))
end

println("\n=== D(L^2 - dot(p,p))  [minus on the LEFT, exactly eq.rhs-eq.lhs shape for dot(p,p)~L^2] ===")
try
    r = Symbolics.derivative(L^2 - dot(p,p), Symbolics.unwrap(t); throw_no_derivative=true)
    println("OK: ", r)
catch e
    println("THREW: ", sprint(showerror, e))
end

println("\n=== D(L^2 - p'*p)  [exactly eq.rhs-eq.lhs shape for p'*p ~ L^2] ===")
try
    r = Symbolics.derivative(L^2 - p'*p, Symbolics.unwrap(t); throw_no_derivative=true)
    println("OK: ", r)
catch e
    println("THREW: ", sprint(showerror, e))
end

println("\n=== inspect structure of L^2 - dot(p,p) ===")
ex = Symbolics.unwrap(L^2 - dot(p,p))
println("ex = ", ex)
println("operation(ex) = ", operation(ex))
for a in arguments(ex)
    println("  addend: ", a, "   operation=", iscall(a) ? operation(a) : "N/A")
end
