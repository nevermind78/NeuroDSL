import Pkg
Pkg.activate(joinpath(@__DIR__, "mtk_translator_env"))
using Symbolics
using LinearAlgebra

println("Symbolics version: ", Pkg.dependencies()[Base.UUID("0c5d862f-8b57-4792-8d23-62f2024744c7")].version)
println("SymbolicUtils version: ", Pkg.dependencies()[Base.UUID("d1185830-fcd6-423d-90d6-eec64667417b")].version)

@variables t
@variables p(t)[1:3]
D = Differential(t)

println("\n--- Test A: D(dot(p,p)) current (unpatched) behavior ---")
try
    ex = D(dot(p, p))
    println("Built D(dot(p,p)) = ", ex)
    res = expand_derivatives(ex)
    println("expand_derivatives result = ", res)
    println("get_variables(res) = ", Symbolics.get_variables(res))
catch e
    println("THREW: ", sprint(showerror, e))
end

println("\n--- Test B: bare array-shaped Differential(t)(p) expand_derivatives ---")
try
    ex2 = D(p)
    println("Built D(p) = ", ex2, "  (shape-array term)")
    res2 = expand_derivatives(ex2)
    println("expand_derivatives(D(p)) = ", res2)
catch e
    println("THREW: ", sprint(showerror, e))
end

println("\n--- Test C: p'*p current behavior ---")
try
    ex3 = D(p'*p)
    println("Built D(p'*p) = ", ex3)
    res3 = expand_derivatives(ex3)
    println("expand_derivatives(p'*p) = ", res3)
catch e
    println("THREW: ", sprint(showerror, e))
end

println("\n--- Test D: D(norm(p)) current behavior ---")
try
    ex4 = D(norm(p))
    res4 = expand_derivatives(ex4)
    println("expand_derivatives(norm(p)) = ", res4)
    println("get_variables = ", Symbolics.get_variables(res4))
catch e
    println("THREW: ", sprint(showerror, e))
end

println("\n--- Test E: raw executediff on bare array Sym (not via D()) ---")
try
    res5 = Symbolics.executediff(D, Symbolics.unwrap(p))
    println("executediff(D, p) = ", res5)
catch e
    println("THREW: ", sprint(showerror, e))
end
