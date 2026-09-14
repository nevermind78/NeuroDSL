import Pkg
Pkg.activate(joinpath(@__DIR__, "mtk_translator_env"))
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra

@parameters L=1.0 g=9.81
@variables p(t)[1:3] = [1.0, 0.0, 0.0]
@variables v(t)[1:3] = [0.0, 0.0, 0.0]
@variables lambda(t) = 1.0

pv = collect(p)
vv = collect(v)

# Literally p'*p, the idiom pass 6 flagged as crashing today ("Differentiation with array
# expressions is not yet supported", matching JuliaSymbolics/Symbolics.jl#1039 / #567).
eqs = [
    D.(pv) .~ vv;
    D.(vv) .~ -lambda .* pv .+ [0, 0, -g];
    pv' * pv ~ L^2;
]

@named sys = ODESystem(eqs, t)

println("=== mtkcompile with literal p'*p ~ L^2 constraint ===")
try
    simplified = ModelingToolkit.mtkcompile(sys)
    println("SUCCEEDED")
    println("n_eqs = ", length(equations(simplified)))
    println("unknowns = ", unknowns(simplified))
catch e
    println("THREW: ", sprint(showerror, e))
end
