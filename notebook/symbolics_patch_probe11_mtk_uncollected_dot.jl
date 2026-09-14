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

# dot(p, p) with the UNCOLLECTED Arr `p` directly -- exercises Symbolics' own lazy
# `LinearAlgebra.dot(::BasicSymbolic,::BasicSymbolic)` term, not Base's eager Vector dot.
eqs = [
    D.(pv) .~ vv;
    D.(vv) .~ -lambda .* pv .+ [0, 0, -g];
    dot(p, p) ~ L^2;
]

@named sys = ODESystem(eqs, t)
println("stored constraint equation: ", equations(sys)[end])

println("\n=== mtkcompile with UNCOLLECTED dot(p,p) ~ L^2 ===")
try
    simplified = ModelingToolkit.mtkcompile(sys)
    println("SUCCEEDED")
    println("n_eqs = ", length(equations(simplified)))
    println("unknowns = ", unknowns(simplified))
catch e
    println("THREW: ", sprint(showerror, e))
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        i > 30 && break
        println("  ", frame)
    end
end
