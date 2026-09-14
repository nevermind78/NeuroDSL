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

# Constraint uses the UNCOLLECTED Arr `p` directly (not `pv = collect(p)`), matching the
# exact lazy adjoint-matmul Term shape that triggers the array-differentiation crash at
# the bare-Symbolics level (see symbolics_patch_probe0_baseline.jl Test C).
eqs = [
    D.(pv) .~ vv;
    D.(vv) .~ -lambda .* pv .+ [0, 0, -g];
    p' * p ~ L^2;
]

@named sys = ODESystem(eqs, t)

println("=== mtkcompile with UNCOLLECTED p'*p ~ L^2 (Arr, not collected Vector) ===")
try
    simplified = ModelingToolkit.mtkcompile(sys)
    println("SUCCEEDED")
    println("n_eqs = ", length(equations(simplified)))
    println("unknowns = ", unknowns(simplified))
catch e
    println("THREW: ", sprint(showerror, e))
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        i > 40 && break
        println("  ", frame)
    end
end
