import Pkg
Pkg.activate(joinpath(@__DIR__, "mtk_translator_env"))
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra
using Symbolics

println("ModelingToolkit version: ", Pkg.dependencies()[Base.UUID("961ee093-0014-501f-94e3-6117800e7a78")].version)

# Classic index-3 spherical pendulum DAE, written with the reduction-shaped constraint
# dot(p,p) - L^2 ~ 0, exactly the motivating example from pass 6 / this task.
@parameters L=1.0 g=9.81
@variables p(t)[1:3] = [1.0, 0.0, 0.0]
@variables v(t)[1:3] = [0.0, 0.0, 0.0]
@variables lambda(t) = 1.0

pv = collect(p)
vv = collect(v)

eqs = [
    D.(pv) .~ vv;
    D.(vv) .~ -lambda .* pv .+ [0, 0, -g];
    dot(pv, pv) ~ L^2;
]

println("\nBuilding ODESystem with ", length(eqs), " equations (", length(pv), " position + ",
        length(vv), " velocity + 1 constraint)")

@named sys = ODESystem(eqs, t)

println("\n=== Attempting structural_simplify / mtkcompile (real Pantelides index reduction) ===")
simplified = nothing
try
    if isdefined(ModelingToolkit, :mtkcompile)
        simplified = ModelingToolkit.mtkcompile(sys)
        println("mtkcompile SUCCEEDED")
    else
        simplified = structural_simplify(sys)
        println("structural_simplify SUCCEEDED")
    end
    println(simplified)
    println("\nnumber of equations after simplification: ", length(equations(simplified)))
    println("unknowns: ", unknowns(simplified))
catch e
    println("THREW: ", sprint(showerror, e))
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        i > 25 && break
        println("  ", frame)
    end
end
