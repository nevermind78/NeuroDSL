import Pkg
Pkg.activate(joinpath(@__DIR__, "mtk_translator_env"))
using Symbolics
using LinearAlgebra
using Test

println("Symbolics version: ", Pkg.dependencies()[Base.UUID("0c5d862f-8b57-4792-8d23-62f2024744c7")].version)

@variables t
@variables p(t)[1:3]
@variables q(t)[1:3]
D = Differential(t)

println("\n=== ROUND 1: D(dot(p,p)) ===")
ex = D(dot(p, p))
res1 = expand_derivatives(ex)
println("result = ", res1)
vars1 = Symbolics.get_variables(res1)
println("get_variables = ", vars1)
println("n_vars = ", length(vars1))

println("\n=== ROUND 1: D(dot(p,q)) (general, a != b) ===")
exg = D(dot(p, q))
resg = expand_derivatives(exg)
println("result = ", resg)
println("get_variables = ", Symbolics.get_variables(resg))

println("\n=== ROUND 1: D(norm(p)) ===")
exn = D(norm(p))
resn = expand_derivatives(exn)
println("result = ", resn)
println("get_variables = ", Symbolics.get_variables(resn))

println("\n=== ROUND 1: D(p'*p)  (bonus bug check) ===")
try
    exadj = D(p' * p)
    resadj = expand_derivatives(exadj)
    println("result = ", resadj)
    println("get_variables = ", Symbolics.get_variables(resadj))
catch e
    println("THREW: ", sprint(showerror, e))
end

println("\n=== ROUND 2: D(D(dot(p,p))) -- apply patched D again to round-1 result ===")
try
    res2 = expand_derivatives(D(res1))
    println("round2 result = ", res2)
    println("round2 get_variables = ", Symbolics.get_variables(res2))
catch e
    println("ROUND2 THREW: ", sprint(showerror, e))
end

println("\n=== ROUND 2 (equation form): D(D(dot(p,p)) - 1) as Pantelides would do (via scalar eqn) ===")
try
    eq_rhs = dot(p,p) - 1
    d1 = expand_derivatives(D(eq_rhs))
    println("d1 = ", d1)
    d2 = expand_derivatives(D(d1))
    println("d2 = ", d2)
    println("d2 vars = ", Symbolics.get_variables(d2))
catch e
    println("EQN ROUND2 THREW: ", sprint(showerror, e))
end

println("\n=== NUMERICAL EQUIVALENCE CHECK vs ORIGINAL (unpatched) formula ===")
# Manually build the OLD scalarized result formula for dot(p,p) and compare numerically
@variables p1 p2 p3 Dp1 Dp2 Dp3
old_dotpp = 2*Dp1*p1 + 2*Dp2*p2 + 2*Dp3*p3
# our patched compact result, substituted down to scalars for comparison:
# res1 should be something like 2*dot(p, D(p)); substitute p[i]->pi, D(p)[i]->Dpi via scalarize
res1_scalarized = Symbolics.scalarize(res1)
println("res1 (compact) scalarized = ", res1_scalarized)

subs = Dict(
    Symbolics.scalarize(p)[1] => p1, Symbolics.scalarize(p)[2] => p2, Symbolics.scalarize(p)[3] => p3,
)
# find the D(p)[i] symbols inside res1_scalarized's variable list and map them to Dp1,Dp2,Dp3 positionally
vs = collect(Symbolics.get_variables(res1_scalarized))
dvars = filter(v -> occursin("Differential", string(operation(v))) || occursin("Differential", string(v)), string.(vs))
println("raw var strings = ", string.(vs))

using Random
rng = Random.default_rng()
Random.seed!(rng, 1234)
randvals = Dict(p1=>rand(rng), p2=>rand(rng), p3=>rand(rng), Dp1=>rand(rng), Dp2=>rand(rng), Dp3=>rand(rng))
old_val = Symbolics.substitute(old_dotpp, randvals)
old_val = Symbolics.value(old_val)
println("old scalarized formula value at random point = ", old_val)

# substitute compact result's p[i] and D(p)[i] with same random numeric values
pvec = Symbolics.scalarize(p)
Dpvec = Symbolics.scalarize(D(p))
compact_subs = Dict(pvec[1]=>randvals[p1], pvec[2]=>randvals[p2], pvec[3]=>randvals[p3],
                     Dpvec[1]=>randvals[Dp1], Dpvec[2]=>randvals[Dp2], Dpvec[3]=>randvals[Dp3])
compact_val = Symbolics.substitute(res1_scalarized, compact_subs)
compact_val = Symbolics.value(compact_val)
println("compact formula value at SAME random point = ", compact_val)
println("MATCH: ", isapprox(Float64(old_val), Float64(compact_val); atol=1e-12))
