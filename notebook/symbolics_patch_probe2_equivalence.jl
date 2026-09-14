import Pkg
Pkg.activate(joinpath(@__DIR__, "mtk_translator_env"))
using Symbolics
using LinearAlgebra
using Random

@variables t
@variables p(t)[1:3]
@variables q(t)[1:3]
D = Differential(t)

println("=== Numerical equivalence: compact vs. hand-built scalarized formula (dot(p,p)) ===")
res1 = expand_derivatives(D(dot(p, p)))
println("compact result: ", res1)

pv = Symbolics.scalarize(p)              # [p[1], p[2], p[3]]
Dpv = Symbolics.scalarize(D(p))          # [D(p)[1], D(p)[2], D(p)[3]] == [D(p[1]), D(p[2]), D(p[3])] structurally after scalarize
res1_scalar = Symbolics.scalarize(res1)
println("compact result scalarized: ", res1_scalar)

# Old formula, hand-built directly with the SAME underlying scalar symbols pv/Dpv
old_formula = 2*Dpv[1]*pv[1] + 2*Dpv[2]*pv[2] + 2*Dpv[3]*pv[3]

rng = Random.default_rng(); Random.seed!(rng, 42)
vals = Dict(pv[1]=>rand(rng), pv[2]=>rand(rng), pv[3]=>rand(rng),
            Dpv[1]=>rand(rng), Dpv[2]=>rand(rng), Dpv[3]=>rand(rng))

old_val = Float64(Symbolics.value(Symbolics.substitute(old_formula, vals)))
new_val = Float64(Symbolics.value(Symbolics.substitute(res1_scalar, vals)))
println("old scalarized formula @ random point = ", old_val)
println("new compact formula   @ random point = ", new_val)
println("MATCH dot(p,p): ", isapprox(old_val, new_val; atol=1e-12))

println("\n=== Numerical equivalence: dot(p,q), a != b ===")
resg = expand_derivatives(D(dot(p, q)))
resg_scalar = Symbolics.scalarize(resg)
qv = Symbolics.scalarize(q)
Dqv = Symbolics.scalarize(D(q))
old_formula_g = Dpv[1]*qv[1] + pv[1]*Dqv[1] + Dpv[2]*qv[2] + pv[2]*Dqv[2] + Dpv[3]*qv[3] + pv[3]*Dqv[3]
vals_g = merge(vals, Dict(qv[1]=>rand(rng), qv[2]=>rand(rng), qv[3]=>rand(rng),
                           Dqv[1]=>rand(rng), Dqv[2]=>rand(rng), Dqv[3]=>rand(rng)))
old_val_g = Float64(Symbolics.value(Symbolics.substitute(old_formula_g, vals_g)))
new_val_g = Float64(Symbolics.value(Symbolics.substitute(resg_scalar, vals_g)))
println("old = ", old_val_g, "  new = ", new_val_g)
println("MATCH dot(p,q): ", isapprox(old_val_g, new_val_g; atol=1e-12))

println("\n=== Numerical equivalence: norm(p) ===")
resn = expand_derivatives(D(norm(p)))
resn_scalar = Symbolics.scalarize(resn)
old_norm_formula = (2*Dpv[1]*pv[1] + 2*Dpv[2]*pv[2] + 2*Dpv[3]*pv[3]) / (2*sqrt(pv[1]^2+pv[2]^2+pv[3]^2))
old_val_n = Float64(Symbolics.value(Symbolics.substitute(old_norm_formula, vals)))
new_val_n = Float64(Symbolics.value(Symbolics.substitute(resn_scalar, vals)))
println("old = ", old_val_n, "  new = ", new_val_n)
println("MATCH norm(p): ", isapprox(old_val_n, new_val_n; atol=1e-10))

println("\n=== Round-2 numerical equivalence: D(D(dot(p,p))) vs fully scalarized 2nd derivative ===")
res2 = expand_derivatives(D(res1))
res2_scalar = Symbolics.scalarize(res2)
println("round2 compact scalarized = ", res2_scalar)
D2pv = Symbolics.scalarize(D(D(p)))
old_formula_2 = 2*(Dpv[1]^2 + Dpv[2]^2 + Dpv[3]^2) + 2*(pv[1]*D2pv[1] + pv[2]*D2pv[2] + pv[3]*D2pv[3])
vals2 = merge(vals, Dict(D2pv[1]=>rand(rng), D2pv[2]=>rand(rng), D2pv[3]=>rand(rng)))
old_val_2 = Float64(Symbolics.value(Symbolics.substitute(old_formula_2, vals2)))
new_val_2 = Float64(Symbolics.value(Symbolics.substitute(res2_scalar, vals2)))
println("old = ", old_val_2, "  new = ", new_val_2)
println("MATCH round2: ", isapprox(old_val_2, new_val_2; atol=1e-10))

println("\n=== Regression: ordinary scalar differentiation is unaffected ===")
@variables x y z k
f1 = k*(abs(x-y)/y - z)^2
Dx = Differential(x)
r1 = expand_derivatives(Dx(f1))
println("k*(abs(x-y)/y-z)^2 wrt x -> ", r1)

f2 = x^3*sin(y) + exp(x*z)
r2 = expand_derivatives(Dx(f2))
println("x^3*sin(y)+exp(x*z) wrt x -> ", r2)
expected2 = 3*x^2*sin(y) + z*exp(x*z)
diffcheck = Symbolics.simplify(r2 - expected2)
println("difference from hand-derived expected (should simplify near 0): ", diffcheck)

f3 = (x + y)/(x - y)
r3 = expand_derivatives(Dx(f3))
println("(x+y)/(x-y) wrt x -> ", r3)

@variables xx[1:3]
f4 = sum(xx.^2)   # ordinary (non-dot/norm) array reduction, should still scalarize (unaffected by patch)
Dxx1 = Differential(xx[1])
r4 = expand_derivatives(Dxx1(f4))
println("sum(xx.^2) wrt xx[1] -> ", r4, "   (expect 2*xx[1])")

println("\n=== Bonus: Symbolics.jacobian silent-zero check (unpatched pre-scan) ===")
@variables t2
@variables p2(t2)[1:2]
D2 = Differential(t2)
eqs_jac = [dot(p2,p2) - 1]
vars_jac = Symbolics.scalarize(p2)
J = Symbolics.jacobian(eqs_jac, vars_jac)
println("jacobian(dot(p2,p2)-1, [p2[1],p2[2]]) = ", J)
