import Pkg
Pkg.activate(joinpath(@__DIR__, "mtk_translator_env"))
using Symbolics
using LinearAlgebra
using Random

@variables t
@variables p(t)[1:3]
D = Differential(t)
pv = Symbolics.scalarize(p)
Dpv = Symbolics.scalarize(D(p))
D2pv = Symbolics.scalarize(D(D(p)))

function evalnum(expr, vals::Dict)
    vs = collect(keys(vals))
    f = eval(Symbolics.build_function(expr, vs...; expression=Val{true}))
    return Base.invokelatest(f, (vals[v] for v in vs)...)
end

rng = Random.default_rng(); Random.seed!(rng, 7)
vals = Dict(pv[1]=>rand(rng)+0.1, pv[2]=>rand(rng)+0.1, pv[3]=>rand(rng)+0.1,
            Dpv[1]=>rand(rng), Dpv[2]=>rand(rng), Dpv[3]=>rand(rng),
            D2pv[1]=>rand(rng), D2pv[2]=>rand(rng), D2pv[3]=>rand(rng))

println("=== norm(p) equivalence ===")
resn = expand_derivatives(D(norm(p)))
resn_scalar = Symbolics.scalarize(resn)
old_norm_formula = (2*Dpv[1]*pv[1] + 2*Dpv[2]*pv[2] + 2*Dpv[3]*pv[3]) / (2*sqrt(pv[1]^2+pv[2]^2+pv[3]^2))
old_val = evalnum(old_norm_formula, vals)
new_val = evalnum(resn_scalar, vals)
println("old=", old_val, "  new=", new_val, "  MATCH=", isapprox(old_val, new_val; atol=1e-10))

println("\n=== round-2 D(D(dot(p,p))) equivalence ===")
res1 = expand_derivatives(D(dot(p, p)))
res2 = expand_derivatives(D(res1))
res2_scalar = Symbolics.scalarize(res2)
old_formula_2 = 2*(Dpv[1]^2 + Dpv[2]^2 + Dpv[3]^2) + 2*(pv[1]*D2pv[1] + pv[2]*D2pv[2] + pv[3]*D2pv[3])
old_val2 = evalnum(old_formula_2, vals)
new_val2 = evalnum(res2_scalar, vals)
println("old=", old_val2, "  new=", new_val2, "  MATCH=", isapprox(old_val2, new_val2; atol=1e-10))

println("\n=== 100 random trials for dot(p,p), dot(p,q), norm(p), round2 ===")
@variables q(t)[1:3]
qv = Symbolics.scalarize(q)
Dqv = Symbolics.scalarize(D(q))
resg = expand_derivatives(D(dot(p, q)))
resg_scalar = Symbolics.scalarize(resg)
old_formula_g = Dpv[1]*qv[1] + pv[1]*Dqv[1] + Dpv[2]*qv[2] + pv[2]*Dqv[2] + Dpv[3]*qv[3] + pv[3]*Dqv[3]

all_ok = true
for trial in 1:100
    v = Dict(pv[1]=>rand(rng)+0.05, pv[2]=>rand(rng)+0.05, pv[3]=>rand(rng)+0.05,
             qv[1]=>rand(rng)+0.05, qv[2]=>rand(rng)+0.05, qv[3]=>rand(rng)+0.05,
             Dpv[1]=>rand(rng)-0.5, Dpv[2]=>rand(rng)-0.5, Dpv[3]=>rand(rng)-0.5,
             Dqv[1]=>rand(rng)-0.5, Dqv[2]=>rand(rng)-0.5, Dqv[3]=>rand(rng)-0.5,
             D2pv[1]=>rand(rng)-0.5, D2pv[2]=>rand(rng)-0.5, D2pv[3]=>rand(rng)-0.5)
    v1o = evalnum(2*Dpv[1]*pv[1] + 2*Dpv[2]*pv[2] + 2*Dpv[3]*pv[3], v)
    v1n = evalnum(Symbolics.scalarize(res1), v)
    v_g_o = evalnum(old_formula_g, v)
    v_g_n = evalnum(resg_scalar, v)
    v_n_o = evalnum((2*Dpv[1]*pv[1]+2*Dpv[2]*pv[2]+2*Dpv[3]*pv[3])/(2*sqrt(pv[1]^2+pv[2]^2+pv[3]^2)), v)
    v_n_n = evalnum(resn_scalar, v)
    v_r2_o = evalnum(old_formula_2, v)
    v_r2_n = evalnum(res2_scalar, v)
    ok = isapprox(v1o,v1n;atol=1e-8) && isapprox(v_g_o,v_g_n;atol=1e-8) &&
         isapprox(v_n_o,v_n_n;atol=1e-8) && isapprox(v_r2_o,v_r2_n;atol=1e-8)
    global all_ok &= ok
    if !ok
        println("trial $trial MISMATCH: ", (v1o,v1n,v_g_o,v_g_n,v_n_o,v_n_n,v_r2_o,v_r2_n))
    end
end
println("ALL 100 TRIALS MATCH: ", all_ok)
