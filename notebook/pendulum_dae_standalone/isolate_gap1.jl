import Pkg
Pkg.activate(@__DIR__; io=devnull)
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra
using Symbolics
const ExpandDerivativeDict = ModelingToolkit.MTKBase.ExpandDerivativeDict

@variables p(t)[1:3]
@variables ptdum(t)  # stand-in dummy derivative for p[1], scalar
pv = collect(p)
Dp = Differential(t, 1)(p)   # the WHOLE-ARRAY compact Differential, exactly what
                              # compact_dot_derivative produces -- NOT D.(pv)

# The compact, un-substituted whole-array term exactly as retained in eq[6]:
raw = 0 ~ -2 * dot(p, Dp)
println("raw equation: ", raw)

# What generate_system_equations! would have accumulated in total_sub by that point
# (scalar keys only -- this is the shape total_sub actually has):
D1p1 = Differential(t, 1)(pv[1])
D1p2 = Differential(t, 1)(pv[2])
D1p3 = Differential(t, 1)(pv[3])
@variables v2(t) v3(t)

println("\n--- Test A: plain Symbolics.substitute with a single Pair (mirrors substitute_derivatives_algevars!) ---")
r1 = substitute(raw, Dict(D1p1 => ptdum))
println(r1)

println("\n--- Test B: Symbolics.fixpoint_sub with a plain Dict (mirrors generate_system_equations! / total_sub shape, if it were a plain Dict) ---")
r2 = Symbolics.fixpoint_sub(raw, Dict(D1p1 => ptdum, D1p2 => v2, D1p3 => v3))
println(r2)

println("\n--- Test C: Symbolics.fixpoint_sub with MTKBase.ExpandDerivativeDict (mirrors ACTUAL total_sub type) ---")
dd = ExpandDerivativeDict(Dict{Symbolics.SymbolicT, Symbolics.SymbolicT}(unwrap(D1p1) => unwrap(ptdum), unwrap(D1p2) => unwrap(v2), unwrap(D1p3) => unwrap(v3)))
r3 = Symbolics.fixpoint_sub(raw, dd)
println(r3)
