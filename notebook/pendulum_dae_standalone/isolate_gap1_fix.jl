import Pkg
Pkg.activate(@__DIR__; io=devnull)
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra
using Symbolics
import SymbolicUtils as SU

function _expand_compact_array_differential(x)
    SU.iscall(x) || return nothing
    op = SU.operation(x)
    op isa Differential || return nothing
    args = SU.arguments(x)
    arr = args[1]
    SU.is_array_shape(SU.shape(arr)) || return nothing
    scal_args = Symbolics.SArgsT()
    sizehint!(scal_args, length(arr)::Int + 1)
    push!(scal_args, Symbolics.SConst(size(arr)))
    for i in SU.stable_eachindex(arr)
        push!(scal_args, op(arr[i]))
    end
    return Symbolics.STerm(SU.array_literal, scal_args; type = SU.symtype(x), shape = SU.shape(x))
end

const EXPAND = SU.Rewriters.Postwalk(_expand_compact_array_differential)

@variables p(t)[1:3]
Dp = Differential(t, 1)(p)
DDp = Differential(t, 1)(Dp)   # compact SECOND derivative, mirrors the eq[7] scenario

println("=== single compact D(p) inside dot ===")
raw1 = 0 ~ -2 * dot(p, Dp)
println("before: ", raw1)
exp1 = EXPAND(raw1.rhs)
println("after : ", exp1)
println("iscall(exp1)=", SU.iscall(exp1), "  op=", SU.iscall(exp1) ? SU.operation(exp1) : nothing)

println("\n=== doubly-compact D(D(p)) inside dot (single Postwalk pass) ===")
raw2 = 0 ~ dot(Dp, Dp) + dot(p, DDp)
println("before: ", raw2)
exp2 = EXPAND(raw2.rhs)
println("after : ", exp2)

println("\n=== now check substitutability: scalar keys for D(p[1]), D(p[2]), D(p[3]) ===")
@variables ptdum(t) v2(t) v3(t)
D1p1 = Differential(t, 1)(p[1]); D1p2 = Differential(t, 1)(p[2]); D1p3 = Differential(t, 1)(p[3])
sub1 = substitute(exp1, Dict(unwrap(D1p1) => unwrap(ptdum), unwrap(D1p2) => unwrap(v2), unwrap(D1p3) => unwrap(v3)))
println("substituted exp1: ", sub1)

println("\n=== and for the doubly-compact case, also need D(D(p[i])) keys ===")
@variables pdd1(t)
D2p1 = Differential(t, 2)(p[1]); D2p2 = Differential(t, 2)(p[2]); D2p3 = Differential(t, 2)(p[3])
sub2 = substitute(exp2, Dict(
    unwrap(D1p1) => unwrap(ptdum), unwrap(D1p2) => unwrap(v2), unwrap(D1p3) => unwrap(v3),
    unwrap(D2p1) => unwrap(pdd1), unwrap(D2p2) => unwrap(v2), unwrap(D2p3) => unwrap(v3),  # placeholder rhs for 2,3
))
println("substituted exp2: ", sub2)
