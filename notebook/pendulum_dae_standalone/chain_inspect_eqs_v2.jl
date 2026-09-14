import Pkg
Pkg.activate(@__DIR__; io=devnull)
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra
using Symbolics

N = 2
@parameters g = 9.81
@parameters L[1:N] = ones(N)
@variables p(t)[1:N, 1:3]
@variables v(t)[1:N, 1:3]
@variables lambda(t)[1:N]
Lp = collect(L)
lamv = collect(lambda)

# Use genuine symbolic array SLICES (p[i,:]) instead of a plain Vector{Num} of
# individually-extracted scalars, so `r_i` stays an array-shaped symbolic TERM (the shape
# dot()/compact_dot_derivative actually pattern-match on) rather than eagerly collapsing to
# a concrete Vector{Num} that Julia's generic dot() loop evaluates immediately.
pview(i) = p[i, :]
vview(i) = v[i, :]
println("typeof(pview(1)) = ", typeof(pview(1)))
println("pview(1) = ", pview(1))
r1 = pview(1) .- zeros(3)
println("typeof(r1) = ", typeof(r1))
println("r1 = ", r1)
d11 = dot(r1, r1)
println("typeof(dot(r1,r1)) = ", typeof(d11))
println("dot(r1,r1) = ", d11)

eqs = Equation[]
for i in 1:N; append!(eqs, collect(D.(pview(i)) .~ vview(i))); end
for i in 1:N
    rprev = i == 1 ? zeros(3) : pview(i - 1)
    r_i = pview(i) .- rprev
    reaction = i < N ? lamv[i + 1] .* (pview(i + 1) .- pview(i)) : zeros(3)
    append!(eqs, collect(D.(vview(i)) .~ -lamv[i] .* r_i .+ reaction .+ [0, 0, -g]))
end
for i in 1:N
    rprev = i == 1 ? zeros(3) : pview(i - 1)
    r_i = pview(i) .- rprev
    push!(eqs, dot(r_i, r_i) ~ Lp[i]^2)
end
println("length(eqs) = ", length(eqs))
for (i, eq) in enumerate(eqs)
    println("raw eq[$i]: ", eq)
end
@named sys = ODESystem(eqs, t)
simplified = ModelingToolkit.mtkcompile(sys)
println("\n=== simplified equations (N=2, array-slice construction) ===")
for (i, eq) in enumerate(equations(simplified))
    println("eq[$i]: ", eq)
end
