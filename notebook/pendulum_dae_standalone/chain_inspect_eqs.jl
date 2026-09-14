import Pkg
Pkg.activate(@__DIR__; io=devnull)
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra

N = 2
@parameters g = 9.81
@parameters L[1:N] = ones(N)
@variables p(t)[1:N, 1:3]
@variables v(t)[1:N, 1:3]
@variables lambda(t)[1:N]
Lp = collect(L)
lamv = collect(lambda)
pvecs = [[p[i,1], p[i,2], p[i,3]] for i in 1:N]
vvecs = [[v[i,1], v[i,2], v[i,3]] for i in 1:N]
anchor = zeros(3)
eqs = Equation[]
for i in 1:N; append!(eqs, D.(pvecs[i]) .~ vvecs[i]); end
for i in 1:N
    rprev = i == 1 ? anchor : pvecs[i-1]
    r_i = pvecs[i] .- rprev
    reaction = i < N ? lamv[i+1] .* (pvecs[i+1] .- pvecs[i]) : zeros(3)
    append!(eqs, D.(vvecs[i]) .~ -lamv[i] .* r_i .+ reaction .+ [0,0,-g])
end
for i in 1:N
    rprev = i == 1 ? anchor : pvecs[i-1]
    r_i = pvecs[i] .- rprev
    push!(eqs, dot(r_i, r_i) ~ Lp[i]^2)
end
@named sys = ODESystem(eqs, t)
simplified = ModelingToolkit.mtkcompile(sys)
println("=== simplified equations (N=2) ===")
for (i, eq) in enumerate(equations(simplified))
    println("eq[$i]: ", eq)
end
