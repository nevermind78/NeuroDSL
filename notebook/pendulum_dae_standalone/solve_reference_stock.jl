t0 = time()
function logmsg(s)
    println(s, "  [+", round(time() - t0, digits=1), "s]")
    flush(stdout)
end
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "mtk_translator_env"); io=devnull)   # stock (unpatched) registry Symbolics 7.39.0
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using LinearAlgebra
using DelimitedFiles
logmsg("using done (STOCK / unpatched Symbolics -- ground-truth reference)")

@parameters L = 1.0 g = 9.81
@variables p(t)[1:3] = [0.6, 0.0, -0.8]
@variables v(t)[1:3] = [0.0, 0.9, 0.0]
@variables lambda(t) = 1.0
pv = collect(p)
vv = collect(v)

eqs = [
    D.(pv) .~ vv;
    D.(vv) .~ -lambda .* pv .+ [0, 0, -g];
    dot(p, p) ~ L^2;
]
@named sys = ODESystem(eqs, t)
logmsg("ODESystem built")

simplified = ModelingToolkit.mtkcompile(sys)
logmsg("mtkcompile SUCCEEDED (stock Symbolics scalarizes dot's derivative eagerly -- no compact-derivative optimization, but numerically equivalent)")
logmsg("n_eqs = $(length(equations(simplified))), unknowns = $(length(unknowns(simplified)))")

# lambda's consistent value at t=0, from differentiating the constraint twice:
# dot(p, D(v)) + dot(v,v) = 0  =>  -lambda*dot(p,p) - g*p3 + dot(v,v) = 0
# => lambda = (dot(v,v) - g*p3) / L^2
lambda0 = (0.81 - 9.81 * (-0.8)) / 1.0
u0 = [pv[1] => 0.6, pv[2] => 0.0, pv[3] => -0.8, vv[1] => 0.0, vv[2] => 0.9, vv[3] => 0.0, lambda => lambda0]
tspan = (0.0, 1.0)
prob = ODEProblem(simplified, u0, tspan)
logmsg("ODEProblem built")

sol = solve(prob, Rodas5P(); abstol = 1e-9, reltol = 1e-9, saveat = 0.01, maxiters = 10_000_000, dtmax = 0.01)
logmsg("solve SUCCEEDED, retcode = $(sol.retcode), n_points = $(length(sol.t))")

# Dump a dense, common-grid trajectory for p and v for later cross-checking against the
# patched run. Written as plain whitespace-separated columns: t p1 p2 p3 v1 v2 v3
out = Matrix{Float64}(undef, length(sol.t), 7)
for (i, ti) in enumerate(sol.t)
    pt = sol(ti; idxs = pv)
    vt = sol(ti; idxs = vv)
    out[i, 1] = ti
    out[i, 2:4] .= pt
    out[i, 5:7] .= vt
end
writedlm(joinpath(@__DIR__, "reference_stock_trajectory.tsv"), out)
logmsg("reference trajectory written to reference_stock_trajectory.tsv ($(size(out,1)) rows)")
