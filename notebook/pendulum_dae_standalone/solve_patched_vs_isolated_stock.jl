t0 = time()
function logmsg(s)
    println(s, "  [+", round(time() - t0, digits=1), "s]")
    flush(stdout)
end
import Pkg
Pkg.activate(@__DIR__; io=devnull)   # patched Symbolics (dev'd to mLvup) + our ModelingToolkitTearing fix
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using LinearAlgebra
using DelimitedFiles
logmsg("using done (PATCHED Symbolics: compact dot/norm derivative + ModelingToolkitTearing dummy-derivative fix) -- FRESH re-solve")

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
logmsg("ODESystem built ($(length(eqs)) equations)")

simplified = ModelingToolkit.mtkcompile(sys)
logmsg("mtkcompile SUCCEEDED")
logmsg("n_eqs = $(length(equations(simplified))), unknowns = $(length(unknowns(simplified)))")

lambda0 = (0.81 - 9.81 * (-0.8)) / 1.0
u0 = [pv[1] => 0.6, pv[2] => 0.0, pv[3] => -0.8, vv[1] => 0.0, vv[2] => 0.9, vv[3] => 0.0, lambda => lambda0]
tspan = (0.0, 1.0)

prob = ODEProblem(simplified, u0, tspan)
logmsg("ODEProblem SUCCEEDED")

sol = solve(prob, Rodas5P(); abstol = 1e-9, reltol = 1e-9, saveat = 0.01, maxiters = 10_000_000, dtmax = 0.01)
logmsg("solve SUCCEEDED, retcode = $(sol.retcode), n_points = $(length(sol.t))")

Lval = 1.0
resid = Float64[]
P = Matrix{Float64}(undef, length(sol.t), 3)
V = Matrix{Float64}(undef, length(sol.t), 3)
for (i, ti) in enumerate(sol.t)
    pt = sol(ti; idxs = pv)
    vt = sol(ti; idxs = vv)
    P[i, :] .= pt
    V[i, :] .= vt
    push!(resid, dot(pt, pt) - Lval^2)
end
logmsg("constraint residual (patched, fresh): max|dot(p,p)-L^2| = $(maximum(abs, resid)), mean = $(sum(abs, resid)/length(resid))")

# ------------------------------------------------------------------
# Cross-check vs the NEW, genuinely-isolated stock reference
# (built in pendulum_dae_standalone_STOCK_ISOLATED via path= into reverted,
# non-shared copies of Symbolics/ModelingToolkitTearing -- verified NOT to
# resolve to the live patched ~/.julia/packages/{Symbolics/mLvup,ModelingToolkitTearing/veEKu})
# ------------------------------------------------------------------
refpath = joinpath(@__DIR__, "..", "pendulum_dae_standalone_STOCK_ISOLATED", "reference_stock_trajectory_ISOLATED.tsv")
ref = readdlm(refpath)
reft = ref[:, 1]
refP = ref[:, 2:4]
refV = ref[:, 5:7]
n = min(length(sol.t), length(reft))
@assert maximum(abs, sol.t[1:n] .- reft[1:n]) < 1e-9 "time grids must match for a direct comparison"
dP = P[1:n, :] .- refP[1:n, :]
dV = V[1:n, :] .- refV[1:n, :]
logmsg("=== cross-check: PATCHED (fresh) vs TRUE ISOLATED STOCK reference, $n common time points ===")
logmsg("  max|Δp| = $(maximum(abs, dP)), max|Δv| = $(maximum(abs, dV))")
logmsg("  rms|Δp| = $(sqrt(sum(dP.^2)/length(dP))), rms|Δv| = $(sqrt(sum(dV.^2)/length(dV)))")

# ------------------------------------------------------------------
# Also load the OLD (contaminated) reference for a direct side-by-side, to see
# whether the contamination happened to change the reported numbers materially.
# ------------------------------------------------------------------
oldrefpath = joinpath(@__DIR__, "reference_stock_trajectory.tsv")
if isfile(oldrefpath)
    oldref = readdlm(oldrefpath)
    oldreft = oldref[:, 1]
    oldrefP = oldref[:, 2:4]
    oldrefV = oldref[:, 5:7]
    n2 = min(length(sol.t), length(oldreft))
    dP2 = P[1:n2, :] .- oldrefP[1:n2, :]
    dV2 = V[1:n2, :] .- oldrefV[1:n2, :]
    logmsg("=== for comparison: PATCHED (fresh) vs OLD CONTAMINATED reference, $n2 common time points ===")
    logmsg("  max|Δp| = $(maximum(abs, dP2)), max|Δv| = $(maximum(abs, dV2))")

    # And: does the OLD contaminated reference itself differ from the NEW isolated stock
    # reference? This isolates "was the contamination numerically consequential" from
    # "is the patched pipeline correct".
    n3 = min(length(oldreft), length(reft))
    @assert maximum(abs, oldreft[1:n3] .- reft[1:n3]) < 1e-9
    dPref = oldrefP[1:n3, :] .- refP[1:n3, :]
    dVref = oldrefV[1:n3, :] .- refV[1:n3, :]
    logmsg("=== OLD CONTAMINATED reference vs NEW TRUE ISOLATED STOCK reference, $n3 common time points ===")
    logmsg("  max|Δp| = $(maximum(abs, dPref)), max|Δv| = $(maximum(abs, dVref))")
end

logmsg("ALL DONE")
