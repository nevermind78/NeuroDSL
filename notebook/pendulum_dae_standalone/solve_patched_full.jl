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
using Plots
using DelimitedFiles
logmsg("using done (PATCHED Symbolics: compact dot/norm derivative + ModelingToolkitTearing dummy-derivative fix)")

@parameters L = 1.0 g = 9.81
@variables p(t)[1:3] = [0.6, 0.0, -0.8]
@variables v(t)[1:3] = [0.0, 0.9, 0.0]
@variables lambda(t) = 1.0
pv = collect(p)
vv = collect(v)

# Spherical pendulum, index-3 DAE, written with the UNCOLLECTED dot(p,p) ~ L^2 constraint
# (the exact shape that previously triggered the compact-derivative mtkcompile hang, now
# fixed). Released off-axis with a purely tangential initial velocity (dot(p0,v0) = 0, so
# the holonomic constraint is consistent from t=0) -- this makes it swing AND precess in
# 3D rather than staying in a single vertical plane like a simple planar pendulum.
eqs = [
    D.(pv) .~ vv;
    D.(vv) .~ -lambda .* pv .+ [0, 0, -g];
    dot(p, p) ~ L^2;
]
@named sys = ODESystem(eqs, t)
logmsg("ODESystem built ($(length(eqs)) equations)")

logmsg("=== mtkcompile (real Pantelides index reduction on the compact dot(p,p) constraint) ===")
simplified = ModelingToolkit.mtkcompile(sys)
logmsg("mtkcompile SUCCEEDED")
logmsg("n_eqs = $(length(equations(simplified))), unknowns = $(length(unknowns(simplified)))")
for (i, eq) in enumerate(equations(simplified))
    logmsg("  eq[$i]: $eq")
end

# lambda's consistent value at t=0, from differentiating the constraint twice:
# dot(p, D(v)) + dot(v,v) = 0  =>  -lambda*dot(p,p) - g*p3 + dot(v,v) = 0
# => lambda = (dot(v,v) - g*p3) / L^2
lambda0 = (0.81 - 9.81 * (-0.8)) / 1.0
u0 = [pv[1] => 0.6, pv[2] => 0.0, pv[3] => -0.8, vv[1] => 0.0, vv[2] => 0.9, vv[3] => 0.0, lambda => lambda0]
tspan = (0.0, 1.0)

logmsg("=== building ODEProblem ===")
prob = ODEProblem(simplified, u0, tspan)
logmsg("ODEProblem SUCCEEDED (this is exactly the step that used to throw before gap 1/2 were fixed)")

logmsg("=== solving with Rodas5P (mass-matrix Rosenbrock solver, appropriate for this index-1-after-dummy-derivative-reduction DAE) ===")
sol = solve(prob, Rodas5P(); abstol = 1e-9, reltol = 1e-9, saveat = 0.01, maxiters = 10_000_000, dtmax = 0.01)
logmsg("solve SUCCEEDED, retcode = $(sol.retcode), n_points = $(length(sol.t))")

# ------------------------------------------------------------------
# Step 4a: holonomic constraint residual dot(p,p) - L^2 over the whole trajectory
# ------------------------------------------------------------------
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
logmsg("constraint residual: max|dot(p,p)-L^2| = $(maximum(abs, resid)), mean = $(sum(abs, resid)/length(resid))")

# ------------------------------------------------------------------
# Step 4b: numerical comparison against the STOCK-Symbolics reference trajectory
# ------------------------------------------------------------------
refpath = joinpath(@__DIR__, "reference_stock_trajectory.tsv")
if isfile(refpath)
    ref = readdlm(refpath)
    reft = ref[:, 1]
    refP = ref[:, 2:4]
    refV = ref[:, 5:7]
    n = min(length(sol.t), length(reft))
    @assert maximum(abs, sol.t[1:n] .- reft[1:n]) < 1e-9 "time grids must match for a direct comparison"
    dP = P[1:n, :] .- refP[1:n, :]
    dV = V[1:n, :] .- refV[1:n, :]
    logmsg("cross-check vs stock-Symbolics reference over $n common time points:")
    logmsg("  max|Δp| = $(maximum(abs, dP)), max|Δv| = $(maximum(abs, dV))")
    logmsg("  rms|Δp| = $(sqrt(sum(dP.^2)/length(dP))), rms|Δv| = $(sqrt(sum(dV.^2)/length(dV)))")
else
    logmsg("WARNING: reference_stock_trajectory.tsv not found -- run solve_reference_stock.jl first")
end

# ------------------------------------------------------------------
# Step 5: plots
# ------------------------------------------------------------------
plotdir = joinpath(@__DIR__, "plots")
mkpath(plotdir)

gr()  # GR backend, renders cleanly to PNG headless

plt_traj3d = plot(P[:, 1], P[:, 2], P[:, 3];
    label = "bob trajectory", lw = 1.5, xlabel = "x", ylabel = "y", zlabel = "z",
    title = "Spherical pendulum: 3D trajectory (patched)",
    size = (700, 500), titlefontsize = 11,
    camera = (45, 25), legend = :topright)
scatter!(plt_traj3d, [P[1,1]], [P[1,2]], [P[1,3]]; label = "start", color = :green, ms = 5)
scatter!(plt_traj3d, [P[end,1]], [P[end,2]], [P[end,3]]; label = "end", color = :red, ms = 5)
savefig(plt_traj3d, joinpath(plotdir, "01_trajectory_3d.png"))
logmsg("saved 01_trajectory_3d.png")

plt_xy = plot(P[:, 1], P[:, 2]; label = "xy projection", lw = 1.2, size = (700, 500), titlefontsize = 11,
    xlabel = "x", ylabel = "y", title = "Spherical pendulum: top-down (xy) view",
    aspect_ratio = :equal, legend = :topright)
scatter!(plt_xy, [P[1,1]], [P[1,2]]; label = "start", color = :green, ms = 5)
savefig(plt_xy, joinpath(plotdir, "02_trajectory_xy_topdown.png"))
logmsg("saved 02_trajectory_xy_topdown.png")

plt_resid = plot(sol.t, resid; label = "dot(p,p) - L^2", lw = 1.2, color = :purple,
    xlabel = "t", ylabel = "constraint residual", size = (700, 450), titlefontsize = 11,
    title = "Holonomic constraint residual dot(p,p)-L^2 (patched solve)")
hline!(plt_resid, [0.0]; label = nothing, color = :black, ls = :dash, lw = 0.8)
savefig(plt_resid, joinpath(plotdir, "03_constraint_residual.png"))
logmsg("saved 03_constraint_residual.png (max|resid| = $(maximum(abs, resid)))")

if isfile(refpath)
    ref = readdlm(refpath)
    n = min(length(sol.t), size(ref, 1))
    plt_cmp = plot(sol.t[1:n], P[1:n, 1]; label = "p_x patched", lw = 2, color = :blue,
        xlabel = "t", ylabel = "p_x(t)", size = (700, 450), titlefontsize = 11,
        title = "Patched vs stock-Symbolics reference (p_x)")
    plot!(plt_cmp, ref[1:n, 1], ref[1:n, 2]; label = "p_x stock reference", lw = 1, ls = :dash, color = :orange)
    savefig(plt_cmp, joinpath(plotdir, "04_patched_vs_reference_overlay.png"))
    logmsg("saved 04_patched_vs_reference_overlay.png")
end

logmsg("ALL DONE")
