using JSON, Plots

d = JSON.parsefile(joinpath(@__DIR__, "attribution_curve_results_net2net.json"))

function unpack(traj)
    t = Float64[row[1] for row in traj]
    A = Float64[row[2] for row in traj]
    alpha = Float64[row[3] for row in traj]
    return t, A, alpha
end

t_e, A_e, alpha_e = unpack(d["traj_early"])
t_l, A_l, alpha_l = unpack(d["traj_late"])

gr()
p1 = plot(t_e, alpha_e; label="Early (t_ins=500)", lw=2.5, color=:steelblue, marker=:circle,
          ylabel="alpha(t)", title="Gate magnitude", legend=:topright)
plot!(p1, t_l, alpha_l; label="Late (t_ins=2500)", lw=2.5, color=:darkorange, marker=:circle)
hline!(p1, [0.0]; color=:gray, ls=:dash, label=nothing)

p2 = plot(t_e, A_e; label="Early (t_ins=500)", lw=2.5, color=:steelblue, marker=:circle,
          xlabel="steps since graft (t - t_ins)", ylabel="A(t) = L(alpha:=0) - L(alpha_t)",
          title="Ablation gap", legend=:bottomright)
plot!(p2, t_l, A_l; label="Late (t_ins=2500)", lw=2.5, color=:darkorange, marker=:circle)
hline!(p2, [0.0]; color=:gray, ls=:dash, label=nothing)

fig = plot(p1, p2; layout=(2,1), size=(700, 700), left_margin=5Plots.mm, bottom_margin=5Plots.mm)

mkpath(joinpath(@__DIR__, "..", "figures"))
savefig(fig, joinpath(@__DIR__, "..", "figures", "fig_e6_gate_dynamics.pdf"))
savefig(fig, joinpath(@__DIR__, "..", "figures", "fig_e6_gate_dynamics.png"))
println("Écrit -> figures/fig_e6_gate_dynamics.pdf (+.png)")
