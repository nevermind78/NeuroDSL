using JSON, Plots

dcold = JSON.parsefile(joinpath(@__DIR__, "attribution_curve_results_net2net.json"))
dwarm = JSON.parsefile(joinpath(@__DIR__, "e6_v_warmstart_results.json"))

function unpack_alpha(traj)
    t = Float64[row[1] for row in traj]
    alpha = Float64[row[3] for row in traj]
    return t, alpha
end
function unpack_alpha_dict(traj)
    t = Float64[row["t"] for row in traj]
    alpha = Float64[row["alpha"] for row in traj]
    return t, alpha
end

t_ec, a_ec = unpack_alpha(dcold["traj_early"])
t_lc, a_lc = unpack_alpha(dcold["traj_late"])
t_ew, a_ew = unpack_alpha_dict(dwarm["traj_early_warm"])
t_lw, a_lw = unpack_alpha_dict(dwarm["traj_late_warm"])

gr()
p = plot(t_ec, a_ec; label="Early, cold-start (v=0)", lw=2.5, color=:steelblue, ls=:dash,
         xlabel="steps since graft", ylabel="alpha(t)", title="Effect of AdamW second-moment warm-start on gate rejection",
         legend=:right, size=(750,500), ylim=(0,1.05))
plot!(p, t_lc, a_lc; label="Late, cold-start (v=0)", lw=2.5, color=:darkorange, ls=:dash)
plot!(p, t_ew, a_ew; label="Early, warm-start (v=v_lmhead)", lw=2.5, color=:steelblue)
plot!(p, t_lw, a_lw; label="Late, warm-start (v=v_lmhead)", lw=2.5, color=:darkorange)
hline!(p, [1.0]; color=:gray, ls=:dot, label=nothing)

mkpath(joinpath(@__DIR__, "..", "figures"))
savefig(p, joinpath(@__DIR__, "..", "figures", "fig_e8_warmstart.pdf"))
savefig(p, joinpath(@__DIR__, "..", "figures", "fig_e8_warmstart.png"))
println("Écrit -> figures/fig_e8_warmstart.pdf (+.png)")
