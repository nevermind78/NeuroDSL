# ══════════════════════════════════════════════════════════════════════════════
# pcb_routing_diagnose.jl — instrumentation du piège de minimum local (composant
# 3) SUR LE CODE ACTUEL DE LA CELLULE PCB (reproduit fidèlement, pas modifié),
# avant tout correctif. Log par époque : position du waypoint le plus proche du
# centre du composant 3, distance à ce centre, gradient BRUT (avant clip) de
# l'exclusion sur ce point (isolé des autres forces), et le pas effectif après
# clip -- pour trancher laquelle des 3 causes possibles (force trop faible,
# clip trop agressif, piège multi-points par tension) est la vraie.
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, NeuroDSL

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=1))s] ", msg); flush(stdout))

PAD_START = Float32[2.0, 2.0]
PAD_END   = Float32[38.0, 34.0]
composants = [
    (12.0f0, 18.0f0, 5.0f0, 6.0f0),
    (25.0f0, 8.0f0,  4.0f0, 3.0f0),
    (28.0f0, 26.0f0, 6.0f0, 4.0f0),   # composant 3 -- celui qui piège la trajectoire
]
MARGE_ISOLATION = 0.6f0
RAIDEUR_VIRAGE  = 0.35f0
FORCE_EXCLUSION = 300.0f0
COMP3_IDX = 3
cx3, cy3, hx3, hy3 = composants[COMP3_IDX]
Hx3, Hy3 = hx3 + MARGE_ISOLATION, hy3 + MARGE_ISOLATION

register_op!(:pcb_trace_optimizer_diag, (dev, out, inputs, attrs, out_sym, out_node, ctx) -> begin
    W = inputs[1]; N = size(W, 1)
    S = PAD_START; E = PAD_END
    loss = 0.0f0
    loss += sum((W[1, :] .- S).^2)
    for i in 1:N-1; loss += sum((W[i+1, :] .- W[i, :]).^2); end
    loss += sum((E .- W[N, :]).^2)
    if N >= 3
        for i in 2:N-1
            Dcourb = W[i+1, :] .- 2.0f0 .* W[i, :] .+ W[i-1, :]
            loss += RAIDEUR_VIRAGE * sum(Dcourb.^2)
        end
    end
    for (cx, cy, hx, hy) in composants
        Hx = hx + MARGE_ISOLATION; Hy = hy + MARGE_ISOLATION
        for i in 1:N
            dx = W[i, 1] - cx; dy = W[i, 2] - cy
            if abs(dx) < Hx && abs(dy) < Hy
                loss += FORCE_EXCLUSION * ((Hx - abs(dx))^2 + (Hy - abs(dy))^2)
            end
        end
    end
    out .= loss
    if ctx !== nothing; ctx[out_sym] = Dict(:W => W, :S => S, :E => E, :N => N); end
end)
CUSTOM_SHAPE_RULES[:pcb_trace_optimizer_diag] = (inputs, attrs) -> (1, 1)
GRAD_RULES[:pcb_trace_optimizer_diag] = (dev, dy, ctx, inputs) -> begin
    W = ctx[:W]; S = ctx[:S]; E = ctx[:E]; N = ctx[:N]
    grad_W = zeros(Float32, N, 2)
    grad_W[1, :] .+= 2.0f0 .* (W[1, :] .- S) .- 2.0f0 .* (W[2, :] .- W[1, :])
    for i in 2:N-1
        grad_W[i, :] .+= 2.0f0 .* (W[i, :] .- W[i-1, :]) .- 2.0f0 .* (W[i+1, :] .- W[i, :])
    end
    grad_W[N, :] .+= 2.0f0 .* (W[N, :] .- W[N-1, :]) .- 2.0f0 .* (E .- W[N, :])
    if N >= 3
        for i in 2:N-1
            Dcourb = W[i+1, :] .- 2.0f0 .* W[i, :] .+ W[i-1, :]
            grad_W[i-1, :] .+= 2.0f0 * RAIDEUR_VIRAGE .* Dcourb
            grad_W[i,   :] .+= -4.0f0 * RAIDEUR_VIRAGE .* Dcourb
            grad_W[i+1, :] .+= 2.0f0 * RAIDEUR_VIRAGE .* Dcourb
        end
    end
    for (cx, cy, hx, hy) in composants
        Hx = hx + MARGE_ISOLATION; Hy = hy + MARGE_ISOLATION
        for i in 1:N
            dx = W[i, 1] - cx; dy = W[i, 2] - cy
            if abs(dx) < Hx && abs(dy) < Hy
                grad_W[i, 1] += FORCE_EXCLUSION * 2.0f0 * (Hx - abs(dx)) * (-sign(dx))
                grad_W[i, 2] += FORCE_EXCLUSION * 2.0f0 * (Hy - abs(dy)) * (-sign(dy))
            end
        end
    end
    return (grad_W .* dy[1], )
end

g = NeuroGraph(device=Backend.CPUDevice())
N_waypoints = 20
W_init = zeros(Float32, N_waypoints, 2)
for i in 1:N_waypoints
    fraction = Float32(i) / Float32(N_waypoints + 1)
    W_init[i, 1] = PAD_START[1] + (PAD_END[1] - PAD_START[1]) * fraction
    W_init[i, 2] = PAD_START[2] + (PAD_END[2] - PAD_START[2]) * fraction
end
set!(g, :waypoints, W_init; is_param=true)
addrule!(g, GraphRule(:loss, [:waypoints], :pcb_trace_optimizer_diag))

# Combien de waypoints démarrent déjà dans la zone d'exclusion élargie du composant 3 ?
n_inside_init = count(i -> abs(W_init[i,1]-cx3) < Hx3 && abs(W_init[i,2]-cy3) < Hy3, 1:N_waypoints)
_log("Waypoints démarrant DANS la zone d'exclusion élargie du composant 3 (init ligne droite) : $n_inside_init / $N_waypoints")

epochs = 1200
lr_route = 4f-3
log_epochs = [1, 10, 25, 50, 100, 200, 300, 400, 600, 800, 1000, 1200]

for epoch in 1:epochs
    p_node = node(g, :waypoints)
    p_node.gradient = nothing
    zero_grads!(g)
    demand!(g, :loss)
    backward_graph!(g, :loss)

    W_now = p_node.value
    # -- Diagnostic : waypoint le plus proche du centre du composant 3 --
    dists3 = [hypot(W_now[i,1]-cx3, W_now[i,2]-cy3) for i in 1:N_waypoints]
    i_near = argmin(dists3)
    d_near = dists3[i_near]
    inside3 = abs(W_now[i_near,1]-cx3) < Hx3 && abs(W_now[i_near,2]-cy3) < Hy3
    n_inside_now = count(i -> abs(W_now[i,1]-cx3) < Hx3 && abs(W_now[i,2]-cy3) < Hy3, 1:N_waypoints)

    # -- Diagnostic : gradient BRUT d'exclusion du composant 3 SEUL sur ce point --
    dx = W_now[i_near,1]-cx3; dyv = W_now[i_near,2]-cy3
    gx3 = inside3 ? FORCE_EXCLUSION*2.0f0*(Hx3-abs(dx))*(-sign(dx)) : 0.0f0
    gy3 = inside3 ? FORCE_EXCLUSION*2.0f0*(Hy3-abs(dyv))*(-sign(dyv)) : 0.0f0
    g3_norm = hypot(gx3, gy3)

    grad_total = p_node.gradient
    if grad_total !== nothing
        g_total_norm_i = grad_total===nothing ? NaN32 : hypot(grad_total[i_near,1], grad_total[i_near,2])
        grad_clipped = clamp.(grad_total, -20.0f0, 20.0f0)
        g_clipped_norm_i = hypot(grad_clipped[i_near,1], grad_clipped[i_near,2])
        current_lr = max(lr_route * (1.0f0 - (Float32(epoch)/Float32(epochs))^2), 1f-4)
        step_i = current_lr * g_clipped_norm_i

        if epoch in log_epochs
            _log("epoch=$epoch  i_near=$i_near  pos=($(round(W_now[i_near,1],digits=2)),$(round(W_now[i_near,2],digits=2)))  " *
                 "dist_centre3=$(round(d_near,digits=3))  dans_zone3=$inside3  n_pts_dans_zone3=$n_inside_now  " *
                 "|grad_excl3_seul|=$(round(g3_norm,digits=2))  |grad_TOTAL_avant_clip|=$(round(g_total_norm_i,digits=2))  " *
                 "|grad_TOTAL_apres_clip|=$(round(g_clipped_norm_i,digits=2))  lr=$(round(current_lr,digits=5))  pas_effectif=$(round(step_i,digits=4))")
        end

        new_W = p_node.value .- current_lr .* grad_clipped
        set!(g, :waypoints, new_W; is_param=true)
    end
end

W_final = node(g, :waypoints).value
n_inside_final = count(i -> abs(W_final[i,1]-cx3) < Hx3 && abs(W_final[i,2]-cy3) < Hy3, 1:N_waypoints)
_log("")
_log("VERDICT DIAGNOSTIC : waypoints finaux dans la zone d'exclusion élargie du composant 3 : $n_inside_final / $N_waypoints")
if n_inside_final > 0
    _log("❌ CONFIRMÉ : le code ACTUEL (avant correctif) laisse la trajectoire traverser le composant 3.")
else
    _log("✅ Pas de collision finale avec le composant 3 dans cette configuration (inattendu -- à revérifier).")
end
