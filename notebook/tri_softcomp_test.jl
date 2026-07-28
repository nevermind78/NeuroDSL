# Test diagnostique du "comparateur souple" (:soft_comparator, cellule
# "Comparateur Souple" de tri.ipynb). Deux bugs identifiés par lecture :
#  1. `m = inputs[2]` est un VECTEUR à 1 élément (set! avec Float32[0.8]),
#     pas un scalaire -> `1 - m` plante (Int - Vector{Float32}).
#  2. La règle de gradient dx = copy(dy) est FAUSSE quand should_swap=true :
#     elle ignore complètement le mélange (1-m)/m et traite l'op comme une
#     identité pure, alors que out[i],out[j] dépendent tous les deux de
#     x[i] ET x[j] via le mélange.
using LinearAlgebra, NeuroDSL

register_op!(:soft_comparator_fixed, (dev, out, inputs, attrs, out_sym, out_node, ctx) -> begin
    x = inputs[1]
    m_vec = inputs[2]
    m = m_vec[1]                      # FIX 1 : extraire le scalaire
    i = attrs[:i]; j = attrs[:j]
    val_i = x[i, 1]; val_j = x[j, 1]
    should_swap = val_i > val_j
    out_vals = copy(x)
    if should_swap
        out_vals[i, 1] = val_i * (1f0 - m) + val_j * m
        out_vals[j, 1] = val_j * (1f0 - m) + val_i * m
    end
    out .= out_vals
    if ctx !== nothing
        ctx[out_sym] = Dict{Symbol, Any}(:should_swap => should_swap, :i => i, :j => j, :m => m)
    end
end)

GRAD_RULES[:soft_comparator_fixed] = (dev, dy, ctx, inputs) -> begin
    should_swap = ctx[:should_swap]
    i = ctx[:i]; j = ctx[:j]; m = ctx[:m]
    dx = copy(dy)
    dm = 0.0f0
    if should_swap
        # FIX 2 : router le gradient à travers le mélange (1-m)/m, pas comme une identité.
        dyi, dyj = dy[i, 1], dy[j, 1]
        dx[i, 1] = dyi * (1f0 - m) + dyj * m
        dx[j, 1] = dyj * (1f0 - m) + dyi * m
        xi, xj = inputs[1][i, 1], inputs[1][j, 1]
        dm = (xi - xj) * (dyj - dyi)
    end
    return (dx, fill(dm, size(inputs[2])))
end

function experimenter_reseau_de_tri_fixed(; verbose=true)
    g = NeuroGraph(device=Backend.CPUDevice())
    x_init = Float32[3.0; 1.0; 4.0; 2.0]
    set!(g, :x, x_init; is_param=false)
    comparateurs = [(1, 2), (3, 4), (2, 3), (1, 2), (3, 4), (2, 3)]
    current_sym = :x
    for (idx, (i, j)) in enumerate(comparateurs)
        mask_sym = Symbol(:m_, idx)
        set!(g, mask_sym, Float32[0.8]; is_param=true)
        next_sym = Symbol(:layer_, idx)
        addrule!(g, GraphRule(next_sym, [current_sym, mask_sym], :soft_comparator_fixed; attrs=Dict(:i => i, :j => j)))
        current_sym = next_sym
    end
    final_node = current_sym
    sortie_initiale = demand!(g, final_node)
    verbose && println("Entrée brute       : ", vec(x_init))
    verbose && println("Sortie avant opt.  : ", round.(vec(sortie_initiale), digits=2))

    target = Float32[1.0; 2.0; 3.0; 4.0]
    set!(g, :target, target; is_param=false)
    addrule!(g, GraphRule(:loss, [final_node, :target], :mse_loss))

    lr = 0.1f0
    local loss_val
    for epoch in 1:50
        zero_grads!(g)
        loss_val = demand!(g, :loss)
        backward_graph!(g, :loss)
        for idx in 1:length(comparateurs)
            mask_sym = Symbol(:m_, idx)
            node_m = node(g, mask_sym)
            if node_m.gradient !== nothing
                new_m = node_m.value .- lr .* node_m.gradient
                new_m .= clamp.(new_m, 0.0f0, 1.0f0)
                set!(g, mask_sym, new_m; is_param=true)
            end
        end
        if verbose && epoch % 10 == 0
            println("Époque $epoch | Loss (MSE) : $(round(loss_val[1], digits=5))")
        end
    end
    sortie_finale = vec(demand!(g, final_node))
    gates = [node(g, Symbol(:m_, idx)).value[1] for idx in 1:length(comparateurs)]
    return sortie_finale, gates, loss_val[1]
end

sortie, gates, final_loss = experimenter_reseau_de_tri_fixed()
println("\nSortie finale : ", round.(sortie, digits=3))
println("Portes finales : ", round.(gates, digits=3))
println("Loss finale : ", final_loss)
println("issorted : ", issorted(sortie))
println("est_permutation : ", sort(Float32[3.0,1.0,4.0,2.0]) ≈ sort(sortie))
