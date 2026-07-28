using LinearAlgebra, NeuroDSL, Statistics

register_op!(:neural_sort_v2, (dev, out, inputs, attrs, out_sym, out_node, ctx) -> begin
    x = inputs[1]; n = size(x,1)
    tau = get(attrs, :tau, 0.05f0); gamma = get(attrs, :gamma, 0.01f0)
    s = std(x) + 1f-6
    diffs = (x .- x') ./ s
    S = 1f0 ./ (1f0 .+ exp.(.-diffs ./ tau))
    ranks = sum(S, dims=2) .- 0.5f0
    positions = Float32.(0:n-1)'
    dist_to_pos = -(ranks .- positions).^2
    P = exp.(dist_to_pos ./ gamma); P ./= sum(P, dims=2)
    out .= P' * x
    if ctx !== nothing; ctx[out_sym] = Dict{Symbol,Any}(:P=>P); end
end)
GRAD_RULES[:neural_sort_v2] = (dev, dy, ctx, inputs) -> ((ctx[:P] * dy),)

register_op!(:fast_sort_v2, (dev, out, inputs, attrs, out_sym, out_node, ctx) -> begin
    x = inputs[1]
    idx = sortperm(vec(x))
    out .= x[idx, :]
    if ctx !== nothing; ctx[out_sym] = Dict{Symbol,Any}(:idx=>idx, :n=>size(x,1)); end
end)
GRAD_RULES[:fast_sort_v2] = (dev, dy, ctx, inputs) -> begin
    idx = ctx[:idx]; n = ctx[:n]
    dx = zeros(Float32, n, 1)
    dx[idx, 1] .= dy
    return (dx,)
end

function train_test(op_sym, x_init; epochs=300, lr=0.05f0, attrs=Dict{Symbol,Any}())
    n = length(x_init)
    g = NeuroGraph(device=Backend.CPUDevice())
    set!(g, :x_brut, reshape(x_init, n, 1); is_param=true)
    addrule!(g, GraphRule(:x_trie, [:x_brut], op_sym; attrs=attrs))
    target = Float32.(collect(1:n))
    set!(g, :target, reshape(target, n, 1); is_param=false)
    addrule!(g, GraphRule(:loss, [:x_trie, :target], :mse_loss))
    local lv
    for e in 1:epochs
        zero_grads!(g)
        lv = demand!(g, :loss)
        backward_graph!(g, :loss)
        nd = node(g, :x_brut)
        newx = nd.value .- lr .* nd.gradient
        set!(g, :x_brut, newx; is_param=true)
    end
    xt = vec(demand!(g, :x_trie))
    return xt, lv[1]
end

x0 = Float32[8.4, 2.1, 9.9, 3.14, 5.5]
xt, lv = train_test(:neural_sort_v2, x0; attrs=Dict{Symbol,Any}(:tau=>0.05f0,:gamma=>0.01f0))
println("neural_sort_v2 : x_trie final=", round.(xt,digits=3), " loss=", lv, " issorted=", issorted(xt))

xt2, lv2 = train_test(:fast_sort_v2, x0)
println("fast_sort_v2   : x_trie final=", round.(xt2,digits=3), " loss=", lv2, " issorted=", issorted(xt2))
