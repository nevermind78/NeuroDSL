
@inline function _buf!(ctx, key, proto)
    haskey(ctx, key) && return ctx[key]
    return (ctx[key] = similar(proto))
end

@inline function _buf_zeros!(ctx, key, dev, dims)
    haskey(ctx, key) && return ctx[key]
    return (ctx[key] = Backend.zeros32(dev, dims...))
end

"""
    accum_grad!(nd::GraphNode, g_val)
Accumule le gradient `g_val` dans le nœud `nd` si celui‑ci est backpropable.
Les `Datom` ignorent le gradient.
"""
function accum_grad!(nd::GraphNode, g_val)
    nd.atom_type <: Datom && return
    g_val === nothing     && return
    if nd.gradient === nothing
        nd.gradient = similar(g_val)
        nd.gradient .= g_val
    else
        nd.gradient .+= g_val
    end
end

const GRAD_RULES = Dict{Symbol,Function}()

GRAD_RULES[:matmul] = (dev,dy,ctx,inputs) -> begin
    tb = get(ctx,:trans_b,false)
    A  = get(ctx, :A, inputs[1])
    B  = inputs[2]
    if tb
        dA = dy * B
        dB = (A' * dy)'
    else
        dA = dy * B'
        dB = A' * dy
    end
    (dA, dB)
end

GRAD_RULES[:linear] = (dev, dy, ctx, inputs) -> begin
    X, W = inputs[1], inputs[2]
    dX = _buf!(ctx, :_buf_dX, X)
    mul!(dX, dy, W)                   # dX = dy * W

    # Gradient de W : dy' * X
    dW_calc = dy' * X
    # Si la forme obtenue ne correspond pas à W, c'est que W avait été transposé
    if size(dW_calc) != size(W)
        dW_calc = dW_calc'
    end
    dW = _buf!(ctx, :_buf_dW, W)
    dW .= dW_calc

    if length(inputs) == 3
        b = inputs[3]
        db = sum(dy, dims=1)
        if ndims(b) == 1
            db = vec(db)
        end
        return (dX, dW, db)
    else
        return (dX, dW)
    end
end

GRAD_RULES[:add] = (dev,dy,ctx,inputs) -> (dy, dy)

GRAD_RULES[:mul] = (dev,dy,ctx,inputs) -> begin
    buf1 = _buf!(ctx, :_buf_m1, dy)
    buf2 = _buf!(ctx, :_buf_m2, dy)
    buf1 .= dy .* inputs[2]
    buf2 .= dy .* inputs[1]
    (buf1, buf2)
end

GRAD_RULES[:rmsnorm] = (dev,dy,ctx,inputs) -> begin
    x,gamma = inputs[1],inputs[2]; rms_inv = ctx[:rms_inv]
    dx     = _buf!(ctx, :_buf_dx,     x)
    dgamma = _buf!(ctx, :_buf_dgamma, gamma)
    rmsnorm_bwd!(dev,dx,dgamma,dy,x,gamma,rms_inv)
    (dx, dgamma)
end

GRAD_RULES[:swiglu] = (dev,dy,ctx,inputs) -> begin
    gate,up = inputs[1],inputs[2]
    dgate = _buf!(ctx, :_buf_dgate, gate)
    dup   = _buf!(ctx, :_buf_dup,   up)
    swiglu_bwd!(dev,dgate,dup,dy,gate,up)
    (dgate,dup)
end

GRAD_RULES[:softmax] = (dev,dy,ctx,inputs) -> begin
    out = ctx[:output]
    dx  = _buf!(ctx, :_buf_dx, out)
    softmax_bwd!(dev, dx, dy, out)
    (dx,)
end

GRAD_RULES[:scale_mask] = (dev,dy,ctx,inputs) -> begin
    buf = _buf!(ctx, :_buf_sm, dy)
    buf .= dy .* ctx[:scale]
    (buf,)
end

GRAD_RULES[:rope] = (dev,dy,ctx,inputs) -> begin
    cos_a = ctx[:cos_a]; sin_a = ctx[:sin_a]; half = ctx[:half]
    dx = _buf!(ctx, :_buf_dx, dy)
    dx[:, 1:half]     .=  dy[:, 1:half] .* cos_a .+ dy[:, half+1:end] .* sin_a
    dx[:, half+1:end] .= .-dy[:, 1:half] .* sin_a .+ dy[:, half+1:end] .* cos_a
    (dx,)
end

GRAD_RULES[:slice_cols] = (dev,dy,ctx,inputs) -> begin
    dx = _buf_zeros!(ctx, :_buf_dx, dev, size(inputs[1]))
    dx .= 0f0
    s, e = ctx[:start_col], ctx[:end_col]
    dx[:, s:e] .= dy
    (dx,)
end

GRAD_RULES[:hcat_heads] = (dev,dy,ctx,inputs) -> begin
    d = ctx[:d_head]
    tuple([dy[:, (i-1)*d+1 : i*d] for i in 1:length(inputs)]...)
end

if Backend.CUDA_AVAILABLE
    function _embedding_bwd_kernel!(dE_gpu::CUDA.CuDeviceMatrix{Float32},
                                     dy_gpu::CUDA.CuDeviceMatrix{Float32},
                                     idx_gpu::CUDA.CuDeviceVector{Int},
                                     D_emb::Int, N_batch::Int)
        li = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        li > N_batch * D_emb && return
        batch_row = cld(li, D_emb)
        emb_col   = (li - 1) % D_emb + 1
        target_embedding_row = @inbounds idx_gpu[batch_row]
        CUDA.atomic_add!(view(dE_gpu, target_embedding_row, emb_col), dy_gpu[batch_row, emb_col])
        return
    end
end

GRAD_RULES[:embedding] = (dev, dy, ctx, inputs) -> begin
    E = inputs[1]
    idx = ctx[:idx]  # Déjà des Int
    dE = _buf_zeros!(ctx, :_buf_dE, dev, size(E))
    dE .= 0f0
    # 🔧 Toujours utiliser le CPU pour l'embedding backward (évite atomic_add GPU)
    idx_cpu = collect(Int, idx)
    dE_cpu = Array(dE)
    dy_cpu = Array(dy)
    for (row_out, row_E) in enumerate(idx_cpu)
        dE_cpu[row_E, :] .+= dy_cpu[row_out, :]
    end
    # Recopier sur le device
    dE .= Backend.to_device(dev, dE_cpu)
    (dE, nothing)
end

#GRAD_RULES[:mse_loss] = (dev,dy,ctx,inputs) ->
    #(mse_loss_bwd(ctx[:out], ctx[:target], dy), nothing)
    
#GRAD_RULES[:mse_loss] = (dev,dy,ctx,inputs) ->
    # inputs[1] est 'out' (la prédiction), inputs[2] est 'target'
    #(mse_loss_bwd(inputs[1], inputs[2], dy), nothing)
GRAD_RULES[:mse_loss] = (dev, dy, ctx, inputs) -> begin
    out, target = inputs[1], inputs[2]
    N = length(out)
    diff = out .- target
    loss_grad = sum(dy)
    grad_out = (2.0f0 / N) .* diff .* loss_grad
    return (grad_out, -grad_out)
end


GRAD_RULES[:sum_matrix] = (dev,dy,ctx,inputs) ->
    (sum_matrix_bwd(dev, inputs[1], dy),)

GRAD_RULES[:relu] = (dev,dy,ctx,inputs) -> (dy .* (inputs[1] .> 0f0),)
GRAD_RULES[:dropout] = (dev,dy,ctx,inputs) -> begin
    mask = ctx[:mask]; rate = ctx[:rate]
    (dy .* mask ./ (1f0 - rate),)
end

GRAD_RULES[:fused_matmul_relu] = (dev, dy, ctx, inputs) -> begin
    # 1. Safe retrieval of the output for the ReLU mask
    out = get(ctx, :output, nothing)
    if out === nothing
        error("❌ Backward Error: Context for :fused_matmul_relu is missing :output. " *
              "Ensure _dispatch_op is saving the context correctly.")
    end
    
    # ReLU Backward: gradient is 0 where output was <= 0
    dz = dy .* (out .> 0f0)
    
    # 2. MatMul Backward
    tb = get(ctx, :trans_b, false)
    A  = get(ctx, :A, inputs[1])
    B  = inputs[2]
    
    if tb
        dA = dz * B
        dB = (A' * dz)'
    else
        dA = dz * B'
        dB = A' * dz
    end
    
    (dA, dB)
end

GRAD_RULES[:cross_entropy] = (dev, dy, ctx, inputs) -> begin
    logits = ctx[:logits]
    labels = vec(ctx[:labels])  # S'assurer que c'est un vecteur
    dlogits = cross_entropy_grad(logits, labels)
    dlogits .*= dy[1]
    (dlogits, nothing)
end

"""
    backward_graph!(g, loss_sym; ctx_store, namespace)
"""
function backward_graph!(g::NeuroGraph, loss_sym::Symbol;
                         ctx_store::CtxStore=CtxStore(), namespace=g.active_ns,
                         full::Bool=true,
                         sparse::Bool = false,
                         log::Union{Nothing, ExecutionLog}=nothing)

    if sparse
        return backward_graph_sparse!(g, loss_sym; ctx_store=ctx_store, namespace=namespace)
    end
    ns = namespace

    # Réinitialisation des gradients
    for (_, nd) in g.nodes[ns]
        nd.gradient = nothing
        nd.backwarded = false
    end

    ln = node(g, loss_sym; namespace=ns)
    # Initialisation du gradient de la perte à 1 (scalaire)
    ln.gradient = Backend.ones32(g.device, size(ln.value)...)
    ln.backwarded = false

    # Parcours inverse du graphe
    for out_sym in reverse(topo_order!(g; namespace=ns))
        !haskey(g.rules[ns], out_sym) && continue
        rule   = g.rules[ns][out_sym]
        nd_out = g.nodes[ns][out_sym]

        # Si le nœud n'a pas de gradient (parce qu'il n'est pas sur le chemin), on passe
        if nd_out.gradient === nothing
            continue
        end

        if !haskey(GRAD_RULES, rule.op)
            error("❌ Pas de règle backward pour :$(rule.op)")
        end

        # Récupération du contexte (exécute forward si nécessaire)
        ctx = get(ctx_store, out_sym, nothing)
        if ctx === nothing
            ctx_tmp = CtxStore()
            execute_rule!(g, rule; ctx_store=ctx_tmp, namespace=ns)
            ctx = get(ctx_tmp, out_sym, Dict{Symbol,Any}())
        end

        inputs_vals = [g.nodes[ns][s].value for s in rule.inputs]
        grads = GRAD_RULES[rule.op](g.device, nd_out.gradient, ctx, inputs_vals)

        for (i, in_sym) in enumerate(rule.inputs)
            accum_grad!(g.nodes[ns][in_sym], grads[i])
            g.nodes[ns][in_sym].backwarded = true
        end

        # Libération du gradient de sortie s'il n'est pas un paramètre
        nd_out.backwarded = true
        if !nd_out.is_param
            nd_out.gradient = nothing
        end
    end
    return g
end