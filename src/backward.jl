
"""
    GRAD_POOL_ENABLED

Interrupteur global : si `true` (défaut), les tampons de gradient/scratch
du backward sont emprunté à un `GradPool` par graphe (voir `grad_pool.jl`)
au lieu d'être alloués frais à chaque appel de `_buf!`/`_buf_zeros!`.
`false` restaure exactement le comportement historique (allocation GC pure)
-- utilisé par `test/test_grad_pool.jl` pour comparer les deux chemins
bit-à-bit, et comme repli immédiat en cas de doute.
"""
const GRAD_POOL_ENABLED = Ref(true)

"""
    BACKWARD_RELEASE_VALUES

Interrupteur global (défaut `false` -- zéro changement pour tout appelant
existant) : si `true`, `backward_graph!`/`backward_graph_sparse!` libèrent
(`Backend.free!`) la valeur forward de chaque nœud non-paramètre juste après
sa PROPRE visite dans la boucle inverse (pas après la visite de son
consommateur -- voir la docstring de `backward_graph!` pour la preuve que
c'est le point sûr : à ce moment, `nd_out.value` et les valeurs des entrées
qu'on vient de lire ne seront JAMAIS relues par une itération ultérieure de
CETTE passe, par construction du DAG). Cible le pic de `train_step` (les
activations forward restent résidentes de la transition forward→backward
jusqu'à la fin du backward sinon) -- n'a aucun effet sur un `demand!` seul
(validation/génération, déjà couvert par `demand_release!`).

Contrepartie mesurée : le PAS SUIVANT doit réallouer ces buffers au forward
(`out_node.value === nothing` déclenche un `Backend.zeros32` frais,
`src/dispatch.jl:151`) au lieu de réutiliser le buffer en place -- gain de
pic contre un peu de churn d'allocation, à mesurer, pas à supposer. Sur CPU,
`Backend.free!` est un no-op : ce mode n'a d'effet réel que sur CUDA.
"""
const BACKWARD_RELEASE_VALUES = Ref(false)

@inline function _release_forward_value!(g::NeuroGraph, nd::GraphNode, ns::Symbol, loss_sym::Symbol)
    (nd.is_param || nd.name === loss_sym || nd.value === nothing) && return
    Backend.free!(g.device, nd.value)
    nd.value = nothing
    nd.valid = false
end

"""
    _freed_backward_error(sym)

Lève une erreur claire quand un backward rencontre une valeur forward
manquante (`.value === nothing`) sur le chemin du gradient -- au lieu du
crash cryptique qu'aurait produit `execute_rule!`/`size(nothing)` un peu
plus loin. Le cas normal : `demand_release!`/`release_intermediates!`
(`src/demand_release.jl`) ont libéré cette valeur en supposant qu'aucun
backward ne suivrait sur ce namespace. Zéro faux positif attendu : après un
`demand!` classique (jamais après `demand_release!`), tout nœud sur le
chemin du gradient a une valeur résidente par construction.
"""
_freed_backward_error(sym::Symbol) = error(
    "❌ Valeur forward de :$sym absente pour le backward -- libérée par " *
    "demand_release!/release_intermediates! (ou jamais calculée). Refaire " *
    "un demand!(g, loss_sym) complet avant backward_graph! sur ce " *
    "namespace ; demand_release! est réservé aux passes qui ne seront " *
    "JAMAIS suivies d'un backward_graph! (validation, génération)."
)

@inline function _buf!(ctx, key, proto)
    gpool = get(ctx, :_gpool, nothing)
    if gpool !== nothing
        return grad_acquire!(gpool, size(proto))
    end
    haskey(ctx, key) && return ctx[key]
    return (ctx[key] = similar(proto))
end

@inline function _buf_zeros!(ctx, key, dev, dims)
    gpool = get(ctx, :_gpool, nothing)
    if gpool !== nothing
        buf = grad_acquire!(gpool, dims)
        fill!(buf, 0f0)
        return buf
    end
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

"""
    register_grad!(op::Symbol, fn::Function) -> op

Register the backward rule for `op` (the gradient counterpart of [`register_op!`](@ref)). Prefer this over
a raw `GRAD_RULES[op] = fn` assignment: like `register_op!`, it fires the host registration hook (see
[`set_register_hook!`](@ref)) so a custom gradient defined in a host cell is re-established on every
execution context (a distributed host running backward/training elsewhere) — otherwise cross-context
backward fails with "pas de règle backward pour :op". Zero-dependency and a no-op standalone.
"""
function register_grad!(op::Symbol, fn::Function)
    GRAD_RULES[op] = fn
    _notify_register(op)
    return op
end

GRAD_RULES[:matmul] = (dev,dy,ctx,inputs) -> begin
    tb = get(ctx,:trans_b,false)
    A  = get(ctx, :A, inputs[1])
    B  = inputs[2]
    # dA emprunté au pool de gradients quand actif (voir grad_pool.jl) --
    # sinon la même allocation BLAS fraîche qu'avant (`dy*B`/`dy*B'`), jamais
    # libérée avant le prochain GC. `mul!` calcule directement dans le
    # tampon emprunté, sans étape intermédiaire.
    gpool = get(ctx, :_gpool, nothing)
    if gpool !== nothing
        dA = grad_acquire!(gpool, size(A))
        tb ? mul!(dA, dy, B) : mul!(dA, dy, B')
    else
        dA = tb ? dy * B : dy * B'
    end

    # Si B est un paramètre déjà pourvu d'un buffer de gradient persistant
    # (voir la boucle de réinitialisation dans backward_graph!, qui ne nullifie
    # plus les gradients de paramètres mais les remet à zéro en place),
    # accumuler dB DIRECTEMENT dans ce buffer via mul! évite d'allouer un `dB`
    # temporaire à chaque pas -- ce temporaire, jamais libéré avant le
    # prochain GC, dominait le pic de mémoire mesuré
    # (notebook/real_llm_vram_probe.jl). `dy'*A` est l'identité algébrique de
    # `(A'*dy)'`, calculée directement par BLAS sans matérialiser de transposée
    # intermédiaire.
    in_nodes = get(ctx, :_in_nodes, nothing)
    B_node = in_nodes === nothing ? nothing : in_nodes[2]
    if B_node !== nothing && B_node.is_param && B_node.gradient !== nothing
        if tb
            mul!(B_node.gradient, dy', A, 1f0, 1f0)
        else
            mul!(B_node.gradient, A', dy, 1f0, 1f0)
        end
        dB = nothing
    else
        dB = tb ? (A' * dy)' : A' * dy
    end
    (dA, dB)
end

GRAD_RULES[:linear] = (dev, dy, ctx, inputs) -> begin
    X, W = inputs[1], inputs[2]
    dX = _buf!(ctx, :_buf_dX, X)
    mul!(dX, dy, W)                   # dX = dy * W

    # Gradient de W : dy' * X -- même optimisation que :matmul (voir
    # commentaire ci-dessus) : accumulation directe dans le buffer persistant
    # du paramètre quand disponible, sinon repli sur le calcul classique
    # (première passe jamais encore backpropée, ou appel hors du moteur
    # d'entraînement standard).
    in_nodes = get(ctx, :_in_nodes, nothing)
    W_node = in_nodes === nothing ? nothing : in_nodes[2]
    if W_node !== nothing && W_node.is_param && W_node.gradient !== nothing
        if size(W) == (size(dy,2), size(X,2))
            mul!(W_node.gradient, dy', X, 1f0, 1f0)
        else
            mul!(W_node.gradient, X', dy, 1f0, 1f0)   # (dy'*X)' == X'*dy
        end
        dW = nothing
    else
        dW_calc = dy' * X
        # Si la forme obtenue ne correspond pas à W, c'est que W avait été transposé
        if size(dW_calc) != size(W)
            dW_calc = dW_calc'
        end
        dW = _buf!(ctx, :_buf_dW, W)
        dW .= dW_calc
    end

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


# ATTENTION : les DEUX opérandes doivent être des copies indépendantes de
# `dy`, jamais `dy` lui-même. Les deux entrées d'un `:add` reçoivent chacune
# `in_nd.gradient = grads[i]` (alias direct, sans copie, pour les nœuds
# non-paramètres -- backward.jl, boucle principale) : si l'une des deux
# partage le MÊME objet que `dy` (= `nd_out.gradient`), et que `nd_out` lui-
# même est ensuite recyclé (mutation `.+=` ailleurs dans la boucle inverse,
# OU rendu à un pool de buffers), l'entrée se retrouve silencieusement
# corrompue -- elle croit encore posséder ce tableau alors qu'il a changé de
# mains. Sur la topologie actuelle d'un bloc Transformer standard, l'ordre
# topologique inverse fait qu'aucune des deux entrées ne survit assez
# longtemps pour être touchée après coup -- mais c'est un hasard de
# structure, pas une garantie, et un futur graphe avec un partage différent
# (ou un mécanisme de réutilisation de buffer côté gradient) briserait ça
# silencieusement. Deux `copy(dy)` corrigent la cause à la racine, pour un
# coût négligeable comparé aux autres allocations de la passe.
GRAD_RULES[:add] = (dev,dy,ctx,inputs) -> (copy(dy), copy(dy))

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

# ── Attention multi-têtes batchée : GRAD_RULES pour les ops de vue et
# les 2 matmuls groupés (conçu avec Fable, 2026-07-10, src/kernels.jl). ──
GRAD_RULES[:view_cols] = (dev,dy,ctx,inputs) -> begin
    dx = _buf_zeros!(ctx, :_buf_dx_view_cols, dev, size(inputs[1]))
    dx .= 0f0
    s, e = ctx[:start_col], ctx[:end_col]
    dx[:, s:e] .= dy
    (dx,)
end

GRAD_RULES[:head_view] = (dev,dy,ctx,inputs) -> begin
    dx = _buf_zeros!(ctx, :_buf_dx_head_view, dev, size(inputs[1]))
    dx .= 0f0
    h = ctx[:head]
    dx[:, :, h] .= dy
    (dx,)
end

GRAD_RULES[:batched_qk] = (dev,dy,ctx,inputs) -> begin
    H = length(inputs) ÷ 2
    q_heads = inputs[1:H]
    k_heads = inputs[H+1:2H]
    seq = size(q_heads[1], 1); d_head = size(q_heads[1], 2)
    Qp = _sibling_view_parent(q_heads, (seq, d_head))
    Kp = _sibling_view_parent(k_heads, (seq, d_head))
    Q3 = Qp !== nothing ? reshape(Qp, seq, d_head, H) : _gather3(q_heads, seq, d_head, H, dev)
    K3 = Kp !== nothing ? reshape(Kp, seq, d_head, H) : _gather3(k_heads, seq, d_head, H, dev)
    dQ3 = similar(Q3); dK3 = similar(K3)
    _batched_qk_bwd!(dev, dQ3, dK3, dy, Q3, K3)
    grads = Vector{Any}(undef, 2H)
    for h in 1:H
        grads[h]   = copy(view(dQ3, :, :, h))
        grads[H+h] = copy(view(dK3, :, :, h))
    end
    tuple(grads...)
end

GRAD_RULES[:batched_pv] = (dev,dy,ctx,inputs) -> begin
    H = length(inputs) ÷ 2
    pr_heads = inputs[1:H]
    v_heads = inputs[H+1:2H]
    seq = size(pr_heads[1], 1); d_head = size(v_heads[1], 2)
    Pp = _sibling_view_parent(pr_heads, (seq, seq))
    Vp = _sibling_view_parent(v_heads, (seq, d_head))
    P3 = Pp !== nothing ? reshape(Pp, seq, seq, H) : _gather3(pr_heads, seq, seq, H, dev)
    V3 = Vp !== nothing ? reshape(Vp, seq, d_head, H) : _gather3(v_heads, seq, d_head, H, dev)
    dP3 = similar(P3); dV3 = similar(V3)
    _batched_pv_bwd!(dev, dP3, dV3, dy, P3, V3)
    grads = Vector{Any}(undef, 2H)
    for h in 1:H
        grads[h]   = copy(view(dP3, :, :, h))
        grads[H+h] = copy(view(dV3, :, :, h))
    end
    tuple(grads...)
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
    # En évaluation (training=false), le forward est l'identité -- le
    # gradient doit donc passer inchangé, pas être filtré par un masque tiré
    # au dernier forward d'ENTRAÎNEMENT (bug latent corrigé au passage : ce
    # cas n'était jamais exercé puisqu'on ne backprop pas en évaluation,
    # mais devient atteignable avec CTX_REBUILD qui persiste le masque
    # au-delà d'un seul appel).
    get(ctx, :training, true) || return (dy,)
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

GRAD_RULES[:fused_matmul_add] = (dev, dy, ctx, inputs) -> begin
    tb   = get(ctx, :trans_b, false)
    A    = get(ctx, :A, inputs[1])
    B    = inputs[2]
    bias = inputs[3]

    if tb
        dA = dy * B
        dB = (A' * dy)'
    else
        dA = dy * B'
        dB = A' * dy
    end
    db = sum(dy, dims=1)
    ndims(bias) == 1 && (db = vec(db))

    (dA, dB, db)
end

GRAD_RULES[:fused_matmul_add_relu] = (dev, dy, ctx, inputs) -> begin
    out = get(ctx, :output, nothing)
    if out === nothing
        error("❌ Backward Error: Context for :fused_matmul_add_relu is missing :output. " *
              "Ensure _dispatch_op is saving the context correctly.")
    end

    # ReLU backward : le gradient est nul là où la sortie était <= 0
    dz = dy .* (out .> 0f0)

    tb   = get(ctx, :trans_b, false)
    A    = get(ctx, :A, inputs[1])
    B    = inputs[2]
    bias = inputs[3]

    if tb
        dA = dz * B
        dB = (A' * dz)'
    else
        dA = dz * B'
        dB = A' * dz
    end
    db = sum(dz, dims=1)
    ndims(bias) == 1 && (db = vec(db))

    (dA, dB, db)
end

GRAD_RULES[:fused_qkv_projection] = (dev, dy, ctx, inputs) -> begin
    tb = get(ctx, :trans_b, false)
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

# Fusion algébrique pure (double_transpose_elim, add_zero_elim) : le gradient
# passe inchangé, il n'y a pas de calcul réel derrière :identity.
GRAD_RULES[:identity] = (dev, dy, ctx, inputs) -> (dy,)

GRAD_RULES[:cross_entropy] = (dev, dy, ctx, inputs) -> begin
    logits = ctx[:logits]
    labels = vec(ctx[:labels])  # S'assurer que c'est un vecteur
    dlogits = cross_entropy_grad(logits, labels)
    dlogits .*= dy   # dy est de forme (1,) -- broadcast, pas dy[1] (setindex/getindex
                      # scalaire interdit sur CUDA ; broadcasting donne le même résultat)
    (dlogits, nothing)
end

# ══════════════════════════════════════════════════════════════════════════
# CTX_REBUILD -- reconstruction du contexte de backward SANS ré-exécuter le
# forward (voir _ctx_for_backward! ci-dessous). Chaque entrée reconstruit
# exactement les clés que la GRAD_RULES correspondante lit, depuis
# `rule.attrs` (statique), `aux_data` (persistant par nœud, écrit au
# forward) ou les valeurs déjà connues des nœuds -- jamais en ré-exécutant
# `_dispatch_op`. `nothing` en retour signale "je ne peux pas reconstruire
# sûrement ici" et déclenche le repli (ré-exécution, comportement historique
# inchangé) dans `_ctx_for_backward!`.
#
# Avant le fix, CHAQUE nœud du graphe ré-exécutait son forward pendant le
# backward (ctx_store vide par défaut -> execute_rule! rappelé pour chaque
# nœud) : ça doublait le calcul du backward ET allouait à nouveau les
# copies défensives (`A_ctx = copy(A)`, src/dispatch.jl) à chaque pas,
# jamais libérées avant le prochain GC -- une bonne part du pic mesuré sur
# train_step (notebook/real_llm_vram_probe.jl). Un op non couvert ici
# retombe automatiquement sur l'ancien comportement (aucune régression pour
# les CUSTOM_OPS/@defop qui n'ont pas d'entrée dans ce registre).
# ══════════════════════════════════════════════════════════════════════════
const CTX_REBUILD = Dict{Symbol,Function}()

# Armé (une fois pour tout le processus, jamais désarmé) dès que
# `acquire!` (src/liveness.jl, seul point d'entrée du BufferPool -- vérifié
# par grep, aucun autre appelant dans src/) est utilisé -- désactive
# définitivement le chemin CTX_REBUILD, puisque c'est le seul mécanisme du
# dépôt qui peut faire d'`out_node.value` un buffer partagé/aliasé plutôt
# qu'un tableau alloué en propre pour ce nœud.
const _POOLED_EXECUTION_SEEN = Ref(false)

_ctx_empty(dev, rule, nd_out, inputs_vals) = Dict{Symbol,Any}()
for op in (:add, :mul, :swiglu, :relu, :identity, :mse_loss, :sum_matrix, :linear)
    CTX_REBUILD[op] = _ctx_empty
end

CTX_REBUILD[:matmul] = (dev, rule, nd_out, inputs_vals) ->
    Dict{Symbol,Any}(:trans_b => get(rule.attrs, :trans_b, false))

CTX_REBUILD[:fused_matmul_add] = CTX_REBUILD[:matmul]
CTX_REBUILD[:fused_qkv_projection] = CTX_REBUILD[:matmul]

CTX_REBUILD[:fused_matmul_relu] = (dev, rule, nd_out, inputs_vals) ->
    Dict{Symbol,Any}(:trans_b => get(rule.attrs, :trans_b, false), :output => nd_out.value)
CTX_REBUILD[:fused_matmul_add_relu] = CTX_REBUILD[:fused_matmul_relu]

CTX_REBUILD[:rmsnorm] = (dev, rule, nd_out, inputs_vals) -> begin
    rms_inv = get(nd_out.aux_data, :rms_inv, nothing)
    (rms_inv === nothing || size(rms_inv, 1) != size(inputs_vals[1], 1)) && return nothing
    Dict{Symbol,Any}(:rms_inv => rms_inv)
end

CTX_REBUILD[:softmax] = (dev, rule, nd_out, inputs_vals) ->
    Dict{Symbol,Any}(:output => nd_out.value)

CTX_REBUILD[:scale_mask] = (dev, rule, nd_out, inputs_vals) ->
    Dict{Symbol,Any}(:scale => 1f0 / sqrt(Float32(get(rule.attrs, :d_head, size(inputs_vals[1], 2)))))

CTX_REBUILD[:rope] = (dev, rule, nd_out, inputs_vals) -> begin
    cos_a = get(nd_out.aux_data, :cos_a, nothing)
    sin_a = get(nd_out.aux_data, :sin_a, nothing)
    (cos_a === nothing || sin_a === nothing || size(cos_a, 1) != size(inputs_vals[1], 1)) && return nothing
    Dict{Symbol,Any}(:cos_a => cos_a, :sin_a => sin_a, :half => size(inputs_vals[1], 2) ÷ 2)
end

CTX_REBUILD[:slice_cols] = (dev, rule, nd_out, inputs_vals) -> Dict{Symbol,Any}(
    :start_col => get(rule.attrs, :start_col, 1),
    :end_col   => get(rule.attrs, :end_col, size(inputs_vals[1], 2)),
)

CTX_REBUILD[:hcat_heads] = (dev, rule, nd_out, inputs_vals) ->
    Dict{Symbol,Any}(:d_head => size(inputs_vals[1], 2))

CTX_REBUILD[:view_cols] = (dev, rule, nd_out, inputs_vals) -> Dict{Symbol,Any}(
    :start_col => get(rule.attrs, :start_col, 1),
    :end_col   => get(rule.attrs, :end_col, size(inputs_vals[1], 2)),
)
CTX_REBUILD[:head_view] = (dev, rule, nd_out, inputs_vals) ->
    Dict{Symbol,Any}(:head => rule.attrs[:head])
CTX_REBUILD[:batched_qk] = _ctx_empty
CTX_REBUILD[:batched_pv] = _ctx_empty

CTX_REBUILD[:embedding] = (dev, rule, nd_out, inputs_vals) ->
    Dict{Symbol,Any}(:idx => Int.(vec(Array(inputs_vals[2]))))

CTX_REBUILD[:cross_entropy] = (dev, rule, nd_out, inputs_vals) ->
    Dict{Symbol,Any}(:logits => inputs_vals[1], :labels => Int.(vec(inputs_vals[2])))

# :dropout -- seul op où le contexte forward (le masque tiré aléatoirement)
# n'est pas reconstructible depuis rule.attrs/les valeurs de nœuds : il DOIT
# être persisté au forward (src/dispatch.jl, branche :dropout) puis relu
# ici. Absence de persistance ou de correspondance de forme -> repli
# ré-exécution (identique au comportement historique, y compris son bug
# connu de masque non déterministe -- documenté, pas corrigé par ce fix).
CTX_REBUILD[:dropout] = (dev, rule, nd_out, inputs_vals) -> begin
    mask = get(nd_out.aux_data, :dropout_mask, nothing)
    (mask === nothing || size(mask) != size(inputs_vals[1])) && return nothing
    Dict{Symbol,Any}(:mask => mask, :rate => get(nd_out.aux_data, :dropout_rate, 0f0),
                      :training => get(nd_out.aux_data, :dropout_training, true))
end

"""
    _ctx_for_backward!(g, rule, ns) -> Dict{Symbol,Any}

Contexte de backward pour `rule`, reconstruit SANS ré-exécuter le forward
quand c'est prouvé sûr, avec repli automatique (ré-exécution, comportement
historique inchangé) sinon. Trois garde-fous, tous nécessaires et vérifiés
suffisants sur le chemin `demand!`/`backward_graph!` (qui n'utilise jamais
`execute_rule_pooled!`/`demand_planned!`) :

1. `_POOLED_EXECUTION_SEEN[]` : armé une fois pour toutes dès qu'un
   `BufferPool` a été utilisé (`acquire!`, src/liveness.jl) -- le seul
   chemin où `out_node.value` pourrait être un buffer partagé/aliasé plutôt
   qu'un tableau alloué en propre pour ce nœud.
2. Validité : `nd_out` doit être `valid` avec une `.value` non nulle --
   garantit que c'est bien la valeur du forward qui a produit la perte en
   cours de backward, pas un résidu d'un appel antérieur. Pour une ENTRÉE,
   ce contrôle ne s'applique que si elle a elle-même une règle (un nœud
   dérivé peut être périmé) -- une feuille (`set!`, sans règle) n'a pas
   cette notion : `_invalidate_downstream!` marque `valid=false` sur TOUTE
   feuille dès qu'elle est affectée (`src/graph_api.jl:60,82`) et rien ne le
   remet jamais à `true` (le parcours de `demand!` ignore les nœuds sans
   règle, `src/dispatch.jl:661`) -- exiger `valid` sur une feuille
   désactiverait le rebuild pour quasiment tout graphe réel (la toute
   première couche consomme toujours directement une feuille). Sa valeur
   reste par construction celle du dernier `set!`, jamais périmée entre un
   forward et son backward (rien ne rappelle `set!` entre les deux dans un
   pas d'entraînement normal).
3. Garde d'alias explicite : `nd_in.value !== nd_out.value` pour chaque
   entrée -- filet de sécurité direct contre le scénario que la copie
   défensive `A_ctx = copy(A)` de `_dispatch_op` protège au forward (le
   rebuild lit `inputs_vals` au lieu de cette copie ; si jamais un alias
   existait malgré la garde 1, il serait quand même détecté ici).
"""
function _ctx_for_backward!(g::NeuroGraph, rule::GraphRule, ns::Symbol)
    out_sym = rule.output
    nd_out  = g.nodes[ns][out_sym]
    rebuild = get(CTX_REBUILD, rule.op, nothing)

    if rebuild !== nothing && !_POOLED_EXECUTION_SEEN[] &&
       nd_out.valid && nd_out.value !== nothing
        safe = true
        for s in rule.inputs
            nd_in = g.nodes[ns][s]
            if nd_in.value === nothing || nd_in.value === nd_out.value
                safe = false
                break
            end
            if haskey(g.rules[ns], s) && !nd_in.valid
                safe = false   # nœud DÉRIVÉ potentiellement périmé -- une feuille n'a pas cette notion (voir docstring)
                break
            end
        end
        if safe
            inputs_vals = AbstractArray{Float32}[g.nodes[ns][s].value for s in rule.inputs]
            ctx = rebuild(g.device, rule, nd_out, inputs_vals)
            ctx !== nothing && return ctx
        end
    end

    # Repli : comportement historique (ré-exécution complète du forward pour
    # ce nœud) -- couvre les CUSTOM_OPS/@defop sans entrée CTX_REBUILD, les
    # gardes échouées, et tout usage du buffer pool.
    ctx_tmp = CtxStore()
    execute_rule!(g, rule; ctx_store=ctx_tmp, namespace=ns)
    return get(ctx_tmp, out_sym, Dict{Symbol,Any}())
end

"""
    _compute_needs_bwd(g, ns) -> Dict{Symbol,Bool}

Calcule, en un seul passage dans l'ordre topologique direct (`topo_order!`, déjà
mis en cache par namespace), quels nœuds ont besoin d'un calcul de gradient lors
du backward : un nœud "a besoin" s'il est lui-même entraînable (`is_param=true`)
ou si au moins une de ses entrées (récursivement) l'est. Les feuilles sans règle
(entrées, labels) utilisent directement `is_param`. Recalculé à chaque appel
(pas de cache) : `is_param` peut changer entre deux appels via `set!`, et un
cache mal invalidé serait pire que le recalcul, qui reste O(nœuds+arêtes).
"""
function _compute_needs_bwd(g::NeuroGraph, ns::Symbol)
    needs_bwd = Dict{Symbol,Bool}()
    for sym in topo_order!(g; namespace=ns)
        nd = g.nodes[ns][sym]
        if haskey(g.rules[ns], sym)
            rule = g.rules[ns][sym]
            needs_bwd[sym] = nd.is_param || any(needs_bwd[in_sym] for in_sym in rule.inputs)
        else
            needs_bwd[sym] = nd.is_param
        end
    end
    return needs_bwd
end

"""
    backward_graph!(g, loss_sym; ctx_store, namespace, prune_frozen=false)

`prune_frozen` (désactivé par défaut) : si `true`, saute le calcul de gradient
pour tout sous-arbre prouvé sans paramètre entraînable en amont (ex. un préfixe
de couches gelées lors d'un fine-tuning partiel). Les valeurs de gradient des
nœuds `is_param=true` sont strictement identiques, activé ou non — ce mot-clé
n'élimine que du calcul dont le résultat serait de toute façon jeté par le
nettoyage final ci-dessous. Désactivé par défaut pour ne rien changer aux
appelants existants.
"""
function backward_graph!(g::NeuroGraph, loss_sym::Symbol;
                         ctx_store::CtxStore=CtxStore(), namespace=g.active_ns,
                         full::Bool=true,
                         sparse::Bool = false,
                         prune_frozen::Bool = false,
                         release_values::Bool = BACKWARD_RELEASE_VALUES[],
                         log::Union{Nothing, ExecutionLog}=nothing)

    if sparse
        return backward_graph_sparse!(g, loss_sym; ctx_store=ctx_store, namespace=namespace,
                                       prune_frozen=prune_frozen, release_values=release_values)
    end
    ns = namespace

    # Réinitialisation des gradients : un nœud paramètre RÉUTILISE son buffer
    # existant (fill! à zéro) plutôt que d'être nullifié. Nullifier forçait
    # `accum_grad!` à réallouer (`similar`+copie) à la prochaine accumulation
    # ET laissait l'ancien buffer résident en VRAM jusqu'au prochain GC (Julia
    # ne libère jamais la mémoire CUDA de façon synchrone) -- deux allocations
    # fantômes par paramètre à chaque pas, dominant le pic mesuré (voir
    # notebook/real_llm_vram_probe.jl). Un nœud non-paramètre reste nullifié :
    # son gradient est de toute façon jeté par le nettoyage final ci-dessous.
    for (_, nd) in g.nodes[ns]
        if nd.is_param && nd.gradient !== nothing
            fill!(nd.gradient, 0f0)
        else
            nd.gradient = nothing
        end
        nd.backwarded = false
    end

    ln = node(g, loss_sym; namespace=ns)
    ln.value === nothing && _freed_backward_error(loss_sym)
    gpool = GRAD_POOL_ENABLED[] ? _grad_pool_for(g) : nothing
    # Initialisation du gradient de la perte à 1 (scalaire) -- emprunté au
    # pool comme tout le reste si actif, pour que sa libération en fin de
    # boucle (nd_out.is_param est faux pour la perte) le rende bien.
    if gpool !== nothing
        ln.gradient = grad_acquire!(gpool, size(ln.value))
        fill!(ln.gradient, 1f0)
    else
        ln.gradient = Backend.ones32(g.device, size(ln.value)...)
    end
    ln.backwarded = false

    needs_bwd = prune_frozen ? _compute_needs_bwd(g, ns) : nothing

    # Parcours inverse du graphe
    for out_sym in reverse(topo_order!(g; namespace=ns))
        !haskey(g.rules[ns], out_sym) && continue
        rule   = g.rules[ns][out_sym]
        nd_out = g.nodes[ns][out_sym]

        # Si le nœud n'a pas de gradient (parce qu'il n'est pas sur le chemin), on passe
        if nd_out.gradient === nothing
            continue
        end

        # Garde défensive : avec prune_frozen=true, la garde symétrique sur les
        # entrées plus bas empêche déjà nd_out.gradient d'être jamais assigné
        # pour un sous-arbre entièrement gelé, donc cette branche ne devrait
        # normalement jamais être prise — gardée pour robustesse.
        if needs_bwd !== nothing && !needs_bwd[out_sym]
            continue
        end

        # Garde-fou explicite (voir _freed_backward_error) : ce nœud est sur
        # le chemin du gradient (il a un gradient non nul ci-dessus), donc sa
        # valeur forward ET celles de ses entrées DOIVENT être résidentes --
        # sauf si demand_release!/release_intermediates! les a libérées en
        # supposant (à tort, ici) qu'aucun backward ne suivrait.
        nd_out.value === nothing && _freed_backward_error(out_sym)
        for s in rule.inputs
            g.nodes[ns][s].value === nothing && _freed_backward_error(s)
        end

        if !haskey(GRAD_RULES, rule.op)
            error("❌ Pas de règle backward pour :$(rule.op)")
        end

        # Logging natif : avant, seul un notebook qui l'écrivait à la main
        # obtenait une trace du backward (et seulement une norme en texte,
        # pas la vraie grille par cellule) — voir l'ancien code de
        # graph_sim.ipynb. Ici ça marche pour n'importe quel graphe, sans
        # rien à ajouter côté appelant.
        log !== nothing && log_event!(log, out_sym, "backward", "starting")

        # Récupération du contexte : priorité absolue à un ctx_store fourni
        # explicitement par l'appelant (zéro changement pour ce cas -- ex.
        # grad_check et tout code qui thread un ctx partagé entre forward et
        # backward) ; sinon reconstruction sans ré-exécution du forward
        # quand c'est prouvé sûr (_ctx_for_backward!), avec repli automatique
        # (comportement historique) sinon.
        ctx = get(ctx_store, out_sym, nothing)
        if ctx === nothing
            ctx = _ctx_for_backward!(g, rule, ns)
        end

        inputs_vals = [g.nodes[ns][s].value for s in rule.inputs]
        # Exposé aux GRAD_RULES (ex. :matmul, :linear) qui veulent accumuler
        # directement dans le buffer de gradient d'un paramètre plutôt que de
        # retourner un tableau temporaire -- ignoré par toute règle qui ne
        # regarde pas cette clé, donc sans effet sur les règles existantes.
        ctx[:_in_nodes] = [g.nodes[ns][s] for s in rule.inputs]
        # Pool de gradients (voir grad_pool.jl) : lu par _buf!/_buf_zeros!
        # pour emprunter leurs tampons scratch au lieu d'allouer frais.
        ctx[:_gpool] = gpool
        dy_val = nd_out.gradient   # conservé : nd_out.gradient peut être réaffecté plus bas
        grads = GRAD_RULES[rule.op](g.device, dy_val, ctx, inputs_vals)

        for (i, in_sym) in enumerate(rule.inputs)
            in_nd = g.nodes[ns][in_sym]
            if needs_bwd !== nothing && !needs_bwd[in_sym]
                # Sous-arbre prouvé sans paramètre entraînable en amont : ne
                # pas assigner de gradient ici. C'est ce skip (pas la garde
                # défensive plus haut) qui empêche réellement in_sym d'être
                # revisité plus loin dans la boucle inverse -- in_nd.backwarded
                # reste false, cohérent avec "jamais touché". `grads[i]` n'est
                # alors JAMAIS adopté par personne : le rendre au pool ici est
                # ce qui ferme la fuite `prune_frozen` trouvée à l'audit --
                # sans ce `continue`, `_buf!` n'aurait même pas dû l'emprunter,
                # mais le calcul de la règle est déjà fait à ce stade.
                if gpool !== nothing && grads[i] !== nothing && grad_owns(gpool, grads[i])
                    grad_release!(gpool, grads[i])
                end
                continue
            end
            if in_nd.is_param
                # accum_grad! ne conserve JAMAIS grads[i] tel quel : soit une
                # copie fraîche (`similar`+`.=`) à la première écriture, soit
                # une accumulation en place (`.+=`) ensuite -- dans les deux
                # cas, grads[i] lui-même devient orphelin juste après, donc
                # toujours sûr à rendre au pool ici.
                accum_grad!(in_nd, grads[i])
                if gpool !== nothing && grads[i] !== nothing && grad_owns(gpool, grads[i])
                    grad_release!(gpool, grads[i])
                end
            else
                # Nœud non entraînable (poids gelé ou activation intermédiaire) :
                # son gradient ne sert qu'à propager plus loin, jamais à un pas
                # d'optimiseur, donc pas besoin de la copie défensive de
                # accum_grad! (similar + .=) sur la première écriture -- c'est
                # cette copie évitable, répétée pour chaque paramètre gelé, qui
                # dominait le coût mesuré (voir notebook.ipynb, gain GPU 37.7%
                # sur backward_graph_sparse!). On garde quand même la sommation
                # (.+=) si une deuxième contribution arrive : un nœud gelé à
                # plusieurs consommateurs (connexion résiduelle, poids partagé)
                # doit quand même recevoir la somme exacte, pas juste la
                # dernière contribution -- backward_graph_sparse! avait ce
                # risque avec une simple affectation à chaque fois.
                if in_nd.gradient === nothing
                    in_nd.gradient = grads[i]   # ADOPTION -- ne jamais relâcher ici, in_nd en devient propriétaire
                else
                    # PAS d'adoption : `.+=` copie le CONTENU dans le buffer déjà
                    # possédé par in_nd, `grads[i]` lui-même devient orphelin --
                    # c'est exactement la fuite fan-out trouvée à l'audit sur la
                    # première ébauche (comptage par objectid) : ici on la
                    # referme en le rendant explicitement au pool.
                    in_nd.gradient .+= grads[i]
                    if gpool !== nothing && grad_owns(gpool, grads[i])
                        grad_release!(gpool, grads[i])
                    end
                end
            end
            in_nd.backwarded = true
            # NB : pour un nœud partagé par plusieurs consommateurs (poids
            # liés, branches multiples), ce log peut être émis plus d'une
            # fois avant que le gradient ne soit définitivement complet —
            # acceptable pour un petit MLP séquentiel (une seule contribution
            # par nœud), à revisiter si on cible des graphes avec partage.
            if log !== nothing && in_nd.gradient !== nothing
                log_event!(log, in_sym, "backward", "finished", format_tensor_grid(in_nd.gradient))
            end
        end

        # Libération du gradient de sortie s'il n'est pas un paramètre.
        # `dy_val` (= l'ancien nd_out.gradient) n'est rendu au pool QUE s'il
        # n'a été adopté par AUCUNE entrée -- seul :identity retourne `dy`
        # inchangé comme un de ses grads (:add a été corrigé pour toujours
        # copier), donc cette adoption reste un cas rare mais réel à
        # détecter, pas à supposer absent.
        nd_out.backwarded = true
        if !nd_out.is_param
            if gpool !== nothing && dy_val !== nothing && grad_owns(gpool, dy_val)
                adopted = any(s -> g.nodes[ns][s].gradient === dy_val, rule.inputs)
                adopted || grad_release!(gpool, dy_val)
            end
            nd_out.gradient = nothing
        end

        # Libération de la VALEUR forward de Y=out_sym (BACKWARD_RELEASE_VALUES,
        # voir sa docstring pour la preuve de sûreté) : au moment où cette
        # itération se termine, ni `nd_out.value` ni les valeurs des entrées
        # qu'on vient de lire ne seront jamais relus par une itération
        # ultérieure de CETTE passe -- Y et ses entrées sont topologiquement
        # antérieurs à tout ce qui reste à visiter dans la boucle inverse, et
        # le repli de `_ctx_for_backward!` pour CETTE itération est déjà passé.
        # Les entrées elles-mêmes ne sont PAS libérées ici : elles seront
        # libérées à LEUR PROPRE visite (si elles ont une règle) ou par le
        # balayage final ci-dessous (si elles sont sautées -- prune_frozen,
        # op sans GRAD_RULES en mode sparse, etc.).
        release_values && _release_forward_value!(g, nd_out, ns, loss_sym)
    end

    # Nettoyage final : une feuille non entraînable (ex. l'entrée :x ou les
    # labels :y) n'est jamais visitée comme nd_out ci-dessus puisqu'elle n'a
    # pas de règle -- sans ce passage, son gradient temporaire ne serait
    # jamais libéré. Redondant mais inoffensif pour les nœuds déjà nettoyés
    # plus haut. Si son gradient est issu du pool (adopté depuis un
    # consommateur mais jamais revisité comme nd_out, puisque les feuilles
    # n'ont pas de règle), le rendre ici plutôt que de juste larguer la
    # référence -- c'est le second site de fuite trouvé à l'audit.
    for (_, nd) in g.nodes[ns]
        if !nd.is_param
            if gpool !== nothing && nd.gradient !== nothing && grad_owns(gpool, nd.gradient)
                grad_release!(gpool, nd.gradient)
            end
            nd.gradient = nothing
        end
    end
    @assert gpool === nothing || isempty(gpool.lent) "GradPool : $(length(gpool.lent)) tampon(s) encore prêté(s) en fin de passe -- fuite de propriété"

    if release_values
        # Balayage final (réutilise demand_release.jl) : rattrape tout nœud
        # à règle jamais visité comme out_sym ci-dessus (nd_out.gradient
        # resté nothing -- hors du chemin du gradient -- ou sauté par
        # prune_frozen) et donc jamais libéré par _release_forward_value!.
        release_intermediates!(g; keep=Symbol[loss_sym], namespace=ns)
        # Vide les listes libres persistantes du GradPool : sans ça, les
        # tampons de gradient/scratch restent résidents d'un pas à l'autre
        # (c'est tout l'intérêt du pool pour LE CALCUL des gradients pendant
        # UNE passe) et annulent une bonne part du gain de résidence qu'on
        # vient de gagner sur les activations -- les deux volets ne sont
        # rentables qu'ENSEMBLE (voir docstring de BACKWARD_RELEASE_VALUES).
        # Le prochain pas les réacquiert depuis le cache du pool CUDA, pas
        # depuis zéro (mêmes formes exactement).
        gpool !== nothing && empty_grad_pool!(gpool)
    end
    return g
end