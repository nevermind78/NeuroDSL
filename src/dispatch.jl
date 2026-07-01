
const DEBUG_MODE = Ref(true)
debug!(v::Bool) = (DEBUG_MODE[] = v)

function _check_matmul(name, A, B, trans_b)
    DEBUG_MODE[] || return
    kA=size(A,2); kB=trans_b ? size(B,2) : size(B,1)
    kA==kB || error("❌ [$name] matmul incompatible A=$(size(A)) B=$(size(B)) trans_b=$trans_b")
end

# Buffer pool pour les tenseurs temporaires
const _BUFFER_POOL = Dict{Tuple,Any}()
function _get_buffer(dev, shape)
    if dev isa Backend.CUDADevice
        return Backend.zeros32(dev, shape...)
    end
    key = (:cpu, shape)
    if haskey(_BUFFER_POOL, key)
        return _BUFFER_POOL[key]
    end
    buf = Backend.zeros32(dev, shape...)
    if length(_BUFFER_POOL) > 20
        empty!(_BUFFER_POOL)
    end
    _BUFFER_POOL[key] = buf
    return buf
end

_store_ctx!(::Nothing, ::Symbol, ::Dict) = nothing
function _store_ctx!(ctx::CtxStore, sym::Symbol, d::Dict)
    ctx[sym] = d
    return nothing
end

# ── Shape inference ────────────────────────────────────────────────────
function _infer_output_shape(op::Symbol, inputs, attrs)
    if op == :matmul
        A, B = inputs[1], inputs[2]
        tb = get(attrs, :trans_b, false)
        return (size(A, 1), tb ? size(B, 1) : size(B, 2))
    elseif op == :fused_matmul_relu
        A, B = inputs[1], inputs[2]
        tb = get(attrs, :trans_b, false)
        return (size(A, 1), tb ? size(B, 1) : size(B, 2))
    elseif op == :fused_relu_matmul
        A, B = inputs[1], inputs[2]  # Premier input a déjà subi relu, second = matrice
        # Ici on suppose que le relu ne change pas la forme, donc shape = matmul
        tb = get(attrs, :trans_b, false)
        return (size(A, 1), tb ? size(B, 1) : size(B, 2))
    elseif op == :randvec
        return (attrs[:vec_size],)
    elseif op == :linear
        X, W = inputs[1], inputs[2]
        return (size(X, 1), size(W, 1))
    elseif op == :hcat_heads
        return (size(inputs[1], 1), sum(size(x, 2) for x in inputs))
    elseif op in (:add, :mul, :rmsnorm, :swiglu, :softmax,:scale_mask, :rope, :dropout, :relu, :wsum, :nsum, :tanh, :identity,:scale_add,:linear2)
        return size(inputs[1])
    elseif op == :embedding
        E, idx = inputs[1], inputs[2]
        return (length(idx), size(E, 2))
    elseif op in (:mse_loss, :sum_matrix, :cce)
        return (1,)
    elseif op == :slice_cols
        return (size(inputs[1], 1), attrs[:end_col] - attrs[:start_col] + 1)
    elseif op == :cross_entropy
        return (1,)
    elseif op == :flash_attn
        return size(inputs[1])
    elseif op == :fused_add_relu
        return size(inputs[1])
    elseif op == :fused_matmul_add
        A, B = inputs[1], inputs[2]
        tb = get(attrs, :trans_b, false)
        return (size(A, 1), tb ? size(B, 1) : size(B, 2))
    elseif op == :fused_matmul_add_relu
        A, B = inputs[1], inputs[2]
        tb = get(attrs, :trans_b, false)
        return (size(A, 1), tb ? size(B, 1) : size(B, 2))
    elseif op == :fused_qkv_projection
        A, B = inputs[1], inputs[2]
        tb = get(attrs, :trans_b, false)
        return (size(A, 1), tb ? size(B, 1) : size(B, 2))
    elseif op == :fused_swiglu
        return size(inputs[1])
    elseif op == :fused_sdpa
        return size(inputs[1])
    elseif op in (:slice_cols, :slice_view)
        A = inputs[1]
        s = get(attrs, :start_col, 1)
        e = get(attrs, :end_col, size(A, 2))
        return (size(A, 1), e - s + 1)
    else
        @warn "Shape inference non implémentée pour :$op, utilisation de la forme du premier argument"
        return size(inputs[1])
    end
end

_infer_output_type(::Symbol, ::Any, ::Any) = Float32

function _run_kernel!(dev, output_buffer::T, rule, inputs_vals, out_sym, out_node, ctx_store) where {T <: AbstractArray}
    _dispatch_op(dev, output_buffer, rule.op, inputs_vals, rule.attrs, out_sym, out_node, ctx_store)
end

# ── execute_rule! avec buffer pool ─────────────────────────────────────
function execute_rule!(g::NeuroGraph, rule::GraphRule;
                       ctx_store::Union{CtxStore,Nothing}=nothing, 
                       namespace=g.active_ns,
                       log::Union{Nothing, ExecutionLog}=nothing) # <--- AJOUT
    dev = g.device
    ns = rule.namespace
    out_sym = rule.output
    out_node = g.nodes[ns][out_sym]

    # LOG : Début du calcul
    if log !== nothing
        log_event!(log, out_sym, "forward", "starting")
    end

    n = length(rule.inputs)
    if !haskey(out_node.aux_data, :_inputs_buf) || length(out_node.aux_data[:_inputs_buf]) != n
        out_node.aux_data[:_inputs_buf] = Vector{AbstractArray{Float32}}(undef, n)
    end
    inputs_vals = out_node.aux_data[:_inputs_buf]::Vector{AbstractArray{Float32}}
    for (i, s) in enumerate(rule.inputs)
        inputs_vals[i] = g.nodes[ns][s].value::AbstractArray{Float32}
    end

    out_shape = _infer_output_shape(rule.op, inputs_vals, rule.attrs)
    out_type  = _infer_output_type(rule.op, inputs_vals, rule.attrs)

    if out_node.value === nothing || size(out_node.value) != out_shape || eltype(out_node.value) != out_type
        out_node.value = Backend.zeros32(dev, out_shape...)
    end
    output_buffer = out_node.value

    _run_kernel!(dev, output_buffer, rule, inputs_vals, out_sym, out_node, ctx_store)
    out_node.valid = true

    # LOG : Fin du calcul avec résumé de la valeur
        # LOG : Fin du calcul avec résumé de la valeur
    if log !== nothing
        val_summary = try
            val_flat = vec(Array(out_node.value))
            join([@sprintf("%.4f", Float64(x)) for x in val_flat[1:min(4, length(val_flat))]], ", ")
        catch
            "error"
        end
        log_event!(log, out_sym, "forward", "finished", val_summary)
    end
    return out_node.value
end

# ── execute_rule_pooled! : variante avec BufferPool (utilisée par CompiledPlan) ──
#
# Parallèle à execute_rule! ci-dessus, qui reste totalement inchangée — demand! et
# backward_graph! continuent de l'appeler directement et ne sont donc jamais affectés
# par cette fonction ni par un bug éventuel dans son chemin. La seule différence : le
# buffer de sortie est emprunté/rendu à un BufferPool plutôt qu'alloué frais à chaque
# appel via Backend.zeros32 — c'est ce qui élimine réellement les allocations
# intermédiaires répétées pendant l'exécution d'un CompiledPlan. Les paramètres
# (is_param) ne sont jamais empruntés au pool : ils gardent un stockage stable.
function execute_rule_pooled!(g::NeuroGraph, rule::GraphRule, pool;
                              ctx_store::Union{CtxStore,Nothing}=nothing,
                              log::Union{Nothing, ExecutionLog}=nothing)
    dev = g.device
    ns = rule.namespace
    out_sym = rule.output
    out_node = g.nodes[ns][out_sym]

    if log !== nothing
        log_event!(log, out_sym, "forward", "starting")
    end

    n = length(rule.inputs)
    if !haskey(out_node.aux_data, :_inputs_buf) || length(out_node.aux_data[:_inputs_buf]) != n
        out_node.aux_data[:_inputs_buf] = Vector{AbstractArray{Float32}}(undef, n)
    end
    inputs_vals = out_node.aux_data[:_inputs_buf]::Vector{AbstractArray{Float32}}
    for (i, s) in enumerate(rule.inputs)
        inputs_vals[i] = g.nodes[ns][s].value::AbstractArray{Float32}
    end

    out_shape = _infer_output_shape(rule.op, inputs_vals, rule.attrs)
    out_type  = _infer_output_type(rule.op, inputs_vals, rule.attrs)
    is_poolable = !out_node.is_param

    if out_node.value === nothing || size(out_node.value) != out_shape || eltype(out_node.value) != out_type
        if is_poolable && out_node.value !== nothing
            release!(pool, out_node.value)
        end
        out_node.value = is_poolable ? acquire!(pool, out_shape) : Backend.zeros32(dev, out_shape...)
    end
    output_buffer = out_node.value

    _run_kernel!(dev, output_buffer, rule, inputs_vals, out_sym, out_node, ctx_store)
    out_node.valid = true

    if log !== nothing
        val_summary = try
            val_flat = vec(Array(out_node.value))
            join([@sprintf("%.4f", Float64(x)) for x in val_flat[1:min(4, length(val_flat))]], ", ")
        catch
            "error"
        end
        log_event!(log, out_sym, "forward", "finished", val_summary)
    end
    return out_node.value
end

# ── _dispatch_op ───────────────────────────────────────────────────────
function _dispatch_op(dev, output_buffer, op::Symbol, inputs, attrs, out_sym, out_node::GraphNode, ctx_store)
    if op == :matmul
        A, B = inputs[1], inputs[2]
        tb = get(attrs, :trans_b, false)
        _check_matmul(out_sym, A, B, tb)
        # Sauvegarder A AVANT tout calcul (le buffer pool peut écraser A)
        A_ctx = ctx_store !== nothing ? copy(A) : A

        if output_buffer === A || output_buffer === B
            if !haskey(out_node.aux_data, :_alias_buf) || size(out_node.aux_data[:_alias_buf]) != size(output_buffer)
                out_node.aux_data[:_alias_buf] = similar(output_buffer)
            end
            tmp_buf = out_node.aux_data[:_alias_buf]
            if tb
                LinearAlgebra.mul!(tmp_buf, A, B')
            else
                LinearAlgebra.mul!(tmp_buf, A, B)
            end
            output_buffer .= tmp_buf
        else
            if tb
                LinearAlgebra.mul!(output_buffer, A, B')
            else
                LinearAlgebra.mul!(output_buffer, A, B)
            end
        end

        if ctx_store !== nothing
            # Utiliser A_ctx (copie faite AVANT l'écrasement du buffer)
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(:trans_b => tb, :A => A_ctx, :B => B))
        end
        return output_buffer
    elseif op == :fused_matmul_relu
        A, B = inputs[1], inputs[2]
        tb = get(attrs, :trans_b, false)
        M, K_A = size(A)
        K_B, N_B = size(B)
        
        N = tb ? K_B : N_B 
        K = tb ? N_B : K_B

        if dev isa Backend.CUDADevice
            threads_x, threads_y = 16, 16
            blocks_x, blocks_y = cld(M, threads_x), cld(N, threads_y)
            @cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) _fused_matmul_relu_kernel!(
                output_buffer, A, B, M, N, K, tb
            )
        else
            temp = similar(output_buffer)
            if tb
                LinearAlgebra.mul!(temp, A, B')
            else
                LinearAlgebra.mul!(temp, A, B)
            end
            output_buffer .= max.(temp, 0f0)
        end

        # 🚀 THIS IS THE MISSING PART 🚀
        # We MUST save the output and metadata for the backward pass to work
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(
                :trans_b => tb, 
                :A => A, 
                :B => B, 
                :output => output_buffer  # This is the key that was missing!
            ))
        end
        
        return output_buffer
    elseif op == :fused_matmul_add
        A, B, bias = inputs[1], inputs[2], inputs[3]
        tb = get(attrs, :trans_b, false)
        A_ctx = ctx_store !== nothing ? copy(A) : A

        if tb
            LinearAlgebra.mul!(output_buffer, A, B')
        else
            LinearAlgebra.mul!(output_buffer, A, B)
        end
        output_buffer .+= reshape(vec(bias), 1, :)

        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(:trans_b => tb, :A => A_ctx, :B => B))
        end
        return output_buffer
    elseif op == :fused_matmul_add_relu
        A, B, bias = inputs[1], inputs[2], inputs[3]
        tb = get(attrs, :trans_b, false)
        A_ctx = ctx_store !== nothing ? copy(A) : A
        M, K_A = size(A)
        K_B, N_B = size(B)

        N = tb ? K_B : N_B
        K = tb ? N_B : K_B

        if dev isa Backend.CUDADevice
            bias_vec = vec(bias)
            threads_x, threads_y = 16, 16
            blocks_x, blocks_y = cld(M, threads_x), cld(N, threads_y)
            @cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) _fused_matmul_add_relu_kernel!(
                output_buffer, A, B, bias_vec, M, N, K, tb
            )
        else
            temp = similar(output_buffer)
            if tb
                LinearAlgebra.mul!(temp, A, B')
            else
                LinearAlgebra.mul!(temp, A, B)
            end
            temp .+= reshape(vec(bias), 1, :)
            output_buffer .= max.(temp, 0f0)
        end

        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(
                :trans_b => tb,
                :A => A_ctx,
                :B => B,
                :output => output_buffer
            ))
        end
        return output_buffer
    elseif op == :fused_qkv_projection
        A, B = inputs[1], inputs[2]
        tb = get(attrs, :trans_b, false)
        A_ctx = ctx_store !== nothing ? copy(A) : A

        if tb
            LinearAlgebra.mul!(output_buffer, A, B')
        else
            LinearAlgebra.mul!(output_buffer, A, B)
        end

        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(:trans_b => tb, :A => A_ctx, :B => B))
        end
        return output_buffer
    elseif op == :hcat_heads
        offset = 0
        for inp in inputs
            inp_arr = inp::AbstractArray{Float32}
            d = size(inp_arr, 2)
            output_buffer[:, offset+1:offset+d] .= inp_arr
            offset += d
        end
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(
                :n_heads => length(inputs), :d_head => size(inputs[1], 2)))
        end
        return output_buffer

    elseif op == :linear
        X, W = inputs[1], inputs[2]

        if output_buffer === X || output_buffer === W
            if !haskey(out_node.aux_data, :_alias_buf) || size(out_node.aux_data[:_alias_buf]) != size(output_buffer)
                out_node.aux_data[:_alias_buf] = similar(output_buffer)
            end
            tmp_buf = out_node.aux_data[:_alias_buf]
            LinearAlgebra.mul!(tmp_buf, X, W')
            output_buffer .= tmp_buf
        else
            LinearAlgebra.mul!(output_buffer, X, W')
        end

        if length(inputs) == 3
            output_buffer .+= inputs[3]'
        end
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(
                :X => X, :W => W, :n_inputs => length(inputs)))
        end
        return output_buffer

    elseif op == :add
        a = inputs[1]
        b = inputs[2]
        # Handle common bias add patterns safely (matrix + vector)
        try
            if ndims(a) == 2 && ndims(b) == 1
                # b may be (n_cols,) or (n_rows,) — prefer broadcasting on cols when lengths match
                if size(b, 1) == size(a, 2)
                    output_buffer .= a .+ reshape(b, 1, :)
                elseif size(b, 1) == size(a, 1)
                    output_buffer .= a .+ reshape(b, :, 1)
                else
                    output_buffer .= a .+ b
                end
            elseif ndims(a) == 1 && ndims(b) == 2
                # reverse order
                if size(a, 1) == size(b, 2)
                    output_buffer .= reshape(a, 1, :) .+ b
                elseif size(a, 1) == size(b, 1)
                    output_buffer .= reshape(a, :, 1) .+ b
                else
                    output_buffer .= a .+ b
                end
            else
                output_buffer .= a .+ b
            end
        catch err
            # Fallback to safe broadcasting using temporary reshape on CPU if needed
            @warn "add op broadcast failed, falling back to safe broadcast: $(err)"
            output_buffer .= a .+ b
        end
        return output_buffer

    elseif op == :mul
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(:a => inputs[1], :b => inputs[2]))
        end
        output_buffer .= inputs[1] .* inputs[2]
        return output_buffer

    elseif op == :rmsnorm
        x, gamma = inputs[1], inputs[2]
        nr, nc = size(x)
        if !haskey(out_node.aux_data, :rms_inv) || size(out_node.aux_data[:rms_inv]) != (nr,)
            out_node.aux_data[:rms_inv] = Backend.zeros32(dev, nr)
        end
        rms_inv = out_node.aux_data[:rms_inv]
        rmsnorm_fwd!(dev, output_buffer, rms_inv, x, gamma)
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(
                :x => x, :gamma => gamma, :rms_inv => rms_inv))
        end
        return output_buffer

    elseif op == :swiglu
        gate, up = inputs[1], inputs[2]
        swiglu_fwd!(dev, output_buffer, gate, up)
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(:gate => gate, :up => up))
        end
        return output_buffer

    elseif op == :relu
        x = inputs[1]
        output_buffer .= max.(x, 0f0)
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(:x => x))
        end
        return output_buffer

    elseif op == :softmax
        x = inputs[1]
        softmax_fwd!(dev, output_buffer, x)
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(:output => output_buffer))
        end
        return output_buffer

    elseif op == :scale_mask
        scores = inputs[1]
        d_head = get(attrs, :d_head, size(scores, 2))
        seqlen = size(scores, 1)
        mask = causal_mask(dev, seqlen)
        scale_mask_fwd!(output_buffer, scores, d_head, mask)
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(
                :scale => 1f0/sqrt(Float32(d_head))))
        end
        return output_buffer

    elseif op == :embedding
        E, idx_raw = inputs[1], inputs[2]
        idx_cpu = Int.(vec(Array(idx_raw)))
        n_batch = length(idx_cpu)
        d_emb = size(E, 2)
        E_cpu = Array(E)
        out_cpu = Array(output_buffer)
        for (i, row) in enumerate(idx_cpu)
            out_cpu[i, :] .= E_cpu[row, :]
        end
        output_buffer .= Backend.to_device(dev, out_cpu)
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(
                :idx => idx_cpu, :E_size => size(E)))
        end
        return output_buffer
    elseif op == :rope
        x = inputs[1]::AbstractArray{Float32}
        seqlen, d = size(x)
        half = d ÷ 2
        if !haskey(out_node.aux_data, :cos_a) || size(out_node.aux_data[:cos_a], 1) != seqlen
            pos   = Backend.to_device(dev, Float32.(0:seqlen-1))
            theta = Backend.to_device(dev, Float32.(1f0 ./ (10000f0 .^ ((0:half-1) ./ half))))
            angles = reshape(pos, :, 1) * reshape(theta, 1, :)
            out_node.aux_data[:cos_a] = cos.(angles)
            out_node.aux_data[:sin_a] = sin.(angles)
        end
        cos_a = out_node.aux_data[:cos_a]::AbstractArray{Float32}
        sin_a = out_node.aux_data[:sin_a]::AbstractArray{Float32}
        output_buffer[:, 1:half]      .= x[:, 1:half] .* cos_a .- x[:, half+1:end] .* sin_a
        output_buffer[:, half+1:end]  .= x[:, 1:half] .* sin_a .+ x[:, half+1:end] .* cos_a
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(
                :cos_a => cos_a, :sin_a => sin_a, :half => half))
        end
        return output_buffer

    elseif op == :mse_loss
        out, target = inputs[1], inputs[2]
        fill!(output_buffer, mse_loss_fwd(out, target)[1])
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(:out => out, :target => target))
        end
        return output_buffer

    elseif op == :sum_matrix
        x = inputs[1]
        fill!(output_buffer, sum_matrix_fwd(x)[1])
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(:x_val => x))
        end
        return output_buffer

    elseif op == :dropout
        x = inputs[1]
        rate = Float32(get(attrs, :rate, 0.1))
        training = get(attrs, :training, true)
        mask = Backend.rand32(dev, size(x)...) .> rate
        if training
            output_buffer .= x .* mask ./ (1f0 - rate)
        else
            output_buffer .= x
        end
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(:mask => mask, :rate => rate))
        end
        return output_buffer

    elseif op == :slice_cols
        x = inputs[1]
        s = get(attrs, :start_col, 1)
        e = get(attrs, :end_col, size(x, 2))
        output_buffer .= x[:, s:e]
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(
                :start_col => s, :end_col => e))
        end
        return output_buffer
    elseif op == :cross_entropy
        logits, labels_raw = inputs[1], inputs[2]
        labels = Int.(vec(labels_raw))
        loss = cross_entropy_loss(logits, labels)
        # S'assurer que la loss est dans un tenseur (1,1)
        if size(output_buffer) == (1,)
            output_buffer[1] = loss
        else
            output_buffer[1, 1] = loss
        end
        if ctx_store !== nothing
            _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(
                :logits => logits, 
                :labels => labels))
        end
        return output_buffer

    elseif haskey(CUSTOM_OPS, op)
        CUSTOM_OPS[op](dev, output_buffer, inputs, attrs, out_sym, out_node, ctx_store)
        return output_buffer

    elseif op == :identity
        # Élimination algébrique pure (double_transpose_elim, add_zero_elim) : recopie
        # simple, sans dépendre d'un @defop custom enregistré par l'utilisateur (comme
        # le faisait chaque cellule du notebook jusqu'ici).
        output_buffer .= inputs[1]
        return output_buffer

    else
        error("❌ Opérateur inconnu : :$op. Utilisez register_op! pour enregistrer un op custom.")
    end
end

const CUSTOM_OPS = Dict{Symbol,Function}()
register_op!(name::Symbol, fn::Function) = (CUSTOM_OPS[name] = fn; println("✅ Op :$name registered"))

_unwrap_value(v::T) where {T} = v

function demand!(g::NeuroGraph, name::Symbol;
                 ctx_store::Union{CtxStore,Nothing}=nothing, 
                 namespace=g.active_ns,
                 log::Union{Nothing, ExecutionLog}=nothing) # <--- AJOUT
    ns = namespace
    haskey(g.nodes, ns) && haskey(g.nodes[ns], name) ||
        error("❌ :$name introuvable dans :$ns")

    nd = g.nodes[ns][name]
    nd.valid && nd.value !== nothing && return _unwrap_value(nd.value)

    order = topo_order!(g; namespace=ns)
    for sym in order
        nd_i = g.nodes[ns][sym]
        nd_i.valid && nd_i.value !== nothing && continue
        haskey(g.rules[ns], sym) || continue
        
        # On passe le log à execute_rule!
        execute_rule!(g, g.rules[ns][sym]; ctx_store=ctx_store, namespace=ns, log=log)
        sym == name && break
    end

    return _unwrap_value(g.nodes[ns][name].value)
end
