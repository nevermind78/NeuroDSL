# ════════════════════════════════════════════════════════════════════════════════
# NeuroDSL — flash_attention.jl
# Attention tuilée (CPU) + fallback simple (GPU)
# ════════════════════════════════════════════════════════════════════════════════

const FLASH_BLOCK = 64

# ----- Forward CPU tuilé --------------------------------------------------------
function flash_attn_fwd_cpu!(out::AbstractMatrix{Float32},
                              Q::AbstractMatrix{Float32},
                              K::AbstractMatrix{Float32},
                              V::AbstractMatrix{Float32},
                              d_head::Int;
                              causal::Bool = true,
                              block_size::Int = FLASH_BLOCK)
    N = size(Q, 1)
    scale = 1f0 / sqrt(Float32(d_head))
    l = fill(-Inf32, N)
    m = fill(-Inf32, N)
    fill!(out, 0f0)

    for j_start in 1:block_size:N
        j_end = min(j_start + block_size - 1, N)
        Kj = K[j_start:j_end, :]
        Vj = V[j_start:j_end, :]
        S = (Q * Kj') .* scale
        if causal
            for qi in 1:N, ki in 1:(j_end - j_start + 1)
                kj_abs = j_start + ki - 1
                if kj_abs > qi
                    S[qi, ki] = -Inf32
                end
            end
        end
        m_new = max.(m, dropdims(maximum(S; dims=2); dims=2))
        exp_diff = exp.(m .- m_new)
        out  .*= exp_diff
        l     .= exp.(l .- m_new) .* (l .!= -Inf32)
        S_shifted = exp.(S .- m_new)
        out .+= S_shifted * Vj
        l   .+= dropdims(sum(S_shifted; dims=2); dims=2)
        m .= m_new
    end
    out ./= max.(l, 1f-38)
    return out, l, m
end

# ----- Forward CPU simple (référence O(N²)) ------------------------------------
function flash_attn_fwd_cpu_simple!(out::AbstractMatrix{Float32},
                                     Q::AbstractMatrix{Float32},
                                     K::AbstractMatrix{Float32},
                                     V::AbstractMatrix{Float32},
                                     d_head::Int;
                                     causal::Bool = true)
    N, D = size(Q)
    scale = 1f0 / sqrt(Float32(d_head))
    S = (Q * K') .* scale
    if causal
        for i in 1:N, j in (i+1):N
            S[i, j] = -Inf32
        end
    end
    m_vec = dropdims(maximum(S; dims=2); dims=2)
    S_shifted = exp.(S .- m_vec)
    l_vec = dropdims(sum(S_shifted; dims=2); dims=2)
    P = S_shifted ./ max.(l_vec, 1f-38)
    out .= P * V
    return out, log.(l_vec) .+ m_vec, m_vec
end

# ----- Backward CPU ------------------------------------------------------------
function flash_attn_bwd_cpu!(dQ::AbstractMatrix{Float32},
                              dK::AbstractMatrix{Float32},
                              dV::AbstractMatrix{Float32},
                              dout::AbstractMatrix{Float32},
                              Q::AbstractMatrix{Float32},
                              K::AbstractMatrix{Float32},
                              V::AbstractMatrix{Float32},
                              out::AbstractMatrix{Float32},
                              l::AbstractVector{Float32},
                              m::AbstractVector{Float32},
                              d_head::Int;
                              causal::Bool = true)
    N, D = size(Q)
    scale = 1f0 / sqrt(Float32(d_head))
    # Recompute P
    S = (Q * K') .* scale
    if causal
        for i in 1:N, j in (i+1):N
            S[i, j] = -Inf32
        end
    end
    m_vec = dropdims(maximum(S; dims=2); dims=2)
    S_shifted = exp.(S .- m_vec)
    l_vec = dropdims(sum(S_shifted; dims=2); dims=2)
    P = S_shifted ./ max.(l_vec, 1f-38)

    dV .= P' * dout
    D_vec = dropdims(sum(dout .* out; dims=2); dims=2)
    dP = dout * V'
    dS = P .* (dP .- D_vec) .* scale
    if causal
        for i in 1:N, j in (i+1):N
            dS[i, j] = 0f0
        end
    end
    dQ .= dS * K
    dK .= dS' * Q
    return nothing
end

# ----- Dispatch CPU ------------------------------------------------------------
function flash_attn_fwd!(::Backend.CPUDevice, out, Q, K, V, d_head; causal=true)
    flash_attn_fwd_cpu_simple!(out, Q, K, V, d_head; causal=causal)
end
function flash_attn_fwd!(::Backend.CUDADevice, out, Q, K, V, d_head; causal=true)
    N     = size(Q, 1)
    scale = 1f0 / sqrt(Float32(d_head))

    if causal
        S = (Q * K') .* scale .+ causal_mask(Backend.CUDADevice(), N)
    else
        S = (Q * K') .* scale
    end

    P = similar(S)
    softmax_fwd!(Backend.CUDADevice(), P, S)
    out .= P * V

    # Sentinelles pour l et m (le backward les recompute)
    l_dummy = CUDA.zeros(Float32, N)
    m_dummy = CUDA.fill(-Inf32, N)
    return out, l_dummy, m_dummy
end

function flash_attn_bwd!(::Backend.CPUDevice, dQ, dK, dV, dout, Q, K, V, out, l, m, d_head; causal=true)
    flash_attn_bwd_cpu!(dQ, dK, dV, dout, Q, K, V, out, l, m, d_head; causal=causal)
end
function flash_attn_bwd!(::Backend.CUDADevice, dQ, dK, dV, dout,
                          Q, K, V, out, l, m, d_head; causal=true)
    N, _  = size(Q)
    scale = 1f0 / sqrt(Float32(d_head))

    # Recompute P (identique au forward)
    if causal
        S = (Q * K') .* scale .+ causal_mask(Backend.CUDADevice(), N)
    else
        S = (Q * K') .* scale
    end
    P = similar(S)
    softmax_fwd!(Backend.CUDADevice(), P, S)

    dV    .= P' * dout
    D_vec  = sum(dout .* out; dims=2)
    dP     = dout * V'
    dS     = P .* (dP .- D_vec) .* scale

    if causal
        # Masque triangulaire inférieur : 1 où j ≤ i, 0 ailleurs
        mask = causal_mask(Backend.CUDADevice(), N)
        tril = Float32.(mask .> -Inf32)   # 1.0f0 pour les positions autorisées
        dS .*= tril
    end

    dQ .= dS * K
    dK .= dS' * Q
    return nothing
end



# ----- Intégration dans le système de règles (doit être après GRAD_RULES) -----
# L'enregistrement de l'op se fait via register_op! dans dispatch.jl
# On définit ici la fonction d'interface qui sera appelée par _dispatch_op
function _dispatch_flash_attn!(dev, output_buffer, inputs, attrs, out_sym, out_node, ctx_store)
    Q, K, V = inputs[1], inputs[2], inputs[3]
    d_head  = get(attrs, :d_head, size(Q, 2))
    causal  = get(attrs, :causal, true)

    N = size(Q, 1)
    if !haskey(out_node.aux_data, :flash_l) || length(out_node.aux_data[:flash_l]) != N
        out_node.aux_data[:flash_l] = Backend.zeros32(dev, N)
        out_node.aux_data[:flash_m] = fill(-Inf32, N)
        dev isa Backend.CUDADevice && (out_node.aux_data[:flash_m] = CUDA.fill(-Inf32, N))
    end
    l_buf = out_node.aux_data[:flash_l]
    m_buf = out_node.aux_data[:flash_m]

    _, l_new, m_new = flash_attn_fwd!(dev, output_buffer, Q, K, V, d_head; causal=causal)
    l_buf .= l_new isa AbstractArray ? l_new : Backend.to_device(dev, l_new)
    m_buf .= m_new isa AbstractArray ? m_new : Backend.to_device(dev, m_new)

    if ctx_store !== nothing
        _store_ctx!(ctx_store, out_sym, Dict{Symbol,Any}(
            :Q => Q, :K => K, :V => V,
            :out => output_buffer, :l => l_buf, :m => m_buf,
            :d_head => d_head, :causal => causal))
    end
    return output_buffer
end

# Enregistrement de l'op (ne doit être fait qu'une fois)
const _FLASH_ATTN_OP_REGISTERED = Ref(false)
function _register_flash_attn_in_dispatch!()
    _FLASH_ATTN_OP_REGISTERED[] && return
    register_op!(:flash_attn, _dispatch_flash_attn!)
    _FLASH_ATTN_OP_REGISTERED[] = true
end

# Règle backward (dépend de GRAD_RULES, qui doit être défini dans backward.jl)
# Cette ligne doit être exécutée APRÈS que GRAD_RULES soit disponible.
# Pour éviter les erreurs de précompilation, on utilise un bloc `if isdefined`
if isdefined(@__MODULE__, :GRAD_RULES)
    GRAD_RULES[:flash_attn] = (dev, dy, ctx, inputs) -> begin
        Q, K, V = inputs[1], inputs[2], inputs[3]
        out = ctx[:out]; l = ctx[:l]; m = ctx[:m]
        d_head = ctx[:d_head]; causal = ctx[:causal]
        dQ = similar(Q); dK = similar(K); dV = similar(V)
        flash_attn_bwd!(dev, dQ, dK, dV, dy, Q, K, V, out, l, m, d_head; causal=causal)
        (dQ, dK, dV)
    end
end

# ----- Couche MultiHeadFlashAttention ------------------------------------------
struct MultiHeadFlashAttention
    dim     :: Int
    n_heads :: Int
    d_head  :: Int
end
MultiHeadFlashAttention(dim, n_heads) = MultiHeadFlashAttention(dim, n_heads, dim ÷ n_heads)

function (m::MultiHeadFlashAttention)(g::NeuroGraph, x_sym::Symbol, prefix::Symbol;
                                       namespace = g.active_ns)
    _register_flash_attn_in_dispatch!()
    q_full = Linear(m.dim, m.dim, bias=false)(g, x_sym, Symbol(prefix, :_q); namespace=namespace)
    k_full = Linear(m.dim, m.dim, bias=false)(g, x_sym, Symbol(prefix, :_k); namespace=namespace)
    v_full = Linear(m.dim, m.dim, bias=false)(g, x_sym, Symbol(prefix, :_v); namespace=namespace)

    head_outputs = Symbol[]
    for h in 1:m.n_heads
        s = (h-1)*m.d_head + 1
        e =  h   *m.d_head
        qh = Symbol(prefix, :_q_h, h)
        kh = Symbol(prefix, :_k_h, h)
        vh = Symbol(prefix, :_v_h, h)
        addrule!(g, GraphRule(qh, [q_full], :slice_cols; attrs=Dict(:start_col=>s,:end_col=>e), namespace=namespace))
        addrule!(g, GraphRule(kh, [k_full], :slice_cols; attrs=Dict(:start_col=>s,:end_col=>e), namespace=namespace))
        addrule!(g, GraphRule(vh, [v_full], :slice_cols; attrs=Dict(:start_col=>s,:end_col=>e), namespace=namespace))
        ao_h = Symbol(prefix, :_ao_h, h)
        addrule!(g, GraphRule(ao_h, [qh, kh, vh], :flash_attn; attrs=Dict(:d_head=>m.d_head, :causal=>true), namespace=namespace))
        push!(head_outputs, ao_h)
    end
    concat_sym = Symbol(prefix, :_concat)
    addrule!(g, GraphRule(concat_sym, head_outputs, :hcat_heads; namespace=namespace))
    return Linear(m.dim, m.dim, bias=false)(g, concat_sym, Symbol(prefix, :_output); namespace=namespace)
end