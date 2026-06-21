if Backend.CUDA_AVAILABLE; using CUDA; end

# ── Cache pour le masque causal (LRU avec taille max) ─────────────────
const _MASK_CACHE = Dict{Tuple{Symbol,Int},Any}()
const _MASK_CACHE_MAXSIZE = 10

function causal_mask_cached(device, seqlen::Int)
    key = (device isa Backend.CUDADevice ? :cuda : :cpu, seqlen)
    if haskey(_MASK_CACHE, key)
        return _MASK_CACHE[key]
    end
    mask = Float32[j<=i ? 0f0 : -Inf32 for i in 1:seqlen, j in 1:seqlen]
    if device isa Backend.CUDADevice
        mask = CUDA.cu(mask)
    end
    # LRU simple : suppression du plus ancien si le cache est trop grand
    if length(_MASK_CACHE) >= _MASK_CACHE_MAXSIZE
        pop!( _MASK_CACHE, first(keys(_MASK_CACHE)) )
    end
    _MASK_CACHE[key] = mask
    return mask
end

# ── Helpers warp reduction ─────────────────────────────────────────────
if Backend.CUDA_AVAILABLE
    const WARPSIZE = 32

    @inline function _warp_reduce_add(v::Float32)::Float32
        v += CUDA.shfl_down_sync(0xffffffff, v, 16)
        v += CUDA.shfl_down_sync(0xffffffff, v,  8)
        v += CUDA.shfl_down_sync(0xffffffff, v,  4)
        v += CUDA.shfl_down_sync(0xffffffff, v,  2)
        v += CUDA.shfl_down_sync(0xffffffff, v,  1)
        return v
    end

    @inline function _warp_reduce_max(v::Float32)::Float32
        v = max(v, CUDA.shfl_down_sync(0xffffffff, v, 16))
        v = max(v, CUDA.shfl_down_sync(0xffffffff, v,  8))
        v = max(v, CUDA.shfl_down_sync(0xffffffff, v,  4))
        v = max(v, CUDA.shfl_down_sync(0xffffffff, v,  2))
        v = max(v, CUDA.shfl_down_sync(0xffffffff, v,  1))
        return v
    end
end

# ── RMSNorm CPU (boucles explicites, sans allocations) ────────────────
function rmsnorm_fwd!(::Backend.CPUDevice, out, rms_inv, x, gamma; eps=1f-6)
    nr, nc = size(x)
    for i in 1:nr
        s = 0.0f0
        for j in 1:nc
            v = x[i,j]
            s += v * v
        end
        rms_inv[i] = 1f0 / sqrt(s / nc + eps)
    end
    for j in 1:nc
        gj = gamma[j]
        for i in 1:nr
            out[i,j] = x[i,j] * rms_inv[i] * gj
        end
    end
end

function rmsnorm_bwd!(::Backend.CPUDevice, dx, dgamma, dout, x, gamma, rms_inv)
    nr, nc = size(x)
    xn = similar(x)
    for i in 1:nr
        inv = rms_inv[i]
        for j in 1:nc
            xn[i,j] = x[i,j] * inv
        end
    end
    # dgamma
    fill!(dgamma, 0f0)
    for j in 1:nc
        s = 0.0f0
        for i in 1:nr
            s += dout[i,j] * xn[i,j]
        end
        dgamma[j] = s
    end
    # corr
    corr = zeros(Float32, nr)
    for i in 1:nr
        s = 0.0f0
        for j in 1:nc
            s += dout[i,j] * gamma[j] * xn[i,j]
        end
        corr[i] = s / nc
    end
    # dx
    for i in 1:nr
        inv = rms_inv[i]
        ci = corr[i]
        for j in 1:nc
            dx[i,j] = inv * (dout[i,j] * gamma[j] - ci * xn[i,j])
        end
    end
end

# ── RMSNorm CUDA (kernel optimisé, inchangé) ─────────────────────────
if Backend.CUDA_AVAILABLE
    function _rmsnorm_fwd_kernel!(out::CUDA.CuDeviceMatrix{Float32}, rms_inv::CUDA.CuDeviceVector{Float32}, x::CUDA.CuDeviceMatrix{Float32}, gamma::CUDA.CuDeviceVector{Float32}, eps, nr, nc)
        row = blockIdx().x; row > nr && return
        tid = threadIdx().x; nth = blockDim().x
        nw  = cld(nth, WARPSIZE)
        wid = (tid-1) ÷ WARPSIZE + 1; lane = (tid-1) % WARPSIZE
        smem = CUDA.CuDynamicSharedArray(Float32, nw)
        ss = 0f0; j = tid
        while j <= nc; @inbounds v = x[row,j]; ss = fma(v,v,ss); j += nth; end
        ss = _warp_reduce_add(ss)
        lane == 0 && (@inbounds smem[wid] = ss)
        sync_threads()
        if wid == 1
            val = tid <= nw ? @inbounds(smem[tid]) : 0f0
            val = _warp_reduce_add(val)
            tid == 1 && (@inbounds rms_inv[row] = 1f0 / sqrt(val/nc + eps))
        end
        sync_threads()
        @inbounds inv_v = rms_inv[row]; j = tid
        while j <= nc; @inbounds out[row,j] = x[row,j]*inv_v*gamma[j]; j += nth; end
        return
    end

    function rmsnorm_fwd!(::Backend.CUDADevice, out, rms_inv, x, gamma; eps=1f-6)
        nr, nc  = size(x); threads = min(256, nextpow(2, nc)); nw = cld(threads, WARPSIZE)
        @cuda threads=threads blocks=nr shmem=(nw*sizeof(Float32)) _rmsnorm_fwd_kernel!(
            out, rms_inv, x, gamma, eps, nr, nc)
    end

    function _rmsnorm_bwd_xn_kernel!(xn_cu::CUDA.CuDeviceMatrix{Float32}, x::CUDA.CuDeviceMatrix{Float32}, rms_inv::CUDA.CuDeviceVector{Float32}, nr, nc)
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if idx <= nr * nc
            row = (idx - 1) % nr + 1
            col = (idx - 1) ÷ nr + 1
            @inbounds xn_cu[row, col] = x[row, col] * rms_inv[row]
        end
        return
    end

    function _rmsnorm_bwd_dgamma_kernel!(dgamma::CUDA.CuDeviceVector{Float32}, dout::CUDA.CuDeviceMatrix{Float32}, xn_cu::CUDA.CuDeviceMatrix{Float32}, nr, nc)
        col = blockIdx().x
        if col <= nc
            tid = threadIdx().x; nth = blockDim().x
            val = 0f0
            for row = tid:nth:nr
                @inbounds val += dout[row, col] * xn_cu[row, col]
            end
            wid = (tid - 1) ÷ WARPSIZE + 1; lane = (tid - 1) % WARPSIZE
            warp_sum = _warp_reduce_add(val)
            smem = CUDA.CuDynamicSharedArray(Float32, cld(nth, WARPSIZE))
            if lane == 0; smem[wid] = warp_sum; end
            sync_threads()
            if tid == 1
                block_sum = 0f0
                for i = 1:cld(nth, WARPSIZE)
                    block_sum += smem[i]
                end
                @inbounds dgamma[col] = block_sum
            end
        end
        return
    end

    function _rmsnorm_bwd_corr_kernel!(corr_cu::CUDA.CuDeviceVector{Float32}, dout::CUDA.CuDeviceMatrix{Float32}, gamma::CUDA.CuDeviceVector{Float32}, xn_cu::CUDA.CuDeviceMatrix{Float32}, nr, nc)
        row = blockIdx().x
        if row <= nr
            tid = threadIdx().x; nth = blockDim().x
            val = 0f0
            for col = tid:nth:nc
                @inbounds val += dout[row, col] * gamma[col] * xn_cu[row, col]
            end
            wid = (tid - 1) ÷ WARPSIZE + 1; lane = (tid - 1) % WARPSIZE
            warp_sum = _warp_reduce_add(val)
            smem = CUDA.CuDynamicSharedArray(Float32, cld(nth, WARPSIZE))
            if lane == 0; smem[wid] = warp_sum; end
            sync_threads()
            if tid == 1
                block_sum = 0f0
                for i = 1:cld(nth, WARPSIZE)
                    block_sum += smem[i]
                end
                @inbounds corr_cu[row] = block_sum / nc
            end
        end
        return
    end

    function _rmsnorm_bwd_dx_kernel!(dx::CUDA.CuDeviceMatrix{Float32}, dout::CUDA.CuDeviceMatrix{Float32}, gamma::CUDA.CuDeviceVector{Float32}, rms_inv::CUDA.CuDeviceVector{Float32}, corr_cu::CUDA.CuDeviceVector{Float32}, xn_cu::CUDA.CuDeviceMatrix{Float32}, nr, nc)
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if idx <= nr * nc
            row = (idx - 1) % nr + 1
            col = (idx - 1) ÷ nr + 1
            @inbounds inv_val = rms_inv[row]
            @inbounds dout_val = dout[row, col]
            @inbounds gamma_val = gamma[col]
            @inbounds corr_val = corr_cu[row]
            @inbounds xn_val = xn_cu[row, col]
            @inbounds dx[row, col] = inv_val * (dout_val * gamma_val - corr_val * xn_val)
        end
        return
    end

    function rmsnorm_bwd!(::Backend.CUDADevice, dx, dgamma, dout, x, gamma, rms_inv)
        nr, nc = size(x)
        xn_cu = CUDA.zeros(Float32, nr, nc)
        threads_pb_xn = 256
        blocks_xn = cld(nr * nc, threads_pb_xn)
        @cuda threads=threads_pb_xn blocks=blocks_xn _rmsnorm_bwd_xn_kernel!(xn_cu, x, rms_inv, nr, nc)

        threads_pb_dgamma = min(256, nr)
        blocks_dgamma = nc
        smem_size_dgamma = cld(threads_pb_dgamma, WARPSIZE) * sizeof(Float32)
        @cuda threads=threads_pb_dgamma blocks=blocks_dgamma shmem=smem_size_dgamma _rmsnorm_bwd_dgamma_kernel!(dgamma, dout, xn_cu, nr, nc)

        corr_cu = CUDA.zeros(Float32, nr)
        threads_pb_corr = min(256, nc)
        blocks_corr = nr
        smem_size_corr = cld(threads_pb_corr, WARPSIZE) * sizeof(Float32)
        @cuda threads=threads_pb_corr blocks=blocks_corr shmem=smem_size_corr _rmsnorm_bwd_corr_kernel!(corr_cu, dout, gamma, xn_cu, nr, nc)

        threads_pb_dx = 256
        blocks_dx = cld(nr * nc, threads_pb_dx)
        @cuda threads=threads_pb_dx blocks=blocks_dx _rmsnorm_bwd_dx_kernel!(dx, dout, gamma, rms_inv, corr_cu, xn_cu, nr, nc)
        return
    end
end

# ── SwiGLU CPU (boucles explicites) ───────────────────────────────────
function swiglu_fwd!(::Backend.CPUDevice, out, gate, up)
    n = length(out)
    for i in 1:n
        g = gate[i]
        sig = 1f0 / (1f0 + exp(-g))
        out[i] = g * sig * up[i]
    end
end

function swiglu_bwd!(::Backend.CPUDevice, dgate, dup, dout, gate, up)
    n = length(dout)
    for i in 1:n
        g = gate[i]
        sig = 1f0 / (1f0 + exp(-g))
        dup[i] = dout[i] * g * sig
        dgate[i] = dout[i] * up[i] * sig * (1f0 + g * (1f0 - sig))
    end
end

if Backend.CUDA_AVAILABLE
    function _swiglu_fwd_kernel!(out::CUDA.CuDeviceMatrix{Float32}, gate::CUDA.CuDeviceMatrix{Float32}, up::CUDA.CuDeviceMatrix{Float32}, n)
        i=(blockIdx().x-1)*blockDim().x+threadIdx().x; i>n && return
        @inbounds g=gate[i]; sig=1f0/(1f0+exp(-g)); out[i]=g*sig*up[i]; return
    end
    function _swiglu_bwd_kernel!(dgate::CUDA.CuDeviceMatrix{Float32}, dup::CUDA.CuDeviceMatrix{Float32}, dout::CUDA.CuDeviceMatrix{Float32}, gate::CUDA.CuDeviceMatrix{Float32}, up::CUDA.CuDeviceMatrix{Float32}, n)
        i=(blockIdx().x-1)*blockDim().x+threadIdx().x; i>n && return
        @inbounds begin g=gate[i]; sig=1f0/(1f0+exp(-g))
            dup[i]=dout[i]*gate[i]*sig; dgate[i]=dout[i]*up[i]*sig*(1f0+g*(1f0-sig)) end
        return
    end
    swiglu_fwd!(::Backend.CUDADevice, out, gate, up) = (n=length(out);
        @cuda threads=256 blocks=cld(n,256) _swiglu_fwd_kernel!(out,gate,up,n))
    swiglu_bwd!(::Backend.CUDADevice, dgate, dup, dout, gate, up) = (n=length(dout);
        @cuda threads=256 blocks=cld(n,256) _swiglu_bwd_kernel!(dgate,dup,dout,gate,up,n))
end

# ── Softmax CPU (boucles explicites) ───────────────────────────────────
function softmax_fwd!(::Backend.CPUDevice, out, x)
    nr, nc = size(x)
    for i in 1:nr
        max_val = x[i,1]
        for j in 2:nc
            v = x[i,j]; if v > max_val; max_val = v; end
        end
        s = 0.0f0
        for j in 1:nc
            e = exp(x[i,j] - max_val)
            out[i,j] = e
            s += e
        end
        inv_s = 1f0 / s
        for j in 1:nc
            out[i,j] *= inv_s
        end
    end
end

function softmax_bwd!(::Backend.CPUDevice, dx, dout, out)
    nr, nc = size(out)
    for i in 1:nr
        dot = 0.0f0
        for j in 1:nc
            dot += dout[i,j] * out[i,j]
        end
        for j in 1:nc
            dx[i,j] = out[i,j] * (dout[i,j] - dot)
        end
    end
end

# ── Softmax CUDA (warp‑per‑row, inchangé) ─────────────────────────────
if Backend.CUDA_AVAILABLE
    function _softmax_fwd_kernel!(out::CUDA.CuDeviceMatrix{Float32}, x::CUDA.CuDeviceMatrix{Float32}, nrows, ncols)
        row = blockIdx().x
        if row > nrows; return; end
        tid = threadIdx().x
        nth = blockDim().x

        local_max = -Inf32
        for col = tid:nth:ncols
            @inbounds v = x[row, col]
            if v > local_max; local_max = v; end
        end
        local_max = _warp_reduce_max(local_max)
        warps_per_block = cld(nth, WARPSIZE)
        smem = CUDA.CuDynamicSharedArray(Float32, warps_per_block + 2)
        wid = (tid-1) ÷ WARPSIZE + 1
        lane = (tid-1) % WARPSIZE
        if lane == 0
            smem[wid] = local_max
        end
        sync_threads()
        if wid == 1
            block_max = tid <= warps_per_block ? smem[tid] : -Inf32
            block_max = _warp_reduce_max(block_max)
            if tid == 1
                smem[warps_per_block+1] = block_max
            end
        end
        sync_threads()
        max_val = smem[warps_per_block+1]

        local_sum = 0f0
        for col = tid:nth:ncols
            @inbounds e = exp(x[row, col] - max_val)
            @inbounds out[row, col] = e
            local_sum += e
        end
        local_sum = _warp_reduce_add(local_sum)
        if lane == 0
            smem[wid] = local_sum
        end
        sync_threads()
        if wid == 1
            block_sum = 0f0
            for i = 1:warps_per_block
                block_sum += smem[i]
            end
            if tid == 1
                smem[warps_per_block+2] = block_sum
            end
        end
        sync_threads()
        total_sum = smem[warps_per_block+2]

        for col = tid:nth:ncols
            @inbounds out[row, col] /= total_sum
        end
        return
    end
    function _ce_subtract_one_hot_kernel!(
        g::CUDA.CuDeviceMatrix{Float32},
        labels::CUDA.CuDeviceVector{Int32},
        n::Int)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        i > n && return
        @inbounds g[i, labels[i]] -= 1f0
        return
    end

    function softmax_fwd!(::Backend.CUDADevice, out, x)
        nr, nc = size(x)
        threads = min(256, nextpow(2, nc))
        warps = cld(threads, WARPSIZE)
        shmem = (warps + 2) * sizeof(Float32)
        @cuda threads=threads blocks=nr shmem=shmem _softmax_fwd_kernel!(out, x, nr, nc)
    end

    function _softmax_bwd_kernel!(dx::CUDA.CuDeviceMatrix{Float32}, dout::CUDA.CuDeviceMatrix{Float32}, out::CUDA.CuDeviceMatrix{Float32}, nrows, ncols)
        row = blockIdx().x
        if row > nrows; return; end
        tid = threadIdx().x
        nth = blockDim().x

        local_dot = 0f0
        for col = tid:nth:ncols
            @inbounds local_dot += dout[row, col] * out[row, col]
        end
        local_dot = _warp_reduce_add(local_dot)
        warps_per_block = cld(nth, WARPSIZE)
        smem = CUDA.CuDynamicSharedArray(Float32, warps_per_block + 1)
        wid = (tid-1) ÷ WARPSIZE + 1
        lane = (tid-1) % WARPSIZE
        if lane == 0
            smem[wid] = local_dot
        end
        sync_threads()
        if wid == 1
            block_dot = 0f0
            for i = 1:warps_per_block
                block_dot += smem[i]
            end
            if tid == 1
                smem[warps_per_block+1] = block_dot
            end
        end
        sync_threads()
        total_dot = smem[warps_per_block+1]

        for col = tid:nth:ncols
            @inbounds dx[row, col] = out[row, col] * (dout[row, col] - total_dot)
        end
        return
    end

    function softmax_bwd!(::Backend.CUDADevice, dx, dout, out)
        nr, nc = size(out)
        threads = min(256, nextpow(2, nc))
        warps = cld(threads, WARPSIZE)
        shmem = (warps + 1) * sizeof(Float32)
        @cuda threads=threads blocks=nr shmem=shmem _softmax_bwd_kernel!(dx, dout, out, nr, nc)
    end
end

# ── Masque causal (utilise le cache) ───────────────────────────────────
causal_mask(::Backend.CPUDevice, seqlen::Int) = causal_mask_cached(Backend.CPUDevice(), seqlen)
causal_mask(::Backend.CUDADevice, seqlen::Int) = causal_mask_cached(Backend.CUDADevice(), seqlen)
scale_mask_fwd!(out, scores, d_head::Int, mask) =
    (out .= scores .* (1f0/sqrt(Float32(d_head))) .+ mask)

# ── AdamW (version corrigée) ───────────────────────────────────────────
function adamw_step_cpu!(W, dW, m1, m2, lr, b1, b2, eps_v, t, clip, wd)
    # 1. Clipping sécurisé
    gc = clamp.(dW, -Float32(clip), Float32(clip))
    
    # 2. Mise à jour des moments
    m1 .= Float32(b1) .* m1 .+ (1f0 - Float32(b1)) .* gc
    m2 .= Float32(b2) .* m2 .+ (1f0 - Float32(b2)) .* (gc .* gc)
    
    # 3. Correction de biais (t doit être >= 1)
    t_f = Float32(t)
    mh = m1 ./ (1f0 - Float32(b1)^t_f)
    vh = m2 ./ (1f0 - Float32(b2)^t_f)
    
    # 4. Mise à jour des poids avec Weight Decay dissocié
    # W = W - lr * (wd * W + update)
    W .= W .* (1f0 - Float32(lr) * Float32(wd)) .- Float32(lr) .* (mh ./ (sqrt.(vh) .+ Float32(eps_v)))
    
    # 5. Reset gradient
    fill!(dW, 0f0)
end

adamw_step!(::Backend.CPUDevice, W,dW,m1,m2,lr,b1,b2,eps_v,t,clip,wd) =
    adamw_step_cpu!(W,dW,m1,m2,lr,b1,b2,eps_v,t,clip,wd)

if Backend.CUDA_AVAILABLE
    function _adamw_fused_kernel!(W::CUDA.CuDeviceVector{Float32}, 
                                  dW::CUDA.CuDeviceVector{Float32}, 
                                  m1::CUDA.CuDeviceVector{Float32}, 
                                  m2::CUDA.CuDeviceVector{Float32}, 
                                  lr::Float32, b1::Float32, b2::Float32, 
                                  eps_v::Float32, t::Int32, clip::Float32, wd::Float32, n::Int32)
        
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if i <= n
            @inbounds begin
                # 1. Chargement dans les registres
                g_val = dW[i]
                
                # 2. Gradient clipping
                g_clip = max(-clip, min(clip, g_val))
                
                # 3. Mise à jour des moments (dans les registres)
                m1_curr = m1[i]
                m2_curr = m2[i]
                
                m1_new = b1 * m1_curr + (1f0 - b1) * g_clip
                m2_new = b2 * m2_curr + (1f0 - b2) * g_clip^2
                
                # Sauvegarde VRAM
                m1[i] = m1_new
                m2[i] = m2_new
                
                # 4. Correction du biais (conversion explicite de t en Float32)
                t_f32 = Float32(t)
                mh = m1_new / (1f0 - b1^t_f32)
                vh = m2_new / (1f0 - b2^t_f32)
                
                # 5. Weight Decay et Step (dans les registres)
                w_curr = W[i]
                w_new = w_curr * (1f0 - lr * wd) - lr * mh / (sqrt(vh) + eps_v)
                
                # 6. Écriture finale et remise à zéro du gradient
                W[i] = w_new
                dW[i] = 0f0
            end
        end
        return
    end

    function adamw_step!(::Backend.CUDADevice, W, dW, m1, m2, lr, b1, b2, eps_v, t, clip, wd)
        n = Int32(length(W))
        threads = 256
        blocks = cld(n, threads)
        
        # On aplatit les tenseurs pour le kernel 1D
        W_vec  = vec(W)
        dW_vec = vec(dW)
        m1_vec = vec(m1)
        m2_vec = vec(m2)
        
        @cuda threads=threads blocks=blocks _adamw_fused_kernel!(
            W_vec, dW_vec, m1_vec, m2_vec, 
            Float32(lr), Float32(b1), Float32(b2), Float32(eps_v), 
            Int32(t), Float32(clip), Float32(wd), n
        )
    end
end

# ── Cross-entropy optimisée (GPU sans copie, CPU sans allocations excessives) ──
function cross_entropy_loss(logits::AbstractMatrix{Float32}, labels::AbstractVector)
    lb = Int.(vec(labels))

    if Backend.CUDA_AVAILABLE && logits isa CUDA.CuArray
        max_vals = maximum(logits, dims=2)
        shifted = logits .- max_vals
        e = exp.(shifted)
        p = e ./ sum(e, dims=2)
        n = size(logits, 1)
        # Éviter la copie inutile si labels est déjà sur GPU
        labels_cpu = collect(Int, labels)
        p_cpu = Array(p)
        logp = log.(max.(p_cpu[CartesianIndex.(1:n, labels_cpu)], 1f-10))
        return Float32(-mean(logp))
    else
        lh = Backend.to_cpu(logits)
        lb = collect(Int, labels)
        e = exp.(lh .- maximum(lh, dims=2))
        p = e ./ sum(e, dims=2)
        n = size(lh, 1)
        -mean(log.(max.(p[CartesianIndex.(1:n, lb)], 1f-10)))
    end
end

function cross_entropy_grad(logits::AbstractMatrix{Float32}, labels::AbstractVector)
    lb = Int.(vec(labels))

    if Backend.CUDA_AVAILABLE && logits isa CUDA.CuArray
        max_vals = maximum(logits, dims=2)
        shifted = logits .- max_vals
        e = exp.(shifted)
        p = e ./ sum(e, dims=2)
        g = copy(p)
        n = size(logits, 1)
        # Gestion efficace des labels GPU (sans copie CPU inutile)
        labels_gpu = if labels isa CUDA.CuArray
            Int32.(labels)   # conversion de type sans passage CPU
        else
            CUDA.cu(Int32.(collect(labels)))
        end
        threads = min(256, n)
        blocks  = cld(n, threads)
        @cuda threads=threads blocks=blocks _ce_subtract_one_hot_kernel!(g, labels_gpu, n)
        g ./= Float32(n)
        return g
    else
        lh = Backend.to_cpu(logits)
        lb = collect(Int, labels)
        e = exp.(lh .- maximum(lh, dims=2))
        p = e ./ sum(e, dims=2)
        g = copy(p)
        n = size(lh, 1)
        for i in 1:n; g[i, lb[i]] -= 1f0; end
        g ./= n
    end
end

# ── MSE loss (corrigé) ───────────────────────────────────────────────
mse_loss_fwd(out, target) = [sum((out .- target).^2) / length(out)]
#mse_loss_bwd(out, target, dy) = (2f0 ./ length(out)) .* (out .- target) .* first(dy)
function mse_loss_bwd(out, target, dy)
    N = length(out)
    grad_out = (2f0 / N) .* (out .- target) .* dy   # dy scalaire ou 0‑dim
    return (grad_out, -grad_out)
end

# ── Sum of matrix elements (identique) ─────────────────────────────────
sum_matrix_fwd(x) = [sum(x)]
sum_matrix_bwd(dev, x_val, dy) = Backend.ones32(dev, size(x_val)...) .* sum(dy)

# ── Forward embedding CUDA kernel (corrigé) ──────────────────────────
if Backend.CUDA_AVAILABLE
    function _embedding_fwd_kernel!(out::CUDA.CuDeviceMatrix{Float32}, E::CUDA.CuDeviceMatrix{Float32}, idx::CUDA.CuDeviceVector{Int}, n_batch, d_emb)
        li = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if li > n_batch * d_emb
            return
        end
        batch_row = (li - 1) ÷ d_emb + 1  # division entière plus efficace
        emb_col   = (li - 1) % d_emb + 1
        target_row = @inbounds idx[batch_row]
        @inbounds out[batch_row, emb_col] = E[target_row, emb_col]
        return
    end
end

if Backend.CUDA_AVAILABLE
    function _fused_matmul_relu_kernel!(out::CUDA.CuDeviceMatrix{Float32},
                                        A::CUDA.CuDeviceMatrix{Float32},
                                        B::CUDA.CuDeviceMatrix{Float32},
                                        M::Int, N::Int, K::Int, trans_b::Bool)
        row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        col = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        if row <= M && col <= N
            acc = 0.0f0
            if trans_b
                # B est (N, K)
                for k in 1:K
                    acc += A[row, k] * B[col, k]
                end
            else
                # B est (K, N)
                for k in 1:K
                    acc += A[row, k] * B[k, col]
                end
            end
            out[row, col] = max(acc, 0.0f0)
        end
        return nothing   # ← AJOUT INDISPENSABLE
    end
end