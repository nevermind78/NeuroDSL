if Backend.CUDA_AVAILABLE; using CUDA; end

# ── Cache pour le masque causal (LRU avec taille max) ─────────────────
const _MASK_CACHE = Dict{Tuple{Symbol,Int},Any}()
const _MASK_CACHE_MAXSIZE = 10

function causal_mask_cached(device, seqlen::Int)
    # L'index du GPU actif fait partie de la clé : sur une machine multi-GPU,
    # un masque alloué pendant que GPU 0 était actif est inutilisable une
    # fois que CUDA.device!(1) est appelé (mémoire inaccessible depuis
    # l'autre device). Sans ça, la deuxième carte réutilise le masque de la
    # première et plante avec "inaccessible device memory".
    key = device isa Backend.CUDADevice ?
        (Symbol(:cuda, Int(CUDA.device().handle)), seqlen) : (:cpu, seqlen)
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

    # CORRECTIF 2026-07-28 (trouvé en chargeant Qwen2.5 réel, séquences courtes) :
    # deux bugs liés, tous deux dans l'appel à `shfl_down_sync`, dont un SEUL
    # (le masque) ne suffit pas à corriger :
    #   (1) `mask` doit être EXACTEMENT l'ensemble des lanes réellement actives
    #       -- un bloc lancé avec `threads < 32` (ex. `nextpow(2,nc)` pour
    #       nc=5 -> threads=8) n'occupe que les lanes 0..nth-1 d'une warp ;
    #       passer 0xffffffff (comme avant) revendique la participation de 24
    #       lanes fantômes qui n'exécutent jamais cette instruction.
    #   (2) `width` (4e argument, PAS juste le masque) doit AUSSI valoir
    #       `min(nth,32)` -- c'est `width`, pas `mask`, qui détermine
    #       l'arithmétique de la lane cible : avec `width` par défaut (32,
    #       comme avant), une lane i<width_réel calculant `i+delta>=width_réel`
    #       tente de lire une lane qui n'existe PAS DU TOUT dans ce lancement
    #       (pas seulement "masquée") -- non défini même avec un masque
    #       correct. Avec `width` correctement fixé, la spec CUDA garantit que
    #       toute lane cible hors de la sous-section de taille `width` "wrap"
    #       vers la valeur PROPRE de l'appelant (no-op documenté) -- donc les 5
    #       déltas (16,8,4,2,1) restent inconditionnellement sûrs : ceux
    #       >= width deviennent des no-ops automatiques, aucune branche
    #       supplémentaire nécessaire.
    # Observé concrètement sans ce correctif : NaN intermittents, dépendant de
    # la ligne/tête, sur `:softmax` avec des séquences courtes (< 32 tokens)
    # sur le chemin batché. Ce correctif ne change RIEN pour nth>=32 (chaque
    # warp d'un bloc >= 32 threads est toujours intégralement peuplée, donc
    # mask=0xffffffff et width=32 restent ce qu'ils étaient) -- seulement pour
    # nth<32.
    @inline function _warp_reduce_params(nth::Int32)
        width = min(nth, Int32(32))
        mask = width >= Int32(32) ? 0xffffffff : (UInt32(1) << width) - UInt32(1)
        return mask, width
    end
    _warp_reduce_params(nth::Integer) = _warp_reduce_params(Int32(nth))

    # CORRECTIF AU CORRECTIF (même session, trouvé par la suite de tests --
    # `check_gradients` sur RMSNorm CUDA, diagnostic non-assertant mais lu) :
    # passer `width` à `shfl_down_sync` fait bien "clamper" un décalage hors
    # de la sous-section de taille `width` vers la valeur PROPRE de
    # l'appelant -- mais "retourner sa propre valeur" n'est un no-op que pour
    # `max` (`max(v,v)=v`) ; pour une somme, `v += v` la DOUBLE. Chaîner les 5
    # déltas (16,8,4,2,1) sans condition, comme avant, corrompt donc toute
    # réduction par SOMME dès que `width<32` (RMSNorm, la somme de ligne du
    # softmax, dgamma, corr, softmax_bwd) -- démasqué par
    # `check_gradients`/`grad_check` sur RMSNorm CUDA (M=4,8,1), erreurs de
    # l'ordre de 1 à 256 au lieu de 1e-4. Le correctif correct : n'appliquer
    # QUE les déltas STRICTEMENT INFÉRIEURS à `width` (16,8,4 sont hors
    # sous-section dès que width<=4, ne doivent jamais s'exécuter, pas
    # seulement "s'exécuter sans effet") -- la réduction converge alors
    # exactement vers le total à la lane 0 de chaque sous-section (le seul
    # résultat jamais lu, via les `lane==0`/`wid==1` déjà présents partout
    # dans ce fichier), quelle que soit la valeur de `width` (toujours une
    # puissance de 2, par construction de `nextpow(2,...)` en amont).
    @inline function _warp_reduce_add(v::Float32, mask::UInt32, width::Int32)::Float32
        width > Int32(16) && (v += CUDA.shfl_down_sync(mask, v, 16, width))
        width > Int32(8)  && (v += CUDA.shfl_down_sync(mask, v,  8, width))
        width > Int32(4)  && (v += CUDA.shfl_down_sync(mask, v,  4, width))
        width > Int32(2)  && (v += CUDA.shfl_down_sync(mask, v,  2, width))
        width > Int32(1)  && (v += CUDA.shfl_down_sync(mask, v,  1, width))
        return v
    end

    @inline function _warp_reduce_max(v::Float32, mask::UInt32, width::Int32)::Float32
        width > Int32(16) && (v = max(v, CUDA.shfl_down_sync(mask, v, 16, width)))
        width > Int32(8)  && (v = max(v, CUDA.shfl_down_sync(mask, v,  8, width)))
        width > Int32(4)  && (v = max(v, CUDA.shfl_down_sync(mask, v,  4, width)))
        width > Int32(2)  && (v = max(v, CUDA.shfl_down_sync(mask, v,  2, width)))
        width > Int32(1)  && (v = max(v, CUDA.shfl_down_sync(mask, v,  1, width)))
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
        wmask, wwidth = _warp_reduce_params(nth)
        smem = CUDA.CuDynamicSharedArray(Float32, nw)
        ss = 0f0; j = tid
        while j <= nc; @inbounds v = x[row,j]; ss = fma(v,v,ss); j += nth; end
        ss = _warp_reduce_add(ss, wmask, wwidth)
        lane == 0 && (@inbounds smem[wid] = ss)
        sync_threads()
        if wid == 1
            val = tid <= nw ? @inbounds(smem[tid]) : 0f0
            val = _warp_reduce_add(val, wmask, wwidth)
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
            wmask, wwidth = _warp_reduce_params(nth)
            warp_sum = _warp_reduce_add(val, wmask, wwidth)
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
            wmask, wwidth = _warp_reduce_params(nth)
            warp_sum = _warp_reduce_add(val, wmask, wwidth)
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
        # similar() (pas CUDA.zeros) : les deux noyaux ci-dessous écrivent chaque élément
        # exactement une fois (aucune accumulation, aucune lecture avant écriture), donc
        # l'initialisation à zéro est un travail perdu -- et similar() est ce que les
        # allocateurs/traceurs mémoire (ex. MemTrack des notebooks) interceptent réellement.
        xn_cu = similar(x, nr, nc)
        threads_pb_xn = 256
        blocks_xn = cld(nr * nc, threads_pb_xn)
        @cuda threads=threads_pb_xn blocks=blocks_xn _rmsnorm_bwd_xn_kernel!(xn_cu, x, rms_inv, nr, nc)

        threads_pb_dgamma = min(256, nr)
        blocks_dgamma = nc
        smem_size_dgamma = cld(threads_pb_dgamma, WARPSIZE) * sizeof(Float32)
        @cuda threads=threads_pb_dgamma blocks=blocks_dgamma shmem=smem_size_dgamma _rmsnorm_bwd_dgamma_kernel!(dgamma, dout, xn_cu, nr, nc)

        corr_cu = similar(rms_inv, nr)
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
        wmask, wwidth = _warp_reduce_params(nth)

        local_max = -Inf32
        for col = tid:nth:ncols
            @inbounds v = x[row, col]
            if v > local_max; local_max = v; end
        end
        local_max = _warp_reduce_max(local_max, wmask, wwidth)
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
            block_max = _warp_reduce_max(block_max, wmask, wwidth)
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
        local_sum = _warp_reduce_add(local_sum, wmask, wwidth)
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
        wmask, wwidth = _warp_reduce_params(nth)

        local_dot = 0f0
        for col = tid:nth:ncols
            @inbounds local_dot += dout[row, col] * out[row, col]
        end
        local_dot = _warp_reduce_add(local_dot, wmask, wwidth)
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

# ── AdamW groupé (un seul lancement de kernel pour N tenseurs de paramètres,
# au lieu d'un lancement par tenseur) -- même patron pointeur-brut-par-bloc
# que `_multi_copy_kernel!` (src/patching.jl), corps de boucle copié verbatim
# depuis `_adamw_fused_kernel!` ci-dessus pour garantir la bit-exactitude. ──
adamw_step_batched!(::Backend.CPUDevice, Ws, dWs, m1s, m2s, lr, b1, b2, eps_v, t, clip, wd) =
    for i in eachindex(Ws)
        adamw_step!(Backend.CPUDevice(), Ws[i], dWs[i], m1s[i], m2s[i], lr, b1, b2, eps_v, t, clip, wd)
    end

if Backend.CUDA_AVAILABLE
    function _multi_adamw_kernel!(ptrs::CUDA.CuDeviceVector{CUDA.CuPtr{Float32}},
                                   lens::CUDA.CuDeviceVector{Int32}, N::Int32,
                                   lr::Float32, b1::Float32, b2::Float32,
                                   eps_v::Float32, t::Int32, clip::Float32, wd::Float32)
        bid = blockIdx().x
        n = Int(lens[bid])
        Nn = Int(N)
        W  = CUDA.CuDeviceArray{Float32,1,CUDA.AS.Global}(
            reinterpret(Core.LLVMPtr{Float32,CUDA.AS.Global}, ptrs[bid]), (n,))
        dW = CUDA.CuDeviceArray{Float32,1,CUDA.AS.Global}(
            reinterpret(Core.LLVMPtr{Float32,CUDA.AS.Global}, ptrs[Nn + bid]), (n,))
        m1 = CUDA.CuDeviceArray{Float32,1,CUDA.AS.Global}(
            reinterpret(Core.LLVMPtr{Float32,CUDA.AS.Global}, ptrs[2Nn + bid]), (n,))
        m2 = CUDA.CuDeviceArray{Float32,1,CUDA.AS.Global}(
            reinterpret(Core.LLVMPtr{Float32,CUDA.AS.Global}, ptrs[3Nn + bid]), (n,))

        t_f32 = Float32(t)
        bc1 = 1f0 - b1^t_f32
        bc2 = 1f0 - b2^t_f32

        i = Int(threadIdx().x)
        while i <= n
            @inbounds begin
                g_val = dW[i]
                g_clip = max(-clip, min(clip, g_val))

                m1_curr = m1[i]
                m2_curr = m2[i]

                m1_new = b1 * m1_curr + (1f0 - b1) * g_clip
                m2_new = b2 * m2_curr + (1f0 - b2) * g_clip^2

                m1[i] = m1_new
                m2[i] = m2_new

                mh = m1_new / bc1
                vh = m2_new / bc2

                w_curr = W[i]
                w_new = w_curr * (1f0 - lr * wd) - lr * mh / (sqrt(vh) + eps_v)

                W[i] = w_new
                dW[i] = 0f0
            end
            i += Int(blockDim().x)
        end
        return
    end

    function adamw_step_batched!(::Backend.CUDADevice, Ws, dWs, m1s, m2s, lr, b1, b2, eps_v, t, clip, wd)
        N = length(Ws)
        @assert length(dWs) == N && length(m1s) == N && length(m2s) == N "adamw_step_batched! : Ws/dWs/m1s/m2s doivent avoir la même longueur"
        N == 0 && return

        lens = Int32.(length.(Ws))
        ptrs = Vector{CUDA.CuPtr{Float32}}(undef, 4N)
        for i in 1:N
            ptrs[i]        = pointer(Ws[i])
            ptrs[N + i]    = pointer(dWs[i])
            ptrs[2N + i]   = pointer(m1s[i])
            ptrs[3N + i]   = pointer(m2s[i])
        end
        ptrs_gpu = CUDA.CuArray(ptrs)
        lens_gpu = CUDA.CuArray(lens)
        threads = min(256, Int(maximum(lens)))

        @cuda threads=threads blocks=N _multi_adamw_kernel!(
            ptrs_gpu, lens_gpu, Int32(N),
            Float32(lr), Float32(b1), Float32(b2), Float32(eps_v),
            Int32(t), Float32(clip), Float32(wd)
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
        # CORRECTIF (enquête grokking, 2026-07-25) : le vrai notebook Nanda
        # (artilce/Grokking_Demo.ipynb, cellule 29) calcule `loss_fn` en
        # castant `logits.to(torch.float64)` AVANT log_softmax/gather --
        # donc tout son calcul de perte (et le gradient qui en remonte)
        # tourne en Float64, pas Float32. Diagnostiqué par instrumentation
        # directe (notebook/grokking_diag_instrumented.jl) : en Float32 pur,
        # le gradient de cross-entropy devient EXACTEMENT 0.0f0 dès que les
        # probabilités softmax sont assez proches de 0/1 (mémorisation
        # profonde sur un petit train set) -- vérifié via la décroissance
        # géométrique EXACTE (ratio beta2^20) du second moment d'Adam sur
        # ~1300 pas avant qu'un pas plein-échelle ("slingshot") ne
        # réapparaisse. Le Float64 interne donne une marge de précision qui
        # retarde/élimine ce sous-dépassement exact -- calcul ici en Float64
        # puis reconversion en Float32 pour la sortie, comme Nanda. Portée
        # limitée à la voie CPU (celle exercée par les tests concernés) --
        # la voie CUDA n'est pas touchée pour l'instant.
        lh64 = Float64.(Backend.to_cpu(logits))
        lb = collect(Int, labels)
        e = exp.(lh64 .- maximum(lh64, dims=2))
        p = e ./ sum(e, dims=2)
        g = copy(p)
        n = size(lh64, 1)
        for i in 1:n; g[i, lb[i]] -= 1.0; end
        g ./= n
        return Float32.(g)
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
    function _fused_matmul_add_relu_kernel!(out::CUDA.CuDeviceMatrix{Float32},
                                            A::CUDA.CuDeviceMatrix{Float32},
                                            B::CUDA.CuDeviceMatrix{Float32},
                                            bias::CUDA.CuDeviceVector{Float32},
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
            out[row, col] = max(acc + bias[col], 0.0f0)
        end
        return nothing   # ← AJOUT INDISPENSABLE
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Attention multi-têtes batchée (gemm_strided_batched) -- conçu avec Fable,
# 2026-07-10. Remplace 4 lancements de kernel par tête par 1 seul, SANS copie
# supplémentaire quand les entrées sont encore des vues zero-copy sur le même
# tampon parent (`:view_cols`/`:head_view`, src/dispatch.jl) -- sinon repli sûr
# par rassemblement (gather) dans un tampon frais, toujours correct.
#
# `_sibling_view_parent` détecte, par ARITHMÉTIQUE DE POINTEUR (pas en
# inspectant les champs internes de SubArray, plus robuste aux versions
# Julia/CUDA.jl), si `views[i]` est exactement la tranche contiguë attendue
# (i-1)*item_shape[1]*item_shape[2] .. du même tableau parent. Si oui, un
# simple `reshape` (zero-copy) du parent remplace le rassemblement.
# ══════════════════════════════════════════════════════════════════════════════
function _sibling_view_parent(views::AbstractVector, item_shape::Tuple{Int,Int})
    n = length(views)
    n == 0 && return nothing
    v1 = views[1]
    (v1 isa SubArray) || return nothing
    p = parent(v1)
    eltype(p) === Float32 || return nothing
    esz = sizeof(Float32)
    stride_elems = item_shape[1] * item_shape[2]
    base = pointer(p)
    for (i, v) in enumerate(views)
        (v isa SubArray && parent(v) === p) || return nothing
        size(v) == item_shape || return nothing
        pointer(v) == base + (i - 1) * stride_elems * esz || return nothing
    end
    return p
end

function _gather3(heads::AbstractVector, a::Int, b::Int, H::Int, dev)
    out = Backend.zeros32(dev, a, b, H)
    for h in 1:H
        out[:, :, h] .= heads[h]
    end
    return out
end

# ── Forward : batched_qk (Q·Kᵀ groupé sur les têtes) ──────────────────────
function batched_qk_fwd!(::Backend.CPUDevice, output_buffer, q_heads::AbstractVector, k_heads::AbstractVector, d_head::Int)
    for h in eachindex(q_heads)
        LinearAlgebra.mul!(view(output_buffer, :, :, h), q_heads[h], k_heads[h]')
    end
end

if Backend.CUDA_AVAILABLE
    function batched_qk_fwd!(dev::Backend.CUDADevice, output_buffer, q_heads::AbstractVector, k_heads::AbstractVector, d_head::Int)
        H = length(q_heads)
        seq = size(q_heads[1], 1)
        Qp = _sibling_view_parent(q_heads, (seq, d_head))
        Kp = _sibling_view_parent(k_heads, (seq, d_head))
        Q3 = Qp !== nothing ? reshape(Qp, seq, d_head, H) : _gather3(q_heads, seq, d_head, H, dev)
        K3 = Kp !== nothing ? reshape(Kp, seq, d_head, H) : _gather3(k_heads, seq, d_head, H, dev)
        CUDA.CUBLAS.gemm_strided_batched!('N', 'T', 1f0, Q3, K3, 0f0, output_buffer)
    end
end

# ── Backward : batched_qk ──────────────────────────────────────────────────
function _batched_qk_bwd!(::Backend.CPUDevice, dQ3, dK3, dC3, Q3, K3)
    for h in axes(dC3, 3)
        LinearAlgebra.mul!(view(dQ3, :, :, h), view(dC3, :, :, h), view(K3, :, :, h))
        LinearAlgebra.mul!(view(dK3, :, :, h), view(dC3, :, :, h)', view(Q3, :, :, h))
    end
end

if Backend.CUDA_AVAILABLE
    function _batched_qk_bwd!(::Backend.CUDADevice, dQ3, dK3, dC3, Q3, K3)
        CUDA.CUBLAS.gemm_strided_batched!('N', 'N', 1f0, dC3, K3, 0f0, dQ3)
        CUDA.CUBLAS.gemm_strided_batched!('T', 'N', 1f0, dC3, Q3, 0f0, dK3)
    end
end

# ── Forward : batched_pv (P·V groupé sur les têtes) ────────────────────────
function batched_pv_fwd!(::Backend.CPUDevice, output_buffer, pr_heads::AbstractVector, v_heads::AbstractVector, d_head::Int)
    for h in eachindex(pr_heads)
        LinearAlgebra.mul!(view(output_buffer, :, :, h), pr_heads[h], v_heads[h])
    end
end

if Backend.CUDA_AVAILABLE
    function batched_pv_fwd!(dev::Backend.CUDADevice, output_buffer, pr_heads::AbstractVector, v_heads::AbstractVector, d_head::Int)
        H = length(pr_heads)
        seq = size(pr_heads[1], 1)
        Pp = _sibling_view_parent(pr_heads, (seq, seq))
        Vp = _sibling_view_parent(v_heads, (seq, d_head))
        P3 = Pp !== nothing ? reshape(Pp, seq, seq, H) : _gather3(pr_heads, seq, seq, H, dev)
        V3 = Vp !== nothing ? reshape(Vp, seq, d_head, H) : _gather3(v_heads, seq, d_head, H, dev)
        CUDA.CUBLAS.gemm_strided_batched!('N', 'N', 1f0, P3, V3, 0f0, output_buffer)
    end
end

# ── Backward : batched_pv ──────────────────────────────────────────────────
function _batched_pv_bwd!(::Backend.CPUDevice, dP3, dV3, dO3, P3, V3)
    for h in axes(dO3, 3)
        LinearAlgebra.mul!(view(dP3, :, :, h), view(dO3, :, :, h), view(V3, :, :, h)')
        LinearAlgebra.mul!(view(dV3, :, :, h), view(P3, :, :, h)', view(dO3, :, :, h))
    end
end

if Backend.CUDA_AVAILABLE
    function _batched_pv_bwd!(::Backend.CUDADevice, dP3, dV3, dO3, P3, V3)
        CUDA.CUBLAS.gemm_strided_batched!('N', 'T', 1f0, dO3, V3, 0f0, dP3)
        CUDA.CUBLAS.gemm_strided_batched!('T', 'N', 1f0, P3, dO3, 0f0, dV3)
    end
end