# ════════════════════════════════════════════════════════════════════════════════
# NeuroDSL — mixed_precision.jl
# Mixed precision : loss scaling dynamique, détection overflow,
# utilitaires de cast Float16/Float32.
# Les kernels BLAS existants restent Float32 ; la précision mixte est
# activable nœud par nœud via cast_fp16! sur les outputs.
# ════════════════════════════════════════════════════════════════════════════════

# ── Utilitaires de cast ───────────────────────────────────────────────────────
"""
    cast_fp16(x) → AbstractArray{Float16}
Convertit un tenseur Float32 en Float16 (CPU ou CuArray).
"""
function cast_fp16(x::AbstractArray{Float32})
    if Backend.CUDA_AVAILABLE && x isa CUDA.CuArray
        return CUDA.cu(Float16.(Array(x)))
    end
    return Float16.(x)
end
cast_fp16(x::AbstractArray{Float16}) = x

"""
    cast_fp32(x) → AbstractArray{Float32}
Convertit un tenseur Float16 en Float32.
"""
function cast_fp32(x::AbstractArray{Float16})
    if Backend.CUDA_AVAILABLE && x isa CUDA.CuArray
        return Float32.(x)
    end
    return Float32.(x)
end
cast_fp32(x::AbstractArray{Float32}) = x

"""
    has_inf_or_nan(x) → Bool
Détecte les inf/nan dans un tenseur (CPU ou GPU).
"""
function has_inf_or_nan(x::AbstractArray)
    arr = (x isa Array) ? x : Array(x)
    return any(isnan, arr) || any(isinf, arr)
end

# ── Tracker de loss scale dynamique ─────────────────────────────────────────────
"""
    LossScaleTracker
Gère le loss scale dynamique selon l'algorithme standard :
- Si le gradient est valide N fois de suite → ×2
- Si overflow détecté → ÷2 (et step ignoré)
"""
mutable struct LossScaleTracker
    scale            :: Float32
    growth_interval  :: Int      # steps sans overflow avant de doubler
    step_count       :: Int      # steps valides consécutifs
    n_overflows      :: Int      # compteur d'overflows (pour diagnostics)
    n_growths        :: Int      # compteur de doublements
end

function LossScaleTracker(; scale::Float32 = Float32(2^15),
                            growth_interval::Int = 2000)
    return LossScaleTracker(scale, growth_interval, 0, 0, 0)
end

"""
    update!(tracker, grads_ok) → (Float32, Bool)
Met à jour le loss scale. Retourne (nouveau_scale, step_valide).
Si step_valide = false, le step optimizer doit être ignoré.
"""
function update!(tracker::LossScaleTracker, grads_ok::Bool)
    if !grads_ok
        tracker.scale     = max(1f0, tracker.scale / 2f0)
        tracker.step_count = 0
        tracker.n_overflows += 1
        @warn "Loss scale overflow détecté → scale=$(tracker.scale)"
        return tracker.scale, false
    end

    tracker.step_count += 1
    if tracker.step_count >= tracker.growth_interval
        tracker.scale = min(Float32(2^24), tracker.scale * 2f0)
        tracker.step_count = 0
        tracker.n_growths += 1
    end
    return tracker.scale, true
end

function Base.show(io::IO, t::LossScaleTracker)
    @printf(io, "LossScaleTracker(scale=%.0f, overflows=%d, growths=%d)",
            t.scale, t.n_overflows, t.n_growths)
end

# ── Backward avec loss scaling ──────────────────────────────────────────────────
"""
    backward_with_loss_scaling!(g, loss_sym; ctx_store, mpc, namespace)

Backward pass avec loss scaling dynamique pour MixedPrecData.
1. Initialise le gradient de la loss à `mpc.loss_scale` (au lieu de 1.0)
2. Propage le backward normalement (via GRAD_RULES existantes)
3. Unscale tous les gradients par ÷ loss_scale
4. Détecte les inf/nan → met à jour mpc._ok

Retourne true si le step optimizer est valide (pas d'overflow).
"""
function backward_with_loss_scaling!(g::NeuroGraph, loss_sym::Symbol;
                                      ctx_store::CtxStore = CtxStore(),
                                      mpc::MixedPrecData,
                                      namespace::Symbol = g.active_ns)
    zero_grads!(g; namespace = namespace)

    ln = node(g, loss_sym; namespace = namespace)
    @assert length(ln.value) == 1 "loss doit être scalaire"

    # Gradient initial scalé au lieu de 1.0
    ln.gradient = Backend.ones32(g.device, size(ln.value)...) .* mpc.loss_scale

    # Backward standard (code backward.jl inchangé)
    for out_sym in reverse(topo_order!(g; namespace = namespace))
        !haskey(g.rules[namespace], out_sym) && continue
        rule   = g.rules[namespace][out_sym]
        nd_out = g.nodes[namespace][out_sym]
        nd_out.gradient === nothing && continue
        !haskey(GRAD_RULES, rule.op) &&
            error("❌ Pas de règle backward pour :$(rule.op)")

        ctx         = get(ctx_store, out_sym, Dict{Symbol, Any}())
        inputs_vals = [g.nodes[namespace][s].value for s in rule.inputs]
        grads       = GRAD_RULES[rule.op](g.device, nd_out.gradient, ctx, inputs_vals)

        for (i, in_sym) in enumerate(rule.inputs)
            accum_grad!(g.nodes[namespace][in_sym], grads[i])
        end
        delete!(ctx_store, out_sym)
        nd_out.gradient = nothing
    end

    # Unscale
    grads_ok = _unscale_and_check!(g, mpc.loss_scale; namespace = namespace)
    mpc._ok  = grads_ok
    return grads_ok
end

"""
    _unscale_and_check!(g, scale; namespace) → Bool
Divise tous les gradients par `scale`. Retourne false si inf/nan détecté.
"""
function _unscale_and_check!(g::NeuroGraph, scale::Float32;
                               namespace::Symbol = g.active_ns)
    grads_ok = true
    for (_, nd) in g.nodes[namespace]
        nd.gradient === nothing && continue
        nd.gradient ./= scale
        if has_inf_or_nan(nd.gradient)
            grads_ok = false
        end
    end
    return grads_ok
end

# ── Wrapper complet : boucle d'entraînement mixed-precision ─────────────────
"""
    mixed_precision_step!(g, loss_sym, ctx_store, mpc, tracker; optimizer_fn!, namespace)

Effectue un step complet d'entraînement mixed-precision :
1. Backward avec loss scaling
2. Unscale + détection overflow
3. Met à jour le tracker de loss scale
4. Appelle optimizer_fn!(g) uniquement si les gradients sont valides

`optimizer_fn!` : fonction (g, namespace) → nothing qui appelle adamw_step!
ou l'optimizer de votre choix.

Retourne (scale_valid::Bool, current_scale::Float32).
"""
function mixed_precision_step!(g::NeuroGraph, loss_sym::Symbol,
                                ctx_store::CtxStore,
                                mpc::MixedPrecData,
                                tracker::LossScaleTracker;
                                optimizer_fn!::Function,
                                namespace::Symbol = g.active_ns)
    # 1. Backward scalé
    grads_ok = backward_with_loss_scaling!(g, loss_sym;
                                            ctx_store = ctx_store,
                                            mpc       = mpc,
                                            namespace = namespace)

    # 2. Mise à jour du loss scale
    new_scale, step_valid = update!(tracker, grads_ok)
    mpc.loss_scale = new_scale

    # 3. Optimizer step uniquement si gradients valides
    if step_valid
        optimizer_fn!(g, namespace)
    else
        # Step ignoré — remettre les gradients à zéro
        zero_grads!(g; namespace = namespace)
    end

    return step_valid, new_scale
end
