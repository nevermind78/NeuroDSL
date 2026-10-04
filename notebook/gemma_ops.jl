# ══════════════════════════════════════════════════════════════════════════════
# gemma_ops.jl -- ops Gemma-2 au niveau du script (aucune modification de src/), avec règles de gradient :
#   :softcap_mask   scores -> cap·tanh(scores·scale/cap), puis masque causal (-Inf au-dessus de la diagonale)
#                   (ordre de transformers : mise à l'échelle, plafonnement, PUIS masque)
#   :geglu          (gate, up) -> gelu_tanh(gate) ⊙ up   (hidden_activation = gelu_pytorch_tanh)
#   :scalar_mul     x -> c·x                              (plongement × sqrt(d))
#   :logit_softcap  x -> cap·tanh(x/cap)                  (plafonnement final des logits)
# Les RMSNorm de Gemma (1 + w) utilisent l'op :rmsnorm standard avec γ = 1 + w replié au chargement.
# ══════════════════════════════════════════════════════════════════════════════
const GEMMA_GELU_C = Float32(sqrt(2 / pi))
const _GEMMA_MASKS = Dict{Int,Any}()
_causal_keep(dev, n) = get!(_GEMMA_MASKS, n) do
    M = [j <= i for i in 1:n, j in 1:n]
    NeuroDSL.Backend.to_device(dev, M)
end
_gelu_t(x) = 0.5f0 .* x .* (1f0 .+ tanh.(GEMMA_GELU_C .* (x .+ 0.044715f0 .* x .^ 3)))
function _dgelu_t(x)
    t = tanh.(GEMMA_GELU_C .* (x .+ 0.044715f0 .* x .^ 3))
    0.5f0 .* (1f0 .+ t) .+ 0.5f0 .* x .* (1f0 .- t .^ 2) .* GEMMA_GELU_C .* (1f0 .+ 3f0 * 0.044715f0 .* x .^ 2)
end

NeuroDSL.register_op!(:softcap_mask, (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> begin
    s = Float32(attrs[:scale]); cap = Float32(attrs[:cap]); x = inputs[1]
    out_buf .= cap .* tanh.(x .* (s / cap))
    out_buf .= ifelse.(_causal_keep(dev, size(x, 1)), out_buf, -Inf32)
    out_buf
end)
NeuroDSL.GRAD_RULES[:softcap_mask] = (dev, dy, ctx, inputs) -> begin
    x = inputs[1]; s = Float32(ctx[:scale]); cap = Float32(ctx[:cap])
    t = tanh.(x .* (s / cap))
    (ifelse.(_causal_keep(dev, size(x, 1)), dy .* s .* (1f0 .- t .^ 2), 0f0),)
end
NeuroDSL.CTX_REBUILD[:softcap_mask] = (dev, rule, nd, iv) -> Dict{Symbol,Any}(:scale => rule.attrs[:scale], :cap => rule.attrs[:cap])

NeuroDSL.register_op!(:geglu, (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> begin
    out_buf .= _gelu_t(inputs[1]) .* inputs[2]; out_buf
end)
NeuroDSL.GRAD_RULES[:geglu] = (dev, dy, ctx, inputs) -> (dy .* inputs[2] .* _dgelu_t(inputs[1]), dy .* _gelu_t(inputs[1]))
NeuroDSL.CTX_REBUILD[:geglu] = (dev, rule, nd, iv) -> Dict{Symbol,Any}()

NeuroDSL.register_op!(:scalar_mul, (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> begin
    out_buf .= Float32(attrs[:c]) .* inputs[1]; out_buf
end)
NeuroDSL.GRAD_RULES[:scalar_mul] = (dev, dy, ctx, inputs) -> (Float32(ctx[:c]) .* dy,)
NeuroDSL.CTX_REBUILD[:scalar_mul] = (dev, rule, nd, iv) -> Dict{Symbol,Any}(:c => rule.attrs[:c])

NeuroDSL.register_op!(:logit_softcap, (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> begin
    cap = Float32(attrs[:cap]); out_buf .= cap .* tanh.(inputs[1] ./ cap); out_buf
end)
NeuroDSL.GRAD_RULES[:logit_softcap] = (dev, dy, ctx, inputs) -> begin
    cap = Float32(ctx[:cap]); (dy .* (1f0 .- tanh.(inputs[1] ./ cap) .^ 2),)
end
NeuroDSL.CTX_REBUILD[:logit_softcap] = (dev, rule, nd, iv) -> Dict{Symbol,Any}(:cap => rule.attrs[:cap])

for op in (:softcap_mask, :geglu, :scalar_mul, :logit_softcap)
    NeuroDSL.CUSTOM_SHAPE_RULES[op] = (inputs, attrs) -> size(inputs[1])
end
