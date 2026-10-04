# ══════════════════════════════════════════════════════════════════════════════
# gpt2_ops.jl -- ops GPT-2 au niveau du script (aucune modification de src/) :
#   :layernorm_gpt2        y = (x − μ)·inv·γ + β, inv = 1/sqrt(var + ε) (stocké dans aux_data[:rms_inv],
#                          même clé que la RMSNorm de Qwen, pour que la chaîne WIND fonctionne telle quelle)
#   :layernorm_gpt2_frozen même chose avec inv figé à sa valeur propre (FROZEN_LN[sym]) -- variante « normes figées »
#   :gelu_tanh             gelu_new de GPT-2 : 0,5·x·(1 + tanh(√(2/π)(x + 0,044715 x³)))
# Chaque op a sa règle de gradient (pour les attributions) ; à inclure AVANT load_graph!.
# ══════════════════════════════════════════════════════════════════════════════
const FROZEN_LN = Dict{Symbol,Any}()
const GELU_C = Float32(sqrt(2 / pi))

function _ln_fwd!(out_buf, x, γ, β, inv)
    μ = sum(x; dims=2) ./ size(x, 2)
    out_buf .= (x .- μ) .* inv .* reshape(γ, 1, :) .+ reshape(β, 1, :)
    out_buf
end
NeuroDSL.register_op!(:layernorm_gpt2, (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> begin
    x, γ, β = inputs[1], inputs[2], inputs[3]
    eps = Float32(get(attrs, :eps, 1f-5))
    μ = sum(x; dims=2) ./ size(x, 2)
    var = sum((x .- μ) .^ 2; dims=2) ./ size(x, 2)
    inv = vec(1f0 ./ sqrt.(var .+ eps))
    out_node.aux_data[:rms_inv] = inv
    _ln_fwd!(out_buf, x, γ, β, inv)
end)
NeuroDSL.register_op!(:layernorm_gpt2_frozen, (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) ->
    _ln_fwd!(out_buf, inputs[1], inputs[2], inputs[3], FROZEN_LN[out_sym]))
NeuroDSL.register_op!(:gelu_tanh, (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> begin
    x = inputs[1]
    out_buf .= 0.5f0 .* x .* (1f0 .+ tanh.(GELU_C .* (x .+ 0.044715f0 .* x .^ 3)))
    out_buf
end)
for op in (:layernorm_gpt2, :layernorm_gpt2_frozen, :gelu_tanh)
    NeuroDSL.CUSTOM_SHAPE_RULES[op] = (inputs, attrs) -> size(inputs[1])
end
NeuroDSL.CTX_REBUILD[:layernorm_gpt2] = (dev, rule, nd, iv) -> Dict{Symbol,Any}(:inv => nd.aux_data[:rms_inv], :frozen => false)
NeuroDSL.CTX_REBUILD[:layernorm_gpt2_frozen] = (dev, rule, nd, iv) -> Dict{Symbol,Any}(:inv => FROZEN_LN[rule.output], :frozen => true)
NeuroDSL.CTX_REBUILD[:gelu_tanh] = (dev, rule, nd, iv) -> Dict{Symbol,Any}()
function _ln_bwd(dy, ctx, inputs)
    x, γ = inputs[1], inputs[2]; inv = ctx[:inv]; D = size(x, 2)
    xc = x .- sum(x; dims=2) ./ D
    xh = xc .* inv
    g = dy .* reshape(γ, 1, :)
    gc = g .- sum(g; dims=2) ./ D                       # projection hors de la direction constante (centrage)
    dx = ctx[:frozen] ? gc .* inv :                     # dénominateur figé : seul le centrage reste
         inv .* (gc .- xh .* (sum(g .* xh; dims=2) ./ D))
    (dx, vec(sum(dy .* xh; dims=1)), vec(sum(dy; dims=1)))
end
NeuroDSL.GRAD_RULES[:layernorm_gpt2] = (dev, dy, ctx, inputs) -> _ln_bwd(dy, ctx, inputs)
NeuroDSL.GRAD_RULES[:layernorm_gpt2_frozen] = (dev, dy, ctx, inputs) -> _ln_bwd(dy, ctx, inputs)
NeuroDSL.GRAD_RULES[:gelu_tanh] = (dev, dy, ctx, inputs) -> begin
    x = inputs[1]; u = GELU_C .* (x .+ 0.044715f0 .* x .^ 3); t = tanh.(u)
    (dy .* (0.5f0 .* (1f0 .+ t) .+ 0.5f0 .* x .* (1f0 .- t .^ 2) .* GELU_C .* (1f0 .+ 3f0 * 0.044715f0 .* x .^ 2)),)
end
