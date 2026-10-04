# ══════════════════════════════════════════════════════════════════════════════
# gemma_build.jl -- construit Gemma-2-2B (poids HF réels) dans un NeuroGraph, en posant DIRECTEMENT les vrais poids
# (aucune initialisation aléatoire : économise la mémoire GPU). Noms de nœuds alignés sur Qwen pour la chaîne WIND :
#   tok_out, layer_l_norm1_out, layer_l_mha_pr_h{h}, layer_l_mha_output_out (o_proj brut),
#   layer_l_attnpost_out (norme post-attention = ce qui est AJOUTÉ au résiduel), layer_l_res1,
#   layer_l_norm2_out (pré-MLP), layer_l_gate/_up/_geglu, layer_l_mlp_out (down_proj brut),
#   layer_l_mlppost_out (norme post-MLP = ce qui est ajouté), layer_l_out, final_norm_out,
#   lm_head_raw (logits avant plafonnement), lm_head_out (logits plafonnés, comme transformers).
# Particularités : RMSNorm (1 + w) repliée en γ ; 4 normes par bloc ; têtes de dimension 256 (≠ d/n_heads) ; GQA 8/4 ;
# RoPE θ = 10000 ; scores × 256^-0,5 puis plafonnés à 50 ; plongement × sqrt(d) ; tête de sortie = plongement (liée) ;
# logits plafonnés à 30. Fenêtre glissante (4096) sans effet sur des prompts courts.
# USAGE : julia --project=. notebook/gemma_build.jl
# ══════════════════════════════════════════════════════════════════════════════
using NeuroDSL, JSON, CUDA
include(joinpath(@__DIR__, "safetensors_reader.jl"))
include(joinpath(@__DIR__, "gemma_ops.jl"))
const MD = joinpath(@__DIR__, "gemma-2-2b")
const C = JSON.parsefile(joinpath(MD, "config.json"))
const DIM, NL, NH, NKV, DH = C["hidden_size"], C["num_hidden_layers"], C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"]
const EPS, THETA = Float32(C["rms_norm_eps"]), Float32(get(C, "rope_theta", 10000.0))
const SCALE, ACAP, FCAP = Float32(C["query_pre_attn_scalar"])^-0.5f0, C["attn_logit_softcapping"], C["final_logit_softcapping"]
println("Gemma-2 : dim $DIM, $NL couches, $NH têtes ($NKV KV) de dim $DH, eps $EPS, θ $THETA, échelle $SCALE, plafonds $ACAP / $FCAP")
st = open_any_safetensors(MD)
dev = NeuroDSL.Backend.CUDADevice(); ns = :gemma2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:8); atom_type=NeuroDSL.Datom, namespace=ns)
P(name, v) = NeuroDSL.set!(g, name, v; is_param=true, namespace=ns)
R(out, ins, op; attrs=Dict{Symbol,Any}()) = NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out, ins, op; attrs=attrs, namespace=ns))
T(n) = read_tensor(st, n)
TB = Dict{Symbol,Any}(:trans_b => true)
function rmsn!(x, prefix, hf)
    P(Symbol(prefix, :_gamma), 1f0 .+ Float32.(T(hf)))
    R(Symbol(prefix, :_out), [x, Symbol(prefix, :_gamma)], :rmsnorm; attrs=Dict{Symbol,Any}(:eps => EPS))
    Symbol(prefix, :_out)
end
lin!(x, out, W) = (P(Symbol(out, :_W), W); R(Symbol(out, :_out), [x, Symbol(out, :_W)], :matmul; attrs=TB); Symbol(out, :_out))

P(:tokE_E, T("model.embed_tokens.weight"))
R(:tokE_out, [:tokE_E, :token_ids], :embedding)
R(:tok_out, [:tokE_out], :scalar_mul; attrs=Dict{Symbol,Any}(:c => Float32(sqrt(DIM))))
cur = :tok_out
for l in 1:NL
    pf = "layer_$(l)"; hf = "model.layers.$(l-1)"; mp = Symbol(pf, :_mha)
    xn1 = rmsn!(cur, Symbol(pf, :_norm1), hf * ".input_layernorm.weight")
    q = lin!(xn1, Symbol(mp, :_q), T(hf * ".self_attn.q_proj.weight"))
    k = lin!(xn1, Symbol(mp, :_k), T(hf * ".self_attn.k_proj.weight"))
    v = lin!(xn1, Symbol(mp, :_v), T(hf * ".self_attn.v_proj.weight"))
    qh = Symbol[]; kh = Symbol[]; vh = Symbol[]
    for h in 1:NH
        s = Symbol(mp, :_q_h, h); R(s, [q], :view_cols; attrs=Dict{Symbol,Any}(:start_col => (h-1)*DH+1, :end_col => h*DH))
        R(Symbol(s, :_rope), [s], :rope; attrs=Dict{Symbol,Any}(:theta => THETA)); push!(qh, Symbol(s, :_rope))
    end
    for h in 1:NKV
        s = Symbol(mp, :_k_h, h); R(s, [k], :view_cols; attrs=Dict{Symbol,Any}(:start_col => (h-1)*DH+1, :end_col => h*DH))
        R(Symbol(s, :_rope), [s], :rope; attrs=Dict{Symbol,Any}(:theta => THETA)); push!(kh, Symbol(s, :_rope))
        s2 = Symbol(mp, :_v_h, h); R(s2, [v], :view_cols; attrs=Dict{Symbol,Any}(:start_col => (h-1)*DH+1, :end_col => h*DH)); push!(vh, s2)
    end
    gs = NH ÷ NKV
    khr = [kh[(h-1)÷gs+1] for h in 1:NH]; vhr = [vh[(h-1)÷gs+1] for h in 1:NH]
    R(Symbol(mp, :_sc3), vcat(qh, khr), :batched_qk; attrs=Dict{Symbol,Any}(:d_head => DH))
    prs = Symbol[]
    for h in 1:NH
        sc = Symbol(mp, :_sc_h, h); sk = Symbol(mp, :_sk_h, h); pr = Symbol(mp, :_pr_h, h)
        R(sc, [Symbol(mp, :_sc3)], :head_view; attrs=Dict{Symbol,Any}(:head => h))
        R(sk, [sc], :softcap_mask; attrs=Dict{Symbol,Any}(:scale => SCALE, :cap => Float32(ACAP)))
        R(pr, [sk], :softmax); push!(prs, pr)
    end
    R(Symbol(mp, :_ao3), vcat(prs, vhr), :batched_pv; attrs=Dict{Symbol,Any}(:d_head => DH))
    aos = Symbol[]
    for h in 1:NH
        a = Symbol(mp, :_ao_h, h); R(a, [Symbol(mp, :_ao3)], :head_view; attrs=Dict{Symbol,Any}(:head => h)); push!(aos, a)
    end
    R(Symbol(mp, :_concat), aos, :hcat_heads)
    ao = lin!(Symbol(mp, :_concat), Symbol(mp, :_output), T(hf * ".self_attn.o_proj.weight"))
    ap = rmsn!(ao, Symbol(pf, :_attnpost), hf * ".post_attention_layernorm.weight")
    r1 = Symbol(pf, :_res1); R(r1, [cur, ap], :add)
    xn2 = rmsn!(r1, Symbol(pf, :_norm2), hf * ".pre_feedforward_layernorm.weight")
    P(Symbol(pf, :_mlp_w1), T(hf * ".mlp.gate_proj.weight")); P(Symbol(pf, :_mlp_w3), T(hf * ".mlp.up_proj.weight"))
    P(Symbol(pf, :_mlp_w2), T(hf * ".mlp.down_proj.weight"))
    R(Symbol(pf, :_gate), [xn2, Symbol(pf, :_mlp_w1)], :matmul; attrs=TB)
    R(Symbol(pf, :_up), [xn2, Symbol(pf, :_mlp_w3)], :matmul; attrs=TB)
    R(Symbol(pf, :_geglu), [Symbol(pf, :_gate), Symbol(pf, :_up)], :geglu)
    R(Symbol(pf, :_mlp_out), [Symbol(pf, :_geglu), Symbol(pf, :_mlp_w2)], :matmul; attrs=TB)
    mpo = rmsn!(Symbol(pf, :_mlp_out), Symbol(pf, :_mlppost), hf * ".post_feedforward_layernorm.weight")
    R(Symbol(pf, :_out), [r1, mpo], :add)
    global cur = Symbol(pf, :_out)
    l % 4 == 0 && (GC.gc(); CUDA.reclaim())
end
fn = rmsn!(cur, :final_norm, "model.norm.weight")
R(:lm_head_raw, [fn, :tokE_E], :matmul; attrs=TB)
R(:lm_head_out, [:lm_head_raw], :logit_softcap; attrs=Dict{Symbol,Any}(:cap => Float32(FCAP)))
GC.gc(); CUDA.reclaim()
NeuroDSL.save_graph!(g, ns, joinpath(MD, "gemma2_neurodsl"))
println("OK -> ", joinpath(MD, "gemma2_neurodsl"), ".json/.bin")
