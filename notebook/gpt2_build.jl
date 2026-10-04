# ══════════════════════════════════════════════════════════════════════════════
# gpt2_build.jl -- construit GPT-2 small (poids HF réels) dans un NeuroGraph, avec les MÊMES noms de nœuds que Qwen
# (tok_out, layer_l_norm1_out, layer_l_mha_*, layer_l_res1, layer_l_norm2_out, layer_l_mlp_out, layer_l_out,
# final_norm_out, lm_head_out) pour que la chaîne WIND s'applique telle quelle. Sauvegarde gpt2/gpt2_neurodsl.
# Différences avec Qwen : LayerNorm avec centrage et biais, plongements de position appris, biais partout,
# MLP SANS porte (c_fc -> gelu_new -> c_proj), poids Conv1D HF stockés (in, out) -> transposés.
# USAGE : julia --project=. notebook/gpt2_build.jl
# ══════════════════════════════════════════════════════════════════════════════
using NeuroDSL, JSON, CUDA
include(joinpath(@__DIR__, "safetensors_reader.jl"))
include(joinpath(@__DIR__, "gpt2_ops.jl"))
const MD = joinpath(@__DIR__, "gpt2")
const C = JSON.parsefile(joinpath(MD, "config.json"))
const DIM, NL, NH, VOC, NPOS = C["n_embd"], C["n_layer"], C["n_head"], C["vocab_size"], C["n_positions"]
const HID, EPS = 4DIM, Float32(C["layer_norm_epsilon"])
st = open_any_safetensors(MD)
dev = NeuroDSL.Backend.CUDADevice(); ns = :gpt2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:8); atom_type=NeuroDSL.Datom, namespace=ns)
P(name, v) = NeuroDSL.set!(g, name, v; is_param=true, namespace=ns)
R(out, ins, op; attrs=Dict{Symbol,Any}()) = NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out, ins, op; attrs=attrs, namespace=ns))
T(n) = read_tensor(st, n)
function lnorm!(x, prefix, hf)
    P(Symbol(prefix, :_gamma), T(hf * ".weight")); P(Symbol(prefix, :_beta), T(hf * ".bias"))
    R(Symbol(prefix, :_out), [x, Symbol(prefix, :_gamma), Symbol(prefix, :_beta)], :layernorm_gpt2; attrs=Dict{Symbol,Any}(:eps => EPS))
    Symbol(prefix, :_out)
end
# plongements : tok_out = wte[ids] + wpe[pos]
P(:tokE_E, T("wte.weight")); P(:posE_E, T("wpe.weight"))
R(:tokE_out, [:tokE_E, :token_ids], :embedding); R(:posE_out, [:posE_E, :pos_ids], :embedding)
R(:tok_out, [:tokE_out, :posE_out], :add)
cur = :tok_out
for l in 1:NL
    pf = "layer_$(l)"; hf = "h.$(l-1)"
    xn1 = lnorm!(cur, Symbol(pf, :_norm1), hf * ".ln_1")
    ao = NeuroDSL.MultiHeadAttention(DIM, NH; batched=true, qkv_bias=true)(g, xn1, Symbol(pf, :_mha); namespace=ns)
    Wqkv = permutedims(T(hf * ".attn.c_attn.weight")); bqkv = T(hf * ".attn.c_attn.bias")
    for (i, s) in enumerate(("q", "k", "v"))
        r = (i-1)*DIM+1:i*DIM
        P(Symbol(pf, "_mha_", s, "_W"), Wqkv[r, :]); P(Symbol(pf, "_mha_", s, "_b"), bqkv[r])
    end
    # projection de sortie AVEC biais (le constructeur générique n'en met pas) : même nœud, règle :linear
    P(Symbol(pf, :_mha_output_W), permutedims(T(hf * ".attn.c_proj.weight"))); P(Symbol(pf, :_mha_output_b), T(hf * ".attn.c_proj.bias"))
    R(ao, [Symbol(pf, :_mha_concat), Symbol(pf, :_mha_output_W), Symbol(pf, :_mha_output_b)], :linear)
    r1 = Symbol(pf, :_res1); R(r1, [cur, ao], :add)
    xn2 = lnorm!(r1, Symbol(pf, :_norm2), hf * ".ln_2")
    P(Symbol(pf, :_fc_W), permutedims(T(hf * ".mlp.c_fc.weight"))); P(Symbol(pf, :_fc_b), T(hf * ".mlp.c_fc.bias"))
    R(Symbol(pf, :_fc_out), [xn2, Symbol(pf, :_fc_W), Symbol(pf, :_fc_b)], :linear)
    R(Symbol(pf, :_gelu), [Symbol(pf, :_fc_out)], :gelu_tanh)
    P(Symbol(pf, :_mlp_W), permutedims(T(hf * ".mlp.c_proj.weight"))); P(Symbol(pf, :_mlp_b), T(hf * ".mlp.c_proj.bias"))
    R(Symbol(pf, :_mlp_out), [Symbol(pf, :_gelu), Symbol(pf, :_mlp_W), Symbol(pf, :_mlp_b)], :linear)
    R(Symbol(pf, :_out), [r1, Symbol(pf, :_mlp_out)], :add)
    global cur = Symbol(pf, :_out)
end
fn = lnorm!(cur, :final_norm, "ln_f")
P(:lm_head_W, T("wte.weight"))
R(:lm_head_out, [fn, :lm_head_W], :matmul; attrs=Dict{Symbol,Any}(:trans_b => true))
GC.gc(); CUDA.reclaim()
NeuroDSL.save_graph!(g, ns, joinpath(MD, "gpt2_neurodsl"))
println("OK -> ", joinpath(MD, "gpt2_neurodsl"), ".json/.bin  (", NL, " couches, dim ", DIM, ")")
