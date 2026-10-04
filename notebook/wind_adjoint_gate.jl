# ══════════════════════════════════════════════════════════════════════════════
# WIND-1 -- PORTE G5 : ADJOINT (chemin totalement indépendant de la collecte)
#
# La collecte (wind_jacobian_collect.jl) matérialise J_k par différences finies AVANT.
# Ici : le moteur NeuroDSL rétro-propage (mode inverse, GRAD_RULES) la cross-entropy
# jusqu'à chaque sortie de couche ; le cotangent au dernier token, g_k^bwd, est capturé
# (op identité insérée sur la branche MLP : cotangent à la branche = cotangent au join
# layer_k_out). Prédiction indépendante : g_k^pred = J_kᵀ J_{k+1}ᵀ ··· J_27ᵀ g_28^bwd
# (même graine g_28, produits des matrices DF). Seuil pré-enregistré : erreur relative
# < 1e-2 pour tout k ≥ 1.
#
# USAGE : WIND_OUTDIR=<dossier> WIND_PROMPT=5 julia --project=. notebook/wind_adjoint_gate.jl
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, LinearAlgebra, Statistics

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const OUTDIR    = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data"))
const PIDX      = parse(Int, get(ENV, "WIND_PROMPT", "5"))
const RES       = joinpath(@__DIR__, "wind_adjoint_gate_results.txt")

const CAPTURED = Dict{Symbol,Array{Float32}}()
function ensure_cap_op!(branch::Symbol)
    op = Symbol("windcap_", branch)
    haskey(NeuroDSL.CUSTOM_OPS, op) && return op
    NeuroDSL.register_op!(op,
        (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> (out_buf .= inputs[1]))
    NeuroDSL.CUSTOM_SHAPE_RULES[op] = (inputs, attrs) -> size(inputs[1])
    NeuroDSL.GRAD_RULES[op] = (dev, dy, ctx, inputs) -> begin
        CAPTURED[branch] = copy(Array(dy))
        return (dy,)
    end
    return op
end
reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-1 -- PORTE G5 (ADJOINT) : backward moteur vs produit des Jacobiennes DF")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    dev = NeuroDSL.Backend.CUDADevice()
    ns, L, D = :qwen2, 28, 1536
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.load_graph!(g, ns, CKPT)
    reclaim()
    keep = Set(vcat([Symbol("layer_", i, "_norm1_gamma") for i in 1:L],
                    [Symbol("layer_", i, "_norm2_gamma") for i in 1:L], [:final_norm_gamma]))
    for (s, nd) in g.nodes[ns]
        nd.is_param && !(s in keep) && (nd.is_param = false)
    end
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:ce_loss, [:lm_head_out, :labels], :cross_entropy; namespace=ns))
    for i in 1:L
        join_sym = Symbol("layer_", i, "_out"); bsym = Symbol("layer_", i, "_mlp_out")
        r = g.rules[ns][join_sym]
        r.inputs[2] == bsym || error("$join_sym : branche inattendue")
        cs = Symbol("windcap_", bsym)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(cs, [bsym], ensure_cap_op!(bsym); namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(join_sym, [r.inputs[1], cs], r.op;
                                                attrs=r.attrs, namespace=ns, atom_type=r.atom_type))
    end
    prompts = JSON.parsefile(joinpath(@__DIR__, "qwen_sweep_prompts.json"))
    ids = Int.(prompts[PIDX]["token_ids"]) .+ 1; n = length(ids)
    emit(@sprintf("Prompt %d : %s (%d tokens)", PIDX, repr(prompts[PIDX]["prompt"]), n))
    NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :labels, vcat(ids[2:end], ids[end]); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    # porte : le recâblage (op identité) ne change pas le forward
    meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(PIDX).json"))
    NeuroDSL.demand!(g, :ce_loss; namespace=ns)
    x28 = Float64.(vec(Array(NeuroDSL.demand!(g, :layer_28_out; namespace=ns)[n:n, :])))
    emit(@sprintf("Porte forward : x_28 (recâblé) vs x_28 (collecte) : err rel = %.2e",
                  norm(x28 .- Float64.(meta["X"][end])) / norm(x28)))
    empty!(CAPTURED)
    NeuroDSL.backward_graph!(g, :ce_loss; namespace=ns)
    length(CAPTURED) == L || error("capturés : $(length(CAPTURED)) != $L")
    gb = [Float64.(vec(CAPTURED[Symbol("layer_", k, "_mlp_out")][n, :])) for k in 1:L]

    Js = Array{Float32}(undef, D, D, L)
    open(joinpath(OUTDIR, "wind_J_p$(PIDX)_T.bin"), "r") do f; read!(f, Js); end
    pred = gb[L]
    worst = 0.0
    emit(@sprintf("\n%4s %14s %14s %12s %10s", "k", "||g_k bwd||", "||g_k pred||", "err rel", "cos"))
    for k in L-1:-1:1
        pred = Float64.(Js[:, :, k+1])' * pred        # J_k = ∂x_{k+1}/∂x_k  -> Js[:,:,k+1]
        e = norm(pred .- gb[k]) / norm(gb[k]); worst = max(worst, e)
        emit(@sprintf("%4d %14.6e %14.6e %12.3e %10.6f", k, norm(gb[k]), norm(pred), e,
                      dot(pred, gb[k]) / (norm(pred) * norm(gb[k]))))
    end
    emit(@sprintf("\nG5 : pire erreur relative = %.3e (seuil 1e-2) -> %s", worst, worst < 1e-2 ? "OK" : "ÉCHEC"))
    reclaim()
end
println("\nÉcrit : ", RES)
