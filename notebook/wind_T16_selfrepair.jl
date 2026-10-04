# ══════════════════════════════════════════════════════════════════════════════
# WIND-T16 -- les normes comme rétroaction négative (pré-enregistré : notebook/wind_T16_preregistration.md).
# Par prompt et par composante (56 sous-couches, dernier token), six effets d'ablation exacts (vrai modèle) :
#   TE, TE_fn (toutes normes figées), TE_fb (blocs), TE_ff (finale), TE_fa (attention épinglée), TE_fan (les deux).
# Portes G0 (TE = vérité WIND-T9) et G1 (chaque variante sans ablation = R propre). Saute les prompts déjà faits.
# USAGE : WIND_PROMPTS=1,...,50 julia --project=. notebook/wind_T16_selfrepair.jl
# ══════════════════════════════════════════════════════════════════════════════
using NeuroDSL, JSON, LinearAlgebra, Statistics, Printf, Dates
const CKPT = joinpath(@__DIR__, "qwen2.5-1.5b-instruct", "qwen2_neurodsl")
const T8DIR = joinpath(@__DIR__, "wind_data_T8")
const PROMPTS = JSON.parsefile(joinpath(@__DIR__, "wind_T8_prompts.json"))
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const OUT = get(ENV, "WIND_T16_OUT", joinpath(@__DIR__, "wind_T16_selfrepair.json"))
const RES = get(ENV, "WIND_T16_RES", joinpath(@__DIR__, "wind_T16_selfrepair_results.txt"))
const T9REF = merge(JSON.parsefile(joinpath(@__DIR__, "wind_T9_attrib.json")),
                    isfile(joinpath(@__DIR__, "wind_T9_attrib_smoke.json")) ? JSON.parsefile(joinpath(@__DIR__, "wind_T9_attrib_smoke.json")) : Dict())
const L, D, NH, ns = 28, 1536, 12, :qwen2

const FROZEN_RMS = Dict{Symbol,Any}()
NeuroDSL.register_op!(:rmsnorm_frozen, (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> begin
    out_buf .= inputs[1] .* FROZEN_RMS[out_sym] .* reshape(inputs[2], 1, :); out_buf
end)
NeuroDSL.CUSTOM_SHAPE_RULES[:rmsnorm_frozen] = (inputs, attrs) -> size(inputs[1])

mha_sym(l) = Symbol("layer_", l, "_mha_output_out"); mlp_sym(l) = Symbol("layer_", l, "_mlp_out")
pr_sym(l, h) = Symbol("layer_", l, "_mha_pr_h", h)
metas = Dict(p => JSON.parsefile(joinpath(T8DIR, "wind_meta_p$(p).json")) for p in 1:50)
const μMHA = [sum(Float64.(metas[p]["MHA"][l]) for p in 1:50) ./ 50 for l in 1:L]
const μMLP = [sum(Float64.(metas[p]["MLP"][l]) for p in 1:50) ./ 50 for l in 1:L]

out = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T16 auto-réparation par les normes -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ") * "   prompts $(PIDX)")
    dev = NeuroDSL.Backend.CUDADevice()
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev); NeuroDSL.load_graph!(g, ns, CKPT)
    norm_syms = vcat([Symbol("layer_", l, "_norm", j, "_out") for l in 1:L for j in 1:2], [:final_norm_out])
    orig = Dict(s => g.rules[ns][s] for s in norm_syms)
    all(r.op == :rmsnorm for r in values(orig)) || error("règle de norme inattendue")
    function set_frozen!(fset)
        for s in norm_syms
            r = orig[s]; want = s in fset ? :rmsnorm_frozen : :rmsnorm
            g.rules[ns][s].op == want && continue
            NeuroDSL.addrule!(g, want == :rmsnorm_frozen ? NeuroDSL.GraphRule(s, r.inputs, :rmsnorm_frozen; attrs=r.attrs,
                                                                                namespace=ns, atom_type=r.atom_type) : r)
        end
    end
    W_U = NeuroDSL.node(g, :lm_head_W; namespace=ns).value
    ALL = Set(norm_syms); FINAL = Set([:final_norm_out]); BLOCKS = setdiff(ALL, FINAL); NONE = Set{Symbol}()
    VARS = [("TE", NONE, false), ("TE_fn", ALL, false), ("TE_fb", BLOCKS, false), ("TE_ff", FINAL, false),
            ("TE_fa", NONE, true), ("TE_fan", ALL, true)]
    for p in PIDX
        haskey(out, string(p)) && continue
        t0 = time(); pr = PROMPTS[p]; ids = Int.(pr["token_ids"]) .+ 1; n = length(ids)
        set_frozen!(NONE)
        NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.demand!(g, :lm_head_out; namespace=ns)
        clean = NeuroDSL.capture_activations(g, ns)
        for s in norm_syms; FROZEN_RMS[s] = copy(NeuroDSL.node(g, s; namespace=ns).aux_data[:rms_inv]); end
        a_id = pr["answer_id"] + 1; c_id = pr["cf_id"] + 1
        w64 = Float64.(vec(Array(W_U[a_id:a_id, :]))) .- Float64.(vec(Array(W_U[c_id:c_id, :])))
        readout() = dot(Float64.(vec(Array(NeuroDSL.demand!(g, :final_norm_out; namespace=ns)[n:n, :]))), w64)
        Rc = readout()
        pin!(from) = for l2 in from:L, h in 1:NH; NeuroDSL.patch_node!(g, pr_sym(l2, h), clean; namespace=ns); end
        comps = vcat([(mha_sym(l), μMHA[l], l) for l in 1:L], [(mlp_sym(l), μMLP[l], l) for l in 1:L])
        rec = Dict{String,Any}(); g1 = 0.0
        for (nm, fs, pin) in VARS
            set_frozen!(fs)
            # G1 : variante sans ablation = R propre
            NeuroDSL.patch_node!(g, :tok_out, clean; namespace=ns); pin && pin!(1)
            g1 = max(g1, abs(readout() - Rc))
            te = Float64[]
            for (sym, μ, l) in comps
                P = copy(clean[sym]); P[n:n, :] .= reshape(NeuroDSL.Backend.to_device(dev, Float32.(μ)), 1, :)
                NeuroDSL.patch_node!(g, sym, Dict(sym => P); namespace=ns)
                pin && l < L && pin!(l + 1)
                push!(te, Rc - readout())
                NeuroDSL.patch_node!(g, sym, clean; namespace=ns)
            end
            rec[nm] = te
        end
        set_frozen!(NONE); NeuroDSL.patch_node!(g, :tok_out, clean; namespace=ns)
        ref = Float64.(T9REF[string(p)]["dR"])
        g0 = maximum(abs.(rec["TE"] .- ref)) / maximum(abs.(ref))
        ok = g0 < 1e-5 && g1 < 1e-4 * max(1.0, abs(Rc))
        rec["R"] = Rc; rec["G0"] = g0; rec["G1"] = g1; rec["ok"] = ok
        out[string(p)] = rec
        open(OUT, "w") do f; JSON.print(f, out); end
        emit(@sprintf("  p%-3d R %6.2f | Σ|TE| %.2f Σ|TE_fn| %.2f Σ|TE_fa| %.2f | G0 %.1e G1 %.1e | %s | %.0f s", p, Rc,
                      sum(abs, rec["TE"]), sum(abs, rec["TE_fn"]), sum(abs, rec["TE_fa"]), g0, g1, ok ? "retenu" : "EXCLU", time() - t0))
    end
end
println("Écrit : ", OUT)
