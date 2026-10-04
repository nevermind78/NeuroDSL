# ══════════════════════════════════════════════════════════════════════════════
# WIND-T13 (GPU) -- correctif « norme du MLP vivante » (pré-enregistré : notebook/wind_T13_preregistration.md).
# Par prompt et par bloc k = 1..27 (couche l = k+1), 3 JVP centrées exactes (outils WIND-3) :
#   a1 = A₁x̂ (attention, probabilités épinglées, norme 1 gelée), b1 = MQ₂ a1 (MLP, norme 2 vivante),
#   mh = Mĥ (MLP, norme 2 gelée).  Porte G2 : JVP directes des blocs AA et AM (blocs 2, 8, 14, 20, 26).
# Gradients exacts de R pour AN (tout gelé), AA (normes 1 + finale gelées), AM (normes 2 + finale gelées),
# probabilités d'attention gelées dans les trois -> attributions des 56 sous-couches (vérité terrain : WIND-T9).
# USAGE : WIND_PROMPTS=1,...,50 julia --project=. notebook/wind_T13_factors.jl
# ══════════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "wind3_common.jl"))
using Dates

const T8DIR = joinpath(@__DIR__, "wind_data_T8")
const PROMPTS = JSON.parsefile(joinpath(@__DIR__, "wind_T8_prompts.json"))
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const OUT = get(ENV, "WIND_T13_OUT", joinpath(@__DIR__, "wind_T13_factors.json"))
const RES = get(ENV, "WIND_T13_RES", joinpath(@__DIR__, "wind_T13_factors_results.txt"))
const EPS_REL = 5e-3
const CHECK_BLOCKS = [2, 8, 14, 20, 26]
mlp_sym(l) = Symbol("layer_", l, "_mlp_out")
n1_sym(l) = Symbol("layer_", l, "_norm1_out"); n2_sym(l) = Symbol("layer_", l, "_norm2_out")
const F_ALL = Set(NORM_SYMS); const F_NONE = Set{Symbol}()
const F_AA = Set(vcat([n1_sym(l) for l in 1:L], [:final_norm_out]))
const F_AM = Set(vcat([n2_sym(l) for l in 1:L], [:final_norm_out]))

metas = Dict(p => JSON.parsefile(joinpath(T8DIR, "wind_meta_p$(p).json")) for p in 1:50)
const μMHA = [sum(Float64.(metas[p]["MHA"][l]) for p in 1:50) ./ 50 for l in 1:L]
const μMLP = [sum(Float64.(metas[p]["MLP"][l]) for p in 1:50) ./ 50 for l in 1:L]
const T9 = merge(JSON.parsefile(joinpath(@__DIR__, "wind_T9_attrib.json")),
                 isfile(joinpath(@__DIR__, "wind_T9_attrib_smoke.json")) ? JSON.parsefile(joinpath(@__DIR__, "wind_T9_attrib_smoke.json")) : Dict())

function set_frozen!(g, orig, fset)
    for s in NORM_SYMS
        r = orig[s]; want = s in fset ? :rmsnorm_frozen : :rmsnorm
        g.rules[NS][s].op == want && continue
        NeuroDSL.addrule!(g, want == :rmsnorm_frozen ? NeuroDSL.GraphRule(s, r.inputs, :rmsnorm_frozen; attrs=r.attrs,
                                                                            namespace=NS, atom_type=r.atom_type) : r)
    end
end
"""JVP centrée : perturbe la ligne n de `src` le long de v (pas eps), lit la somme des nœuds `reads` (ligne n)."""
function jvp!(g, cache, src, reads, v, eps; pin=0)
    n = READ[:n]; r = Vector{Vector{Float64}}(undef, 2)
    for (i, sg) in enumerate((1.0, -1.0))
        P = zeros(Float32, n, D); P[n, :] .= Float32.(sg * eps .* v)
        NeuroDSL.patch_node!(g, src, Dict(src => cache[src] .+ NeuroDSL.Backend.to_device(g.device, P)); namespace=NS)
        if pin > 0; for h in 1:NH; NeuroDSL.patch_node!(g, pr_sym(pin, h), cache; namespace=NS); end; end
        r[i] = sum(Float64.(vec(Array(NeuroDSL.demand!(g, s; namespace=NS)[n:n, :]))) for s in reads)
    end
    NeuroDSL.patch_node!(g, src, cache; namespace=NS)
    if pin > 0; for h in 1:NH; NeuroDSL.patch_node!(g, pr_sym(pin, h), cache; namespace=NS); end; end
    (r[1] .- r[2]) ./ (2eps)
end
function grads_subset!(g, orig, fset)
    SG_ON[] = true; set_frozen!(g, orig, fset)
    NeuroDSL.demand!(g, :w3_R; namespace=NS)
    NeuroDSL.backward_graph!(g, :w3_R; namespace=NS, prune_frozen=true)
    Gx = [Float64.(Array(NeuroDSL.node(g, zx(k); namespace=NS).gradient)) for k in 0:L]
    Gr = [Float64.(Array(NeuroDSL.node(g, zr(l); namespace=NS).gradient)) for l in 1:L]
    SG_ON[] = false
    Gx, Gr
end

out = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T13 (GPU) facteurs exacts + gradients AA/AM/AN -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ") * "   prompts $(PIDX)")
    g, dev, orig, W_U = load_instrumented()
    for p in PIDX
        haskey(out, string(p)) && continue
        t0 = time()
        pr = PROMPTS[p]; ids = Int.(pr["token_ids"]) .+ 1
        set_frozen!(g, orig, F_NONE); SG_ON[] = false
        n = set_prompt!(g, ids)
        NeuroDSL.demand!(g, :lm_head_out; namespace=NS)
        a_id = pr["answer_id"] + 1; c_id = pr["cf_id"] + 1
        w64 = Float64.(vec(Array(W_U[a_id:a_id, :]))) .- Float64.(vec(Array(W_U[c_id:c_id, :])))
        set_readout!(dev, w64)
        cache = capture_rule_nodes(g)
        for s in NORM_SYMS; FROZEN_RMS[s] = copy(NeuroDSL.node(g, s; namespace=NS).aux_data[:rms_inv]); end
        row(s) = Float64.(vec(Array(cache[s][n:n, :])))
        rinv(s) = Float64(Array(FROZEN_RMS[s])[n])
        # ── facteurs exacts par bloc ───────────────────────────────────────────
        A1 = zeros(D, 27); B1 = zeros(D, 27); MH = zeros(D, 27); s1 = zeros(27); s2 = zeros(27)
        for k in 1:27
            l = k + 1
            x = row(res_sym(k)); h = row(r1_sym(l)); xh = x ./ norm(x); hh = h ./ norm(h)
            s1[k] = sum(abs2, x) * rinv(n1_sym(l))^2 / D; s2[k] = sum(abs2, h) * rinv(n2_sym(l))^2 / D
            set_frozen!(g, orig, F_ALL)
            A1[:, k] = jvp!(g, cache, res_sym(k), (mha_sym(l),), xh, EPS_REL * norm(x); pin=l)
            MH[:, k] = jvp!(g, cache, r1_sym(l), (mlp_sym(l),), hh, EPS_REL * norm(h))
            set_frozen!(g, orig, F_NONE)
            na = norm(A1[:, k])
            B1[:, k] = na .* jvp!(g, cache, r1_sym(l), (mlp_sym(l),), A1[:, k] ./ na, EPS_REL * norm(h))
        end
        # ── porte G2 : JVP directes des blocs AA et AM ─────────────────────────
        rng = MersenneTwister(4000 + p); chk = Any[]
        for k in CHECK_BLOCKS, _ in 1:2
            l = k + 1; x = row(res_sym(k)); v = randn(rng, D); v ./= norm(v)
            set_frozen!(g, orig, F_AA); jaa = v .+ jvp!(g, cache, res_sym(k), (mha_sym(l), mlp_sym(l)), v, EPS_REL * norm(x); pin=l)
            set_frozen!(g, orig, F_AM); jam = v .+ jvp!(g, cache, res_sym(k), (mha_sym(l), mlp_sym(l)), v, EPS_REL * norm(x); pin=l)
            push!(chk, Dict("k" => k, "v" => v, "JAA_v" => jaa, "JAM_v" => jam))
        end
        set_frozen!(g, orig, F_NONE); NeuroDSL.demand!(g, :w3_R; namespace=NS)
        # ── gradients et attributions ──────────────────────────────────────────
        oM = [row(mha_sym(l)) for l in 1:L]; oP = [row(mlp_sym(l)) for l in 1:L]
        dM = oM .- μMHA; dP = oP .- μMLP
        att = Dict{String,Any}(); glast = Dict{String,Any}()
        for (nm, fs) in (("AN", F_ALL), ("AA", F_AA), ("AM", F_AM))
            Gx, Gr = grads_subset!(g, orig, fs)
            att[nm] = vcat([dot(Gr[l][n, :], dM[l]) for l in 1:L], [dot(Gx[l+1][n, :], dP[l]) for l in 1:L])
            glast[nm] = [Gx[k+1][n, :] for k in 0:L]
        end
        set_frozen!(g, orig, F_NONE); NeuroDSL.demand!(g, :w3_R; namespace=NS)
        g4 = maximum(abs.(att["AN"] .- Float64.(T9[string(p)]["aAN"]))) / maximum(abs.(Float64.(T9[string(p)]["aAN"])))
        # binaire : A1, B1, MH (D×27 chacun) ; gradients au dernier token AA, AM (D×29) ; contrôles (v, JAA v, JAM v) × 10
        open(joinpath(T8DIR, "wind_T13_factors_p$(p).bin"), "w") do f
            write(f, A1); write(f, B1); write(f, MH)
            for nm in ("AA", "AM"); write(f, reduce(hcat, glast[nm])); end
            for c in chk; write(f, c["v"]); write(f, c["JAA_v"]); write(f, c["JAM_v"]); end
        end
        out[string(p)] = Dict("s1" => s1, "s2" => s2, "check_blocks" => [c["k"] for c in chk], "attr" => att, "gate_G4" => g4)
        open(OUT, "w") do f; JSON.print(f, out); end
        emit(@sprintf("  p%-3d facteurs + %d contrôles + gradients | G4 (AN vs WIND-T9) %.1e | %.0f s", p, length(chk), g4, time() - t0))
    end
end
println("Écrit : ", OUT)
