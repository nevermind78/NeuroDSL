# ══════════════════════════════════════════════════════════════════════════════
# WIND-T9 -- l'effondrement fausse-t-il de vraies attributions ? (pré-enregistré : notebook/wind_T9_preregistration.md)
# Par prompt : attributions linéaires a^V_c = ⟨∂R/∂(point d'écriture de c), o_c − μ_c⟩ pour V ∈ {T, A, AN}
# (backward exact, outils WIND-3) et vérité terrain ΔR_c = R(propre) − R(o_c → μ_c) (vrai modèle), pour les 56
# sous-couches au dernier token. R = logit(réponse) − logit(contrefactuel). Portes GR, GV, GA.
#
# USAGE : WIND_PROMPTS=1,...,50 julia --project=. notebook/wind_T9_attrib.jl
# ══════════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "wind3_common.jl"))
using Dates

const T8DIR = joinpath(@__DIR__, "wind_data_T8")
const PROMPTS = JSON.parsefile(joinpath(@__DIR__, "wind_T8_prompts.json"))
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const OUT = get(ENV, "WIND_T9_OUT", joinpath(@__DIR__, "wind_T9_attrib.json"))
const RES = get(ENV, "WIND_T9_RES", joinpath(@__DIR__, "wind_T9_attrib_results.txt"))
const EPS = 0.05
mlp_sym(l) = Symbol("layer_", l, "_mlp_out")

metas = Dict(p => JSON.parsefile(joinpath(T8DIR, "wind_meta_p$(p).json")) for p in 1:50)
const μMHA = [sum(Float64.(metas[p]["MHA"][l]) for p in 1:50) ./ 50 for l in 1:L]
const μMLP = [sum(Float64.(metas[p]["MLP"][l]) for p in 1:50) ./ 50 for l in 1:L]

out = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T9 -- attributions vs ablations -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ") * "   prompts $(PIDX)")
    g, dev, orig, W_U = load_instrumented()
    for p in PIDX
        haskey(out, string(p)) && continue
        t0 = time()
        pr = PROMPTS[p]; ids = Int.(pr["token_ids"]) .+ 1
        meta = get(metas, p, nothing)
        swap_norms!(g, orig, false); SG_ON[] = false
        n = set_prompt!(g, ids)
        lg = Float64.(Array(NeuroDSL.demand!(g, :lm_head_out; namespace=NS))[n, :])
        a_id = pr["answer_id"] + 1; c_id = pr["cf_id"] + 1
        w64 = Float64.(vec(Array(W_U[a_id:a_id, :]))) .- Float64.(vec(Array(W_U[c_id:c_id, :])))
        set_readout!(dev, w64)
        cacheT = capture_rule_nodes(g)
        for s in NORM_SYMS; FROZEN_RMS[s] = copy(NeuroDSL.node(g, s; namespace=NS).aux_data[:rms_inv]); end
        Rc = readout64(g, w64)
        refdiff = meta === nothing ? lg[a_id] - lg[c_id] : meta["logit_answer"] - meta["logit_cf"]
        refscale = meta === nothing ? abs(lg[a_id]) : abs(meta["logit_answer"])
        gr = argmax(lg) == a_id && abs(Rc - refdiff) <= 1e-4 * max(1.0, refscale)
        # ── gradients exacts par variante ──────────────────────────────────────
        G = Dict{Symbol,Any}()
        for V in (:T, :A, :AN)
            G[V] = variant_grads!(g, orig, V)                # (Gx, Gr)
        end
        swap_norms!(g, orig, false); SG_ON[] = false; NeuroDSL.demand!(g, :w3_R; namespace=NS)
        # ── porte GV : gradient au dernier token = produit des Jacobiennes WIND-T8 ─
        gv = Dict{String,Float64}()
        for V in (:T, :A, :AN)
            fJ = joinpath(T8DIR, "wind_J_p$(p)_$(V).bin")
            isfile(fJ) || (gv[string(V)] = NaN; continue)          # (prompt 51 de test : pas de T)
            Js = Array{Float32}(undef, D, D, L)
            open(fJ, "r") do f; read!(f, Js); end
            Gx = G[V][1]; worst = 0.0
            for k in 1:L-1
                pred = Float64.(Js[:, :, k+1])' * Gx[k+2][n, :]
                worst = max(worst, norm(pred .- Gx[k+1][n, :]) / norm(Gx[k+1][n, :]))
            end
            gv[string(V)] = worst; Js = nothing
        end
        reclaim()
        # ── attributions linéaires ─────────────────────────────────────────────
        oM = [Float64.(vec(Array(cacheT[mha_sym(l)][n:n, :]))) for l in 1:L]
        oP = [Float64.(vec(Array(cacheT[mlp_sym(l)][n:n, :]))) for l in 1:L]
        dM = oM .- μMHA; dP = oP .- μMLP
        attr(V) = vcat([dot(G[V][2][l][n, :], dM[l]) for l in 1:L], [dot(G[V][1][l+1][n, :], dP[l]) for l in 1:L])
        aT, aA, aN = attr(:T), attr(:A), attr(:AN)
        # ── vérité terrain (vrai modèle) et porte GA (différence locale) ────────
        function Rwith(sym, v)
            P = copy(cacheT[sym])
            P[n:n, :] .= reshape(NeuroDSL.Backend.to_device(dev, Float32.(v)), 1, :)
            NeuroDSL.patch_node!(g, sym, Dict(sym => P); namespace=NS)
            r = readout64(g, w64)
            NeuroDSL.patch_node!(g, sym, cacheT; namespace=NS)
            r
        end
        comps = vcat([(mha_sym(l), oM[l], dM[l]) for l in 1:L], [(mlp_sym(l), oP[l], dP[l]) for l in 1:L])
        dR = Float64[]; fd = Float64[]
        for (sym, o, d) in comps
            push!(dR, Rc - Rwith(sym, o .- d))                           # o − d = μ
            push!(fd, (Rwith(sym, o .+ EPS .* d) - Rwith(sym, o .- EPS .* d)) / (2EPS))
        end
        sel = findall(abs.(aT) .>= 1e-3 * maximum(abs.(aT)))
        ga = median(abs.(fd[sel] .- aT[sel]) ./ abs.(aT[sel]))
        ok = gr && all(v < 1e-2 for v in values(gv) if !isnan(v)) && ga < 2e-2
        out[string(p)] = Dict("task" => pr["task"], "R" => Rc, "aT" => aT, "aA" => aA, "aAN" => aN, "dR" => dR, "fd" => fd,
                              "gate_GR" => gr, "gate_GV" => gv, "gate_GA" => ga, "ok" => ok)
        open(OUT, "w") do f; JSON.print(f, out); end
        sp(a, b) = cor(sortperm(sortperm(a)), sortperm(sortperm(b)))
        emit(@sprintf("  p%-3d %-9s R %6.2f | ρ T %.2f A %.2f AN %.2f | GV T %.1e A %.1e AN %.1e | GA %.1e | GR %s | %s | %.0f s",
                      p, pr["task"], Rc, sp(aT, dR), sp(aA, dR), sp(aN, dR), gv["T"], gv["A"], gv["AN"], ga,
                      gr ? "OK" : "ÉCHEC", ok ? "retenu" : "EXCLU", time() - t0))
    end
end
println("Écrit : ", OUT)
