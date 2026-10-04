# ══════════════════════════════════════════════════════════════════════════════
# WIND-3 -- COLLECTE (vent horizontal) : carte 1 (gradients toutes positions, T/A/AN)
# + carte 2 (knockouts d'attention exacts, dernier token ← i, par couche) + portes.
# Pré-enregistrement : notebook/wind3_preregistration.md (écrit AVANT ce script).
# Les verdicts sont calculés par wind3_analysis.jl à partir des fichiers écrits ici.
#
# USAGE : julia --project=. notebook/wind3_collect.jl      (WIND3_PROMPTS=5,6,12,17,22)
# ══════════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "wind3_common.jl"))

const OUTDIR = joinpath(@__DIR__, "wind_data")
const WIND1_DIR = get(ENV, "WIND1_DIR", joinpath(@__DIR__, "wind_data"))
const RES = get(ENV, "WIND3_RES", joinpath(@__DIR__, "wind3_collect_results.txt"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND3_PROMPTS", "5,6,12,17,22"), ","))
const EPS_REL = 1e-3          # fixé par la sonde de calibration (voir pré-enregistrement)
const VARIANTS = (:T, :A, :AN)
const G4B = Ref(NaN)

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-3 -- COLLECTE (vent horizontal)")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("VRAM avant chargement : $(gpu_used_mib()) MiB")
    g, dev, orig, W_U = load_instrumented()
    prompts = JSON.parsefile(joinpath(@__DIR__, "qwen_sweep_prompts.json"))

    for pidx in PROMPT_IDXS
        t_p = time()
        ids = Int.(prompts[pidx]["token_ids"]) .+ 1
        emit("\n" * "="^100)
        emit(@sprintf("PROMPT %d : %s (%d tokens)", pidx, repr(prompts[pidx]["prompt"]), length(ids)))
        emit("="^100)
        swap_norms!(g, orig, false); SG_ON[] = false
        n = set_prompt!(g, ids)
        lg = Array(NeuroDSL.demand!(g, :lm_head_out; namespace=NS))
        meta1 = JSON.parsefile(joinpath(WIND1_DIR, "wind_meta_p$(pidx).json"))
        ln = Float64.(lg[n, :]); order = sortperm(ln; rev=true); y1, y2 = order[1], order[2]
        (y1 - 1 == meta1["top1_id"] && y2 - 1 == meta1["top2_id"]) || error("top-1/top-2 ≠ WIND-1")
        w64 = Float64.(meta1["w_read"])
        set_readout!(dev, w64)
        cacheT = capture_rule_nodes(g)
        for s in NORM_SYMS
            FROZEN_RMS[s] = copy(NeuroDSL.node(g, s; namespace=NS).aux_data[:rms_inv])
        end
        X = [Float64.(Array(cacheT[res_sym(k)])) for k in 0:L]          # n×D chacun
        # ── porte R0 : forward instrumenté identique à WIND-1 (bit à bit) ─────────
        r0 = all(X[k+1][n, :] == Float64.(meta1["X"][k+1]) for k in 0:L) &&
             ln[y1] == meta1["logit_top1"] && ln[y2] == meta1["logit_top2"]
        emit("Porte R0 (x_{n,k} k=0..28 et logits top-1/top-2 == WIND-1, bit à bit) : " * (r0 ? "OK" : "ÉCHEC"))
        R_clean = readout64(g, w64)
        emit(@sprintf("R propre = %.6f (logit-diff WIND-1 = %.6f)", R_clean, meta1["logit_top1"] - meta1["logit_top2"]))
        cleanP = Dict((l, h) => Array(cacheT[pr_sym(l, h)]) for l in 1:L for h in 1:NH)
        attn_last = [Float64.(cleanP[(l, h)][n, :]) for l in 1:L, h in 1:NH]   # L×NH de vecteurs n

        # ── carte 1 : gradients de R, toutes positions, 3 variantes ──────────────
        caches = Dict{Symbol,Any}(:T => cacheT, :A => cacheT)
        Gx = Dict{Symbol,Any}(); Gr = Dict{Symbol,Any}()
        for V in VARIANTS
            if V == :AN
                swap_norms!(g, orig, true)
                lgf = Array(NeuroDSL.demand!(g, :lm_head_out; namespace=NS))
                g4b = norm(Float64.(lgf) .- Float64.(lg)) / norm(Float64.(lg))
                emit(@sprintf("  Porte G4b (forward normes gelées vs propre) : err rel logits = %.2e (seuil 1e-5) -> %s",
                              g4b, g4b < 1e-5 ? "OK" : "ÉCHEC"))
                G4B[] = g4b
                caches[:AN] = capture_rule_nodes(g)
            end
            t1 = time()
            Gx[V], Gr[V] = variant_grads!(g, orig, V)
            emit(@sprintf("  backward [%s] : %.1f s", V, time() - t1))
            open(joinpath(OUTDIR, "wind3_G_p$(pidx)_$(V).bin"), "w") do f
                for k in 0:L; write(f, Float32.(Gx[V][k+1])); end
                for l in 1:L; write(f, Float32.(Gr[V][l])); end
            end
        end
        swap_norms!(g, orig, false); NeuroDSL.demand!(g, :w3_R; namespace=NS)

        # ── porte GV : lignes n des gradients vs produits des Jacobiennes DF de WIND-1 ─
        gates = Dict{String,Any}("R0" => r0, "G4b" => G4B[])
        for V in VARIANTS
            Js = Array{Float32}(undef, D, D, L)
            open(joinpath(WIND1_DIR, "wind_J_p$(pidx)_$(V).bin"), "r") do f; read!(f, Js); end
            pred = Gx[V][L+1][n, :]; worst = 0.0; worst0 = 0.0
            for k in L-1:-1:0
                pred = Float64.(Js[:, :, k+1])' * pred
                e = norm(pred .- Gx[V][k+1][n, :]) / norm(Gx[V][k+1][n, :])
                k >= 1 ? (worst = max(worst, e)) : (worst0 = e)
            end
            Js = nothing; reclaim()
            emit(@sprintf("  Porte GV [%-2s] : pire err rel (s=1..27) = %.2e (seuil 1e-2) -> %s ; s=0 : %.2e",
                          V, worst, worst < 1e-2 ? "OK" : "ÉCHEC", worst0))
            gates["GV_$(V)"] = worst; gates["GV0_$(V)"] = worst0
        end

        # ── porte GF : JVP DF horizontales vs gradients (cellules fixées a priori) ─
        cells = [(1, 3), (1, 14), (2, 1), (cld(n, 2), 7), (n - 1, 20), (n - 1, 26), (n, 10)]
        rng = MersenneTwister(3000 + pidx)
        for V in VARIANTS
            swap_norms!(g, orig, V == :AN); NeuroDSL.demand!(g, :w3_R; namespace=NS)
            errs = Float64[]; unres = 0; allerrs = Float64[]
            for (i, s) in cells
                gv = Gx[V][s+1][i, :]; gn = norm(gv); xn = norm(X[s+1][i, :])
                dirs = [gv ./ gn]
                for _ in 1:3; v = randn(rng, D); push!(dirs, v ./ norm(v)); end
                cl = @sprintf("    GF [%-2s] i=%2d s=%2d ‖g‖=%.3e :", V, i, s, gn)
                for v in dirs
                    d1 = fd_jvp(g, caches[V], V, s, i, v, EPS_REL * xn, w64)
                    d2 = fd_jvp(g, caches[V], V, s, i, v, 2EPS_REL * xn, w64)
                    e = abs(d1 - dot(gv, v)) / gn; c = abs(d1 - d2) / gn
                    push!(allerrs, e)
                    if !isfinite(e) || !isfinite(c) || c > 5e-2
                        unres += 1; cl *= @sprintf(" [non résolue e=%.1e c=%.1e]", e, c)
                    else
                        push!(errs, e); cl *= @sprintf(" %.1e", e)
                    end
                end
                emit(cl)
            end
            ok = !isempty(errs) && median(errs) < 1e-2 && maximum(errs) < 5e-2
            md = isempty(errs) ? NaN : median(errs); mx = isempty(errs) ? NaN : maximum(errs)
            emit(@sprintf("  Porte GF [%-2s] : e_v médiane %.2e max %.2e sur %d résolues ; %d non résolues (seuils 1e-2 / 5e-2) -> %s",
                          V, md, mx, length(errs), unres, ok ? "OK" : "ÉCHEC"))
            gates["GF_$(V)_med"] = md; gates["GF_$(V)_max"] = mx
            gates["GF_$(V)_unresolved"] = unres; gates["GF_$(V)_ok"] = ok
        end
        swap_norms!(g, orig, false); SG_ON[] = false; NeuroDSL.demand!(g, :w3_R; namespace=NS)

        # ── porte K0 + carte 2 : knockouts exacts ────────────────────────────────
        k0 = true
        for l in (1, 14, 28)
            R0_, δ0 = knockout(g, cleanP, cacheT, [l], 1, :none, w64)
            k0 &= (R0_ == R_clean) && all(δ0 .== 0)
        end
        emit("  Porte K0 (knockout sans blocage bit-identique) : " * (k0 ? "OK" : "ÉCHEC"))
        gates["K0"] = k0
        t1 = time()
        Krem = zeros(L, n - 1); Kmask = zeros(L, n - 1); Kwin = zeros(L, n - 1)
        LIN = Dict(V => zeros(L, n - 1) for V in VARIANTS)       # <c^V_l[n], δ_rem>
        LINm = Dict(V => zeros(L, n - 1) for V in VARIANTS)      # <c^V_l[n], δ_mask>
        for l in L:-1:1, i in 1:n-1
            Rr, δr = knockout(g, cleanP, cacheT, [l], i, :remove, w64)
            Rm, δm = knockout(g, cleanP, cacheT, [l], i, :mask, w64)
            win = max(1, l - 2):min(L, l + 2)
            Rw, _ = knockout(g, cleanP, cacheT, collect(win), i, :mask, w64)
            Krem[l, i] = Rr - R_clean; Kmask[l, i] = Rm - R_clean; Kwin[l, i] = Rw - R_clean
            for V in VARIANTS
                c = Gr[V][l][n, :]
                LIN[V][l, i] = dot(c, δr); LINm[V][l, i] = dot(c, δm)
            end
        end
        kres = readout64(g, w64) == R_clean
        emit(@sprintf("  knockouts : %d × 3 en %.0f s ; état restauré bit-identique : %s", L * (n - 1), time() - t1, kres))
        gates["K_restore"] = kres

        out = Dict("prompt_idx" => pidx, "prompt" => prompts[pidx]["prompt"], "n" => n,
                   "token_ids" => ids .- 1, "top1_id" => y1 - 1, "top2_id" => y2 - 1, "R_clean" => R_clean,
                   "xnorm" => [[norm(X[k+1][i, :]) for i in 1:n] for k in 0:L],
                   "attn_last" => [[attn_last[l, h] for h in 1:NH] for l in 1:L],
                   "Krem" => [Krem[l, :] for l in 1:L], "Kmask" => [Kmask[l, :] for l in 1:L],
                   "Kwin" => [Kwin[l, :] for l in 1:L],
                   "LIN" => Dict(string(V) => [LIN[V][l, :] for l in 1:L] for V in VARIANTS),
                   "LINm" => Dict(string(V) => [LINm[V][l, :] for l in 1:L] for V in VARIANTS),
                   "gates" => gates, "eps_rel" => EPS_REL)
        open(joinpath(OUTDIR, "wind3_meta_p$(pidx).json"), "w") do f; JSON.print(f, out); end
        emit(@sprintf("Écrit : wind3_G_p%d_{T,A,AN}.bin + wind3_meta_p%d.json  (%.0f s)", pidx, pidx, time() - t_p))
        cacheT = nothing; caches = nothing; reclaim()
    end
end
println("\nÉcrit : ", RES)
