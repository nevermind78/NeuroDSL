# ══════════════════════════════════════════════════════════════════════════════
# WIND-3 -- SONDE DE CALIBRATION (prompt 4, HORS des 5 prompts mesurés ;
# aucune quantité d'hypothèse HH* n'est calculée ici)
#
# Vérifie : recâblage bit-exact ; lecture = différence de logits ; backward
# (T, A, AN) fonctionne, temps, VRAM ; bruit plancher des JVP par différences finies
# (choix de eps et des seuils des portes) ; knockout bit-identique quand rien n'est
# bloqué, confiné à la ligne n ; temps d'un knockout et d'un backward de sonde.
#
# USAGE : julia --project=. notebook/wind3_calib_probe.jl
# ══════════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "wind3_common.jl"))

const RES = joinpath(@__DIR__, "wind3_calib_probe_results.txt")
const PIDX = 4

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-3 -- SONDE DE CALIBRATION (prompt $PIDX, hors échantillon mesuré)")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("VRAM avant chargement : $(gpu_used_mib()) MiB")
    prompts = JSON.parsefile(joinpath(@__DIR__, "qwen_sweep_prompts.json"))
    ids = Int.(prompts[PIDX]["token_ids"]) .+ 1

    # ── 1. logits de référence SANS instrumentation ───────────────────────────
    t0 = time()
    dev = NeuroDSL.Backend.CUDADevice()
    g0 = NeuroDSL.NeuroGraph(namespace=NS, device=dev)
    NeuroDSL.load_graph!(g0, NS, W3_CKPT)
    n = length(ids)
    NeuroDSL.set!(g0, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=NS)
    NeuroDSL.set!(g0, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=NS)
    NeuroDSL.invalidate_all!(g0; namespace=NS)
    ref_logits = Array(NeuroDSL.demand!(g0, :lm_head_out; namespace=NS))
    ref_fn = Array(NeuroDSL.node(g0, :final_norm_out; namespace=NS).value)
    g0 = nothing; reclaim()
    emit(@sprintf("Référence non instrumentée : %.1f s", time() - t0))

    # ── 2. graphe instrumenté ─────────────────────────────────────────────────
    t0 = time()
    g, dev, orig, W_U = load_instrumented()
    emit(@sprintf("Chargement + recâblage : %.1f s ; VRAM : %d MiB", time() - t0, gpu_used_mib()))
    for k in (0, 5, 28)
        emit("  consommateurs de $(capx(k)) : $(get(NeuroDSL._consumers_index!(g, NS), capx(k), Symbol[]))")
    end
    emit("  consommateurs de $(capr(7)) : $(get(NeuroDSL._consumers_index!(g, NS), capr(7), Symbol[]))")
    emit("  consommateurs de $(sg_sym(7,3)) : $(get(NeuroDSL._consumers_index!(g, NS), sg_sym(7,3), Symbol[]))")
    set_prompt!(g, ids)
    lg = Array(NeuroDSL.demand!(g, :lm_head_out; namespace=NS))
    emit("Porte R0 : logits instrumentés == référence (bit à bit) : $(lg == ref_logits)" *
         "   final_norm_out bit à bit : $(Array(NeuroDSL.node(g, :final_norm_out; namespace=NS).value) == ref_fn)")
    ln = Float64.(lg[n, :]); order = sortperm(ln; rev=true); y1, y2 = order[1], order[2]
    w64 = Float64.(vec(Array(W_U[y1:y1, :]))) .- Float64.(vec(Array(W_U[y2:y2, :])))
    set_readout!(dev, w64)
    Rnode = Float64(Array(NeuroDSL.demand!(g, :w3_R; namespace=NS))[1])
    R64 = readout64(g, w64)
    emit(@sprintf("Lecture : logit-diff (logits) = %.6f ; nœud w3_R (fp32) = %.6f ; readout64 = %.6f",
                  ln[y1] - ln[y2], Rnode, R64))

    cacheT = capture_rule_nodes(g)
    for s in NORM_SYMS
        FROZEN_RMS[s] = copy(NeuroDSL.node(g, s; namespace=NS).aux_data[:rms_inv])
    end
    X = [Float64.(Array(cacheT[res_sym(k)])) for k in 0:L]

    # ── 3. backward des 3 variantes ───────────────────────────────────────────
    G = Dict{Symbol,Any}()
    caches = Dict{Symbol,Any}(:T => cacheT, :A => cacheT)
    for V in (:T, :A, :AN)
        if V == :AN
            swap_norms!(g, orig, true)
            lgf = Array(NeuroDSL.demand!(g, :lm_head_out; namespace=NS))
            emit(@sprintf("Porte G4b-like : forward normes gelées vs propre, err rel logits = %.2e",
                          norm(Float64.(lgf) .- Float64.(lg)) / norm(Float64.(lg))))
            caches[:AN] = capture_rule_nodes(g)
        end
        variant_grads!(g, orig, V)            # chauffe (JIT)
        t1 = time(); vb = gpu_used_mib()
        Gx, Gr = variant_grads!(g, orig, V)
        emit(@sprintf("Backward [%s] : %.2f s (post-JIT) ; VRAM après : %d MiB", V, time() - t1, gpu_used_mib()))
        G[V] = (Gx, Gr)
        z = maximum(abs.(Gx[L+1][1:n-1, :]))
        emit(@sprintf("  [%s] max|∂R/∂x_{i<n,28}| = %.1e (attendu 0) ; ‖∂R/∂x_{n,28}‖ = %.4e ; ‖∂R/∂x_{i<n,27}‖ max = %.3e",
                      V, z, norm(Gx[L+1][n, :]), maximum(norm.(eachrow(Gx[L][1:n-1, :])))))
    end
    swap_norms!(g, orig, false); NeuroDSL.demand!(g, :w3_R; namespace=NS)

    # ── 4. plancher de bruit des JVP DF (choix d'eps) ─────────────────────────
    emit("\nJVP DF de R vs <g,v> (backward) ; erreur e_v = |DF − <g,v>| / ‖g‖, v unitaire")
    rng = MersenneTwister(4242)
    cells = [(1, 3), (3, 8), (n - 2, 16), (n - 1, 24), (n, 12), (2, 0)]
    for V in (:T, :A, :AN)
        swap_norms!(g, orig, V == :AN)
        NeuroDSL.demand!(g, :w3_R; namespace=NS)
        Gx = G[V][1]
        for (i, s) in cells
            gv = Gx[s+1][i, :]; gn = norm(gv); xn = norm(X[s+1][i, :])
            dirs = [gv ./ gn]
            for _ in 1:3; v = randn(rng, D); push!(dirs, v ./ norm(v)); end
            line = @sprintf("  [%-2s] i=%2d s=%2d ‖g‖=%.3e ‖x‖=%9.2f |", V, i, s, gn, xn)
            for er in (1e-3, 3e-3, 1e-2, 3e-2)
                errs = [abs(fd_jvp(g, caches[V], V, s, i, v, er * xn, w64) - dot(gv, v)) / gn for v in dirs]
                line *= @sprintf(" eps=%.0e: ĝ %.1e, rnd max %.1e |", er, errs[1], maximum(errs[2:end]))
            end
            emit(line)
        end
    end
    swap_norms!(g, orig, false); NeuroDSL.demand!(g, :w3_R; namespace=NS)
    t1 = time(); fd_jvp(g, cacheT, :T, 10, 2, randn(rng, D) ./ 40, 1.0, w64)
    emit(@sprintf("Temps d'une JVP DF (T, s=10) : %.0f ms", 1000 * (time() - t1)))
    t1 = time(); fd_jvp(g, cacheT, :A, 10, 2, randn(rng, D) ./ 40, 1.0, w64)
    emit(@sprintf("Temps d'une JVP DF (A, s=10, 216 pr épinglés) : %.0f ms", 1000 * (time() - t1)))

    # ── 5. knockout ──────────────────────────────────────────────────────────
    cleanP = Dict((l, h) => Array(cacheT[pr_sym(l, h)]) for l in 1:L for h in 1:NH)
    Rc = readout64(g, w64)
    fn_c = Array(NeuroDSL.demand!(g, :final_norm_out; namespace=NS))
    ok = true
    for l in (1, 14, 28)
        R0, δ0 = knockout(g, cleanP, cacheT, [l], 2, :none, w64)
        ok &= (R0 == Rc) && all(δ0 .== 0) &&
              (Array(NeuroDSL.demand!(g, :final_norm_out; namespace=NS)) == fn_c)
    end
    emit("\nPorte K0 : knockout sans blocage (pr ré-imposés à leur valeur propre) bit-identique : $ok")
    for (l, i) in ((1, 1), (14, 2), (27, n - 1))
        t1 = time()
        Rr, δr = knockout(g, cleanP, cacheT, [l], i, :remove, w64)
        tr = time() - t1
        # confinement : relance, lit la sortie MHA complète
        for h in 1:NH
            P = copy(cleanP[(l, h)]); P[n, i] = 0f0
            NeuroDSL.patch_node!(g, pr_sym(l, h), Dict(pr_sym(l, h) => P); namespace=NS)
        end
        M = Array(NeuroDSL.demand!(g, mha_sym(l); namespace=NS)); Mc = Array(cacheT[mha_sym(l)])
        conf = (M[1:n-1, :] == Mc[1:n-1, :])
        for h in 1:NH; NeuroDSL.patch_node!(g, pr_sym(l, h), cacheT; namespace=NS); end
        Rm, δm = knockout(g, cleanP, cacheT, [l], i, :mask, w64)
        emit(@sprintf("  knockout l=%2d i=%2d : ΔR(remove)=%+.4f ΔR(mask)=%+.4f ‖δ‖ rem/mask = %.3f/%.3f ; lignes i<n inchangées : %s ; %.0f ms",
                      l, i, Rr - Rc, Rm - Rc, norm(δr), norm(δm), conf, 1000tr))
    end
    emit(@sprintf("  vérif. restauration : R après knockouts == R propre : %s", readout64(g, w64) == Rc))

    # ── 6. backward de sonde (esquisse) ───────────────────────────────────────
    set_probe!(dev, randn(rng, D))
    variant_grads!(g, orig, :T; seed=:w3_P)
    t1 = time(); Gp, _ = variant_grads!(g, orig, :T; seed=:w3_P)
    emit(@sprintf("\nBackward de sonde (T) : %.2f s ; ‖∂P/∂x_{n,28}‖ = %.3f (attendu ‖u‖ = %.3f)",
                  time() - t1, norm(Gp[L+1][n, :]), norm(Float64.(Array(READ[:u])))))
    emit("VRAM fin : $(gpu_used_mib()) MiB")
end
println("\nÉcrit : ", RES)
