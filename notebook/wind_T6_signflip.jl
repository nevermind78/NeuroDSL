# ══════════════════════════════════════════════════════════════════════════════
# WIND-T6 -- tests pré-enregistrés P6.1–P6.4 (notebook/wind_T3_preregistration.md, section WIND-T6).
# EXPLORATOIRE : les 5 prompts ont révélé l'effet.
#   P6.1 ensemble QK   P_Q(ε) = Π(A_k + ε_k E_k), E_k = J_T − J_A  ; ε ≡ +1 → T
#   P6.2 ensemble norm P_R(ε) = Π(A_k + ε_k R_k), R_k = J_AN − J_A ; ε ≡ +1 → AN
#   P6.3 ‖U_kᵀ Δ̂_k‖² (colonnes de R_k vs mise à jour du résiduel)
#   P6.4 cos(u₁(N·P_AN), z), z = N·Σ_{k≤10} P^AN_{k+1} Δ_k
# Descriptif : masses résiduelles τ, approximations d'ordre 1 (Walsh), bornes d'entropie, entrelacement, encadrement de τ_AN.
# USAGE : julia --project=. notebook/wind_T6_signflip.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON, Random

const OUTDIR = joinpath(@__DIR__, "wind_data")
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const RES = joinpath(@__DIR__, "wind_T6_signflip_results.txt")
const JOUT = joinpath(OUTDIR, "wind_T6_signflip.json")
const L, D = 28, 1536
const K = parse(Int, get(ENV, "WIND_K", "40"))
BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ 2))
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
ent(λ) = (q = max.(λ, 0) ./ sum(max.(λ, 0)); q = q[q .> 0]; exp(-sum(q .* log.(q))))
function st(N, P)
    λ = eigvals(Symmetric((N * P) * (N * P)'))
    return (ern = ent(λ), s1 = sqrt(maximum(λ)), p1 = maximum(λ) / sum(max.(λ, 0)), fro = sqrt(sum(max.(λ, 0))))
end
τ(s) = sqrt(max(s.fro^2 - s.s1^2, 0.0))
prodof(f) = (P = f(1); for k in 2:27; P = f(k) * P; end; P)
hbin(p) = (p <= 0 || p >= 1) ? 0.0 : -p * log(p) - (1 - p) * log(1 - p)
pct_below(obs, xs) = count(<(obs), xs) / length(xs)

rng = MersenneTwister(20261001)
out = Dict{String,Any}()
verd = Dict{String,Any}()

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T6 -- tests pré-enregistrés P6.1–P6.4 (EXPLORATOIRE) -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    emit("K = $K tirages de signes par ensemble ; q = fraction des tirages strictement inférieurs à la vraie configuration")
    for p in PROMPT_IDXS
        t0 = time()
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        X = [Float64.(v) for v in meta["X"]]
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x = X[29]
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        JA = loadJ(p, "A"); JT = loadJ(p, "T"); JN = loadJ(p, "AN")
        PA = prodof(k -> JA[k]); PT = prodof(k -> JT[k]); PN = prodof(k -> JN[k])
        sA, sT, sN = st(N, PA), st(N, PT), st(N, PN)
        emit(@sprintf("\n==================== PROMPT %d ====================", p))
        emit(@sprintf("  réel : ern T/A/AN = %.2f / %.2f / %.2f ; σ₁ = %.3e / %.3e / %.3e ; τ = %.2f / %.2f / %.2f (τ_A/τ_T = %.2f, τ_AN/τ_T = %.2f)",
                      sT.ern, sA.ern, sN.ern, sT.s1, sA.s1, sN.s1, τ(sT), τ(sA), τ(sN), τ(sA) / τ(sT), τ(sN) / τ(sT)))
        # bornes d'entropie (Lemme 1) : 1/p₁ ≤ ern ≤ exp(h(p₁))·(n−1)^{1−p₁}
        for (lab, s) in (("T", sT), ("A", sA), ("AN", sN))
            lo = 1 / s.p1; hi = exp(hbin(s.p1)) * (D - 1)^(1 - s.p1)
            emit(@sprintf("  Lemme 1 [%s] : 1/p₁ = %.3f ≤ ern = %.3f ≤ %.3f -> %s", lab, lo, s.ern, hi, (lo <= s.ern + 1e-9 && s.ern <= hi + 1e-9) ? "OK" : "VIOLÉ"))
        end
        # ── P6.1 / P6.2 : ensembles de signes ────────────────────────────────────
        res_ens = Dict{String,Any}()
        for (lab, Jp, obs) in (("QK", JT, sT), ("norm", JN, sN))
            ers = Float64[]; s1s = Float64[]
            for _ in 1:K
                ε = rand(rng, (-1, 1), 27)
                P = prodof(k -> ε[k] == 1 ? Jp[k] : 2 .* JA[k] .- Jp[k])
                s = st(N, P); push!(ers, s.ern); push!(s1s, s.s1)
            end
            qe = pct_below(obs.ern, ers); qs = pct_below(obs.s1, s1s)
            emit(@sprintf("  ensemble %-4s : vrai ern = %7.2f (q = %.3f) ; nul ern médiane %.2f [min %.2f, max %.2f] | vrai σ₁ = %.3e (q = %.3f) ; nul σ₁ médiane %.3e [min %.3e, max %.3e] ; σ₁(N·P_A) = %.3e",
                          lab, obs.ern, qe, median(ers), minimum(ers), maximum(ers), obs.s1, qs, median(s1s), minimum(s1s), maximum(s1s), sA.s1))
            res_ens[lab] = Dict("ern" => ers, "s1" => s1s, "q_ern" => qe, "q_s1" => qs)
        end
        # ── P6.3 : colonnes de R_k vs mise à jour ────────────────────────────────
        f3 = zeros(27)
        for k in 1:27
            U = svd(JN[k] - JA[k]).U[:, 1:2]
            Δ = X[k+2] - X[k+1]; Δ ./= norm(Δ)
            f3[k] = sum(abs2, U' * Δ)
        end
        emit("  P6.3 ‖U_kᵀΔ̂_k‖² k=1..27 : " * join([@sprintf("%.2f", v) for v in f3], " "))
        emit(@sprintf("  P6.3 médiane = %.3f (couches 1–10 : %.3f ; 11–27 : %.3f)", median(f3), median(f3[1:10]), median(f3[11:27])))
        # ── P6.4 + ordre 1 (Walsh) ───────────────────────────────────────────────
        Q = Vector{Matrix{Float64}}(undef, 27); Q[1] = Matrix{Float64}(I, D, D)
        for k in 2:27; Q[k] = JA[k-1] * Q[k-1]; end
        z = zeros(D); S = Matrix{Float64}(I, D, D)            # S = P^AN_{k+1}
        SA = Matrix{Float64}(I, D, D)                          # SA = P^A_{k+1}
        P1R = copy(PA); P1Q = copy(PA)
        for k in 27:-1:1
            k <= 10 && (z .+= S * (X[k+2] - X[k+1]))
            P1R .+= SA * ((JN[k] - JA[k]) * Q[k])
            P1Q .+= SA * ((JT[k] - JA[k]) * Q[k])
            S = S * JN[k]; SA = SA * JA[k]
        end
        z = N * z
        u1 = svd(N * PN).U[:, 1]
        c4 = abs(dot(u1, z)) / norm(z)
        emit(@sprintf("  P6.4 |cos(u₁(N·P_AN), z)| = %.3f   (hasard ≈ %.3f)", c4, 1 / sqrt(D)))
        s1R = st(N, P1R); s1Q = st(N, P1Q)
        emit(@sprintf("  ordre 1 (Walsh) : ern(N·(P_A + Σ_k W_{k}^R)) = %.2f (AN %.2f) ; ern(N·(P_A + Σ_k W_{k}^Q)) = %.2f (T %.2f)",
                      s1R.ern, sN.ern, s1Q.ern, sT.ern))
        # ── entrelacement (Weyl) et encadrement de τ_AN (perturbation quasi de rang 1) ─
        MA = N * PA; MN = N * PN; ΔN = MN - MA
        sa = svdvals(MA); sn = svdvals(MN)
        viol = maximum(max(sn[i+54] - sa[i], sa[i+54] - sn[i]) for i in 1:(D-54))
        Fd = svd(ΔN); Drest = ΔN - Fd.S[1] * Fd.U[:, 1] * Fd.V[:, 1]'
        sr = svdvals(MA + Drest)
        lo = sqrt(sum(abs2, sr[3:end])); hi = norm(MA + Drest)
        emit(@sprintf("  entrelacement σ_{i+54}(AN) ≤ σ_i(A) et réciproque : violation max = %.2e (relative à σ₁(A) : %.1e)", viol, viol / sa[1]))
        emit(@sprintf("  encadrement τ_AN : √Σ_{i≥3}σ_i(N P_A + D_rest)² = %.2f ≤ τ_AN = %.2f ≤ ‖N P_A + D_rest‖_F = %.2f ; σ₂/σ₁(NΔ) = %.3f ; ‖D_rest‖_F/‖N P_A‖_F = %.3f",
                      lo, τ(sN), hi, Fd.S[2] / Fd.S[1], norm(Drest) / norm(MA)))
        out["p$p"] = Dict("ens" => res_ens, "f3" => f3, "c4" => c4, "ern" => [sT.ern, sA.ern, sN.ern],
                          "s1" => [sT.s1, sA.s1, sN.s1], "tau" => [τ(sT), τ(sA), τ(sN)],
                          "ern_ord1" => [s1Q.ern, s1R.ern])
        verd["p$p"] = (qe_Q = res_ens["QK"]["q_ern"], qs_Q = res_ens["QK"]["q_s1"],
                       qe_R = res_ens["norm"]["q_ern"], qs_R = res_ens["norm"]["q_s1"], f3 = median(f3), c4 = c4)
        emit(@sprintf("  (%.0f s)", time() - t0))
        JA = nothing; JT = nothing; JN = nothing; Q = nothing; S = nothing; SA = nothing; GC.gc()
    end
    # ── verdicts ────────────────────────────────────────────────────────────────
    emit("\nVERDICTS (pré-enregistrés) :")
    v = verd
    has(p) = haskey(v, "p$p")
    if all(has, (5, 6, 12, 17, 22))
        p61 = v["p12"].qe_Q >= 0.975 && v["p12"].qs_Q <= 0.025 &&
              all(0.025 < v["p$p"].qe_Q < 0.975 for p in (5, 6, 17, 22))
        emit(@sprintf("P6.1 : %s  (p12 : q_ern = %.3f, q_σ₁ = %.3f ; autres q_ern : p5 %.3f p6 %.3f p17 %.3f p22 %.3f)",
                      p61 ? "VRAI" : "FAUX", v["p12"].qe_Q, v["p12"].qs_Q, v["p5"].qe_Q, v["p6"].qe_Q, v["p17"].qe_Q, v["p22"].qe_Q))
        p62 = all(v["p$p"].qs_R >= 0.975 && v["p$p"].qe_R <= 0.025 for p in (12, 22)) &&
              all(0.025 < v["p$p"].qe_R < 0.975 for p in (5, 6))
        emit(@sprintf("P6.2 : %s  (q_σ₁ / q_ern : p12 %.3f/%.3f, p22 %.3f/%.3f ; p5 q_ern %.3f, p6 q_ern %.3f ; p17 (descriptif) %.3f/%.3f)",
                      p62 ? "VRAI" : "FAUX", v["p12"].qs_R, v["p12"].qe_R, v["p22"].qs_R, v["p22"].qe_R,
                      v["p5"].qe_R, v["p6"].qe_R, v["p17"].qs_R, v["p17"].qe_R))
        f3s = [v["p$p"].f3 for p in (5, 6, 12, 17, 22)]
        p63 = all(f3s .>= 0.5) ? "VRAI" : (all(f3s .>= 0.1) ? "PARTIEL" : "FAUX")
        emit("P6.3 : $p63  (médianes : " * join([@sprintf("p%d %.3f", p, v["p$p"].f3) for p in (5, 6, 12, 17, 22)], ", ") * ")")
        p64 = v["p12"].c4 >= 0.7 && v["p22"].c4 >= 0.7
        emit(@sprintf("P6.4 : %s  (p12 %.3f, p22 %.3f ; descriptif p5 %.3f p6 %.3f p17 %.3f)", p64 ? "VRAI" : "FAUX",
                      v["p12"].c4, v["p22"].c4, v["p5"].c4, v["p6"].c4, v["p17"].c4))
    end
end
open(JOUT, "w") do f; JSON.print(f, out); end
println("Écrit : ", RES)
