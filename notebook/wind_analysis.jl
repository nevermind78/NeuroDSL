# ══════════════════════════════════════════════════════════════════════════════
# WIND-1 -- ANALYSE (CPU, Float64) des Jacobiennes collectées -- verdicts H1, H2, H3
# contre les seuils de notebook/wind_preregistration.md (écrits avant la collecte).
#
# USAGE : WIND_OUTDIR=<dossier> julia --project=. notebook/wind_analysis.jl
# ══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra, Statistics, Printf, JSON

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const RES  = joinpath(@__DIR__, "wind_analysis_results.txt")
const JOUT = joinpath(@__DIR__, "wind_analysis_results.json")
const L, D = 28, 1536
BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ 2))

loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 1:L])          # Js[k+1] = J_k = ∂x_{k+1}/∂x_k
erank2(σ) = (p = σ .^ 2 ./ sum(σ .^ 2); p = p[p .> 0]; exp(-sum(p .* log.(p))))
opn(M) = svdvals(M)[1]
function prod_range(Js, s, t)           # J_{s→t} = J_{t-1} ··· J_s
    M = Matrix{Float64}(I, D, D)
    for k in s:t-1; M = Js[k+1] * M; end
    M
end
function trunc_prod_backward(Js, s, r)   # protocole Fernando & Guitchounts : P ← trunc_r(P·J_ℓ), ℓ = 27 … s
    P = Matrix{Float64}(I, D, D)
    for k in L-1:-1:s
        P = P * Js[k+1]
        F = svd(P); P = F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
    end
    P
end

out = Dict{String,Any}()
open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-1 -- ANALYSE (verdicts contre wind_preregistration.md)")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))

    G_1_28 = Float64[]; G_s7 = Dict(s => Float64[] for s in (1, 7, 14, 21))
    erank_pp = Float64[]; erank_pp_tr = Dict(r => Float64[] for r in (192, 512))
    err_read = Dict{Int,Vector{Float64}}(); errA_read = Dict{Int,Vector{Float64}}()
    Jmean = [zeros(D, D) for _ in 1:L]
    per_prompt = Dict{String,Any}()

    for p in PROMPT_IDXS
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        X = [Float64.(x) for x in meta["X"]]
        emit("\n" * "="^92)
        emit(@sprintf("PROMPT %d : %s   (top-1 %d vs top-2 %d)", p, repr(meta["prompt"]),
                      meta["top1_id"], meta["top2_id"]))
        emit("="^92)
        JT = loadJ(p, "T")
        for k in 1:L; Jmean[k] .+= JT[k] ./ length(PROMPT_IDXS); end
        pp = Dict{String,Any}()

        # ── par couche : spectre de la vraie Jacobienne ──────────────────────
        nrm = zeros(L); smin = zeros(L); namp = zeros(Int, L); ndamp = zeros(Int, L); bF = zeros(L)
        for k in 1:L
            σ = svdvals(JT[k]); nrm[k] = σ[1]; smin[k] = σ[end]
            namp[k] = count(>(1.5), σ); ndamp[k] = count(<(0.5), σ)
            bF[k] = norm(JT[k] - I)
        end
        emit("\nPar couche (T) : k -> ‖J_k‖₂  σ_min  #σ>1.5  #σ<0.5  ‖J_k−I‖_F")
        for k in 1:L
            emit(@sprintf("  %2d->%2d  %9.3f  %8.4f  %5d  %5d  %9.3f", k-1, k, nrm[k], smin[k], namp[k], ndamp[k], bF[k]))
        end
        pp["opnorm_layer"] = nrm; pp["smin_layer"] = smin; pp["nres_F"] = bF

        # ── H1 : chaînage ───────────────────────────────────────────────────
        Gcurves = Dict{String,Any}()
        for s in (0, 1, 7, 14, 21)
            M = Matrix{Float64}(I, D, D); logprod = 0.0; Gs = Float64[]; Ns = Float64[]
            for t in s+1:L
                M = JT[t] * M; logprod += log(nrm[t])
                nm = opn(M); push!(Ns, nm); push!(Gs, exp(logprod - log(nm)))
            end
            Gcurves[string(s)] = Dict("G" => Gs, "norm" => Ns)
            emit(@sprintf("  H1 s=%2d : ‖J_{s→t}‖₂ pour t=s+1.. : %s", s,
                          join([@sprintf("%.3g", x) for x in Ns[[1, min(2,end), min(4,end), min(7,end), end]]], " ")))
            emit(@sprintf("           G(s,t)            pour t=s+1.. : %s   [t = s+1, s+2, s+4, s+7, 28]",
                          join([@sprintf("%.3g", x) for x in Gs[[1, min(2,end), min(4,end), min(7,end), end]]], " ")))
            s == 1 && push!(G_1_28, Gs[end])
            s in keys(G_s7) && push!(G_s7[s], Gs[7])
        end
        Gadj = [nrm[k] * nrm[k+1] / opn(JT[k+1] * JT[k]) for k in 1:L-1]
        emit("  G(k,k+2) couches adjacentes : " * join([@sprintf("%.2f", x) for x in Gadj], " "))
        pp["G_curves"] = Gcurves; pp["G_adjacent"] = Gadj

        # ── H2 : entonnoir ─────────────────────────────────────────────────
        P = [Matrix{Float64}(I, D, D) for _ in 1:L+1]      # P[s+1] = J_{s→28}
        for s in L-1:-1:0; P[s+1] = P[s+2] * JT[s+1]; end
        er = zeros(L); top5 = Dict{Int,Vector{Float64}}()
        for s in 0:L-1
            σ = svdvals(P[s+1]); er[s+1] = erank2(σ)
            s in (0, 1, 7, 14, 21) && (top5[s] = σ[1:5])
        end
        emit("  H2 erank₂(J_{s→28}) pour s=0..27 : " * join([@sprintf("%.1f", x) for x in er], " "))
        for s in (0, 1, 7, 14, 21)
            emit(@sprintf("     top-5 σ(J_{%d→28}) = %s", s, join([@sprintf("%.3g", x) for x in top5[s]], ", ")))
        end
        push!(erank_pp, er[2])
        for r in (192, 512)
            push!(erank_pp_tr[r], erank2(svdvals(trunc_prod_backward(JT, 1, r))))
        end
        emit(@sprintf("     erank₂(J_{1→28}) exact = %.2f ; tronqué-192 = %.2f ; tronqué-512 = %.2f",
                      er[2], erank_pp_tr[192][end], erank_pp_tr[512][end]))
        pp["erank2_s_to_28"] = er

        # ── direction la plus amplifiée : rotation, alignement ──────────────
        gT28 = let x = X[end], γ = Float64.(meta["gamma_final"]), w = Float64.(meta["w_read"])
            ρ = 1 / sqrt(mean(x .^ 2) + 1e-6)
            ρ .* (γ .* w) .- (ρ^3 / D) .* x .* dot(x, γ .* w)
        end
        gAN28 = Float64(meta["rms_inv_final"]) .* (Float64.(meta["gamma_final"]) .* Float64.(meta["w_read"]))
        for s in (1, 7, 14, 21)
            F = svd(P[s+1]); u1 = F.U[:, 1]; v1 = F.V[:, 1]
            gs = P[s+1]' * gT28
            emit(@sprintf("     s=%2d : cos(u₁,v₁)=%+.3f  |cos(v₁,x_s)|=%.3f  |cos(v₁,g_s)|=%.3f  |cos(u₁,x_28)|=%.3f",
                          s, dot(u1, v1), abs(dot(v1, X[s+1]))/norm(X[s+1]), abs(dot(v1, gs))/norm(gs),
                          abs(dot(u1, X[end]))/norm(X[end])))
        end

        # ── H3 : linéarisation gelée ────────────────────────────────────────
        JA = loadJ(p, "A")
        JAN = loadJ(p, "AN")
        shQK = [norm(JT[k] - JA[k]) / norm(JT[k] - I) for k in 1:L]
        shNo = [norm(JA[k] - JAN[k]) / norm(JT[k] - I) for k in 1:L]
        shN  = [norm(JT[k] - JAN[k]) / norm(JT[k] - I) for k in 1:L]
        PA = [Matrix{Float64}(I, D, D) for _ in 1:L+1]
        PAN = [Matrix{Float64}(I, D, D) for _ in 1:L+1]
        for s in L-1:-1:0; PA[s+1] = PA[s+2] * JA[s+1]; PAN[s+1] = PAN[s+2] * JAN[s+1]; end
        JA = nothing; JAN = nothing; GC.gc()
        emit("\n  H3 part par couche ‖J^T−J^A‖_F/‖J^T−I‖_F (canal QK) : " * join([@sprintf("%.3f", x) for x in shQK], " "))
        emit("  H3 part par couche ‖J^A−J^AN‖_F/‖J^T−I‖_F (canal norme): " * join([@sprintf("%.3f", x) for x in shNo], " "))
        emit("  H3 part par couche ‖J^T−J^AN‖_F/‖J^T−I‖_F (QK+norme)  : " * join([@sprintf("%.3f", x) for x in shN], " "))
        eT = zeros(L); eA = zeros(L); cT = zeros(L)
        for s in 0:L-1
            gs = P[s+1]' * gT28; gA = PA[s+1]' * gT28; gAN = PAN[s+1]' * gAN28
            eT[s+1] = norm(gs .- gAN) / norm(gs); eA[s+1] = norm(gs .- gA) / norm(gs)
            cT[s+1] = dot(gs, gAN) / (norm(gs) * norm(gAN))
            haskey(err_read, s) || (err_read[s] = Float64[]; errA_read[s] = Float64[])
            push!(err_read[s], eT[s+1]); push!(errA_read[s], eA[s+1])
        end
        emit("  H3 err_s = ‖g_s − g_s^AN‖/‖g_s‖, s=0..27 : " * join([@sprintf("%.3f", x) for x in eT], " "))
        emit("     (QK seul) ‖g_s − g_s^A‖/‖g_s‖         : " * join([@sprintf("%.3f", x) for x in eA], " "))
        emit("     cos(g_s, g_s^AN)                      : " * join([@sprintf("%.3f", x) for x in cT], " "))
        # transport multi-couches complet (sans lecture) : erreur relative sur J_{s→28} − I
        for s in (1, 7, 14, 21)
            emit(@sprintf("     s=%2d : ‖J_{s→28}−J^AN_{s→28}‖₂/‖J_{s→28}‖₂ = %.3f   ‖·‖_F/‖J_{s→28}−I‖_F = %.3f",
                          s, opn(P[s+1] - PAN[s+1]) / opn(P[s+1]), norm(P[s+1] - PAN[s+1]) / norm(P[s+1] - I)))
        end
        pp["share_QK"] = shQK; pp["share_norm"] = shNo; pp["share_QKnorm"] = shN; pp["err_read_AN"] = eT; pp["err_read_A"] = eA
        pp["cos_read_AN"] = cT
        per_prompt[string(p)] = pp
        P = nothing; PA = nothing; PAN = nothing; JT = nothing; GC.gc()
    end
    out["per_prompt"] = per_prompt

    # ══════════════════════ H4 : climat vs météo (amendement) ═════════════
    if length(PROMPT_IDXS) >= 2
        emit("\n" * "="^92)
        emit("H4 -- dispersion par prompt du transport J_{s→28} autour de la moyenne (leave-one-out)")
        emit("="^92)
        Pall = Dict{Int,Vector{Matrix{Float64}}}(); Xall = Dict{Int,Any}(); Mall = Dict{Int,Any}()
        for p in PROMPT_IDXS
            JT = loadJ(p, "T")
            P = [Matrix{Float64}(I, D, D) for _ in 1:L+1]
            for s in L-1:-1:0; P[s+1] = P[s+2] * JT[s+1]; end
            Pall[p] = P[1:L]; JT = nothing; P = nothing; GC.gc()
            Mall[p] = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
            Xall[p] = [Float64.(x) for x in Mall[p]["X"]]
        end
        δ = Dict(s => Float64[] for s in 0:L-1)
        for p in PROMPT_IDXS
            others = filter(!=(p), PROMPT_IDXS)
            γf = Float64.(Mall[p]["gamma_final"]); w = Float64.(Mall[p]["w_read"])
            ld(v) = dot(w, γf .* v) / sqrt(mean(v .^ 2) + 1e-6)
            X = Xall[p]; truth = ld(X[end])
            lines = String[]
            for s in 0:L-1
                Pbar = sum(Pall[q][s+1] for q in others) ./ length(others)
                push!(δ[s], norm(Pall[p][s+1] - Pbar) / norm(Pall[p][s+1] - I))
                if s in (1, 7, 14, 21, 25)
                    push!(lines, @sprintf("s=%2d: exacte %+.2f  moyenne %+.2f  logit-lens %+.2f", s,
                                          ld(Pall[p][s+1] * X[s+1]), ld(Pbar * X[s+1]), ld(X[s+1])))
                end
            end
            emit(@sprintf("  prompt %2d : vraie logit-diff finale %+.3f ; lentilles -> %s", p, truth, join(lines, " | ")))
        end
        medδ = [median(δ[s]) for s in 0:L-1]
        emit("  médiane δ_s (s=0..27) : " * join([@sprintf("%.3f", x) for x in medδ], " "))
        fr4 = count(>(0.5), medδ[2:21]) / 20
        h4 = fr4 >= 0.5 ? "VRAIE (transport spécifique au prompt : la moyenne est un climat)" :
             all(<(0.2), medδ[2:21]) ? "FAUSSE (la moyenne représente chaque prompt)" : "PARTIELLE"
        emit(@sprintf("  fraction des s∈1..20 avec médiane δ_s > 0.5 : %.2f", fr4))
        emit("H4 -> " * h4)
        out["H4"] = Dict("median_delta" => medδ, "delta" => Dict(string(s) => δ[s] for s in 0:L-1), "verdict" => h4)
        Pall = nothing; GC.gc()

    end

    # ══════════════════════ VERDICTS ═══════════════════════════════════════
    emit("\n" * "#"^92)
    emit("VERDICTS (seuils pré-enregistrés)")
    emit("#"^92)
    mG = median(G_1_28); mGs = Dict(s => median(v) for (s, v) in G_s7)
    emit(@sprintf("\nH1 : médiane G(1,28) = %.3g ; médianes G(s,s+7) : s=1: %.3g  s=7: %.3g  s=14: %.3g  s=21: %.3g",
                  mG, mGs[1], mGs[7], mGs[14], mGs[21]))
    h1 = (mG > 100 && all(v > 10 for v in values(mGs))) ? "VRAIE (chaînage structurellement inutilisable)" :
         all(v < 3 for v in values(mGs)) ? "FAUSSE (chaînage quasi serré)" : "MIXTE"
    emit("H1 -> " * h1)

    er_mean_full = erank2(svdvals(prod_range(Jmean, 1, L)))
    er_mean_192 = erank2(svdvals(trunc_prod_backward(Jmean, 1, 192)))
    er_mean_512 = erank2(svdvals(trunc_prod_backward(Jmean, 1, 512)))
    mer = median(erank_pp)
    emit(@sprintf("\nH2a : erank₂(J_{1→28}) par prompt = %s ; médiane = %.2f",
                  join([@sprintf("%.2f", x) for x in erank_pp], ", "), mer))
    emit("H2a -> " * (mer < 50 ? "VRAIE (entonnoir présent par prompt)" : mer > 200 ? "FAUSSE" : "INDÉTERMINÉE (50–200)"))
    emit(@sprintf("H2b : produit des moyennes : plein = %.2f ; tronqué-512 = %.2f ; tronqué-192 = %.2f",
                  er_mean_full, er_mean_512, er_mean_192))
    emit(@sprintf("      par prompt tronqué-192 (médiane) = %.2f ; tronqué-512 = %.2f",
                  median(erank_pp_tr[192]), median(erank_pp_tr[512])))
    ratio = er_mean_192 / mer
    emit(@sprintf("      rapport erank(moyennes, tronqué-192) / médiane erank(par prompt) = %.3f", ratio))
    emit("H2b -> " * (ratio < 1/3 ? "VRAIE (moyennage/troncature abaisse substantiellement le rang)" :
                      (1/2 <= ratio <= 2) ? "FAUSSE (pas d'effet notable)" : "INDÉTERMINÉE"))

    med_err = [median(err_read[s]) for s in 0:L-1]
    med_errA = [median(errA_read[s]) for s in 0:L-1]
    emit("\nH3 : médiane (prompts) err_s, s=0..27 : " * join([@sprintf("%.3f", x) for x in med_err], " "))
    emit("     médiane (prompts) err_s QK seul     : " * join([@sprintf("%.3f", x) for x in med_errA], " "))
    frac = count(>(0.3), med_err[2:21]) / 20
    h3 = frac >= 0.5 ? "VRAIE (la linéarisation gelée manque une part substantielle)" :
         all(<(0.1), med_err[2:end]) ? "FAUSSE (fidèle)" : "PARTIELLE"
    emit(@sprintf("     fraction des s∈1..20 avec médiane err_s > 0.3 : %.2f", frac))
    emit("H3 -> " * h3)
    out["verdicts"] = Dict("H1" => h1, "G_1_28" => G_1_28, "G_s7" => Dict(string(k) => v for (k, v) in G_s7),
                           "erank_pp" => erank_pp, "erank_mean_full" => er_mean_full,
                           "erank_mean_512" => er_mean_512, "erank_mean_192" => er_mean_192,
                           "erank_pp_tr192" => erank_pp_tr[192], "erank_pp_tr512" => erank_pp_tr[512],
                           "med_err_read_AN" => med_err, "med_err_read_A" => med_errA, "H3" => h3)
end
open(JOUT, "w") do f; JSON.print(f, out); end
println("\nÉcrit : ", RES, " et ", JOUT)
