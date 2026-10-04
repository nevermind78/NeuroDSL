# ══════════════════════════════════════════════════════════════════════════════
# WIND-T4 -- La linéarisation figée garde-t-elle l'entonnoir ? (pré-enregistré :
# notebook/wind_T3_preregistration.md). Pour T (vrai), A (attention figée), AN (attention +
# normes figées ≈ LRP(AH+LN)) : erank post-norme du produit J_{1→28}, coefficient de cohérence
# c = ern_vrai / ern_substituts-signes, et recouvrement des sous-espaces de sortie/entrée avec T.
# Pour AN la norme finale est figée aussi : N_AN = rms⁻¹·diag(γ) (sans projecteur radial).
#
# USAGE : julia --project=. notebook/wind_frozen_funnel_check.jl   (WIND_NSUR, défaut 4)
# ══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra, Statistics, Printf, Dates, Random, JSON

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const NSUR = parse(Int, get(ENV, "WIND_NSUR", "4"))
const RES = joinpath(@__DIR__, "wind_frozen_funnel_check_results.txt")
const L, D = 28, 1536
const VARIANTS = ["T", "A", "AN"]
const RS = [5, 10, 20]
BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ 2))

loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 1:L])      # Js[k+1] = J_k

erank2(σ) = (q = σ .^ 2 ./ sum(abs2, σ); q = q[q .> 0]; exp(-sum(q .* log.(q))))
overlap(A, B, r) = sum(abs2, A[:, 1:r]' * B[:, 1:r]) / r          # moyenne des cos² principaux

rng = MersenneTwister(20261001)
agg = Dict(V => Dict{String,Vector{Float64}}() for V in VARIANTS)
push_!(V, key, v) = push!(get!(agg[V], key, Float64[]), v)

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T4 -- entonnoir post-norme : vrai (T) vs linéarisations figées (A, AN)")
    emit("Date : " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ") * "   NSUR = $NSUR")
    emit("ern = erank₂(N_V·J_{1→28}) ; ernTN = même produit avec la vraie N (isole l'effet des couches)")
    emit("c = ern / ern_substituts-signes ; ovU_r / ovV_r = cos² moyen des sous-espaces top-r de sortie / d'entrée vs T")
    emit("hasard pour ov : r/n = " * join([@sprintf("%.4f", r / D) for r in RS], ", "))
    for p in PROMPT_IDXS
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        x = Float64.(meta["X"][end]); γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]
        Ntrue = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        Nfroz = Matrix(ri .* Diagonal(γ))
        emit(@sprintf("\nPROMPT %d", p))
        emit("   var     ern    ernTN   ern_S (moy±sd)      c      ovU5   ovU10  ovU20   ovV5   ovV10  ovV20   err_rel")
        FT = nothing; MT = nothing
        for V in VARIANTS
            NV = V == "AN" ? Nfroz : Ntrue
            Js = loadJ(p, V)[2:L]
            T = Js[1]; for k in 2:27; T = Js[k] * T; end
            M = NV * T; F = svd(M)
            ern = erank2(F.S); ernTN = erank2(svdvals(Ntrue * T))
            Us = Vector{Matrix{Float64}}(undef, 26); SVts = Vector{Matrix{Float64}}(undef, 26)
            for k in 1:26
                G = svd(Js[k]); Us[k] = G.U; SVts[k] = Diagonal(G.S) * G.Vt
            end
            ernS = [begin
                        Tk = (Us[1] .* rand(rng, (-1.0, 1.0), 1, D)) * SVts[1]
                        for k in 2:26; Tk = ((Us[k] .* rand(rng, (-1.0, 1.0), 1, D)) * SVts[k]) * Tk; end
                        erank2(svdvals(NV * (Js[27] * Tk)))
                    end for _ in 1:NSUR]
            c = ern / mean(ernS)
            if V == "T"
                FT = F; MT = M
            end
            ovU = [overlap(FT.U, F.U, r) for r in RS]; ovV = [overlap(FT.V, F.V, r) for r in RS]
            err = norm(M - MT) / norm(MT)
            emit(@sprintf("   %-4s  %6.2f  %6.2f   %6.2f ± %-5.2f   %5.2f   %5.3f  %5.3f  %5.3f   %5.3f  %5.3f  %5.3f   %6.3f",
                          V, ern, ernTN, mean(ernS), std(ernS), c, ovU..., ovV..., err))
            for (key, v) in zip(["ern", "ernTN", "ernS", "c", "ovU5", "ovU10", "ovU20", "ovV5", "ovV10", "ovV20", "err"],
                                [ern, ernTN, mean(ernS), c, ovU..., ovV..., err])
                push_!(V, key, v)
            end
            Js = nothing; Us = nothing; SVts = nothing; GC.gc()
        end
        FT = nothing; MT = nothing; GC.gc()
    end
    emit("\nMÉDIANES sur les prompts :")
    emit("   var     ern    ernTN   ern_S      c      ovU5   ovU10  ovU20   ovV5   ovV10  ovV20   err_rel")
    for V in VARIANTS
        m(k) = median(agg[V][k])
        emit(@sprintf("   %-4s  %6.2f  %6.2f   %6.2f   %5.2f   %5.3f  %5.3f  %5.3f   %5.3f  %5.3f  %5.3f   %6.3f",
                      V, m("ern"), m("ernTN"), m("ernS"), m("c"), m("ovU5"), m("ovU10"), m("ovU20"),
                      m("ovV5"), m("ovV10"), m("ovV20"), m("err")))
    end
    eT = median(agg["T"]["ern"]); eAN = median(agg["AN"]["ern"]); cAN = median(agg["AN"]["c"])
    hf1 = eAN <= 1.5 * eT && cAN <= 0.6
    hf2 = median(agg["AN"]["ovU10"]) >= 0.5
    emit(@sprintf("\nHF1 (ern_AN ≤ 1,5·ern_T et c_AN ≤ 0,6) : %s  (%.2f vs %.2f ; c = %.2f)", hf1 ? "VRAI" : "FAUX", eAN, eT, cAN))
    emit(@sprintf("HF2 (cos² moyen top-10 sortie T vs AN ≥ 0,5) : %s  (%.3f ; hasard %.4f)",
                  hf2 ? "VRAI" : "FAUX", median(agg["AN"]["ovU10"]), 10 / D))
end
println("\nÉcrit : ", RES)
