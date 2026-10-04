# ══════════════════════════════════════════════════════════════════════════════
# WIND-T8 -- verdicts H2 (S1–S4, prédictions WIND-T6) et H3 (F1–F2, réplication de l'entonnoir).
# Pré-enregistré : notebook/wind_T8_preregistration.md. Nécessite les variantes T, A, AN.
# N = vraie norme finale (sauf F2 : N_AN = rms⁻¹·diag(γ) pour AN, comme WIND-T4).
#
# USAGE : WIND_PROMPTS=1,...,50 julia --project=. notebook/wind_T8_secondary.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON, Random

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data_T8"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const OUT = get(ENV, "WIND_SEC", joinpath(@__DIR__, "wind_T8_secondary.json"))
const RES = get(ENV, "WIND_SEC_RES", joinpath(@__DIR__, "wind_T8_secondary_results.txt"))
const L, D = 28, 1536
const K_R, K_Q, NSUR = 40, 20, 4
BLAS.set_num_threads(parse(Int, get(ENV, "WIND_BLAS", "16")))

loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])                     # J_1 .. J_27
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
chain(Js) = (P = Js[1]; for k in 2:length(Js); P = Js[k] * P; end; P)
spec(M) = (sv = svdvals(M); τ = sqrt(sum(abs2, sv[2:end])); (sigma1 = sv[1], tau = τ, b = sv[1] / τ, ern = ent(sv)))

sec = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T8 SECONDAIRE (S1–S4, F1–F2) -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PROMPT_IDXS
        haskey(sec, string(p)) && continue
        t0 = time(); rng = MersenneTwister(20261002 + p)
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; X = [Float64.(v) for v in meta["X"]]
        x = X[end]
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D))); Nfz = Matrix(ri .* Diagonal(γ))
        JT = loadJ(p, "T"); JA = loadJ(p, "A"); JN = loadJ(p, "AN")
        PT = chain(JT); PA = chain(JA); PN = chain(JN)
        sT = spec(N * PT); sA = spec(N * PA); sN = spec(N * PN)
        rec = Dict{String,Any}("T" => Dict(pairs(sT)), "A" => Dict(pairs(sA)), "AN" => Dict(pairs(sN)))
        # S3 : nul des signes sur R_k et alignement sur z (seulement si AN s'effondre)
        if sN.ern <= 3
            R = [JN[k] - JA[k] for k in 1:27]
            σd = [svdvals(N * chain([JA[k] + rand(rng, (-1.0, 1.0)) .* R[k] for k in 1:27]))[1] for _ in 1:K_R]
            z = zeros(D)
            for k in 1:10
                v = X[k+2] .- X[k+1]                                  # Δ_k = x_{k+1} − x_k
                for j in k+1:27; v = JN[j] * v; end
                z .+= v
            end
            z = N * z
            u1 = svd(N * PN).U[:, 1]
            rec["S3"] = Dict("q_sigma1" => count(<(sN.sigma1), σd) / K_R, "cos_u1_z" => abs(dot(u1, z)) / norm(z))
        end
        # S4 : nul des signes QK (E_k = J_T − J_A), K = 20
        E = [JT[k] - JA[k] for k in 1:27]
        eq = [ent(svdvals(N * chain([JA[k] + rand(rng, (-1.0, 1.0)) .* E[k] for k in 1:27]))) for _ in 1:K_Q]
        E = nothing
        rec["S4"] = Dict("ern_T" => sT.ern, "null_median" => median(eq), "above" => sT.ern > median(eq))
        # F1 : substituts à signes de WIND-T2 (garde spectres et P_k)
        Fs = [svd(J) for J in JT]
        es = [begin
                  Tk = (Fs[1].U .* rand(rng, (-1.0, 1.0), 1, D)) * Diagonal(Fs[1].S) * Fs[1].Vt
                  for k in 2:27
                      Tk = ((Fs[k].U .* rand(rng, (-1.0, 1.0), 1, D)) * Diagonal(Fs[k].S) * Fs[k].Vt) * Tk
                  end
                  ent(svdvals(N * Tk))
              end for _ in 1:NSUR]
        Fs = nothing
        rec["F1"] = Dict("ern_T" => sT.ern, "sur_mean" => mean(es), "sur_sd" => std(es), "c" => sT.ern / mean(es),
                         "below" => sT.ern < mean(es) - 2std(es))
        # F2 : recouvrement des sous-espaces de sortie top-10, T (vraie N) vs AN (N figée)
        UT = svd(N * PT).U[:, 1:10]; UN = svd(Nfz * PN).U[:, 1:10]
        rec["F2"] = sum(abs2, UT' * UN) / 10
        JT = nothing; JA = nothing; JN = nothing; GC.gc()
        sec[string(p)] = rec
        open(OUT, "w") do f; JSON.print(f, sec); end
        emit(@sprintf("  p%-3d ern T %.2f A %.2f AN %.2f | τ_A/τ_T %.2f τ_AN/τ_T %.2f | S4 %s | F1 c %.2f %s | F2 %.3f%s | %.0f s",
                      p, sT.ern, sA.ern, sN.ern, sA.tau / sT.tau, sN.tau / sT.tau, rec["S4"]["above"] ? "au-dessus" : "en dessous",
                      rec["F1"]["c"], rec["F1"]["below"] ? "<−2σ" : "≥−2σ", rec["F2"],
                      haskey(rec, "S3") ? @sprintf(" | S3 q %.3f cos %.3f", rec["S3"]["q_sigma1"], rec["S3"]["cos_u1_z"]) : "",
                      time() - t0))
    end
    secr = JSON.parsefile(OUT)       # relecture : clés homogènes (les records frais ont des clés Symbol)
    ps = [string(p) for p in PROMPT_IDXS if haskey(secr, string(p))]
    if length(ps) >= 10
        r = [secr[p] for p in ps]
        s1 = count(x -> 0.7 <= x["A"]["tau"] / x["T"]["tau"] <= 1.1 && 0.7 <= x["AN"]["tau"] / x["T"]["tau"] <= 1.1, r) / length(r)
        pairs2 = [(x[V]["ern"] <= 3) == (x[V]["b"] >= 2.6) for x in r for V in ("T", "A", "AN")]
        s2 = count(pairs2) / length(pairs2)
        col = [x for x in r if haskey(x, "S3")]
        s3 = isempty(col) ? NaN : count(x -> x["S3"]["q_sigma1"] >= 0.975 && x["S3"]["cos_u1_z"] >= 0.9, col) / length(col)
        s4 = count(x -> x["S4"]["above"], r) / length(r)
        f1a = count(x -> x["F1"]["below"], r) / length(r); f1c = median(x["F1"]["c"] for x in r)
        f2 = median(x["F2"] for x in r)
        emit(@sprintf("\nVERDICTS H2/H3 (n = %d prompts)", length(ps)))
        emit(@sprintf("S1 (τ_A/τ_T et τ_AN/τ_T ∈ [0,7;1,1] sur ≥ 80 %%) : %s  (%.3f)", s1 >= 0.8 ? "VRAI" : "FAUX", s1))
        emit(@sprintf("S2 (ern ≤ 3 ⟺ b ≥ 2,6 sur ≥ 95 %% des couples) : %s  (%.3f sur %d)", s2 >= 0.95 ? "VRAI" : "FAUX", s2, length(pairs2)))
        emit(length(col) < 3 ? "S3 : non applicable ($(length(col)) prompt(s) effondré(s))" :
             @sprintf("S3 (q ≥ 0,975 et |cos| ≥ 0,9 sur ≥ 80 %% des effondrés) : %s  (%.3f sur %d)", s3 >= 0.8 ? "VRAI" : "FAUX", s3, length(col)))
        emit(@sprintf("S4 (T au-dessus de la médiane du nul QK sur ≥ 80 %%) : %s  (%.3f)", s4 >= 0.8 ? "VRAI" : "FAUX", s4))
        emit(@sprintf("F1 (T < moyenne − 2σ sur ≥ 80 %% ET médiane c ≤ 0,6) : %s  (%.3f ; c médian %.3f)",
                      (f1a >= 0.8 && f1c <= 0.6) ? "VRAI" : "FAUX", f1a, f1c))
        emit(@sprintf("F2 (médiane cos² top-10 T vs AN ≥ 0,5) : %s  (%.3f)", f2 >= 0.5 ? "VRAI" : "FAUX", f2))
    end
end
println("Écrit : ", RES)
