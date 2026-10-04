# ══════════════════════════════════════════════════════════════════════════════
# WIND-T8 -- VÉRITÉ et verdicts H1 (C1–C4, D1) (pré-enregistré : notebook/wind_T8_preregistration.md).
# ern_true(θ) = erank₂(N·Π_k (J_k^A + θ(J_k^AN − J_k^A))) sur θ ∈ {0} ∪ 0,25 : 0,125 : 2,5.
# Porte : relèvement exact avec W complet à θ = 1 (doit égaler le produit vrai).
# Refuse de tourner si les prédictions figées n'existent pas.
#
# USAGE : WIND_PROMPTS=1,...,50 julia --project=. notebook/wind_T8_truth.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data_T8"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const PRED = get(ENV, "WIND_PRED", joinpath(@__DIR__, "wind_T8_predictions.json"))
const TRUTH = get(ENV, "WIND_TRUTH", joinpath(@__DIR__, "wind_T8_truth.json"))
const RES = get(ENV, "WIND_TRUTH_RES", joinpath(@__DIR__, "wind_T8_truth_results.txt"))
const L, D, Lb = 28, 1536, 27
const THETAS = vcat(0.0, collect(0.25:0.125:2.5))
const CAP = 2.5
BLAS.set_num_threads(16)
isfile(PRED) || error("prédictions figées absentes : $PRED -- la vérité ne peut pas être calculée avant")

loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
function θcross(θs, e)
    i = findfirst(<=(3.0), e)
    i === nothing && return Inf
    i == 1 && return θs[1]
    θs[i-1] + (θs[i] - θs[i-1]) * (e[i-1] - 3) / (e[i-1] - e[i])
end
function ranks(v)                       # rangs moyens en cas d'égalité
    o = sortperm(v); r = zeros(length(v)); i = 1
    while i <= length(v)
        j = i; while j < length(v) && v[o[j+1]] == v[o[i]]; j += 1; end
        r[o[i:j]] .= (i + j) / 2; i = j + 1
    end
    r
end
spearman(a, b) = cor(ranks(a), ranks(b))
wilson(k, n; z = 1.96) = (p = k / n; c = (p + z^2 / (2n)) / (1 + z^2 / n);
                          h = z * sqrt(p * (1 - p) / n + z^2 / (4n^2)) / (1 + z^2 / n); (c - h, c + h))

pred = JSON.parsefile(PRED)["predictions"]
truth = isfile(TRUTH) ? JSON.parsefile(TRUTH) : Dict{String,Any}()
open(RES, "a") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("\nWIND-T8 VÉRITÉ -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ") * "   prédictions : $(PRED)")
    for p in PROMPT_IDXS
        haskey(truth, string(p)) && continue
        t0 = time()
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x = Float64.(meta["X"][end])
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        JA = loadJ(p, "A"); JN = loadJ(p, "AN")
        et = Float64[]; sig = Dict{String,Any}(); gate = NaN
        for θ in THETAS
            P = JA[1] + θ .* (JN[1] - JA[1])
            for k in 2:27; P = (JA[k] + θ .* (JN[k] - JA[k])) * P; end
            MT = N * P; sv = svdvals(MT)
            push!(et, ent(sv))
            if θ == 0.0 || θ == 1.0
                τ = sqrt(sum(abs2, sv[2:end]))
                sig[θ == 0 ? "A" : "AN"] = Dict("sigma1" => sv[1], "tau" => τ, "b" => sv[1] / τ, "ern" => ent(sv))
            end
            if θ == 1.0
                buf = Array{Float64}(undef, D * 54 * 2 + D * D + 54 * 54)
                open(joinpath(OUTDIR, "wind_T8_lift_p$(p).bin"), "r") do f; read!(f, buf); end
                o = 0
                𝒰 = reshape(buf[o+1:o+D*54], D, 54); o += D * 54
                𝒱 = reshape(buf[o+1:o+54*D], 54, D); o += 54 * D
                MA = reshape(buf[o+1:o+D*D], D, D); o += D * D
                W = reshape(buf[o+1:o+54*54], 54, 54)
                gate = norm(MA + 𝒰 * inv(I - W) * 𝒱 - MT) / norm(MT)
            end
        end
        JA = nothing; JN = nothing; GC.gc()
        thT = θcross(THETAS[2:end], et[2:end])
        truth[string(p)] = Dict("thetas" => THETAS, "ern_true" => et, "theta_c_true" => thT,
                                "ern_true_1" => et[findfirst(==(1.0), THETAS)], "gate_lift" => gate, "spectra" => sig)
        f(v) = isfinite(v) ? @sprintf("%.3f", v) : "> 2,5"
        emit(@sprintf("  p%-3d θ_c^true %s (mod r=2 : %s) | ern A %.2f AN %.2f | b A %.2f AN %.2f | porte relèvement %.1e | %.0f s",
                      p, f(thT), f(pred[string(p)]["theta_c_mod2"]), sig["A"]["ern"], sig["AN"]["ern"],
                      sig["A"]["b"], sig["AN"]["b"], gate, time() - t0))
        open(TRUTH, "w") do fh; JSON.print(fh, truth); end
    end

    ps = [string(p) for p in PROMPT_IDXS if haskey(truth, string(p))]
    if length(ps) < 10
        emit("Moins de 10 prompts : pas de verdict (test du pipeline).")
    else
        cap(v) = min(v, CAP)
        tt = [cap(truth[p]["theta_c_true"]) for p in ps]
        emit(@sprintf("\nVERDICTS H1 (n = %d prompts ; porte relèvement max %.1e)", length(ps), maximum(truth[p]["gate_lift"] for p in ps)))
        sp = Dict{Int,Float64}(); fr = Dict{Int,Float64}(); nC2 = Dict{Int,Int}()
        for r in (1, 2, 3)
            tm = [cap(pred[p]["theta_c_mod$(r)"]) for p in ps]
            sp[r] = spearman(tm, tt)
            sel = [i for i in eachindex(ps) if tm[i] <= 2.3]
            nC2[r] = length(sel)
            fr[r] = isempty(sel) ? NaN : count(i -> abs(tt[i] - tm[i]) <= 0.15 * tm[i], sel) / length(sel)
            emit(@sprintf("  ordre %d : Spearman = %.3f ; C2 : %d prompts avec θ_c^mod ≤ 2,3, fraction à ±15 %% = %.3f", r, sp[r], nC2[r], fr[r]))
        end
        v1 = sp[2] >= 0.7 ? "VRAI" : sp[2] >= 0.4 ? "PARTIEL" : "FAUX"
        v2 = nC2[2] < 10 ? "PUISSANCE INSUFFISANTE" : fr[2] >= 0.7 ? "VRAI" : fr[2] >= 0.5 ? "PARTIEL" : "FAUX"
        emit("C1 (Spearman ordre 2 ≥ 0,7) : $v1  ($(round(sp[2], digits=3)))")
        emit("C2 (fraction à ±15 % ≥ 0,7) : $v2  ($(round(fr[2], digits=3)) sur $(nC2[2]))")
        pc = [pred[p]["theta_c_mod2"] < 1 for p in ps]; tc = [truth[p]["ern_true_1"] <= 3 for p in ps]
        ntc = count(tc)
        tp = count(pc .& tc); tn = count(.!pc .& .!tc); fp = count(pc .& .!tc); fn = count(.!pc .& tc)
        if ntc >= 3 && length(ps) - ntc >= 3
            ba = (tp / (tp + fn) + tn / (tn + fp)) / 2
            emit(@sprintf("C3 (exactitude équilibrée ≥ 0,8) : %s  (%.3f ; VP %d FN %d VN %d FP %d)", ba >= 0.8 ? "VRAI" : "FAUX", ba, tp, fn, tn, fp))
        else
            emit(@sprintf("C3 : descriptif (effondrements vrais %d) ; VP %d FN %d VN %d FP %d", ntc, tp, fn, tn, fp))
        end
        c4 = sp[1] <= sp[2] - 0.1 || (isfinite(fr[1]) && isfinite(fr[2]) && fr[1] <= fr[2] - 0.2)
        emit(@sprintf("C4 (ordre 2 nécessaire) : %s  (Spearman r=1 %.3f vs r=2 %.3f ; C2 r=1 %.3f vs r=2 %.3f ; r=3 descriptif %.3f / %.3f)",
                      c4 ? "VRAI" : "FAUX", sp[1], sp[2], fr[1], fr[2], sp[3], fr[3]))
        lo, hi = wilson(ntc, length(ps))
        emit(@sprintf("D1 : effondrements AN (ern_true(1) ≤ 3) = %d / %d = %.3f  (IC95 Wilson [%.3f ; %.3f])", ntc, length(ps), ntc / length(ps), lo, hi))
    end
end
println("Écrit : ", RES)
