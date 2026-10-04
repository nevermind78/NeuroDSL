# ══════════════════════════════════════════════════════════════════════════════
# WIND-T5 (EXPLORATOIRE) -- Mécanisme du sur-effondrement de la linéarisation figée.
# Pré-enregistré dans notebook/wind_T3_preregistration.md (section WIND-T5) avant le run.
# Hypothèse dérivée (F1–F3) : figer les normes supprime le projecteur (I − x̂x̂ᵀ) de chaque RMSNorm,
# et le SwiGLU à norme figée renvoie ≈ 2·m(h) le long de l'état -> « mode d'état » auto-amplifié.
# Mesures : gain d'état λ_k = ⟨x̂_{k+1}, J_k x̂_k⟩ ; produits partiels P_k = J_27···J_k :
# s_k = ‖P_k x̂_k‖/σ₁(P_k), |cos(v₁, x̂_k)|, erank post-norme ; masse5(u₁).
#
# USAGE : julia --project=. notebook/wind_collapse_check.jl
# ══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const RES = joinpath(@__DIR__, "wind_collapse_check_results.txt")
const L, D = 28, 1536
const VARIANTS = ["T", "A", "AN"]
const KS = [26, 21, 16, 11, 6, 1]
BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ 2))

loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 1:L])      # Js[k+1] = J_k

erank2(σ) = (q = σ .^ 2 ./ sum(abs2, σ); q = q[q .> 0]; exp(-sum(q .* log.(q))))
mass5(u) = sum(sort(abs2.(u), rev = true)[1:5]) / sum(abs2, u)

res = Dict{Tuple{Int,String},Any}()

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T5 (EXPLORATOIRE) -- mécanisme du sur-effondrement de la linéarisation figée")
    emit("Date : " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    emit("λ_k = ⟨x̂_{k+1}, J_k x̂_k⟩ ; P_k = J_27···J_k ; s_k = ‖P_k x̂_k‖/σ₁(P_k) ; cv = |cos(v₁(P_k), x̂_k)| ; " *
         "cu = |cos(u₁(P_k), x̂_28)| ; ern = erank post-vraie-norme")
    for p in PROMPT_IDXS
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        X = [Float64.(v) for v in meta["X"]]                       # X[k+1] = x_k
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]
        x28 = X[29]
        Ntrue = ri .* (Diagonal(γ) * (I - (x28 * x28') .* (ri^2 / D)))
        xh(k) = X[k+1] ./ norm(X[k+1])
        topx = sortperm(abs.(x28), rev = true)[1:5]
        emit(@sprintf("\nPROMPT %d  « …%s »", p, replace(meta["prompt"][max(1, end - 40):end], "\n" => "\\n")))
        emit(@sprintf("   ‖x_1‖ = %.1f  ‖x_28‖ = %.1f  ; x_28 : masse5 = %.3f, coords %s", norm(X[2]), norm(x28),
                      mass5(x28), string(topx .- 1)))
        for V in VARIANTS
            Js = loadJ(p, V)
            λ = [dot(xh(k + 1), Js[k+1] * xh(k)) for k in 1:27]
            logΛ = sum(log10.(abs.(λ)))
            P = Matrix{Float64}(I, D, D); rows = Dict{Int,NTuple{5,Float64}}()
            u1 = zeros(D)
            for k in 27:-1:1
                P = P * Js[k+1]
                if k in KS
                    F = svd(P)
                    s = norm(P * xh(k)) / F.S[1]
                    rows[k] = (s, abs(dot(F.V[:, 1], xh(k))), abs(dot(F.U[:, 1], xh(28))),
                               F.S[2] / F.S[1], erank2(svdvals(Ntrue * P)))
                    k == 1 && (u1 .= F.U[:, 1])
                end
            end
            m5 = mass5(u1); topu = sortperm(abs.(u1), rev = true)[1:5]
            emit(@sprintf("   [%s] log10 Λ = %+.2f  (λ_k < 0 : %d/27 ; λ médian %.3f, max %.3f @k=%d) ; u₁(P_1) masse5 = %.3f coords %s",
                          V, logΛ, count(<(0), λ), median(λ), maximum(λ), argmax(λ), m5, string(topu .- 1)))
            emit("        k     s_k     cv      cu     σ₂/σ₁    ern")
            for k in KS
                r = rows[k]
                emit(@sprintf("       %2d   %.3f   %.3f   %.3f   %.3f   %7.2f", k, r...))
            end
            res[(p, V)] = (logΛ = logΛ, smax = maximum(first(rows[k]) for k in KS), m5 = m5, λ = λ)
            Js = nothing; GC.gc()
        end
    end
    emit("\nPROFIL DES GAINS D'ÉTAT λ_k (médiane sur les prompts) :")
    emit("   k      T       A       AN")
    for k in 1:27
        emit(@sprintf("  %2d   %6.3f  %6.3f  %6.3f", k, [median(res[(p, V)].λ[k] for p in PROMPT_IDXS) for V in VARIANTS]...))
    end
    d = median(res[(p, "AN")].logΛ - res[(p, "T")].logΛ for p in PROMPT_IDXS)
    sT = median(res[(p, "T")].smax for p in PROMPT_IDXS)
    coll = filter(in(PROMPT_IDXS), [12, 22])
    p2 = all(res[(p, "AN")].smax >= 0.5 for p in coll) && sT < 0.2
    p3 = all(res[(p, "AN")].m5 >= 0.5 for p in coll)
    emit(@sprintf("\nP1 (médiane log10 Λ_AN − log10 Λ_T ≥ 1) : %s  (%.2f)", d >= 1 ? "VRAI" : "FAUX", d))
    emit(@sprintf("P2 (max_k s_k(AN) ≥ 0,5 sur %s et médiane max_k s_k(T) < 0,2) : %s  (AN : %s ; T : %.3f)",
                  string(coll), p2 ? "VRAI" : "FAUX", join([@sprintf("%.3f", res[(p, "AN")].smax) for p in coll], ", "), sT))
    emit(@sprintf("P3 (masse5(u₁ AN) ≥ 0,5 sur %s) : %s  (%s)", string(coll), p3 ? "VRAI" : "FAUX",
                  join([@sprintf("%.3f", res[(p, "AN")].m5) for p in coll], ", ")))
end
println("\nÉcrit : ", RES)
