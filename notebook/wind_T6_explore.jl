# ══════════════════════════════════════════════════════════════════════════════
# WIND-T6 (EXPLORATOIRE, inspiration seulement) -- anatomie de l'effondrement de rang de J_{1→28}
# Pour chaque prompt et variante (T, A, AN) :
#   - par couche k = 1..27 : σ₁, σ₂ de J_k ; σ₁ et rang stable de B_k = J_k − I
#   - produits partiels P_k = J_27···J_k : σ₁, σ₂, ‖·‖_F, p₁ = σ₁²/‖·‖_F², ern post-vraie-norme
#   - alignement du vecteur singulier gauche dominant de J_k avec la sensibilité aval :
#       α_k = ‖P_{k+1} u₁(J_k)‖ / σ₁(P_{k+1})
#   - trajectoire de la direction d'effondrement v = v₁(P_1) : w_k = J_k···J_1 v,
#       gain g_k = ‖w_k‖/‖w_{k−1}‖, persistance cos(ŵ_k, ŵ_{k−1}), cos(ŵ_{k-1}, v₁(J_k)), cos(ŵ_k, u₁(J_k))
# USAGE : julia --project=. notebook/wind_T6_explore.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = joinpath(@__DIR__, "wind_data")
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const RES = joinpath(@__DIR__, get(ENV, "WIND_RESNAME", "wind_T6_explore_results.txt"))
const JOUT = joinpath(OUTDIR, get(ENV, "WIND_JNAME", "wind_T6_explore.json"))
const L, D = 28, 1536
BLAS.set_num_threads(16)

loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 1:L])      # Js[k+1] = J_k
erank2(σ) = (q = σ .^ 2 ./ sum(abs2, σ); q = q[q .> 0]; exp(-sum(q .* log.(q))))

out = Dict{String,Any}()
open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T6 exploration -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    for p in PROMPT_IDXS
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        X = [Float64.(v) for v in meta["X"]]
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x28 = X[29]
        Ntrue = ri .* (Diagonal(γ) * (I - (x28 * x28') .* (ri^2 / D)))
        for V in ("T", "A", "AN")
            t0 = time()
            Js = loadJ(p, V)
            s1 = zeros(27); s2 = zeros(27); b1 = zeros(27); bsr = zeros(27)
            U1 = zeros(D, 27); V1 = zeros(D, 27)
            for k in 1:27
                F = svd(Js[k+1]); s1[k] = F.S[1]; s2[k] = F.S[2]; U1[:, k] = F.U[:, 1]; V1[:, k] = F.V[:, 1]
                sb = svdvals(Js[k+1] - I); b1[k] = sb[1]; bsr[k] = sum(abs2, sb) / sb[1]^2
            end
            # produits partiels aval
            P = Matrix{Float64}(I, D, D)
            Ps1 = zeros(27); Ps2 = zeros(27); PF = zeros(27); Pern = zeros(27); α = zeros(27)
            Pp1 = zeros(27); v1P1 = zeros(D); u1P1 = zeros(D)
            for k in 27:-1:1
                # alignement avant multiplication : P = P_{k+1}
                α[k] = k == 27 ? 1.0 : norm(P * U1[:, k]) / Ps1[k+1]
                P = P * Js[k+1]
                F = svd(P)
                Ps1[k] = F.S[1]; Ps2[k] = F.S[2]; PF[k] = norm(F.S); Pp1[k] = F.S[1]^2 / PF[k]^2
                Pern[k] = erank2(svdvals(Ntrue * P))
                if k == 1; v1P1 .= F.V[:, 1]; u1P1 .= F.U[:, 1]; end
            end
            # trajectoire avant de la direction d'effondrement
            w = copy(v1P1); gk = zeros(27); pers = zeros(27); cv = zeros(27); cu = zeros(27); cx = zeros(27)
            for k in 1:27
                wn = Js[k+1] * w
                gk[k] = norm(wn) / norm(w)
                pers[k] = abs(dot(wn, w)) / (norm(wn) * norm(w))
                cv[k] = abs(dot(V1[:, k], w)) / norm(w)
                cu[k] = abs(dot(U1[:, k], wn)) / norm(wn)
                cx[k] = abs(dot(X[k+2], wn)) / (norm(wn) * norm(X[k+2]))
                w = wn
            end
            emit(@sprintf("\n=== PROMPT %d  variante %s  (%.0f s) ===", p, V, time() - t0))
            emit(@sprintf("   P_1 : σ₂/σ₁ = %.4f  p₁ = %.4f  ern(N·P_1) = %.2f", Ps2[1] / Ps1[1], Pp1[1], Pern[1]))
            emit("    k   σ₁(J)  σ₂/σ₁(J)  σ₁(B)  srk(B)   α_k    σ₁(P_k)   σ₂/σ₁(P)   p₁(P)   ern(NP)   g_k   pers   cos(w,v₁J) cos(Jw,u₁J) cos(Jw,x)")
            for k in 1:27
                emit(@sprintf("   %2d  %6.2f   %.3f   %6.2f  %6.1f   %.3f  %9.3e   %.4f   %.4f   %7.2f   %5.2f  %.3f   %.3f       %.3f       %.3f",
                              k, s1[k], s2[k] / s1[k], b1[k], bsr[k], α[k], Ps1[k], Ps2[k] / Ps1[k], Pp1[k], Pern[k],
                              gk[k], pers[k], cv[k], cu[k], cx[k]))
            end
            out["p$(p)_$(V)"] = Dict("s1" => s1, "s2" => s2, "b1" => b1, "bsr" => bsr, "alpha" => α,
                                     "Ps1" => Ps1, "Ps2" => Ps2, "PF" => PF, "Pp1" => Pp1, "Pern" => Pern,
                                     "g" => gk, "pers" => pers, "cv" => cv, "cu" => cu, "cx" => cx,
                                     "v1P1" => v1P1, "u1P1" => u1P1)
            Js = nothing; GC.gc()
        end
    end
end
open(JOUT, "w") do f; JSON.print(f, out); end
println("Écrit : ", RES, " et ", JOUT)
