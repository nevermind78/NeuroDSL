# ══════════════════════════════════════════════════════════════════════════════
# WIND-T2b -- comme WIND-T2, mais APRÈS la RMSNorm finale, c'est-à-dire ce que la tête de
# lecture voit réellement : N = rms⁻¹·diag(γ)·(I − x xᵀ/(n·rms²)), x = x_28 (dernier token).
# N efface la direction radiale x̂. Question : l'excès de concentration du produit vrai sur les
# substituts à spectres fixés (erank 20 vs ~50 en WIND-T2) survit-il, ou était-il radial ?
# Mêmes substituts (même graine, mêmes tirages) que WIND-T2 :
#   (S) signes : J_k' = (U_k D_k) Σ_k V_kᵀ    (H) Haar : U_27 Σ_27 Q_26 ··· Q_1 Σ_1
# Contrôle intégré : les colonnes "avant" doivent reproduire WIND-T2 à l'identique.
#
# USAGE : julia --project=. notebook/wind_postnorm_check.jl   (WIND_NSUR, défaut 4)
# ══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra, Statistics, Printf, Dates, Random, JSON

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const NSUR = parse(Int, get(ENV, "WIND_NSUR", "4"))
const RES = joinpath(@__DIR__, "wind_postnorm_check_results.txt")
const L, D = 28, 1536
BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ 2))

loadJ(p) = (A = Array{Float32}(undef, D, D, L);
            open(joinpath(OUTDIR, "wind_J_p$(p)_T.bin"), "r") do f; read!(f, A); end;
            [Float64.(A[:, :, k]) for k in 1:L])          # Js[k+1] = J_k

erank2(σ) = (q = σ .^ 2 ./ sum(abs2, σ); q = q[q .> 0]; exp(-sum(q .* log.(q))))
haar(rng) = (F = qr(randn(rng, D, D)); Matrix(F.Q) * Diagonal(sign.(diag(F.R))))

# avant / après la norme finale : erank, σ₂/σ₁, |cos(u₁, x̂)|
function metrics(T, N, xh)
    F = svd(T); Fn = svdvals(N * T)
    (er = erank2(F.S), s21 = F.S[2] / F.S[1], rad = abs(dot(F.U[:, 1], xh)),
     ern = erank2(Fn), s21n = Fn[2] / Fn[1])
end
fmt(v) = @sprintf("%7.2f ± %-5.2f", mean(v), length(v) > 1 ? std(v) : 0.0)

const FIELDS = (:er, :s21, :rad, :ern, :s21n)
rng = MersenneTwister(20260930)
agg = Dict(k => Dict(f => Float64[] for f in FIELDS) for k in (:T, :S, :H))

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T2b -- J_{1→28} avant / après la RMSNorm finale, vrai vs substituts à spectres fixés")
    emit("Date : " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ") * "   NSUR = $NSUR")
    emit("er/s21/rad = avant la norme (rad = |cos(u₁, x̂_28)|) ; ern/s21n = après N = Jacobien de la norme finale")
    for p in PROMPT_IDXS
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        x = Float64.(meta["X"][end]); γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]
        @assert length(x) == D
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        xh = x ./ norm(x)
        Js = loadJ(p)[2:L]                                    # J_1 .. J_27
        Fs = [svd(J) for J in Js]
        T = Js[1]; for k in 2:length(Js); T = Js[k] * T; end
        mT = metrics(T, N, xh)
        mS = [begin
                  Tk = (Fs[1].U .* rand(rng, (-1.0, 1.0), 1, D)) * Diagonal(Fs[1].S) * Fs[1].Vt
                  for k in 2:length(Fs)
                      Jk = (Fs[k].U .* rand(rng, (-1.0, 1.0), 1, D)) * Diagonal(Fs[k].S) * Fs[k].Vt
                      Tk = Jk * Tk
                  end
                  metrics(Tk, N, xh)
              end for _ in 1:NSUR]
        mH = [begin
                  Tk = Matrix(Diagonal(Fs[1].S))
                  for k in 2:length(Fs); Tk = Diagonal(Fs[k].S) * (haar(rng) * Tk); end
                  metrics(Fs[end].U * Tk, N, xh)
              end for _ in 1:NSUR]
        emit(@sprintf("\nPROMPT %d   (‖x_28‖ = %.1f)", p, norm(x)))
        emit("            vrai        (S) signes, P_k gardés    (H) Haar, spectres seuls")
        for f in FIELDS
            emit(@sprintf("  %-4s  %9.3f       %s          %s", f, getfield(mT, f),
                          fmt(getfield.(mS, f)), fmt(getfield.(mH, f))))
            push!(agg[:T][f], getfield(mT, f))
            push!(agg[:S][f], mean(getfield.(mS, f))); push!(agg[:H][f], mean(getfield.(mH, f)))
        end
        Js = nothing; Fs = nothing; GC.gc()
    end
    emit("\nMÉDIANES sur les prompts :")
    emit("            vrai     (S) signes    (H) Haar")
    for f in FIELDS
        emit(@sprintf("  %-4s  %8.3f    %8.3f     %8.3f", f, median(agg[:T][f]), median(agg[:S][f]), median(agg[:H][f])))
    end
end
println("\nÉcrit : ", RES)
