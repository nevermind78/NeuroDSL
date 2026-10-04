# ══════════════════════════════════════════════════════════════════════════════
# WIND-T2 -- Les résultats H1 (G(1,28) ~ 1e17) et H2a (entonnoir, erank ~ 20) de WIND-1
# sont-ils des conséquences GÉNÉRIQUES des spectres par couche, ou une structure apprise ?
# On compare J_{1→28} = J_27···J_1 (variante T, dernier token) à deux substituts qui gardent
# EXACTEMENT le spectre singulier de chaque J_k :
#   (S) signes   : J_k' = (U_k D_k) Σ_k V_kᵀ, D_k = diag(±1) aléatoire -> garde aussi P_k = (V_{k+1}ᵀU_k)∘²
#   (H) Haar     : Σ_27 Q_26 Σ_26 ··· Q_1 Σ_1, Q_k de Haar -> ne garde que les spectres
# Si (H) reproduit G et erank, H1/H2a découlent des spectres seuls (aucun alignement appris requis).
#
# USAGE : julia --project=. notebook/wind_surrogate_check.jl   (WIND_NSUR = nb de substituts, défaut 4)
# ══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra, Statistics, Printf, Dates, Random

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const NSUR = parse(Int, get(ENV, "WIND_NSUR", "4"))
const RES = joinpath(@__DIR__, "wind_surrogate_check_results.txt")
const L, D = 28, 1536
BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ 2))

loadJ(p) = (A = Array{Float32}(undef, D, D, L);
            open(joinpath(OUTDIR, "wind_J_p$(p)_T.bin"), "r") do f; read!(f, A); end;
            [Float64.(A[:, :, k]) for k in 1:L])          # Js[k+1] = J_k

erank2(σ) = (q = σ .^ 2 ./ sum(abs2, σ); q = q[q .> 0]; exp(-sum(q .* log.(q))))
haar(rng) = (F = qr(randn(rng, D, D)); Matrix(F.Q) * Diagonal(sign.(diag(F.R))))

# métriques d'un produit T, étant donné Σ_k log10 σ₁(J_k)
function metrics(T, lsum)
    σ = svdvals(T)
    (G = lsum - log10(σ[1]), er = erank2(σ), fro = log10(sum(abs2, σ)), s21 = σ[2] / σ[1])
end
fmt(v) = @sprintf("%7.2f ± %-5.2f", mean(v), length(v) > 1 ? std(v) : 0.0)

rng = MersenneTwister(20260930)
agg = Dict(k => Dict(f => Float64[] for f in (:G, :er, :fro, :s21)) for k in (:T, :S, :H))

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T2 -- substituts à spectres fixés pour J_{1→28} (couches k = 1..27, variante T)")
    emit("Date : " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ") * "   NSUR = $NSUR")
    emit("G = log10(Π σ₁(J_k) / σ₁(J_{1→28}))   erank = exp(H(σ²/Σσ²))   fro = log10 ‖J_{1→28}‖_F²   s21 = σ₂/σ₁")
    for p in PROMPT_IDXS
        Js = loadJ(p)[2:L]                                    # J_1 .. J_27
        Fs = [svd(J) for J in Js]
        lsum = sum(log10(F.S[1]) for F in Fs)
        T = Js[1]; for k in 2:length(Js); T = Js[k] * T; end
        mT = metrics(T, lsum)
        mS = [begin
                  Tk = (Fs[1].U .* rand(rng, (-1.0, 1.0), 1, D)) * Diagonal(Fs[1].S) * Fs[1].Vt
                  for k in 2:length(Fs)
                      Jk = (Fs[k].U .* rand(rng, (-1.0, 1.0), 1, D)) * Diagonal(Fs[k].S) * Fs[k].Vt
                      Tk = Jk * Tk
                  end
                  metrics(Tk, lsum)
              end for _ in 1:NSUR]
        mH = [begin
                  Tk = Matrix(Diagonal(Fs[1].S))
                  for k in 2:length(Fs); Tk = Diagonal(Fs[k].S) * (haar(rng) * Tk); end
                  metrics(Tk, lsum)
              end for _ in 1:NSUR]
        emit(@sprintf("\nPROMPT %d", p))
        emit("            vrai        (S) signes, P_k gardés    (H) Haar, spectres seuls")
        for f in (:G, :er, :fro, :s21)
            emit(@sprintf("  %-4s  %9.2f       %s          %s", f, getfield(mT, f),
                          fmt(getfield.(mS, f)), fmt(getfield.(mH, f))))
            push!(agg[:T][f], getfield(mT, f))
            push!(agg[:S][f], mean(getfield.(mS, f))); push!(agg[:H][f], mean(getfield.(mH, f)))
        end
        Js = nothing; Fs = nothing; GC.gc()
    end
    emit("\nMÉDIANES sur les prompts :")
    emit("            vrai     (S) signes    (H) Haar")
    for f in (:G, :er, :fro, :s21)
        emit(@sprintf("  %-4s  %8.2f    %8.2f     %8.2f", f, median(agg[:T][f]), median(agg[:S][f]), median(agg[:H][f])))
    end
end
println("\nÉcrit : ", RES)
