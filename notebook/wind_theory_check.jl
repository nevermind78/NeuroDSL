# ══════════════════════════════════════════════════════════════════════════════
# WIND-T1 -- Test, sur les Jacobiennes WIND-1 existantes (aucune nouvelle mesure GPU),
# de trois prédictions de ‖J_{s→t}‖_F² (J_{s→t} = J_{t-1}···J_s, dernier token, variante T) :
#   (P) formule proposée   : M_{s→t} · Π_{k=s}^{t-2} 𝒜_k,  M = Σ_i Π_k σ_i^(k)² (spectres triés)
#   (M) matrices de transfert (espérance EXACTE sous signes ±1 aléatoires aux jonctions) :
#       𝔼 = 1ᵀ S_{t-1} P_{t-2} S_{t-2} ··· P_s s_s,  P_k = (V_{k+1}ᵀ U_k)∘²,  S_k = diag(σ^(k)²)
#   (H) appariement de Haar : Π_k ‖J_k‖_F² / n^{L-1}
# Tout en log pour éviter les débordements. Porte : pour L = 2, (M) doit égaler la vraie valeur
# à la précision machine (identité exacte, déjà vérifiée dans WIND-2).
#
# USAGE : julia --project=. notebook/wind_theory_check.jl
# ══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra, Statistics, Printf, Dates

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const RES = joinpath(@__DIR__, "wind_theory_check_results.txt")
const L, D = 28, 1536
const SPANS = [(1, 3), (1, 5), (1, 8), (1, 15), (1, 28), (7, 14), (14, 21), (21, 28)]
BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ 2))

loadJ(p) = (A = Array{Float32}(undef, D, D, L);
            open(joinpath(OUTDIR, "wind_J_p$(p)_T.bin"), "r") do f; read!(f, A); end;
            [Float64.(A[:, :, k]) for k in 1:L])          # Js[k+1] = J_k

logsumexp(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))

tab = Dict(sp => Dict("P" => Float64[], "M" => Float64[], "H" => Float64[]) for sp in SPANS)

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T1 -- test de trois prédictions de ‖J_{s→t}‖_F² sur les Jacobiennes WIND-1 (variante T)")
    emit("Date : " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    emit("Colonnes : log10(prédiction / vraie valeur). 0 = exact ; +11 = surestimation d'un facteur 1e11.")
    gate = 0.0
    for p in PROMPT_IDXS
        Js = loadJ(p)
        Fs = [svd(Js[k]) for k in 1:L]
        s2 = [F.S .^ 2 for F in Fs]                               # s2[k+1] = σ(J_k)²
        P = [(Fs[k+2].V' * Fs[k+1].U) .^ 2 for k in 0:L-2]        # P[k+1] = P_k (J_k -> J_{k+1})
        A = [dot(s2[k+2], P[k+1] * s2[k+1]) / dot(s2[k+2], s2[k+1]) for k in 0:L-2]
        Fs = nothing; GC.gc()
        emit(@sprintf("\nPROMPT %d : 𝒜_k médian (k=1..26) = %.3f", p, median(A[2:27])))
        emit("   span      log10 vraie ‖·‖_F²   (P) proposée   (M) transfert   (H) Haar")
        for (s, t) in SPANS
            Tm = Matrix{Float64}(I, D, D)
            for k in s:t-1; Tm = Js[k+1] * Tm; end
            lt = log(sum(abs2, Tm))
            # (M) récurrence en log : w ← S_{k+1} P_k w
            w = copy(s2[s+1]); lscale = 0.0
            for k in s:t-2
                w = s2[k+2] .* (P[k+1] * w)
                m = maximum(w); w ./= m; lscale += log(m)
            end
            lM = lscale + log(sum(w))
            # (P) formule proposée
            lMperf = logsumexp(sum(log.(s2[k+1]) for k in s:t-1))
            lP = lMperf + sum(log.(A[k+1]) for k in s:t-2; init = 0.0)
            # (H) Haar
            lH = sum(log(sum(s2[k+1])) for k in s:t-1) - (t - s - 1) * log(D)
            (t - s == 2) && (gate = max(gate, abs(lM - lt)))
            r = (lP - lt, lM - lt, lH - lt) ./ log(10)
            push!(tab[(s, t)]["P"], r[1]); push!(tab[(s, t)]["M"], r[2]); push!(tab[(s, t)]["H"], r[3])
            emit(@sprintf("  %2d→%-2d        %8.2f           %+8.2f        %+8.2f       %+8.2f",
                          s, t, lt / log(10), r[1], r[2], r[3]))
        end
        Js = nothing; P = nothing; GC.gc()
    end
    emit(@sprintf("\nPORTE L=2 : |log (M) − log vraie| max = %.2e  (identité exacte attendue) -> %s",
                  gate, gate < 1e-8 ? "OK" : "ÉCHEC"))
    emit("\nMÉDIANES sur les prompts, log10(prédiction / vraie) :")
    emit("   span      (P) proposée   (M) transfert   (H) Haar")
    for sp in SPANS
        emit(@sprintf("  %2d→%-2d       %+8.2f        %+8.2f       %+8.2f", sp[1], sp[2],
                      median(tab[sp]["P"]), median(tab[sp]["M"]), median(tab[sp]["H"])))
    end
end
println("\nÉcrit : ", RES)
