# ══════════════════════════════════════════════════════════════════════════════
# WIND-T6 (EXPLORATOIRE) -- localisation causale de l'effondrement par produits hybrides.
# Produit J_{1→28} = J_27···J_1 où chaque couche k vient d'une variante choisie (T, A ou AN) ;
# métrique : ern = erank₂(N·produit) avec la VRAIE norme finale N (isole l'effet des couches),
# et p₁ = σ₁²/‖·‖_F² du produit brut.
#   H1 : base A, couche k seule remise en T        (quelles couches QK « protègent » ?)
#   H2 : base T, couche k seule passée en A        (une seule couche figée suffit-elle ?)
#   H3 : base A, couche k seule passée en AN       (gel des normes, une couche)
#   H4 : base AN, couche k seule remise en A
#   W  : fenêtres de couches
# USAGE : julia --project=. notebook/wind_T6_hybrid.jl
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, Statistics, Printf, Dates, JSON

const OUTDIR = joinpath(@__DIR__, "wind_data")
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const RES = joinpath(@__DIR__, get(ENV, "WIND_RESNAME", "wind_T6_hybrid_results.txt"))
const L, D = 28, 1536
BLAS.set_num_threads(16)

loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])      # J[k] = J_k, k = 1..27
ent(λ) = (q = max.(λ, 0) ./ sum(max.(λ, 0)); q = q[q .> 0]; exp(-sum(q .* log.(q))))
function stats(N, P)
    λN = eigvals(Symmetric(N * P * (N * P)'))
    λP = eigvals(Symmetric(P * P'))
    return ent(λN), maximum(λP) / sum(λP)
end

const WINDOWS = [1:1, 1:3, 1:5, 1:10, 6:10, 11:27, 2:27, 6:27, 11:15, 16:20, 21:27]

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T6 hybrides -- " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    emit("ern = erank₂(N_vraie · J_27···J_1) ; p₁ = σ₁²/‖·‖_F² du produit brut")
    for p in PROMPT_IDXS
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; x = Float64.(meta["X"][end])
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        J = Dict(V => loadJ(p, V) for V in ("T", "A", "AN"))
        prodof(ch) = (P = J[ch[1]][1]; for k in 2:27; P = J[ch[k]][k] * P; end; P)
        base = Dict(V => stats(N, prodof(fill(V, 27))) for V in ("T", "A", "AN"))
        emit(@sprintf("\nPROMPT %d : ern T = %.2f  A = %.2f  AN = %.2f   (p₁ : %.3f / %.3f / %.3f)", p,
                      base["T"][1], base["A"][1], base["AN"][1], base["T"][2], base["A"][2], base["AN"][2]))
        for (lab, B, S) in (("H1 base A, k→T", "A", "T"), ("H2 base T, k→A", "T", "A"),
                            ("H3 base A, k→AN", "A", "AN"), ("H4 base AN, k→A", "AN", "A"))
            # préfixes R[k] = J_{k-1}···J_1 (R[1] = I), suffixes S[k] = J_27···J_{k+1} (S[27] = I)
            Jb = J[B]
            R = Vector{Matrix{Float64}}(undef, 27); R[1] = Matrix{Float64}(I, D, D)
            for k in 2:27; R[k] = Jb[k-1] * R[k-1]; end
            Sf = Vector{Matrix{Float64}}(undef, 27); Sf[27] = Matrix{Float64}(I, D, D)
            for k in 26:-1:1; Sf[k] = Sf[k+1] * Jb[k+1]; end
            e = zeros(27); q = zeros(27)
            for k in 1:27
                e[k], q[k] = stats(N, Sf[k] * (J[S][k] * R[k]))
            end
            emit(@sprintf("  %-18s ern_k : ", lab) * join([@sprintf("%6.1f", v) for v in e]))
            emit(@sprintf("  %-18s p₁_k  : ", "") * join([@sprintf("%6.3f", v) for v in q]))
            R = nothing; Sf = nothing; GC.gc()
        end
        emit("  fenêtres W (couches de W prises dans la variante S, le reste dans la base B) :")
        emit("     W          A→T(W)   T→A(W)   A→AN(W)  AN→A(W)   T→AN(W)")
        for W in WINDOWS
            vals = Float64[]
            for (B, S) in (("A", "T"), ("T", "A"), ("A", "AN"), ("AN", "A"), ("T", "AN"))
                ch = [k in W ? S : B for k in 1:27]
                push!(vals, stats(N, prodof(ch))[1])
            end
            emit(@sprintf("     %-9s  %7.2f  %7.2f  %7.2f  %7.2f  %7.2f", string(W), vals...))
        end
        J = nothing; GC.gc()
    end
end
println("Écrit : ", RES)
