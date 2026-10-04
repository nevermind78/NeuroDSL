# ══════════════════════════════════════════════════════════════════════════════
# WIND-T3 -- Où se construit la cohérence de phase de l'entonnoir ? (pré-enregistré :
# notebook/wind_T3_preregistration.md). On garde les vraies phases sur un ensemble W de
# jonctions k (entre J_k et J_{k+1}, k = 1..26) et on randomise les autres :
# J_k -> (U_k D_k) Σ_k V_kᵀ, i.e. réflexion aléatoire R_k = U_k D_k U_kᵀ insérée à la jonction k.
# Métrique : ern = erank₂(N·T), N = Jacobien de la RMSNorm finale.
#
# USAGE : julia --project=. notebook/wind_localize_check.jl   (WIND_NSUR, défaut 3)
# ══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra, Statistics, Printf, Dates, Random, JSON

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const NSUR = parse(Int, get(ENV, "WIND_NSUR", "3"))
const RES = joinpath(@__DIR__, "wind_localize_check_results.txt")
const L, D = 28, 1536
BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ 2))

const CONFIGS = [("aucune", Int[]), ("toutes (porte)", collect(1:26)),
                 ("fenêtre 1-5", collect(1:5)), ("fenêtre 6-10", collect(6:10)),
                 ("fenêtre 11-15", collect(11:15)), ("fenêtre 16-20", collect(16:20)),
                 ("fenêtre 21-26", collect(21:26)),
                 ("suffixe ≥6", collect(6:26)), ("suffixe ≥11", collect(11:26)), ("suffixe ≥16", collect(16:26)),
                 ("préfixe ≤10", collect(1:10)), ("préfixe ≤15", collect(1:15)), ("préfixe ≤20", collect(1:20))]

loadJ(p) = (A = Array{Float32}(undef, D, D, L);
            open(joinpath(OUTDIR, "wind_J_p$(p)_T.bin"), "r") do f; read!(f, A); end;
            [Float64.(A[:, :, k]) for k in 1:L])          # Js[k+1] = J_k

erank2(σ) = (q = σ .^ 2 ./ sum(abs2, σ); q = q[q .> 0]; exp(-sum(q .* log.(q))))

function chain(Js, Us, SVts, keep, rng)       # Js[k] = J_k, k = 1..27
    Jk(k) = (k == 27 || k in keep) ? Js[k] : (Us[k] .* rand(rng, (-1.0, 1.0), 1, D)) * SVts[k]
    T = Jk(1); for k in 2:27; T = Jk(k) * T; end; T
end

rng = MersenneTwister(20260931)
ρs = Dict(name => Float64[] for (name, _) in CONFIGS)
gate = 0.0

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-T3 -- localisation de la cohérence de phase de l'entonnoir (post-norme finale, variante T)")
    emit("Date : " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ") * "   NSUR = $NSUR")
    emit("ρ(W) = [log ern_aucune − log ern_W] / [log ern_aucune − log ern_vrai] ; 1 = entonnoir entièrement récupéré")
    for p in PROMPT_IDXS
        meta = JSON.parsefile(joinpath(OUTDIR, "wind_meta_p$(p).json"))
        x = Float64.(meta["X"][end]); γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]
        N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D)))
        Js = loadJ(p)[2:L]                                    # J_1 .. J_27
        Us = Vector{Matrix{Float64}}(undef, 26); SVts = Vector{Matrix{Float64}}(undef, 26)
        for k in 1:26
            F = svd(Js[k]); Us[k] = F.U; SVts[k] = Diagonal(F.S) * F.Vt
        end
        T = Js[1]; for k in 2:27; T = Js[k] * T; end
        ern_true = erank2(svdvals(N * T))
        ern = Dict{String,Vector{Float64}}()
        for (name, keep) in CONFIGS
            nd = length(keep) == 26 ? 1 : NSUR
            ern[name] = [erank2(svdvals(N * chain(Js, Us, SVts, keep, rng))) for _ in 1:nd]
        end
        global gate = max(gate, abs(ern["toutes (porte)"][1] - ern_true) / ern_true)
        base = log(mean(ern["aucune"]))
        emit(@sprintf("\nPROMPT %d : ern vrai = %.2f   ern aucune = %.2f ± %.2f", p, ern_true,
                      mean(ern["aucune"]), std(ern["aucune"])))
        emit("   configuration        ern (moy ± sd)        ρ")
        for (name, _) in CONFIGS
            ρ = (base - log(mean(ern[name]))) / (base - log(ern_true))
            push!(ρs[name], ρ)
            emit(@sprintf("   %-18s   %7.2f ± %-6.2f     %+.2f", name, mean(ern[name]),
                          length(ern[name]) > 1 ? std(ern[name]) : 0.0, ρ))
        end
        Js = nothing; Us = nothing; SVts = nothing; GC.gc()
    end
    emit(@sprintf("\nPORTE GL : écart relatif max (toutes vs vrai) = %.2e -> %s", gate, gate < 1e-8 ? "OK" : "ÉCHEC"))
    emit("\nMÉDIANES de ρ sur les prompts [min, max] :")
    for (name, _) in CONFIGS
        emit(@sprintf("   %-18s   %+.2f   [%+.2f, %+.2f]", name, median(ρs[name]), minimum(ρs[name]), maximum(ρs[name])))
    end
    s16 = median(ρs["suffixe ≥16"]); p15 = median(ρs["préfixe ≤15"])
    hl1 = (s16 >= 0.7 && p15 <= 0.3) ? "VRAI" : (s16 <= 0.3 && p15 >= 0.7) ? "FAUX" : "PARTIEL"
    wins = ["fenêtre 1-5", "fenêtre 6-10", "fenêtre 11-15", "fenêtre 16-20", "fenêtre 21-26"]
    best = argmax(w -> median(ρs[w]), wins)
    emit(@sprintf("\nHL1 (suffixe ≥16 ≥ 0,7 et préfixe ≤15 ≤ 0,3) : %s  (%.2f / %.2f)", hl1, s16, p15))
    emit(@sprintf("HL2 : %s  (meilleure fenêtre : %s, médiane ρ = %.2f)",
                  median(ρs[best]) >= 0.5 ? "LOCALISÉ" : "DISTRIBUÉ", best, median(ρs[best])))
end
println("\nÉcrit : ", RES)
