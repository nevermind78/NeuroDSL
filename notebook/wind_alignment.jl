# ══════════════════════════════════════════════════════════════════════════════
# WIND-2 -- Alignement spectral des couches adjacentes (CPU, Float64) sur les
# Jacobiennes WIND-1. Verdicts contre notebook/wind_alignment_preregistration.md.
#
# USAGE : julia --project=. notebook/wind_alignment.jl
#         (WIND_OUTDIR pour un autre dossier de données, WIND_PROMPTS="5,6" pour un sous-ensemble)
# ══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra, Statistics, Printf, JSON, Random, Dates

const OUTDIR = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const VARIANTS = split(get(ENV, "WIND_VARIANTS", "T,A,AN"), ",")
const RES  = joinpath(@__DIR__, "wind_alignment_results.txt")
const JOUT = joinpath(@__DIR__, "wind_alignment_results.json")
const W1JSON = joinpath(@__DIR__, "wind_analysis_results.json")
const L, D = 28, 1536
const RS = [1, 2, 5, 10, 20, 50, 100, 200]
const C_FD = 1.2e-3          # proxy de l'erreur de différence finie (porte G1 de WIND-1, max)
const N_NULL = parse(Int, get(ENV, "WIND_NNULL", "3"))   # tirages de Haar par paire (variante T)
BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ 2))

loadJ(p, V) = (A = Array{Float32}(undef, D, D, L);
               open(joinpath(OUTDIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 1:L])          # Js[k+1] = J_k = ∂x_{k+1}/∂x_k

haar(n, rng) = (F = qr(randn(rng, n, n)); Matrix(F.Q) * Diagonal(sign.(diag(F.R))))
nanfree(x) = map(v -> isnan(v) ? nothing : v, x)

# Paire (J_k, J_{k+1}) ; Fa = svd(J_k), Fb = svd(J_{k+1}).
function pair_metrics(Fa, Fb, Ja, Jb, rng)
    σa, σb = Fa.S, Fb.S
    O = Fb.V' * Fa.U                        # O_k = V_{k+1}ᵀ U_k : lignes ↔ v_i^(k+1), colonnes ↔ u_j^(k)
    P = O .^ 2
    M = Jb * Ja                             # J_{k+1} J_k
    fro2 = sum(abs2, M)
    ident = dot(σb .^ 2, P * (σa .^ 2))     # Σ_ij σ_i'² P_ij σ_j²
    gA1 = abs(fro2 - ident) / fro2
    gA3 = max(maximum(abs.(sum(P, dims = 2) .- 1)), maximum(abs.(sum(P, dims = 1) .- 1)))

    sM1 = opnorm(M)
    γ = sM1 / (σb[1] * σa[1])
    𝒜 = fro2 / sum((σb .* σa) .^ 2)
    𝒜r = sum(abs2, σb) * sum(abs2, σa) / (D * sum((σb .* σa) .^ 2))

    ra = count(>(1.5), σa); rb = count(>(1.5), σb); rd = count(<(0.5), σb)
    Eamp  = (ra > 0 && rb > 0) ? (sum(abs2, O[1:rb, 1:ra]) / ra) / (rb / D) : NaN
    Edamp = (ra > 0 && rd > 0) ? (sum(abs2, O[D-rd+1:D, 1:ra]) / ra) / (rd / D) : NaN

    τ = sqrt(dot(σb .^ 2, P[:, 1])) / σb[1]          # transfert de la direction la plus écrite u₁^(k)
    τr = norm(σb) / (sqrt(D) * σb[1])                # sa valeur pour un appariement aléatoire

    α = [(ρ = sum(abs2, O[1:r, 1:r]) / r; (ρ - r / D) / (1 - r / D)) for r in RS]
    csv = [svdvals(O[1:r, 1:r]) for r in RS]
    cmax = [c[1] for c in csv]; cmin = [c[end] for c in csv]
    ηa = [C_FD * norm(σa) / (σa[r] - σa[r+1]) for r in RS]
    ηb = [C_FD * norm(σb) / (σb[r] - σb[r+1]) for r in RS]
    cert = [(ηa[i] < 0.1 && ηb[i] < 0.1) for i in eachindex(RS)]

    γnull = NaN
    if rng !== nothing
        γnull = median([opnorm(Diagonal(σb) * haar(D, rng) * Diagonal(σa)) / (σb[1] * σa[1]) for _ in 1:N_NULL])
    end
    (; γ, γnull, 𝒜, 𝒜r, Eamp, Edamp, ra, rb, rd, τ, τr, α, cmax, cmin, cert, gA1, gA3)
end

w1 = isfile(W1JSON) ? JSON.parsefile(W1JSON) : nothing
rng = MersenneTwister(20260929)
out = Dict{String,Any}("per_prompt" => Dict{String,Any}(), "RS" => RS)
agg = Dict(V => Dict(k => Float64[] for k in ("Eamp", "Edamp", "g_ratio", "Anorm", "tau_ratio")) for V in VARIANTS)
gates = Dict("GA1" => 0.0, "GA2" => 0.0, "GA3" => 0.0)
certfrac = zeros(length(RS)); ncert = Ref(0)
alpha_T = [Float64[] for _ in RS]
pair0 = Dict{String,Any}()

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-2 -- ALIGNEMENT SPECTRAL DES COUCHES ADJACENTES (verdicts contre wind_alignment_preregistration.md)")
    emit("Date : " * Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    emit("Paire k : sortie de J_k (U_k) contre entrée de J_{k+1} (V_{k+1}), dans l'espace x_{k+1}. n = $D.")

    for p in PROMPT_IDXS
        pp = Dict{String,Any}()
        for V in VARIANTS
            t0 = time()
            Js = loadJ(p, V)
            Fs = [svd(Js[k]) for k in 1:L]
            rows = [pair_metrics(Fs[k+1], Fs[k+2], Js[k+1], Js[k+2], V == "T" ? rng : nothing) for k in 0:L-2]
            Js = nothing; Fs = nothing; GC.gc()

            for (i, r) in enumerate(rows)
                k = i - 1
                gates["GA1"] = max(gates["GA1"], r.gA1); gates["GA3"] = max(gates["GA3"], r.gA3)
                if V == "T" && w1 !== nothing
                    Gadj = w1["per_prompt"][string(p)]["G_adjacent"][k+1]
                    gates["GA2"] = max(gates["GA2"], abs(r.γ - 1 / Gadj) * Gadj)
                end
                if k == 0
                    pair0["$(p)_$(V)"] = Dict("Eamp" => r.Eamp, "Edamp" => r.Edamp, "gamma" => r.γ,
                                              "gamma_null" => r.γnull, "A" => r.𝒜, "Arand" => r.𝒜r)
                    continue
                end
                push!(agg[V]["Eamp"], r.Eamp); push!(agg[V]["Edamp"], r.Edamp)
                push!(agg[V]["Anorm"], (r.𝒜 - r.𝒜r) / (1 - r.𝒜r))
                push!(agg[V]["tau_ratio"], r.τ / r.τr)
                V == "T" && push!(agg[V]["g_ratio"], r.γ / r.γnull)
                if V == "T"
                    certfrac .+= r.cert; ncert[] += 1
                    for j in eachindex(RS); push!(alpha_T[j], r.α[j]); end
                end
            end

            emit("\n" * "="^100)
            emit(@sprintf("PROMPT %d -- variante %s   (%.0f s)", p, V, time() - t0))
            emit("="^100)
            emit("   k  r_a  r_b  r_d   E_amp  E_damp    γ_k  γ_null   𝒜_k  𝒜_rand   τ/τ_rand   α_1    α_10   α_50")
            for (i, r) in enumerate(rows)
                emit(@sprintf("  %2d %4d %4d %4d  %6.2f  %6.2f  %5.3f  %6.3f  %5.3f  %6.3f   %7.2f  %+.3f %+.3f %+.3f",
                              i - 1, r.ra, r.rb, r.rd, r.Eamp, r.Edamp, r.γ, r.γnull, r.𝒜, r.𝒜r, r.τ / r.τr,
                              r.α[1], r.α[4], r.α[6]))
            end
            pp[V] = Dict(
                "gamma" => [r.γ for r in rows], "gamma_null" => nanfree([r.γnull for r in rows]),
                "A" => [r.𝒜 for r in rows], "Arand" => [r.𝒜r for r in rows],
                "Eamp" => nanfree([r.Eamp for r in rows]), "Edamp" => nanfree([r.Edamp for r in rows]),
                "ra" => [r.ra for r in rows], "rb" => [r.rb for r in rows], "rd" => [r.rd for r in rows],
                "tau" => [r.τ for r in rows], "tau_rand" => [r.τr for r in rows],
                "alpha" => [r.α for r in rows], "cos_max" => [r.cmax for r in rows],
                "cos_min" => [r.cmin for r in rows], "certified" => [r.cert for r in rows])
        end
        out["per_prompt"][string(p)] = pp
    end

    emit("\n" * "#"^100)
    emit("PORTES")
    emit("#"^100)
    emit(@sprintf("GA1 identité ‖J_{k+1}J_k‖_F² = Σ σ'² P σ² : écart relatif max = %.2e  (seuil 1e-10) -> %s",
                  gates["GA1"], gates["GA1"] < 1e-10 ? "OK" : "ÉCHEC"))
    emit(w1 === nothing ? "GA2 : wind_analysis_results.json absent -> NON ÉVALUÉE" :
         @sprintf("GA2 γ_k (T) vs 1/G_adjacent (WIND-1) : écart relatif max = %.2e  (seuil 1e-6) -> %s",
                  gates["GA2"], gates["GA2"] < 1e-6 ? "OK" : "ÉCHEC"))
    emit(@sprintf("GA3 P doublement stochastique : écart max = %.2e  (seuil 1e-10) -> %s",
                  gates["GA3"], gates["GA3"] < 1e-10 ? "OK" : "ÉCHEC"))

    emit("\n" * "#"^100)
    emit("VERDICTS (variante T, médiane sur 5 prompts × paires k = 1..26)")
    emit("#"^100)
    T = agg["T"]
    mEa = median(filter(!isnan, T["Eamp"])); mEd = median(filter(!isnan, T["Edamp"])); mg = median(T["g_ratio"])
    emit(@sprintf("médiane E_amp = %.3f   médiane E_damp = %.3f   médiane γ/γ_null = %.3f", mEa, mEd, mg))
    emit(@sprintf("  (quartiles E_amp : %.2f / %.2f ; E_damp : %.2f / %.2f ; γ/γ_null : %.2f / %.2f)",
                  quantile(filter(!isnan, T["Eamp"]), 0.25), quantile(filter(!isnan, T["Eamp"]), 0.75),
                  quantile(filter(!isnan, T["Edamp"]), 0.25), quantile(filter(!isnan, T["Edamp"]), 0.75),
                  quantile(T["g_ratio"], 0.25), quantile(T["g_ratio"], 0.75)))
    ha1 = mEa > 2 ? "ALIGNÉ" : (0.67 <= mEa <= 1.5) ? "HASARD" : mEa < 0.67 ? "ANTI-ALIGNÉ" : "INTERMÉDIAIRE"
    ha2 = (mEd > 1.5 && mEa < 1) ? "PRÉSENTE" : mEd < 1.2 ? "ABSENTE" : "AMBIGUË"
    ha3 = mg > 1.2 ? "PLUS ALIGNÉ qu'un appariement aléatoire" : mg < 0.8 ? "MOINS ALIGNÉ qu'un appariement aléatoire" :
          "INDISCERNABLE du hasard"
    emit("HA1 (alignement des sous-espaces amplifiés) -> " * ha1)
    emit("HA2 (signature de correction)               -> " * ha2)
    emit("HA3 (appariement vs Haar)                   -> " * ha3)

    emit("\nDescriptif :")
    emit("  α_r médian (T) : " * join([@sprintf("r=%d: %+.4f", RS[j], median(alpha_T[j])) for j in eachindex(RS)], "  "))
    emit("  fraction certifiée (Wedin, η<0.1 sur J_k et J_{k+1}) : " *
         join([@sprintf("r=%d: %.2f", RS[j], certfrac[j] / ncert[]) for j in eachindex(RS)], "  "))
    for V in VARIANTS
        a = agg[V]
        emit(@sprintf("  %-2s : médiane E_amp = %.3f  E_damp = %.3f  𝒜 normalisé = %.4f  τ/τ_rand = %.2f",
                      V, median(filter(!isnan, a["Eamp"])), median(filter(!isnan, a["Edamp"])),
                      median(a["Anorm"]), median(a["tau_ratio"])))
    end
    emit("  Paire k=0 (embedding, rapportée à part) : " *
         join([@sprintf("p%s: E_amp=%.2f E_damp=%.2f γ/γ_null=%.2f", split(key, "_")[1], v["Eamp"], v["Edamp"],
                        v["gamma"] / v["gamma_null"]) for (key, v) in sort(collect(pair0), by = first) if endswith(key, "_T")], " | "))

    out["gates"] = gates
    out["verdicts"] = Dict("HA1" => ha1, "HA2" => ha2, "HA3" => ha3, "median_Eamp" => mEa,
                           "median_Edamp" => mEd, "median_g_ratio" => mg)
    out["aggregate"] = Dict(V => Dict(k => nanfree(v) for (k, v) in d) for (V, d) in agg)
    out["certified_fraction"] = certfrac ./ ncert[]
    out["pair0"] = pair0
end
open(JOUT, "w") do f; JSON.print(f, out); end
println("\nÉcrit : ", RES, " et ", JOUT)
