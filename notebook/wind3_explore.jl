# ══════════════════════════════════════════════════════════════════════════════
# WIND-3 -- EXPLORATOIRE (POST HOC, NON pré-enregistré) : robustesse des chiffres
# principaux à l'exclusion du puits (position 1), carte « fenêtre » de Geva,
# profil du puits par couche, origine de l'explosion du terme « norme » aux couches 0-2.
# Lit uniquement les fichiers de wind3_collect.jl. USAGE : julia --project=. notebook/wind3_explore.jl
# ══════════════════════════════════════════════════════════════════════════════
using JSON, LinearAlgebra, Statistics, Printf
const DIR = joinpath(@__DIR__, "wind_data")
const RES = joinpath(@__DIR__, "wind3_explore_results.txt")
const PIDX = [5, 6, 12, 17, 22]
const L, D = 28, 1536
const TOK = JSON.parsefile(joinpath(DIR, "wind3_tokens.json"))
function load_G(p, V, n)
    a = Array{Float32}(undef, n * D * (2L + 1))
    open(joinpath(DIR, "wind3_G_p$(p)_$(V).bin"), "r") do f; read!(f, a); end
    [Float64.(reshape(a[(k*n*D+1):((k+1)*n*D)], n, D)) for k in 0:L]
end
tomat(v) = reduce(vcat, [permutedims(Float64.(r)) for r in v])

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-3 -- EXPLORATOIRE (post hoc, NON pré-enregistré)")
    EHx = Dict(k => zeros(L, 5) for k in ("T-AN", "T-A", "A-AN")); hx = zeros(L + 1, 5); vQK = zeros(L, 5)
    sinkprof = zeros(L, 5); sinkmask = zeros(L, 5)
    for (j, p) in enumerate(PIDX)
        m = JSON.parsefile(joinpath(DIR, "wind3_meta_p$(p).json")); n = m["n"]; toks = TOK[string(p)]["tokens"]
        xn = tomat(m["xnorm"]); Krem = tomat(m["Krem"]); Kmask = tomat(m["Kmask"]); Kwin = tomat(m["Kwin"])
        G = Dict(V => load_G(p, V, n) for V in ("T", "A", "AN"))
        dn(V1, V2, i, s) = norm(G[V1][s+1][i, :] .- G[V2][s+1][i, :]); gn(V, i, s) = norm(G[V][s+1][i, :])
        EH(V1, V2, s, idx) = sqrt(sum(xn[s+1, i]^2 * dn(V1, V2, i, s)^2 for i in idx) / sum(xn[s+1, i]^2 * gn(V1, i, s)^2 for i in idx))
        for s in 0:L-1
            EHx["T-AN"][s+1, j] = EH("T", "AN", s, 2:n-1); EHx["T-A"][s+1, j] = EH("T", "A", s, 2:n-1)
            EHx["A-AN"][s+1, j] = EH("A", "AN", s, 2:n-1)
        end
        for s in 0:L
            S = [gn("T", i, s) * xn[s+1, i] for i in 1:n]
            hx[s+1, j] = sum(S[2:n-1]) / sum(S[2:n])
        end
        sinkprof[:, j] = Krem[:, 1]; sinkmask[:, j] = Kmask[:, 1]
        emit("\n── PROMPT $p ($(repr(m["prompt"]))) ; R = $(round(m["R_clean"], digits=3))")
        # origine du terme norme aux couches 0-2
        for s in 0:2
            c = [xn[s+1, i]^2 * dn("A", "AN", i, s)^2 for i in 1:n-1]
            emit(@sprintf("  s=%d : part de Σ‖x‖²‖g^A−g^AN‖² portée par la position 1 = %.3f ; ‖x_{1,s}‖ = %.1f ; ‖g^AN_1‖/‖g^A_1‖ = %.1f",
                          s, c[1] / sum(c), xn[s+1, 1], gn("AN", 1, s) / gn("A", 1, s)))
        end
        # fenêtre de Geva (mask, couches ℓ−2..ℓ+2)
        W2 = abs.(Kwin[:, 2:end]); ci = argmax(W2)
        emit(@sprintf("  fenêtre-5 (mask) : cellule max hors puits ℓ=%d i=%d (%s) ΔR=%+.3f ; puits max |ΔR| = %.3f (ℓ=%d)",
                      ci[1], ci[2] + 1, repr(toks[ci[2]+1]), Kwin[ci[1], ci[2]+1], maximum(abs.(Kwin[:, 1])), argmax(abs.(Kwin[:, 1]))))
        prof = [sum(abs.(Kwin[:, i])) for i in 1:n-1]
        emit("  fenêtre-5, Σ_ℓ|ΔR| par position : " * join([@sprintf("%d:%s %.2f", i, strip(toks[i]), prof[i]) for i in 1:n-1], " | "))
        if p == 5
            subj = [sum(abs.(Kwin[l, 8:11])) for l in 1:L]; rel = [sum(abs.(Kwin[l, 2:6])) for l in 1:L]
            emit("  p5 fenêtre-5 Σ|ΔR| sujet (8-11) par ℓ : " * join([@sprintf("%.2f", v) for v in subj], " "))
            emit("  p5 fenêtre-5 Σ|ΔR| relation (2-6) par ℓ : " * join([@sprintf("%.2f", v) for v in rel], " "))
            emit(@sprintf("  p5 : ℓ du max sujet = %d ; ℓ du max relation = %d", argmax(subj), argmax(rel)))
            s1 = [sum(abs.(Krem[l, 8:11])) for l in 1:L]; r1 = [sum(abs.(Krem[l, 2:6])) for l in 1:L]
            emit(@sprintf("  p5 mono-couche remove : ℓ du max sujet = %d ; ℓ du max relation = %d", argmax(s1), argmax(r1)))
        end
        if p == 22
            emit("  p22 mono-couche remove, position 6 ('mat') par ℓ : " * join([@sprintf("%+.2f", Krem[l, 6]) for l in 1:L], " "))
            emit("  p22 mono-couche remove, position 5 ('the') par ℓ : " * join([@sprintf("%+.2f", Krem[l, 5]) for l in 1:L], " "))
            emit("  p22 fenêtre-5 mask, position 6 ('mat') par ℓ : " * join([@sprintf("%+.2f", Kwin[l, 6]) for l in 1:L], " "))
            emit(@sprintf("  p22 : rang de la position 6 par Σ_ℓ|K^rem| hors puits = %d / %d", findfirst(==(6), sortperm([sum(abs.(Krem[:, i])) for i in 2:n-1]; rev=true) .+ 1), n - 2))
        end
    end
    md(M) = [median(M[s, :]) for s in 1:size(M, 1)]
    emit("\n── Médianes (5 prompts), SOURCES i ∈ 2..n−1 SEULEMENT (puits exclu), s = 0..27")
    emit("  E^H (T vs AN)   : " * join([@sprintf("%.2f", v) for v in md(EHx["T-AN"])], " "))
    emit("  E^{H,QK} (T/A)  : " * join([@sprintf("%.2f", v) for v in md(EHx["T-A"])], " "))
    emit("  E^{H,N} (A/AN)  : " * join([@sprintf("%.2f", v) for v in md(EHx["A-AN"])], " "))
    emit(@sprintf("  fraction s∈1..20 avec médiane E^H > 0.3 (puits exclu) : %.2f", count(md(EHx["T-AN"])[s+1] > 0.3 for s in 1:20) / 20))
    emit("  h^T_s (part horizontale, puits exclu des deux côtés), s=0..28 : " * join([@sprintf("%.2f", v) for v in md(hx)], " "))
    emit("\n── Puits : ΔR (remove) par couche, médiane des |ΔR| : " * join([@sprintf("%.2f", v) for v in md(abs.(sinkprof))], " "))
    emit("   Puits : ΔR (mask)   par couche, médiane des |ΔR| : " * join([@sprintf("%.2f", v) for v in md(abs.(sinkmask))], " "))
    emit(@sprintf("   part de Σ_ℓ|K^rem_{ℓ,1}| venant des couches 19-28 (médiane) : %.2f",
                  median([sum(abs.(sinkprof[19:28, j])) / sum(abs.(sinkprof[:, j])) for j in 1:5])))
    open(joinpath(@__DIR__, "wind3_explore_results.json"), "w") do f
        JSON.print(f, Dict("E_H_nosink" => md(EHx["T-AN"]), "E_HQK_nosink" => md(EHx["T-A"]),
                           "E_HN_nosink" => md(EHx["A-AN"]), "hshare_nosink" => md(hx)))
    end
end
println("\nÉcrit : ", RES)
