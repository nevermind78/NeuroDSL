# ══════════════════════════════════════════════════════════════════════════════
# WIND-3 -- ANALYSE : verdicts contre notebook/wind3_preregistration.md
# Lit wind_data/wind3_meta_p*.json + wind3_G_p*_{T,A,AN}.bin (CPU uniquement).
# USAGE : julia --project=. notebook/wind3_analysis.jl
# ══════════════════════════════════════════════════════════════════════════════
using JSON, LinearAlgebra, Statistics, Printf

const DIR = joinpath(@__DIR__, "wind_data")
const RES = joinpath(@__DIR__, "wind3_analysis_results.txt")
const PIDX = [5, 6, 12, 17, 22]
const L, D = 28, 1536
const TOK = JSON.parsefile(joinpath(DIR, "wind3_tokens.json"))

function load_G(p, V, n)
    a = Array{Float32}(undef, n * D * (L + 1) + n * D * L)
    open(joinpath(DIR, "wind3_G_p$(p)_$(V).bin"), "r") do f; read!(f, a); end
    Gx = [Float64.(reshape(a[(k*n*D+1):((k+1)*n*D)], n, D)) for k in 0:L]
    off = n * D * (L + 1)
    Gr = [Float64.(reshape(a[(off+(l-1)*n*D+1):(off+l*n*D)], n, D)) for l in 1:L]
    return Gx, Gr
end
tomat(v) = reduce(vcat, [permutedims(Float64.(r)) for r in v])        # vecteur de lignes -> matrice
function ranks(x)
    p = sortperm(x); r = similar(x, Float64); i = 1
    while i <= length(x)
        j = i
        while j < length(x) && x[p[j+1]] == x[p[i]]; j += 1; end
        r[p[i:j]] .= (i + j) / 2; i = j + 1
    end
    r
end
spearman(a, b) = cor(ranks(a), ranks(b))
verdict(v, t, f) = v ? "VRAIE" : (f ? "FAUSSE" : "PARTIELLE")

out = Dict{String,Any}("per_prompt" => Dict{String,Any}())
open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-3 -- ANALYSE (verdicts contre wind3_preregistration.md)")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    PP = Dict{Int,Any}()
    for p in PIDX
        m = JSON.parsefile(joinpath(DIR, "wind3_meta_p$(p).json"))
        n = m["n"]; toks = TOK[string(p)]["tokens"]
        xn = tomat(m["xnorm"])                                     # (L+1) × n
        Krem = tomat(m["Krem"]); Kmask = tomat(m["Kmask"]); Kwin = tomat(m["Kwin"])   # L × (n−1)
        LIN = Dict(V => tomat(m["LIN"][V]) for V in ("T", "A", "AN"))
        attn = [Float64.(m["attn_last"][l][h]) for l in 1:L, h in 1:12]
        G = Dict(V => load_G(p, V, n) for V in ("T", "A", "AN"))
        gn(V, i, s) = norm(G[V][1][s+1][i, :])
        S = Dict(V => [gn(V, i, s) * xn[s+1, i] for s in 0:L, i in 1:n] for V in ("T", "A", "AN"))  # (L+1)×n
        diffn(V1, V2, i, s) = norm(G[V1][1][s+1][i, :] .- G[V2][1][s+1][i, :])
        function EH(V1, V2, s, idx)
            num = sum(xn[s+1, i]^2 * diffn(V1, V2, i, s)^2 for i in idx)
            den = sum(xn[s+1, i]^2 * gn(V1, i, s)^2 for i in idx)
            sqrt(num / den)
        end
        hor = 1:n-1
        E_H   = [EH("T", "AN", s, hor) for s in 0:L-1]
        E_HQK = [EH("T", "A", s, hor) for s in 0:L-1]
        E_HN  = [EH("A", "AN", s, hor) for s in 0:L-1]
        errV_QK = [diffn("T", "A", n, s) / gn("T", n, s) for s in 0:L-1]
        errV_AN = [diffn("T", "AN", n, s) / gn("T", n, s) for s in 0:L-1]
        hshare = Dict(V => [sum(S[V][s+1, 1:n-1]) / sum(S[V][s+1, :]) for s in 0:L] for V in ("T", "A", "AN"))
        cellerr = [diffn("T", "AN", i, s) / gn("T", i, s) for s in 0:L-1, i in 1:n-1]
        # sink
        q1 = median([diffn("T", "A", 1, s) / gn("T", 1, s) for s in 3:20])
        qrest = median([EH("T", "A", s, 2:n-1) for s in 3:20])
        π1 = sum(attn[l, h][1] for l in 1:L, h in 1:12) / sum(sum(attn[l, h][1:n-1]) for l in 1:L, h in 1:12)
        κ1 = sum(abs.(Krem[:, 1])) / sum(abs.(Krem))
        μ1 = sum(abs.(Kmask[:, 1])) / sum(abs.(Krem[:, 1]))
        γ1 = Dict(V => sum(S[V][2:L, 1]) / sum(S[V][2:L, 1:n-1]) for V in ("T", "A", "AN"))   # s=1..27
        # concentration
        a = sort(vec(abs.(Krem)); rev=true); ntop = ceil(Int, 0.05 * length(a)); C5 = sum(a[1:ntop]) / sum(a)
        am = sort(vec(abs.(Kmask)); rev=true); C5m = sum(am[1:ntop]) / sum(am)
        # HH4
        sp = Dict(V => spearman(vec(abs.(LIN[V])), vec(abs.(Krem))) for V in ("T", "A", "AN"))
        top = sortperm(vec(abs.(Krem)); rev=true)[1:10]
        ov = Dict(V => length(intersect(top, sortperm(vec(abs.(LIN[V])); rev=true)[1:10])) for V in ("T", "A", "AN"))
        relerr = Dict(V => median(abs.(vec(LIN[V])[top] .- vec(Krem)[top]) ./ abs.(vec(Krem)[top])) for V in ("T", "A", "AN"))
        # sources
        Km2 = abs.(Krem[:, 2:end]); ci = argmax(Km2); lmax, imax = ci[1], ci[2] + 1
        prof = [sum(S["T"][2:L, i]) for i in 1:n-1]                 # s = 1..27
        igrad = argmax(prof[2:end]) + 1
        pos_attn = [mean(attn[l, h][i] for l in 1:L, h in 1:12) for i in 1:n]
        pos_krem = [sum(abs.(Krem[:, i])) for i in 1:n-1]
        pos_kmask = [sum(abs.(Kmask[:, i])) for i in 1:n-1]
        PP[p] = (; n, toks, E_H, E_HQK, E_HN, errV_QK, errV_AN, hshare, q1, qrest, π1, κ1, μ1, γ1, C5, C5m,
                   sp, ov, relerr, lmax, imax, igrad, prof, pos_attn, pos_krem, pos_kmask, Krem, Kmask, Kwin,
                   cellerr, R=m["R_clean"], gates=m["gates"], LIN)

        emit("\n" * "="^100)
        emit(@sprintf("PROMPT %d : %s  (n=%d, R propre = %+.3f ; top-1 %s vs top-2 %s)", p, repr(m["prompt"]), n,
                      m["R_clean"], repr(TOK[string(p)]["top1"]), repr(TOK[string(p)]["top2"])))
        emit("="^100)
        emit("Portes : " * join(["$k=$(v isa Real ? @sprintf("%.2e", v) : v)" for (k, v) in sort(collect(m["gates"]))], "  "))
        emit("Tokens : " * join(["$i:$(repr(toks[i]))" for i in 1:n], " "))
        emit("Par position i : attention moyenne du dernier token (moy. couches×têtes) | Σ_ℓ|K^rem| | Σ_ℓ|K^mask| | Σ_s S^T (s=1..27)")
        for i in 1:n-1
            emit(@sprintf("  %2d %-10s attn %.3f | Krem %.4f | Kmask %.4f | S^T %.3f", i, repr(toks[i]), pos_attn[i],
                          pos_krem[i], pos_kmask[i], prof[i]))
        end
        emit(@sprintf("  %2d %-10s attn %.3f (soi)", n, repr(toks[n]), pos_attn[n]))
        emit(@sprintf("Cellule |K^rem| max (i≥2) : ℓ=%d, i=%d (%s), ΔR=%+.4f ; position de S^T max (i≥2) : %d (%s)",
                      lmax, imax, repr(toks[imax]), Krem[lmax, imax], igrad, repr(toks[igrad])))
        emit("Top-8 cellules |K^rem| : " * join([@sprintf("(ℓ%d,i%d:%+.3f)", c[1], c[2], Krem[c]) for c in
             CartesianIndices(Krem)[sortperm(vec(abs.(Krem)); rev=true)[1:8]]], " "))
        emit("Top-8 cellules |K^mask| : " * join([@sprintf("(ℓ%d,i%d:%+.3f)", c[1], c[2], Kmask[c]) for c in
             CartesianIndices(Kmask)[sortperm(vec(abs.(Kmask)); rev=true)[1:8]]], " "))
        emit(@sprintf("C₅ (remove) = %.3f ; C₅ (mask) = %.3f", C5, C5m))
        emit("E^H_s (T vs AN), s=0..27    : " * join([@sprintf("%.2f", v) for v in E_H], " "))
        emit("E^{H,QK}_s (T vs A)          : " * join([@sprintf("%.2f", v) for v in E_HQK], " "))
        emit("E^{H,N}_s (A vs AN)          : " * join([@sprintf("%.2f", v) for v in E_HN], " "))
        emit("err^{V,QK}_s (vertical, T/A) : " * join([@sprintf("%.2f", v) for v in errV_QK], " "))
        emit("h^T_s (part horizontale)     : " * join([@sprintf("%.2f", v) for v in hshare["T"]], " "))
        emit("h^AN_s                       : " * join([@sprintf("%.2f", v) for v in hshare["AN"]], " "))
        emit(@sprintf("Puits : π₁=%.3f κ₁=%.3f ρ₁=%.3f μ₁=%.2f q₁=%.3f q_rest=%.3f ; part S du puits γ₁ : T %.3f A %.3f AN %.3f",
                      π1, κ1, κ1 / π1, μ1, q1, qrest, γ1["T"], γ1["A"], γ1["AN"]))
        emit(@sprintf("HH4 : Spearman(|LIN|,|K^rem|) T %.3f A %.3f AN %.3f ; top-10 communs T %d A %d AN %d ; err rel médiane top-10 T %.3f A %.3f AN %.3f",
                      sp["T"], sp["A"], sp["AN"], ov["T"], ov["A"], ov["AN"], relerr["T"], relerr["A"], relerr["AN"]))
        out["per_prompt"][string(p)] = Dict("n" => n, "tokens" => toks, "E_H" => E_H, "E_HQK" => E_HQK, "E_HN" => E_HN,
            "errV_QK" => errV_QK, "errV_AN" => errV_AN, "hshare" => hshare, "Krem" => m["Krem"], "Kmask" => m["Kmask"],
            "Kwin" => m["Kwin"], "LIN" => m["LIN"], "pos_attn" => pos_attn, "pos_krem" => pos_krem,
            "pos_kmask" => pos_kmask, "prof" => prof, "cellerr" => [cellerr[:, i] for i in 1:n-1],
            "pi1" => π1, "kappa1" => κ1, "mu1" => μ1, "q1" => q1, "qrest" => qrest, "C5" => C5, "R" => m["R_clean"])
    end

    emit("\n" * "#"^100 * "\nVERDICTS (seuils pré-enregistrés)\n" * "#"^100)
    med(f) = median([f(PP[p]) for p in PIDX])
    # HH1a
    C5s = [PP[p].C5 for p in PIDX]; mC5 = median(C5s)
    emit(@sprintf("HH1a : C₅ par prompt = %s ; médiane = %.3f", join([@sprintf("%.3f", c) for c in C5s], ", "), mC5))
    emit("HH1a -> " * verdict(mC5 >= 0.5, true, mC5 < 0.25))
    # HH1b
    tgt = Dict(22 => (Set([6]), 1:L), 5 => (Set([8, 9, 10, 11]), 8:24))
    hits = Dict{Int,Tuple{Bool,Bool}}()
    for (p, (S_, lr)) in tgt
        P = PP[p]
        a_ = (P.imax in S_) && (P.lmax in lr); b_ = P.igrad in S_
        hits[p] = (a_, b_)
        emit(@sprintf("HH1b p%d : (a) cellule max ℓ=%d i=%d (%s) -> %s ; (b) S^T max i=%d (%s) -> %s", p, P.lmax, P.imax,
                      repr(P.toks[P.imax]), a_, P.igrad, repr(P.toks[P.igrad]), b_))
    end
    allh = all(all(v) for v in values(hits)); noh = !any(any(v) for v in values(hits))
    emit("HH1b -> " * verdict(allh, true, noh))
    # HH2a
    mEH = [median([PP[p].E_H[s+1] for p in PIDX]) for s in 0:L-1]
    frac = count(mEH[s+1] > 0.3 for s in 1:20) / 20
    emit("HH2a : médiane E^H_s, s=0..27 : " * join([@sprintf("%.2f", v) for v in mEH], " "))
    emit(@sprintf("       fraction des s∈1..20 avec médiane > 0.3 : %.2f", frac))
    emit("HH2a -> " * verdict(frac >= 0.5, true, all(mEH[s+1] < 0.1 for s in 1:26)))
    mQK = [median([PP[p].E_HQK[s+1] for p in PIDX]) for s in 0:L-1]
    mN = [median([PP[p].E_HN[s+1] for p in PIDX]) for s in 0:L-1]
    mV = [median([PP[p].errV_QK[s+1] for p in PIDX]) for s in 0:L-1]
    emit("       médiane E^{H,QK}_s : " * join([@sprintf("%.2f", v) for v in mQK], " "))
    emit("       médiane E^{H,N}_s  : " * join([@sprintf("%.2f", v) for v in mN], " "))
    emit("HH2b : médiane err^{V,QK}_s : " * join([@sprintf("%.2f", v) for v in mV], " "))
    nh = count(mQK[s+1] > mV[s+1] for s in 1:20)
    emit(@sprintf("       s∈1..20 avec horizontal QK > vertical QK : %d/20", nh))
    emit("HH2b -> " * (nh >= 14 ? "VRAIE" : (20 - nh >= 14 ? "FAUSSE" : "INDÉTERMINÉE")))
    # HH3
    mπ = med(P -> P.π1); mρ = med(P -> P.κ1 / P.π1); mμ = med(P -> P.μ1); mq1 = med(P -> P.q1); mqr = med(P -> P.qrest)
    emit(@sprintf("HH3a : π₁ = %s (médiane %.3f) ; ρ₁ = %s (médiane %.3f)", join([@sprintf("%.3f", PP[p].π1) for p in PIDX], ", "), mπ,
                  join([@sprintf("%.3f", PP[p].κ1 / PP[p].π1) for p in PIDX], ", "), mρ))
    emit("HH3a -> " * verdict(mπ >= 0.5 && mρ <= 0.25, true, mρ >= 0.75 || mπ < 0.2))
    emit(@sprintf("HH3b : q₁ = %s (médiane %.3f) ; q_rest = %s (médiane %.3f)", join([@sprintf("%.3f", PP[p].q1) for p in PIDX], ", "), mq1,
                  join([@sprintf("%.3f", PP[p].qrest) for p in PIDX], ", "), mqr))
    emit("HH3b -> " * verdict(mq1 > 0.5 && mq1 > mqr, true, mq1 < 0.2 || mq1 < mqr))
    emit(@sprintf("HH3c : μ₁ = %s (médiane %.2f)", join([@sprintf("%.2f", PP[p].μ1) for p in PIDX], ", "), mμ))
    emit("HH3c -> " * verdict(mμ > 3, true, mμ < 1.5))
    # HH4
    msp = med(P -> P.sp["T"])
    emit(@sprintf("HH4 : Spearman(|LIN^T|,|K^rem|) = %s (médiane %.3f) ; A médiane %.3f ; AN médiane %.3f",
                  join([@sprintf("%.3f", PP[p].sp["T"]) for p in PIDX], ", "), msp, med(P -> P.sp["A"]), med(P -> P.sp["AN"])))
    emit(@sprintf("      err rel médiane sur top-10 : T %.3f  A %.3f  AN %.3f ; top-10 communs (médiane) T %.0f A %.0f AN %.0f",
                  med(P -> P.relerr["T"]), med(P -> P.relerr["A"]), med(P -> P.relerr["AN"]),
                  med(P -> P.ov["T"]), med(P -> P.ov["A"]), med(P -> P.ov["AN"])))
    emit("HH4 -> " * verdict(msp >= 0.7, true, msp < 0.4))
    out["median"] = Dict("E_H" => mEH, "E_HQK" => mQK, "E_HN" => mN, "errV_QK" => mV)
end
open(joinpath(@__DIR__, "wind3_analysis_results.json"), "w") do f; JSON.print(f, out); end
println("\nÉcrit : ", RES)
