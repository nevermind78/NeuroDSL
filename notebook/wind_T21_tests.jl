# WIND-T21 (CPU, EXPLORATOIRE) -- tests A, B, C des affirmations de theo.tex (pré-enregistrement wind_T21_preregistration.md).
# USAGE : WIND_BLAS=4 julia --project=. notebook/wind_T21_tests.jl
using LinearAlgebra, Statistics, Printf, JSON
BLAS.set_num_threads(parse(Int, get(ENV, "WIND_BLAS", "4")))
const NB = @__DIR__
const CFG = Dict("qwen" => ("wind_data_T8", 28, 1536), "gpt2" => ("wind_data_GPT2", 12, 768), "gemma" => ("wind_data_GEMMA", 26, 2304))
loadJ(m, p, V) = (dir, L, D) = CFG[m] |> c -> c;
function loadJs(m, p, V)
    dir, L, D = CFG[m]
    A = Array{Float32}(undef, D, D, L); open(joinpath(NB, dir, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end
    [Float64.(A[:, :, k]) for k in 2:L]
end
meta(m, p) = JSON.parsefile(joinpath(NB, CFG[m][1], "wind_meta_p$(p).json"))
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
pr(s) = (q = s .^ 2; sum(q)^2 / sum(q .^ 2))
out = Dict{String,Any}(); lines = String[]
emit(s) = (push!(lines, s); println(s))

# ── Test A ─────────────────────────────────────────────────────────────────────────────────────
emit("=== Test A — ln‖J_k x̂_k‖ (lemme « Lyapunov Inversion »)")
A = Dict{String,Any}()
for (m, ps, vars) in (("qwen", 1:10, ["T", "A", "AN"]), ("gpt2", 1:10, ["T", "A", "AN"]), ("gemma", 1:5, ["A", "AN"]))
    L = CFG[m][2]; rec = Dict(V => Float64[] for V in vars); normsAN = Float64[]
    for p in ps
        X = [Float64.(v) for v in meta(m, p)["X"]]
        for V in vars
            Js = loadJs(m, p, V)
            for k in 1:L-1
                xh = X[k+1] ./ norm(X[k+1]); n = norm(Js[k] * xh)
                push!(rec[V], log(n)); V == "AN" && push!(normsAN, n)
            end
            Js = nothing; GC.gc()
        end
    end
    A[m] = rec
    ex = haskey(rec, "T") ? "T" : "A"
    emit(@sprintf("%-5s exact(%s) : frac s>0 %.3f | médiane s %.3f | max s %.3f   ||  AN : frac s>0 %.3f | médiane ‖J x̂‖ %.3f | min ‖J x̂‖ %.3f",
                  m, ex, mean(rec[ex] .> 0), median(rec[ex]), maximum(rec[ex]), mean(rec["AN"] .> 0), median(normsAN), minimum(normsAN)))
    haskey(rec, "A") && ex != "A" && emit(@sprintf("      A : frac s>0 %.3f | médiane s %.3f", mean(rec["A"] .> 0), median(rec["A"])))
end
a1 = all(all(A[m][haskey(A[m], "T") ? "T" : "A"] .<= 0) for m in keys(A))
a2 = all(all(A[m]["AN"] .> 0) for m in keys(A))
a3 = all(median(exp.(A[m]["AN"])) >= 2 for m in keys(A))
emit("A1 (s^exact ≤ 0 partout) : $(a1 ? "VRAI" : "FAUX") | A2 (s^AN > 0 partout) : $(a2 ? "VRAI" : "FAUX") | A3 (médiane ‖J^AN x̂‖ ≥ 2) : $(a3 ? "VRAI" : "FAUX")")
out["A"] = A

# ── Test B ─────────────────────────────────────────────────────────────────────────────────────
emit("\n=== Test B — θ_c « Ho–Kalman/Sylvester » du texte : det(I₂ − θ H₂) = 0, H₂ = C(I − J^A)⁻¹B, par bloc")
fac = JSON.parsefile(joinpath(NB, "wind_T13_factors.json")); T8 = JSON.parsefile(joinpath(NB, "wind_T8_truth.json"))
D = 1536; Bres = Dict{String,Any}()
for p in 1:50
    fp = fac[string(p)]; mt = meta("qwen", p)
    X = [Float64.(v) for v in mt["X"]]; H = [Float64.(v) for v in mt["H"]]
    nchk = length(fp["check_blocks"])
    buf = Array{Float64}(undef, D * 27 * 3 + D * 29 * 2 + D * 3 * nchk)
    open(joinpath(NB, "wind_data_T8", "wind_T13_factors_p$(p).bin"), "r") do f; read!(f, buf); end
    A1 = reshape(buf[1:D*27], D, 27); B1 = reshape(buf[D*27+1:2D*27], D, 27); MH = reshape(buf[2D*27+1:3D*27], D, 27)
    s1 = Float64.(fp["s1"]); s2 = Float64.(fp["s2"])
    JA = loadJs("qwen", p, "A"); JN = loadJs("qwen", p, "AN")
    θs = Float64[]; gate = 0.0; conds = Float64[]
    for k in 1:27
        xh = X[k+1] ./ norm(X[k+1]); u1 = s1[k] .* (A1[:, k] .+ B1[:, k]); mh = MH[:, k]
        Z = JN[k] - JA[k] - u1 * xh'
        ρv = (Z' * mh) ./ (s2[k] * sum(abs2, mh))
        Bk = hcat(u1, s2[k] .* mh); Ck = vcat(xh', ρv')
        gate = max(gate, norm(JN[k] - JA[k] - Bk * Ck) / norm(JN[k] - JA[k]))
        M = I - JA[k]
        H2 = Ck * (M \ Bk)
        ev = eigvals(H2); rp = [real(e) for e in ev if abs(imag(e)) < 1e-12 && real(e) > 0]
        push!(θs, isempty(rp) ? Inf : 1 / maximum(rp))
        p <= 10 && (sv = svdvals(M); push!(conds, sv[1] / sv[end]))
    end
    fin = filter(isfinite, θs)
    Bres[string(p)] = Dict("theta_k" => [isfinite(t) ? t : -1.0 for t in θs], "theta_min" => isempty(fin) ? Inf : minimum(fin),
                           "theta_med" => isempty(fin) ? Inf : median(fin), "gate" => gate, "cond" => conds, "theta_true" => T8[string(p)]["theta_c_true"])
    JA = nothing; JN = nothing; GC.gc()
end
spear(a, b) = (ra = invperm(sortperm(a)); rb = invperm(sortperm(b)); cor(Float64.(ra), Float64.(rb)))
tt = [Bres[string(p)]["theta_true"] for p in 1:50]
for key in ("theta_min", "theta_med")
    tp = [Bres[string(p)][key] for p in 1:50]
    ok = isfinite.(tp)
    sp = spear(tp[ok], tt[ok]); frac = mean(abs.(tt[ok] .- tp[ok]) .<= 0.15 .* tp[ok])
    emit(@sprintf("%-9s : n fini %d/50 | Spearman %.3f | fraction ±15 %% %.3f | médiane θ_pred %.3f (θ_c vrai médian %.3f)",
                  key, sum(ok), sp, frac, median(tp[ok]), median(tt)))
    Bres[key * "_spearman"] = sp; Bres[key * "_frac15"] = frac
end
emit(@sprintf("porte rang 2 (‖ΔJ − BC‖/‖ΔJ‖) max %.1e | κ(I − J^A_k) médiane %.2e, max %.2e (prompts 1–10)",
              maximum(Bres[string(p)]["gate"] for p in 1:50), median(vcat([Bres[string(p)]["cond"] for p in 1:10]...)),
              maximum(vcat([Bres[string(p)]["cond"] for p in 1:10]...))))
b1 = max(Bres["theta_min_spearman"], Bres["theta_med_spearman"]) >= 0.7
b2 = max(Bres["theta_min_frac15"], Bres["theta_med_frac15"]) >= 0.7
emit("B1 (Spearman ≥ 0,7) : $(b1 ? "VRAI" : "FAUX") | B2 (±15 % sur ≥ 70 %) : $(b2 ? "VRAI" : "FAUX")")
out["B"] = Bres

# ── Test C ─────────────────────────────────────────────────────────────────────────────────────
emit("\n=== Test C — erank de g_k = T_{k→L}ᵀT_{k→L} (« Riemannian Funnel »), Qwen, variante T")
Cres = Dict{String,Any}()
function finalN(mt)
    x = Float64.(mt["X"][end]); γ = Float64.(mt["gamma_final"]); r = mt["rms_inv_final"]
    r .* (Diagonal(γ) * (I - (x * x') .* (r^2 / D)))
end
for p in 1:10
    mt = meta("qwen", p); N = finalN(mt); JT = loadJs("qwen", p, "T")
    P = Matrix{Float64}(I, D, D); prs = zeros(27); hs = zeros(27); prN = zeros(27); hN = zeros(27); smin = zeros(27)
    for k in 27:-1:1
        P = P * JT[k]                       # T_{k→L} = J_27···J_k
        s = svdvals(P); sN = svdvals(N * P)
        prs[k] = pr(s); hs[k] = ent(s); prN[k] = pr(sN); hN[k] = ent(sN); smin[k] = s[end] / s[1]
    end
    Cres[string(p)] = Dict("pr" => prs, "ent" => hs, "prN" => prN, "entN" => hN, "smin_ratio" => smin)
    JT = nothing; GC.gc()
end
med_k(key) = [median(Cres[string(p)][key][k] for p in 1:10) for k in 1:27]
for key in ("pr", "ent", "prN", "entN")
    v = med_k(key)
    emit(@sprintf("%-4s médiane par k (k = 1, 5, 10, 15, 19, 23, 26, 27) : %s", key, join([@sprintf("%.1f", v[k]) for k in (1, 5, 10, 15, 19, 23, 26, 27)], " ; ")))
end
vpr = med_k("pr")
c1 = all(diff(vpr[19:27]) .<= 0) && vpr[27] <= 30
up = mean(Cres[string(p)]["pr"][27] > Cres[string(p)]["pr"][1] for p in 1:10)
c2 = any(minimum(Cres[string(p)]["smin_ratio"]) <= 1e-8 for p in 1:10)
emit(@sprintf("C1 (décroissance pour k ≥ 19 et erank_PR(g_27) ≤ 30) : %s | fraction erank_PR(g_27) > erank_PR(g_1) : %.2f",
              c1 ? "VRAI" : "FAUX", up))
emit(@sprintf("C2 (noyau numérique σ_min/σ_max ≤ 1e-8) : %s | min σ_min/σ_max observé %.2e", c2 ? "VRAI" : "FAUX",
              minimum(minimum(Cres[string(p)]["smin_ratio"]) for p in 1:10)))
out["C"] = Cres
open(joinpath(NB, "wind_T21_tests.json"), "w") do f; JSON.print(f, out); end
open(joinpath(NB, "wind_T21_tests_results.txt"), "w") do f; println(f, join(lines, "\n")); end
