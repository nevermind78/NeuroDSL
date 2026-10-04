# WIND-T20 (CPU) -- linéarisation « REC » (unique linéarisation complète ET transverse-exacte, théorème 1(a)) sur Qwen.
# REC = normes figées, dérivée EXACTE sur le sous-espace transverse ĥ⊥, gain radial du MLP ramené à la complétude
# J̃_m h = m (au lieu du vecteur d'Euler M h). Construction exacte à partir des facteurs T13 :
#   J^AA  = J^A + s₁ (A1 + B1) x̂ᵀ                        (norme d'attention figée : déjà complète et transverse-exacte)
#   J^AN  = J^AA + s₂ MH ρᵀ,  ρ = (I + A₁)ᵀ ĥ              (ρ extrait exactement de R − R¹, rang 1, porte G1)
#   J^REC = J^AA + c ρᵀ,      c = m/‖h‖ − (1 − s₂)·MH      (⇒ J̃_m = M − (Mh − m)hᵀ/‖h‖² = L* du théorème 1)
# θ-chemin J^A + θ(J^REC − J^A) évalué par le relèvement exact (lemme A, T7) avec les facteurs exacts de rang 2
# U_k = [s₁(A1+B1), c], V_k = [x̂, ρ] ; porte : relèvement à θ = 1 = produit direct.
# Attributions des 28 sous-couches MLP (gradient au point d'écriture x_l, norme finale VIVANTE) contre la vérité
# WIND-T9. Valeurs T, A, AN, AH reprises de T8 (secondaire, vérité), T9, T14, T15 (pas recalculées).
# USAGE : WIND_PROMPTS=… WIND_BLAS=4 julia --project=. notebook/wind_T20_rec.jl
using LinearAlgebra, Statistics, Printf, JSON
const T8DIR = joinpath(@__DIR__, "wind_data_T8")
const PIDX = parse.(Int, split(get(ENV, "WIND_PROMPTS", join(1:50, ",")), ","))
const OUT = get(ENV, "WIND_T20_REC_OUT", joinpath(@__DIR__, "wind_T20_rec.json"))
const L, D = 28, 1536
const THETAS = collect(0.25:0.125:2.5)
BLAS.set_num_threads(parse(Int, get(ENV, "WIND_BLAS", "4")))
loadJ(p, V) = (A = Array{Float32}(undef, D, D, L); open(joinpath(T8DIR, "wind_J_p$(p)_$(V).bin"), "r") do f; read!(f, A); end;
               [Float64.(A[:, :, k]) for k in 2:L])
ent(s) = (q = s .^ 2 ./ sum(abs2, s); q = q[q .> 0]; exp(-sum(q .* log.(q))))
θcross(θs, e) = (i = findfirst(<=(3.0), e); i === nothing ? 99.0 : i == 1 ? θs[1] :
                 θs[i-1] + (θs[i] - θs[i-1]) * (e[i-1] - 3) / (e[i-1] - e[i]))
fac = JSON.parsefile(joinpath(@__DIR__, "wind_T13_factors.json"))
t9 = JSON.parsefile(joinpath(@__DIR__, "wind_T9_attrib.json")); t14 = JSON.parsefile(joinpath(@__DIR__, "wind_T14_finalnorm.json"))
metas = Dict(q => JSON.parsefile(joinpath(T8DIR, "wind_meta_p$(q).json")) for q in 1:50)
# lecture contrefactuelle de WIND-T9 : w = W_U[réponse] − W_U[contrefactuel] (la méta stocke top1 − top2)
const WREAD = JSON.parsefile(joinpath(@__DIR__, "wind_T20_wread.json"))
μMLP = [sum(Float64.(metas[q]["MLP"][l]) for q in 1:50) ./ 50 for l in 1:L]
function mlp_attr(Js, Nmat, w, meta)       # a_l = ⟨(N Π_{k≥l} J_k)ᵀ w, MLP_l − μ_l⟩, l = 1..28 (MLP_l écrit x_l)
    g = Nmat' * w; a = zeros(L)
    for l in L:-1:1
        a[l] = dot(g, Float64.(meta["MLP"][l]) .- μMLP[l])
        l > 1 && (g = Js[l-1]' * g)
    end
    a
end
out = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String,Any}()
for p in PIDX
    haskey(out, string(p)) && continue
    t0 = time(); fp = fac[string(p)]; meta = metas[p]
    γ = Float64.(meta["gamma_final"]); ri = meta["rms_inv_final"]; X = [Float64.(v) for v in meta["X"]]
    H = [Float64.(v) for v in meta["H"]]; MLP = [Float64.(v) for v in meta["MLP"]]; x = X[end]
    N = ri .* (Diagonal(γ) * (I - (x * x') .* (ri^2 / D))); Nfz = Matrix(ri .* Diagonal(γ))
    w = Float64.(WREAD[string(p)])
    nchk = length(fp["check_blocks"])
    buf = Array{Float64}(undef, D * 27 * 3 + D * 29 * 2 + D * 3 * nchk)
    open(joinpath(T8DIR, "wind_T13_factors_p$(p).bin"), "r") do f; read!(f, buf); end
    A1 = reshape(buf[1:D*27], D, 27); B1 = reshape(buf[D*27+1:2D*27], D, 27); MH = reshape(buf[2D*27+1:3D*27], D, 27)
    s1 = Float64.(fp["s1"]); s2 = Float64.(fp["s2"])
    JA = loadJ(p, "A"); JN = loadJ(p, "AN")
    U = Vector{Matrix{Float64}}(undef, 27); V = Vector{Matrix{Float64}}(undef, 27)
    JAA = Vector{Matrix{Float64}}(undef, 27); JR = Vector{Matrix{Float64}}(undef, 27)
    g1 = zeros(27); eul = zeros(27); eulperp = zeros(27)
    for k in 1:27
        xh = X[k+1] ./ norm(X[k+1]); hk = H[k+1]; hn = norm(hk); mk = MLP[k+1]   # bloc k : MHA/MLP index k+1
        u1 = s1[k] .* (A1[:, k] .+ B1[:, k])
        JAA[k] = JA[k] + u1 * xh'
        Z = JN[k] - JAA[k]; mh = MH[:, k]
        ρv = (Z' * mh) ./ (s2[k] * sum(abs2, mh))
        g1[k] = norm(Z - s2[k] .* (mh * ρv')) / norm(Z)
        c = mk ./ hn .- (1 - s2[k]) .* mh
        JR[k] = JAA[k] + c * ρv'
        U[k] = hcat(u1, c); V[k] = hcat(xh, ρv)
        ev = hn .* mh
        eul[k] = dot(ev, mk) / sum(abs2, mk); eulperp[k] = norm(ev .- eul[k] .* mk) / norm(ev)
    end
    rec = Dict{String,Any}("gate_G1_max" => maximum(g1), "euler_d" => eul, "euler_perp" => eulperp)
    # attributions (avant de libérer les Jacobiennes)
    rec["attr_ANf"] = mlp_attr(JN, N, w, meta); rec["attr_AAf"] = mlp_attr(JAA, N, w, meta)
    rec["attr_REC"] = mlp_attr(JR, N, w, meta); rec["attr_A"] = mlp_attr(JA, N, w, meta)
    rec["gate_T14"] = max(maximum(abs.(rec["attr_ANf"] .- Float64.(t14[string(p)]["attr"]["ANf"][29:56]))) / maximum(abs.(rec["attr_ANf"])),
                          maximum(abs.(rec["attr_AAf"] .- Float64.(t14[string(p)]["attr"]["AAf"][29:56]))) / maximum(abs.(rec["attr_AAf"])))
    rec["gate_T9A"] = maximum(abs.(rec["attr_A"] .- Float64.(t9[string(p)]["aA"][29:56]))) / maximum(abs.(rec["attr_A"]))
    JN = nothing; JAA = nothing; GC.gc()
    # relèvement exact : N·P(θ) = M_A + θ 𝒰 (I − θW)⁻¹ 𝒱
    n = 54; W = zeros(n, n); 𝒰 = zeros(D, n); 𝒱 = zeros(n, D); Φ = Matrix{Float64}(I, D, D)
    for k in 1:27; 𝒱[2k-1:2k, :] = V[k]' * Φ; Φ = JA[k] * Φ; end
    MA = N * Φ
    for k in 1:27
        Y = U[k]
        for j in k+1:27; W[2j-1:2j, 2k-1:2k] = V[j]' * Y; Y = JA[j] * Y; end
        𝒰[:, 2k-1:2k] = N * Y
    end
    PR = JR[1]; for k in 2:27; PR = JR[k] * PR; end; MR = N * PR
    lift1 = MA + 𝒰 * ((I - W) \ 𝒱)
    rec["gate_lift"] = norm(lift1 - MR) / norm(MR)
    rec["ern_REC"] = ent(svdvals(MR))
    e = [ent(svdvals(MA + θ .* (𝒰 * ((I - θ .* W) \ 𝒱)))) for θ in THETAS]
    rec["path_REC"] = e; rec["theta_c_REC"] = θcross(THETAS, e)
    dR = Float64.(t9[string(p)]["dR"][29:56])
    spear(a, b) = (ra = invperm(sortperm(a)); rb = invperm(sortperm(b)); cor(Float64.(ra), Float64.(rb)))
    for V_ in ("A", "ANf", "AAf", "REC")
        rec["rho_$V_"] = spear(rec["attr_$V_"], dR)
        rec["early_$V_"] = sum(abs.(rec["attr_$V_"][1:10])) / sum(abs.(rec["attr_$V_"]))
    end
    rec["early_truth"] = sum(abs.(dR[1:10])) / sum(abs.(dR))
    out[string(p)] = rec
    open(OUT, "w") do f; JSON.print(f, out); end
    @printf("p%-3d G1 %.1e T14 %.1e T9A %.1e relèvement %.1e | ern REC %.2f | θ_c REC %.3f | ρ A %.2f ANf %.2f AAf %.2f REC %.2f | early vrai %.2f REC %.2f ANf %.2f | d %.2f | %.0f s\n",
            p, rec["gate_G1_max"], rec["gate_T14"], rec["gate_T9A"], rec["gate_lift"], rec["ern_REC"], rec["theta_c_REC"],
            rec["rho_A"], rec["rho_ANf"], rec["rho_AAf"], rec["rho_REC"], rec["early_truth"], rec["early_REC"], rec["early_ANf"], median(eul), time() - t0)
    JA = nothing; JR = nothing; GC.gc()
end
