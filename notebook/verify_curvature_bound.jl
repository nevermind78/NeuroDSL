# ══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION NUMÉRIQUE DE LA BORNE DE COURBURE FERMÉE (artilce/code4.tex,
# Proposition "Closed-form curvature bound") -- aucun réentraînement.
#
# CE QUI EST TESTÉ
# ----------------
# L'article prouve désormais (au lieu de l'estimer heuristiquement) :
#
#   (L1)  ||D^2 N(r)||_op  <=  6 ||gamma||_inf rho(r)^2 / sqrt(d)
#
#   (P)   ||D^2 g(r)||_op  <=  ||gamma||_inf^2 rho(r)^2 ||W1|| ||W2|| ||W3||
#                              * ( 6 + 8 s1 + s2 sqrt(d) ||gamma||_inf ||W1|| )
#
# avec g = M o N, M(z) = W2 (SiLU(W1 z) . W3 z), s1 = sup|SiLU'|, s2 = sup|SiLU''|,
# et ||.|| = norme d'opérateur spectrale.
#
# Ces deux bornes sont des THÉORÈMES : le test ne peut pas les "confirmer", il
# peut seulement les RÉFUTER. C'est exactement son rôle -- une seule violation
# sur des poids réels signalerait une erreur d'algèbre dans la dérivation, qui
# est faite à la main et dont chaque étape (D^2N, D^2h, composition) est ici
# recalculée indépendamment par différences finies du QUATRIÈME point
# (D^2f[u,v] ~ (f(r+hu+hv) - f(r+hu-hv) - f(r-hu+hv) + f(r-hu-hv)) / (4h^2)),
# sans réutiliser une seule ligne de la dérivation analytique.
#
# On rapporte AUSSI le facteur de relâchement (borne / courbure observée) : une
# borne rigoureuse mais lâche de plusieurs ordres de grandeur reste une borne,
# et c'est le chiffre honnête à publier plutôt qu'un silence.
#
# Sondes : graine 20260730, distincte de tous les autres scripts.
# ══════════════════════════════════════════════════════════════════════════════

ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))
using LinearAlgebra, JSON

const CKPT_DIR = get(ENV, "MARKER_CKPT_DIR", joinpath(@__DIR__, "marker_ckpt"))
const INSTANCES = [("inst2",   joinpath(CKPT_DIR, "marker_task"), 1),
                   ("seed_11", joinpath(CKPT_DIR, "seed_11"),     1),
                   ("seed_22", joinpath(CKPT_DIR, "seed_22"),     1),
                   ("seed_33", joinpath(CKPT_DIR, "seed_33"),     1),
                   ("seed_44", joinpath(CKPT_DIR, "seed_44"),     1)]
const EPS_RMS  = 1f-6          # défaut de rmsnorm_fwd!, src/kernels.jl:54
const SEED_PROBE = 20260730
const N_PROBE_R  = 6           # points r_1 réels par instance
const N_DIR      = 40          # couples (u,v) aléatoires par point

# ── s1 = sup|SiLU'|, s2 = sup|SiLU''| : calculés numériquement, pas cités ────
silu(t)  = t / (1 + exp(-t))
sig(t)   = 1 / (1 + exp(-t))
dsilu(t) = sig(t) * (1 + t * (1 - sig(t)))
# SiLU'' par différences finies centrées d'ordre 4 sur une grille fine
d2silu(t; h=1e-4) = (-dsilu(t+2h) + 8dsilu(t+h) - 8dsilu(t-h) + dsilu(t-2h)) / (12h)
const GRID = range(-60.0, 60.0; length=2_400_001)
const S1 = maximum(abs(dsilu(t)) for t in GRID)
const S2 = maximum(abs(d2silu(t)) for t in GRID)

# ── N et M codés à la main (repris de verify_marker_interaction_theorem.jl) ──
rho(r; d=length(r), eps=Float64(EPS_RMS)) = 1.0 / sqrt(sum(r .^ 2) / d + eps)
rmsnorm_hand(r, gamma) = gamma .* r .* rho(r)
mlp_hand(z, W1, W2, W3) = W2 * (silu.(W1 * z) .* (W3 * z))
g_hand(r, gamma, W1, W2, W3) = mlp_hand(rmsnorm_hand(r, gamma), W1, W2, W3)

opn(A) = maximum(svdvals(A))

# D^2 f[u,v] par différences finies à 4 points -- INDÉPENDANT de toute formule.
function d2_fd(f::Function, r::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64}; h=1e-3)
    return (f(r .+ h .* u .+ h .* v) .- f(r .+ h .* u .- h .* v)
          .- f(r .- h .* u .+ h .* v) .+ f(r .- h .* u .- h .* v)) ./ (4h^2)
end

function run_forward!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
end
node_val(g, ns, sym) = Array(NeuroDSL.node(g, sym; namespace=ns).value)

function sample_contrast(rng)
    while true
        k = rand(rng, 1:V); sk = SIGMA[k]
        rest = shuffle(rng, collect(setdiff(1:V, (k, sk))))[1:N_PAIRS-2]
        ks = vcat([k, sk], rest); vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS); push!(ctx, ks[i]); push!(ctx, vs[i]); end
        return (; tokB = vcat(ctx, [MARKER_B, sk]),)
    end
end

randunit(rng, d) = (w = randn(rng, d); w ./ norm(w))

println("Constantes universelles (calculées, non citées) : s1 = sup|SiLU'| = ",
        round(S1, digits=6), "   s2 = sup|SiLU''| = ", round(S2, digits=6))

ALL = NamedTuple[]
for (name, ckpt, l) in INSTANCES
    println("\n", "═"^74); println("INSTANCE $name  (couche $l)"); flush(stdout)
    NeuroDSL.load_graph!(g, ns, ckpt; overwrite=true)

    gamma = vec(Float64.(node_val(g, ns, Symbol("layer_$(l)_norm2_gamma"))))
    W1 = Float64.(node_val(g, ns, Symbol("layer_$(l)_mlp_w1")))
    W2 = Float64.(node_val(g, ns, Symbol("layer_$(l)_mlp_w2")))
    W3 = Float64.(node_val(g, ns, Symbol("layer_$(l)_mlp_w3")))
    d  = length(gamma)
    n1, n2, n3 = opn(W1), opn(W2), opn(W3)
    ginf = maximum(abs.(gamma))
    println("  d=$d  ||gamma||_inf=", round(ginf, digits=4),
            "  ||W1||=", round(n1, digits=3), " ||W2||=", round(n2, digits=3), " ||W3||=", round(n3, digits=3))

    Nfun(r) = rmsnorm_hand(r, gamma)
    gfun(r) = g_hand(r, gamma, W1, W2, W3)

    rng = MersenneTwister(SEED_PROBE)
    # points r_1 réels, pris sur le graphe (pas synthétiques)
    rs = Vector{Float64}[]
    while length(rs) < N_PROBE_R
        p = sample_contrast(rng)
        run_forward!(g, ns, p.tokB)
        r1 = Float64.(node_val(g, ns, Symbol("layer_$(l)_res1")))
        for row in (SEQ_LEN - 1, SEQ_LEN)
            length(rs) < N_PROBE_R && push!(rs, vec(r1[row, :]))
        end
    end

    violN = 0; violG = 0
    ratN = Float64[]; ratG = Float64[]
    obsN_max = 0.0; obsG_max = 0.0; boundN_min = Inf; boundG_min = Inf
    for r in rs
        rr = rho(r; d=d)
        boundN = 6 * ginf * rr^2 / sqrt(d)
        boundG = ginf^2 * rr^2 * n1 * n2 * n3 * (6 + 8*S1 + S2 * sqrt(d) * ginf * n1)
        boundN_min = min(boundN_min, boundN); boundG_min = min(boundG_min, boundG)
        for _ in 1:N_DIR
            u = randunit(rng, d); v = randunit(rng, d)
            oN = norm(d2_fd(Nfun, r, u, v))
            oG = norm(d2_fd(gfun, r, u, v))
            obsN_max = max(obsN_max, oN); obsG_max = max(obsG_max, oG)
            oN > boundN * (1 + 1e-6) && (violN += 1)
            oG > boundG * (1 + 1e-6) && (violG += 1)
            push!(ratN, boundN / max(oN, 1e-300)); push!(ratG, boundG / max(oG, 1e-300))
        end
    end
    nsamp = length(rs) * N_DIR
    println("  (L1) ||D^2 N||   : borne min sur les sondes = ", round(boundN_min, sigdigits=4),
            "   max observé = ", round(obsN_max, sigdigits=4),
            "   violations = $violN / $nsamp",
            "   relâchement min = ", round(minimum(ratN), digits=2), "x")
    println("  (P)  ||D^2 g||   : borne min sur les sondes = ", round(boundG_min, sigdigits=4),
            "   max observé = ", round(obsG_max, sigdigits=4),
            "   violations = $violG / $nsamp",
            "   relâchement min = ", round(minimum(ratG), sigdigits=4), "x")
    flush(stdout)
    push!(ALL, (; name, d, gamma_inf=ginf, nW1=n1, nW2=n2, nW3=n3,
                 boundN_min, obsN_max, violN, slackN_min=minimum(ratN),
                 boundG_min, obsG_max, violG, slackG_min=minimum(ratG), nsamp))
end

println("\n", "═"^74)
println("VERDICT -- bornes de courbure fermées sur poids réels")
println("═"^74)
tv_N = sum(r.violN for r in ALL); tv_G = sum(r.violG for r in ALL); tot = sum(r.nsamp for r in ALL)
println("  échantillons (r, u, v) testés   : ", tot, " par borne")
println("  violations de (L1) ||D^2N||     : ", tv_N, " / ", tot)
println("  violations de (P)  ||D^2g||     : ", tv_G, " / ", tot)
println("  relâchement minimal de (L1)     : ", round(minimum(r.slackN_min for r in ALL), digits=2), "x")
println("  relâchement minimal de (P)      : ", round(minimum(r.slackG_min for r in ALL), sigdigits=4), "x")
println()
println("  LECTURE : zéro violation est la seule issue compatible avec la dérivation ;")
println("  le relâchement dit à quel point la borne est conservatrice sur ces poids.")

out = Dict{String,Any}(
    "constants" => Dict("s1"=>S1, "s2"=>S2, "eps_rms"=>Float64(EPS_RMS),
                         "probe_seed"=>SEED_PROBE, "n_dirs_per_point"=>N_DIR),
    "per_instance" => [Dict(String(k)=>getfield(r, k) for k in keys(r)) for r in ALL],
)
open(joinpath(@__DIR__, "curvature_bound_results.json"), "w") do io
    JSON.print(io, out)
end
println("\nÉcrit -> notebook/curvature_bound_results.json")
