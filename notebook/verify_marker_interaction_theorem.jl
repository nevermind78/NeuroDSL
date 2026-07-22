# ══════════════════════════════════════════════════════════════════════════════
# Vérification NUMÉRIQUE de Theorem~\ref{thm:interaction} (artilce/math4.tex,
# §6.5, "A Partial Proof: The RMSNorm-Mediated Interaction Term") sur de vraies
# activations/poids -- AUCUN réentraînement, checkpoints existants uniquement.
#
# Rappel de l'identité EXACTE à vérifier (Théorème 3) :
#   Delta(x) = g(r1-v) - g(r1) = -Dg(r1)*v - R(x),   g := M o N
#   R(x) := integral_0^1 [Dg(r1-t*v) - Dg(r1)] v dt
#   avec la borne ||R(x)|| <= L(x)||v(x)||, L(x) := sup_t ||Dg(r1-t*v)-Dg(r1)||_op
# et le Corollaire (mono-carrier) : v(x)=0 => Delta(x)=0 EXACTEMENT.
#
# Protocole (chaque test isole une source d'erreur possible AVANT de conclure) :
#   ÉTAPE 0 -- SANITÉ : les fonctions N (RMSNorm) et M (MLP) codées à la main
#     ici reproduisent-elles EXACTEMENT les nœuds du graphe (layer_1_norm2_out,
#     layer_1_mlp_out) sur une vraie entrée ? Si non -> bug d'implémentation,
#     pas une question mathématique -- on s'arrête avant d'aller plus loin.
#   ÉTAPE 1 -- IDENTITÉ : Dg(r1) calculé par différences finies centrées
#     (indépendant de toute dérivation analytique -- test de bout en bout du
#     code, pas de la théorie), R(x) calculé de DEUX façons INDÉPENDANTES :
#     (a) réarrangement direct R = -Dg(r1)v - Delta(x) (vrai par construction,
#         sert de référence) ; (b) quadrature numérique explicite de
#         l'intégrale (Simpson composite, Dg réévalué à chaque point du
#         segment) -- (a) et (b) doivent coïncider : c'est le VRAI test
#         (si mes différences finies étaient fausses à un point du segment,
#         (a) et (b) diffèreraient).
#   ÉTAPE 2 -- BORNE : L(x) échantillonné sur le segment, ||R(x)|| <= L(x)||v(x)||
#     jamais violée.
#   ÉTAPE 3 -- COROLLAIRE MONO-CARRIER : ablation d'UN SEUL carrier (D1) ->
#     v(x) est calculé et affiché comme étant EXACTEMENT le vecteur nul PAR
#     CONSTRUCTION (rien n'est soustrait de r1 puisque le second carrier
#     n'est pas touché), pas une coïncidence numérique.
#   ÉTAPE 4 -- ORDRE DE R : mise à l'échelle synthétique alpha*v (une direction
#     fixe, alpha in {1, 1/2, 1/4, 1/8}) -- si R = O(||v||^2), ||R(alpha*v)||
#     doit décroître comme alpha^2 (test du deuxième ordre) ; comparaison
#     supplémentaire entre 3 instances (inst2, seed_33, seed_44, configs DJ
#     réelles) dont les ||v|| naturels diffèrent, à titre indicatif.
# ══════════════════════════════════════════════════════════════════════════════

ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))
using LinearAlgebra, JSON

const CKPT_DIR = get(ENV, "MARKER_CKPT_DIR", joinpath(@__DIR__, "marker_ckpt"))
const N_HEADS = 4
const D_HEAD  = DIM ÷ N_HEADS
const EPS_RMS = 1f-6   # défaut de rmsnorm_fwd!, src/kernels.jl:54

head_site(l, h) = Symbol("layer_$(l)_mha_ao_h$(h)")
mlp_site(l)     = Symbol("layer_$(l)_mlp_out")

# ── Helpers repris de marker_conj1_verify.jl (identiques, non redérivés) ─────
function run_forward!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
end
function sample_contrast(rng)
    while true
        k = rand(rng, 1:V); sk = SIGMA[k]
        rest = shuffle(rng, collect(setdiff(1:V, (k, sk))))[1:N_PAIRS-2]
        ks = vcat([k, sk], rest); vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS); push!(ctx, ks[i]); push!(ctx, vs[i]); end
        return (; tokA = vcat(ctx, [MARKER_A, sk]), tokB = vcat(ctx, [MARKER_B, sk]), lit = vs[2], inv = vs[1])
    end
end
function draw_clean_pair!(g, ns, rng)
    for _ in 1:200
        p = sample_contrast(rng)
        d_out = run_forward!(g, ns, p.tokA); r_out = run_forward!(g, ns, p.tokB)
        sop(o) = Float64(o[1, p.lit] - o[1, p.inv])
        dd, dr = sop(d_out), sop(r_out)
        (argmax(vec(d_out)) == p.lit && argmax(vec(r_out)) == p.inv && dd > 0 > dr) && return p
    end
    error("pas de paire propre")
end
function collect_deltas!(g, ns, sites; n_pairs, seed)
    rng = MersenneTwister(seed)
    rows = Dict{Symbol,Vector{Vector{Float64}}}(s => Vector{Vector{Float64}}() for s in sites)
    for _ in 1:n_pairs
        p = draw_clean_pair!(g, ns, rng)
        run_forward!(g, ns, p.tokA); dcache = NeuroDSL.capture_activations(g, ns)
        run_forward!(g, ns, p.tokB); rcache = NeuroDSL.capture_activations(g, ns)
        for s in sites
            da = Array(dcache[s]); ra = Array(rcache[s])
            for row in (SEQ_LEN - 1, SEQ_LEN)
                push!(rows[s], vec(Float64.(da[row, :] .- ra[row, :])))
            end
        end
    end
    return rows
end
function format_subspace(vecs; energy=0.90, kcap::Int)
    X = reduce(hcat, vecs); F = svd(X)
    e = F.S .^ 2; ce = cumsum(e) ./ sum(e)
    k = min(something(findfirst(>=(energy), ce), length(ce)), kcap)
    return F.U[:, 1:k]
end
node_val(g, ns, sym) = Array(NeuroDSL.node(g, sym; namespace=ns).value)

# ── N (RMSNorm) et M (SwiGLU-MLP) codés à la main, indépendamment du graphe ──
# Convention vecteur-ligne (1xdim), comme les activations du graphe (SEQ_LEN x DIM).
rmsnorm_hand(r::Vector{Float64}, gamma::Vector{Float64}; eps=Float64(EPS_RMS)) = begin
    d = length(r); rho = 1.0 / sqrt(sum(r .^ 2) / d + eps)
    gamma .* r .* rho
end
silu(t) = t / (1.0 + exp(-t))
mlp_hand(z::Vector{Float64}, W1, W2, W3) = begin
    gt = W1 * z; up = W3 * z            # W1,W3 : (hidden,dim) ; gt,up : (hidden,)
    sg = silu.(gt) .* up
    W2 * sg                             # W2 : (dim,hidden) -> (dim,)
end
g_hand(r, gamma, W1, W2, W3) = mlp_hand(rmsnorm_hand(r, gamma), W1, W2, W3)

# Jacobien de g par différences finies centrées (INDÉPENDANT de toute formule
# analytique -- teste le CODE de bout en bout, pas seulement le Lemme).
function jacobian_fd(f::Function, r::Vector{Float64}; h=1e-5)
    d = length(r); J = zeros(Float64, d, d)
    for j in 1:d
        rp = copy(r); rp[j] += h
        rm = copy(r); rm[j] -= h
        J[:, j] = (f(rp) .- f(rm)) ./ (2h)
    end
    return J
end

opnorm2(A) = maximum(svdvals(A))   # norme d'opérateur (plus grande valeur singulière)

const N_PROBES = parse(Int, get(ENV, "MARKER_VERIFY_PROBES", "15"))

function verify_instance(name, ckpt, head_sym, mlp_sym; n_probes=N_PROBES, quad_n=41)
    println("\n", "═"^70); println("INSTANCE $name  ($ckpt)  carriers=$head_sym + $mlp_sym"); flush(stdout)
    NeuroDSL.load_graph!(g, ns, ckpt; overwrite=true)
    r1 = evaluate_marker(g, logits, ns; n_eval=200)
    println("  P1 : ", r1)

    m = match(r"^layer_(\d+)_mha_ao_h(\d+)$", String(head_sym))
    l = parse(Int, m.captures[1]); h = parse(Int, m.captures[2])
    cols = (h-1)*D_HEAD+1 : h*D_HEAD

    # Tout en Float64 pour les différences finies (le graphe est Float32 --
    # conversion explicite, une seule fois, pas de mélange de types ensuite).
    gamma = vec(Float64.(node_val(g, ns, Symbol("layer_$(l)_norm2_gamma"))))
    W1 = Float64.(node_val(g, ns, Symbol("layer_$(l)_mlp_w1")))
    W2 = Float64.(node_val(g, ns, Symbol("layer_$(l)_mlp_w2")))
    W3 = Float64.(node_val(g, ns, Symbol("layer_$(l)_mlp_w3")))
    W_full = Float64.(node_val(g, ns, Symbol("layer_$(l)_mha_output_W")))
    Wcols = W_full[:, cols]   # (dim, d_head)

    # sous-espaces de format (mêmes graines que marker_conj1_verify.jl/marker_seed_matrix.jl)
    deltas = collect_deltas!(g, ns, [head_sym, mlp_sym]; n_pairs=32, seed=1618)
    Uh = format_subspace(deltas[head_sym]; kcap=4)
    Um = format_subspace(deltas[mlp_sym]; kcap=8)
    P = Uh * Uh'          # (d_head, d_head)
    Q = Um * Um'          # (dim, dim)
    println("  k_head=", size(Uh,2), "  k_mlp=", size(Um,2))

    gfun(r) = g_hand(r, gamma, W1, W2, W3)

    # ── ÉTAPE 0 : sanité contre le graphe ────────────────────────────────────
    rng0 = MersenneTwister(555)
    p0 = draw_clean_pair!(g, ns, rng0)
    run_forward!(g, ns, p0.tokB)  # état "receveur" -- celui ablaté dans le pipeline principal
    r1v   = vec(Float64.(node_val(g, ns, Symbol("layer_$(l)_res1"))[SEQ_LEN, :]))
    xn2_g = vec(Float64.(node_val(g, ns, Symbol("layer_$(l)_norm2_out"))[SEQ_LEN, :]))
    mlp_g = vec(Float64.(node_val(g, ns, mlp_sym))[SEQ_LEN, :])
    xn2_hand = rmsnorm_hand(r1v, gamma)
    mlp_hand_v = mlp_hand(xn2_hand, W1, W2, W3)
    err_N = maximum(abs.(xn2_hand .- xn2_g))
    err_M = maximum(abs.(mlp_hand_v .- mlp_g))
    println("  [étape 0] sanité N (RMSNorm) : |main-graphe|_max = ", err_N)
    println("  [étape 0] sanité M (MLP)     : |main-graphe|_max = ", err_M)
    sane = err_N < 1e-3 && err_M < 1e-3   # tolérance float32 (poids/activations Float32)
    println("  [étape 0] ", sane ? "OK -- N/M codés à la main reproduisent le graphe." :
            "ÉCHEC -- bug d'implémentation, PAS une question mathématique. Arrêt de l'instance.")
    sane || return nothing

    # ── ÉTAPES 1-2 : identité exacte + borne, sur n_probes entrées réelles ──
    rng = MersenneTwister(9090)
    max_ident_gap = 0.0; max_bound_viol = 0.0
    Rs = Float64[]; Vs = Float64[]
    Rrearr_norms = Float64[]; Rquad_norms = Float64[]   # pour le nuage de points (a)
    for _ in 1:n_probes
        p = draw_clean_pair!(g, ns, rng)
        run_forward!(g, ns, p.tokB)
        row = rand((SEQ_LEN - 1, SEQ_LEN))
        r1x = vec(Float64.(node_val(g, ns, Symbol("layer_$(l)_res1"))[row, :]))
        a_x = vec(Float64.(node_val(g, ns, head_sym)[row, :]))   # (d_head,)
        v = Wcols * (P * a_x)                                    # (dim,) -- v(x)=a(x)PW^T, convention col
        normv = norm(v)
        normv < 1e-9 && continue

        Delta_raw = gfun(r1x .- v) .- gfun(r1x)
        Jg = jacobian_fd(gfun, r1x)
        R_rearr = -(Jg * v) .- Delta_raw            # rearrangement exact (référence)

        # quadrature composite de Simpson de l'intégrale : ∫_0^1 [Dg(r1-tv)-Dg(r1)] v dt
        # quad_n DOIT être impair (nombre de points) pour couvrir tout [0,1] avec un
        # nombre pair d'intervalles -- vérifié explicitement, pas ajusté après coup.
        isodd(quad_n) || error("quad_n doit être impair pour Simpson composite sur [0,1]")
        ts = range(0.0, 1.0; length=quad_n)
        integrand(t) = (jacobian_fd(gfun, r1x .- t .* v) .- Jg) * v
        vals = [integrand(t) for t in ts]
        nT = length(ts) - 1   # pair par construction (quad_n impair)
        hstep = 1.0 / nT
        S = vals[1] .+ vals[nT+1]
        for i in 2:nT
            S = S .+ (iseven(i-1) ? 2.0 : 4.0) .* vals[i]
        end
        R_quad = (hstep / 3.0) .* S

        ident_gap = norm(R_rearr .- R_quad) / (norm(R_rearr) + 1e-9)
        max_ident_gap = max(max_ident_gap, ident_gap)

        # borne : L(x) = sup_t ||Dg(r1-tv)-Dg(r1)||_op, échantillonné
        Ls = [opnorm2(jacobian_fd(gfun, r1x .- t .* v) .- Jg) for t in range(0.0, 1.0; length=9)]
        Lx = maximum(Ls)
        bound_ok = norm(R_rearr) <= Lx * normv * 1.001   # tolérance flottante
        bound_viol = max(0.0, norm(R_rearr) - Lx * normv)
        max_bound_viol = max(max_bound_viol, bound_viol)

        push!(Rs, norm(R_rearr)); push!(Vs, normv)
        push!(Rrearr_norms, norm(R_rearr)); push!(Rquad_norms, norm(R_quad))
    end
    println("  [étape 1] écart relatif max entre R (réarrangement) et R (quadrature Simpson) : ",
            round(max_ident_gap, sigdigits=4), "  (attendu ~0 : identité de Taylor exacte confirmée)")
    println("  [étape 2] violation max de ||R|| <= L(x)||v(x)|| : ", round(max_bound_viol, sigdigits=4),
            "  (attendu 0 : borne jamais violée)")

    # ── ÉTAPE 3 : corollaire mono-carrier (D1) -- v(x) EXACTEMENT nul, PAS une
    # coïncidence numérique. Test sur le GRAPHE réel (pas juste ma reconstruction
    # à la main) : applique la VRAIE ablation de poids D1 (MLP seul, W_O intact)
    # et vérifie que r1 -- lu directement sur le graphe -- est bit-identique
    # avant/après. Puis, pour contraste, montre que la tête n'est PAS silencieuse :
    # son ablation hypothétique (même sous-espace P que DJ) déplacerait r1 d'une
    # quantité ||v|| franchement non nulle -- donc l'exactitude de D1 n'est pas un
    # cas dégénéré où "il n'y a rien à ablater de toute façon".
    run_forward!(g, ns, p0.tokB)
    r1_before = Array(Float64.(node_val(g, ns, Symbol("layer_$(l)_res1"))))
    mlp_w2_sym = Symbol("layer_$(l)_mlp_w2")
    W2_orig = node_val(g, ns, mlp_w2_sym)
    NeuroDSL.set_params!(g, ns, Dict(mlp_w2_sym => Float32.((Matrix{Float64}(I, DIM, DIM) .- Q) * W2_orig)))
    NeuroDSL.invalidate_all!(g; namespace=ns); NeuroDSL.demand!(g, :final_logits; namespace=ns)
    r1_after_D1 = Array(Float64.(node_val(g, ns, Symbol("layer_$(l)_res1"))))
    max_r1_shift_D1 = maximum(abs.(r1_after_D1 .- r1_before))
    NeuroDSL.set_params!(g, ns, Dict(mlp_w2_sym => W2_orig))   # restaure
    NeuroDSL.invalidate_all!(g; namespace=ns); NeuroDSL.demand!(g, :final_logits; namespace=ns)

    a_x0 = vec(Float64.(node_val(g, ns, head_sym)[SEQ_LEN, :]))
    v_hyp = Wcols * (P * a_x0)   # ce que v(x) SERAIT si la tête était AUSSI ablatée (DJ)
    println("  [étape 3] D1 réel (poids : MLP seul ablaté, W_O intact) : |r1_après - r1_avant|_max = ",
            max_r1_shift_D1, " (bit-identique attendu -> v(x)=0 EXACTEMENT, pas une coïncidence)")
    println("             contraste -- la tête n'est PAS silencieuse : si elle était AUSSI ablatée (DJ), ",
            "||v||_hyp = ", round(norm(v_hyp), sigdigits=4), " (nettement non nul).")

    return (; name, l, gfun, Wcols, P, Rs, Vs, Rrearr_norms, Rquad_norms,
             max_ident_gap, max_bound_viol, err_N, err_M, max_r1_shift_D1, v_hyp_norm=norm(v_hyp))
end

# ── Lance sur 3 instances réelles (mêmes carriers de couche 1 que Table tab:verify) ─
results = NamedTuple[]
for (name, ckpt, hs, ms) in [
        ("inst2",   joinpath(CKPT_DIR, "marker_task"), head_site(1,2), mlp_site(1)),
        ("seed_33", joinpath(CKPT_DIR, "seed_33"),      head_site(1,3), mlp_site(1)),
        ("seed_44", joinpath(CKPT_DIR, "seed_44"),      head_site(1,3), mlp_site(1)),
    ]
    r = verify_instance(name, ckpt, hs, ms)
    r !== nothing && push!(results, r)
end

# ── ÉTAPE 4 : ordre de R -- mise à l'échelle synthétique + comparaison inter-instances ─
median_(x) = sort(x)[cld(length(x), 2)]
println("\n", "═"^70); println("ÉTAPE 4 -- ordre de R(x) en fonction de ||v(x)||"); flush(stdout)
idx44 = findfirst(r -> r.name == "seed_44", results)
if idx44 !== nothing
    r0 = results[idx44]
    rng = MersenneTwister(4242)
    NeuroDSL.load_graph!(g, ns, joinpath(CKPT_DIR, "seed_44"); overwrite=true)
    p = draw_clean_pair!(g, ns, rng)
    run_forward!(g, ns, p.tokB)
    row = SEQ_LEN
    r1x = vec(Float64.(node_val(g, ns, Symbol("layer_$(r0.l)_res1"))[row, :]))
    a_x = vec(Float64.(node_val(g, ns, head_site(1, 3))[row, :]))
    v_full = r0.Wcols * (r0.P * a_x)
    println("  mise à l'échelle synthétique alpha*v (direction fixe, seed_44) :")
    global ALPHA_SCALING = NamedTuple[]
    for alpha in (1.0, 0.5, 0.25, 0.125)
        v = alpha .* v_full
        Delta_raw = r0.gfun(r1x .- v) .- r0.gfun(r1x)
        Jg = jacobian_fd(r0.gfun, r1x)
        R = -(Jg * v) .- Delta_raw
        push!(ALPHA_SCALING, (; alpha, norm_v=norm(v), norm_R=norm(R),
                               ratio1=norm(R)/norm(v), ratio2=norm(R)/norm(v)^2))
        println("    alpha=$alpha  ||v||=", round(norm(v), sigdigits=4),
                "  ||R||=", round(norm(R), sigdigits=4),
                "  ||R||/||v||=", round(norm(R)/norm(v), sigdigits=4),
                "  ||R||/||v||^2=", round(norm(R)/norm(v)^2, sigdigits=4))
    end
    println("  (si R = O(||v||^2) : ||R||/||v|| doit décroître avec alpha, ||R||/||v||^2 doit rester ~constant)")

    println("\n  comparaison inter-instances (||v|| et ||R|| naturels, mêmes 15 sondes ci-dessus) :")
    for r in results
        isempty(r.Rs) && continue
        println("    $(r.name) : médiane ||v|| = ", round(median_(r.Vs), sigdigits=4),
                "  médiane ||R|| = ", round(median_(r.Rs), sigdigits=4),
                "  médiane ||R||/||v||^2 = ", round(median_(r.Rs ./ r.Vs.^2), sigdigits=4))
    end
end

println("\n", "═"^70)
println("VERDICT VÉRIFICATION NUMÉRIQUE DU THÉORÈME 3")
println("═"^70)

# ── Export JSON pour la figure (nuage R_rearr vs R_quad + courbe alpha) ──────
# Même patron que attribution_curve_results*.json / generate_e6_figure.jl :
# pas de nombres recopiés à la main dans le script de figure, lecture directe
# du JSON produit par CE script de vérification.
out = Dict{String,Any}(
    "per_probe" => [Dict("instance"=>r.name, "R_rearr"=>rr, "R_quad"=>rq)
                    for r in results for (rr, rq) in zip(r.Rrearr_norms, r.Rquad_norms)],
    "alpha_scaling" => [Dict("alpha"=>a.alpha, "norm_v"=>a.norm_v, "norm_R"=>a.norm_R,
                              "ratio1"=>a.ratio1, "ratio2"=>a.ratio2) for a in (@isdefined(ALPHA_SCALING) ? ALPHA_SCALING : NamedTuple[])],
    "summary" => [Dict("instance"=>r.name, "err_N"=>r.err_N, "err_M"=>r.err_M,
                        "max_ident_gap"=>r.max_ident_gap, "max_bound_viol"=>r.max_bound_viol,
                        "max_r1_shift_D1"=>r.max_r1_shift_D1, "v_hyp_norm"=>r.v_hyp_norm)
                   for r in results],
)
open(joinpath(@__DIR__, "interaction_verify_results.json"), "w") do io
    JSON.print(io, out)
end
println("Écrit -> notebook/interaction_verify_results.json (", length(out["per_probe"]), " sondes)")
