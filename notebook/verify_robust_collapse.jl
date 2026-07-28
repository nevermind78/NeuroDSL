# ══════════════════════════════════════════════════════════════════════════════
# TEST DES HYPOTHÈSES DU THÉORÈME DE COLLAPSE, VERSION ROBUSTE
# (artilce/code4.tex, Theorem thm:collapse / Corollary cor:invariance)
# -- aucun réentraînement, checkpoints existants uniquement.
#
# CE QUI EST TESTÉ
# ----------------
# Theorem 1 est un "si et seulement si" pour un collapse EXACT, sous
#   (i) qbar_S = 0     (retrait symétrique sur la paire)
#   (ii) Psi_S  = 0    (aucun contraste ne survit hors de S).
# Ces deux conditions sont de mesure nulle : aucune expérience ne les satisfait
# exactement. La version robuste repose sur l'IDENTITÉ EXACTE (vérifiée
# symboliquement, et re-vérifiée numériquement ci-dessous) :
#
#     s_S(x_A) - sbar = -qbar_S + Psi_S/2
#     s_S(x_B) - sbar = -qbar_S - Psi_S/2
#
# d'où, si |qbar_S| <= eps1 et |Psi_S| <= eps2 :
#     |s_S(x.) - sbar| <= eps1 + eps2/2,
# et la conclusion AU NIVEAU DE LA DÉCISION (les deux entrées basculent sur la
# branche sign(sbar)) tient dès que |sbar| > eps1 + eps2/2.
#
# CE SCRIPT MESURE, par paire appariée et par configuration :
#   eps1 = |qbar_S|,  eps2 = |Psi_S|,  |sbar|,  et la marge |sbar| - (eps1+eps2/2).
#
# Trois choses sont rapportées séparément, sans les confondre :
#   (A) résidu de l'IDENTITÉ (contrôle d'algèbre : doit être ~0 à la précision
#       machine ; un résidu non nul signifierait que la dérivation est fausse) ;
#   (B) taux de paires où l'HYPOTHÈSE robuste tient : |sbar| > eps1 + eps2/2 ;
#   (C) taux de paires où la CONCLUSION tient réellement -- s_S(x_A) et
#       s_S(x_B) de même signe que sbar -- que l'hypothèse tienne ou non.
# (C) sans (B) mesure à quel point la condition suffisante est conservatrice ;
# (B) proche de 0 avec (C) proche de 0 signifierait que le collapse lui-même
# n'a pas lieu. Les deux issues sont rapportées telles quelles.
#
# Quantités opérationnelles (identiques à marker_conj1_verify.jl) :
#   s_op(x) = logit_litteral - logit_inverse en position finale
#   g_A, g_B = s_op(x_A), s_op(x_B)
#   q(x)     = s_op(x) - s_op(x | activations des carriers de S projetées)
#   qbar_S = (q_A+q_B)/2 ; Phi = g_A-g_B ; Phi_S = q_A-q_B ; Psi_S = Phi-Phi_S ;
#   sbar = (g_A+g_B)/2 ; s_S(x) = s_op(x) - q(x).
#
# Configurations : les MÊMES 39 que verify_config_band_oos.jl (tous les
# sous-ensembles non vides de l'ensemble de carriers de chaque instance).
# Sondes : graines 20260801 / 20260802, distinctes de tous les autres scripts.
# ══════════════════════════════════════════════════════════════════════════════

const SMOKE = get(ENV, "MARKER_SMOKE", "0") == "1"
ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))
using LinearAlgebra, JSON

const CKPT_DIR = get(ENV, "MARKER_CKPT_DIR", joinpath(@__DIR__, "marker_ckpt"))
const INSTANCES = [("inst2",   joinpath(CKPT_DIR, "marker_task")),
                   ("seed_11", joinpath(CKPT_DIR, "seed_11")),
                   ("seed_22", joinpath(CKPT_DIR, "seed_22")),
                   ("seed_33", joinpath(CKPT_DIR, "seed_33")),
                   ("seed_44", joinpath(CKPT_DIR, "seed_44"))]

const N_HEADS = 4
const D_HEAD  = DIM ÷ N_HEADS
const N_PAIRS_FAM = SMOKE ? 5 : 60
const N_SWEEP     = SMOKE ? 2 : 8
const N_DELTA     = SMOKE ? 4 : 32
const SEED_SWEEP  = 4242
const SEED_DELTA  = 1618
const SEED_PROBE_A = 20260801
const SEED_PROBE_B = 20260802

head_site(l, h) = Symbol("layer_$(l)_mha_ao_h$(h)")
mlp_site(l)     = Symbol("layer_$(l)_mlp_out")
site_layer(s)   = parse(Int, match(r"^layer_(\d+)_", String(s)).captures[1])
const CANDIDATES = Symbol[]
for l in 1:N_LAYERS
    for h in 1:N_HEADS; push!(CANDIDATES, head_site(l, h)); end
    push!(CANDIDATES, mlp_site(l))
end

function run_forward!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
end
run_and_capture!(g, ns, tokens) = (out = run_forward!(g, ns, tokens); (out, NeuroDSL.capture_activations(g, ns)))
full_reset!(g, ns) = (NeuroDSL.invalidate_all!(g; namespace=ns); NeuroDSL.demand!(g, :final_logits; namespace=ns); g)

function sample_contrast(rng)
    while true
        k = rand(rng, 1:V); sk = SIGMA[k]
        rest = shuffle(rng, collect(setdiff(1:V, (k, sk))))[1:N_PAIRS-2]
        ks = vcat([k, sk], rest); vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS); push!(ctx, ks[i]); push!(ctx, vs[i]); end
        return (; tokA = vcat(ctx, [MARKER_A, sk]), tokB = vcat(ctx, [MARKER_B, sk]),
                 lit = vs[2], inv = vs[1])
    end
end
function sample_contrast_A(rng)
    while true
        k = rand(rng, 1:V); ik = SIGMA_INV[k]
        rest = shuffle(rng, collect(setdiff(1:V, (k, ik))))[1:N_PAIRS-2]
        ks = vcat([k, ik], rest); vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS); push!(ctx, ks[i]); push!(ctx, vs[i]); end
        return (; tokA = vcat(ctx, [MARKER_A, k]), tokB = vcat(ctx, [MARKER_B, k]),
                 lit = vs[1], inv = vs[2])
    end
end

sop(out, p) = Float64(out[1, p.lit] - out[1, p.inv])

function draw_clean_pair!(g, ns, rng)
    for _ in 1:200
        c = sample_contrast(rng)
        d_out = run_forward!(g, ns, c.tokA); r_out = run_forward!(g, ns, c.tokB)
        dd, dr = sop(d_out, c), sop(r_out, c)
        if SMOKE
            abs(dd - dr) > 1e-6 && return c
        else
            (argmax(vec(d_out)) == c.lit && argmax(vec(r_out)) == c.inv && dd > 0 > dr) && return c
        end
    end
    error("pas de paire propre")
end

function single_site_sweep!(g, ns; n_pairs, seed)
    rng = MersenneTwister(seed)
    sums = Dict{Symbol,Float64}(s => 0.0 for s in CANDIDATES)
    for _ in 1:n_pairs
        c = draw_clean_pair!(g, ns, rng)
        d_out, dcache = run_and_capture!(g, ns, c.tokA)
        r_out, rcache = run_and_capture!(g, ns, c.tokB)
        dd, dr = sop(d_out, c), sop(r_out, c)
        for s in CANDIDATES
            NeuroDSL.patch_node!(g, s, dcache; namespace=ns)
            out = Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
            sums[s] += (sop(out, c) - dr) / (dd - dr)
            NeuroDSL.patch_node!(g, s, rcache; namespace=ns)
            NeuroDSL.demand!(g, :final_logits; namespace=ns)
        end
        full_reset!(g, ns)
    end
    return Dict{Symbol,Float64}(s => sums[s] / n_pairs for s in CANDIDATES)
end

function collect_deltas!(g, ns, sites; n_pairs, seed)
    rng = MersenneTwister(seed)
    rows = Dict{Symbol,Vector{Vector{Float64}}}(s => Vector{Vector{Float64}}() for s in sites)
    for _ in 1:n_pairs
        c = draw_clean_pair!(g, ns, rng)
        _, dcache = run_and_capture!(g, ns, c.tokA)
        _, rcache = run_and_capture!(g, ns, c.tokB)
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

function projectors(subs::Dict{Symbol,<:AbstractMatrix})
    P = Dict{Symbol,Matrix{Float32}}()
    for (s, U) in subs
        d = size(U, 1)
        P[s] = Matrix{Float32}(I, d, d) .- Float32.(U) * Float32.(U)'
    end
    return P
end

function measure_gq!(g, ns, tokens, p, P)
    out = run_forward!(g, ns, tokens)
    gval = sop(out, p)
    proj_vals = Dict{Symbol,Any}()
    for (s, Pm) in P
        proj_vals[s] = Array(NeuroDSL.node(g, s; namespace=ns).value) * Pm
    end
    NeuroDSL.patch_nodes!(g, collect(keys(P)), proj_vals; namespace=ns)
    out_p = Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
    return gval, gval - sop(out_p, p)
end

med(v) = isempty(v) ? NaN : sort(v)[cld(length(v), 2)]
q90(v) = isempty(v) ? NaN : sort(v)[max(1, ceil(Int, 0.9*length(v)))]

function eval_config!(g, ns, name, cfgname, in_sample, subs, pairs)
    P = projectors(subs)
    e1 = Float64[]; e2 = Float64[]; sb = Float64[]; slack = Float64[]
    ident = 0.0; nhyp = 0; nconc = 0; ntot = 0
    phis = Float64[]
    for p in pairs
        gA, qA = measure_gq!(g, ns, p.tokA, p, P)
        gB, qB = measure_gq!(g, ns, p.tokB, p, P)
        qbar = (qA + qB) / 2
        Phi  = gA - gB
        PhiS = qA - qB
        Psi  = Phi - PhiS
        sbar = (gA + gB) / 2
        sSA  = gA - qA
        sSB  = gB - qB
        # (A) contrôle d'algèbre : identité exacte
        ident = max(ident,
                    abs((sSA - sbar) - (-qbar + Psi/2)),
                    abs((sSB - sbar) - (-qbar - Psi/2)))
        eps1 = abs(qbar); eps2 = abs(Psi)
        push!(e1, eps1); push!(e2, eps2); push!(sb, abs(sbar)); push!(phis, abs(Phi))
        push!(slack, abs(sbar) - (eps1 + eps2/2))
        ntot += 1
        # (B) hypothèse robuste
        (abs(sbar) > eps1 + eps2/2) && (nhyp += 1)
        # (C) conclusion effective : les deux sélecteurs ablatés du signe de sbar
        (sign(sSA) == sign(sbar) && sign(sSB) == sign(sbar)) && (nconc += 1)
    end
    full_reset!(g, ns)
    hyp = nhyp/ntot; conc = nconc/ntot
    println("  [$name/$cfgname]", in_sample ? " (IN-SAMPLE)" : " (hors éch.)",
            "  eps1(med)=", round(med(e1), digits=2),
            " eps2(med)=", round(med(e2), digits=2),
            " |sbar|(med)=", round(med(sb), digits=2),
            " |Phi|(med)=", round(med(phis), digits=2),
            " -> hypothèse ", round(100*hyp, digits=1), "%",
            "  conclusion ", round(100*conc, digits=1), "%",
            "  (identité max ", round(ident, sigdigits=2), ")")
    flush(stdout)
    return (; name, cfgname, in_sample, n = ntot,
             eps1_med = med(e1), eps2_med = med(e2), sbar_med = med(sb), phi_med = med(phis),
             eps1_q90 = q90(e1), eps2_q90 = q90(e2),
             slack_med = med(slack), slack_max = maximum(slack),
             hyp_rate = hyp, conc_rate = conc, identity_max = ident,
             # versions normalisées par le contraste total (sans échelle)
             eps1_rel = med(e1 ./ max.(phis, 1e-12)), eps2_rel = med(e2 ./ max.(phis, 1e-12)),
             sbar_rel = med(sb ./ max.(phis, 1e-12)))
end

ALL = NamedTuple[]
for (name, ckpt) in INSTANCES
    println("\n", "═"^74); println("INSTANCE $name"); flush(stdout)
    NeuroDSL.load_graph!(g, ns, ckpt; overwrite=true)
    println("  P1 : ", evaluate_marker(g, logits, ns; n_eval=200))
    MEAN_R = single_site_sweep!(g, ns; n_pairs=N_SWEEP, seed=SEED_SWEEP)
    carriers = sort([s for s in CANDIDATES if MEAN_R[s] >= 0.25]; by=s -> -MEAN_R[s])
    isempty(carriers) && continue
    main_layer = minimum(site_layer(s) for s in carriers)
    cmain = [s for s in carriers if site_layer(s) == main_layer]
    println("  Carriers : ", [(String(s), round(MEAN_R[s], digits=3)) for s in carriers])
    deltas = collect_deltas!(g, ns, carriers; n_pairs=N_DELTA, seed=SEED_DELTA)
    subs_all = Dict{Symbol,Matrix{Float64}}(
        s => format_subspace(deltas[s]; kcap = endswith(String(s), "_mlp_out") ? 8 : 4) for s in carriers)

    published = Set{Vector{Symbol}}()
    push!(published, sort([cmain[1]]; by=String))
    push!(published, sort(copy(cmain); by=String))
    length(carriers) > length(cmain) && push!(published, sort(copy(carriers); by=String))

    rngA = MersenneTwister(SEED_PROBE_A); rngB = MersenneTwister(SEED_PROBE_B)
    pairs = vcat([sample_contrast_A(rngA) for _ in 1:N_PAIRS_FAM],
                 [sample_contrast(rngB)   for _ in 1:N_PAIRS_FAM])

    nc = length(carriers)
    for mask in 1:(2^nc - 1)
        sites = [carriers[i] for i in 1:nc if (mask >> (i-1)) & 1 == 1]
        in_sample = sort(copy(sites); by=String) in published
        cfgname = join([replace(String(s), "layer_" => "L", "_mha_ao_h" => "h", "_mlp_out" => "mlp") for s in sites], "+")
        subs = Dict{Symbol,Matrix{Float64}}(s => subs_all[s] for s in sites)
        push!(ALL, eval_config!(g, ns, name, cfgname, in_sample, subs, pairs))
    end
end

println("\n", "═"^74)
println("VERDICT -- hypothèses robustes du théorème de collapse")
println("═"^74)
println("  configurations : ", length(ALL), "   paires appariées par configuration : ", ALL[1].n)
println("  (A) résidu max de l'identité s_S(x)-sbar = -qbar -/+ Psi/2 : ",
        maximum(r.identity_max for r in ALL),
        "  (attendu ~0 : contrôle d'algèbre)")
println()
tot = sum(r.n for r in ALL)
hyp_all  = sum(r.hyp_rate  * r.n for r in ALL) / tot
conc_all = sum(r.conc_rate * r.n for r in ALL) / tot
println("  (B) hypothèse robuste |sbar| > eps1 + eps2/2 : ", round(100*hyp_all, digits=2),
        "% des ", tot, " paires-configurations")
println("      configurations où elle tient sur >= 50% des paires : ",
        count(r -> r.hyp_rate >= 0.5, ALL), " / ", length(ALL))
println("      configurations où elle tient sur >= 90% des paires : ",
        count(r -> r.hyp_rate >= 0.9, ALL), " / ", length(ALL))
println("      taux max sur une configuration : ", round(100*maximum(r.hyp_rate for r in ALL), digits=1), "%")
println()
println("  (C) conclusion effective (les deux sélecteurs du signe de sbar) : ",
        round(100*conc_all, digits=2), "% des paires")
println("      configurations où elle tient sur >= 90% des paires : ",
        count(r -> r.conc_rate >= 0.9, ALL), " / ", length(ALL))
println()
println("  Échelles médianes (normalisées par le contraste total |Phi|) :")
println("      eps1/|Phi| médian = ", round(med([r.eps1_rel for r in ALL]), digits=3))
println("      eps2/|Phi| médian = ", round(med([r.eps2_rel for r in ALL]), digits=3))
println("      |sbar|/|Phi| médian = ", round(med([r.sbar_rel for r in ALL]), digits=3))
println()
println("  LECTURE : sbar est la MOYENNE d'une marge positive et d'une marge négative")
println("  sur une paire appariée -- petite par construction -- tandis que eps1 et eps2")
println("  sont de l'ordre des marges elles-mêmes. Si (B) est faible mais (C) élevé, la")
println("  condition suffisante est conservatrice ; si (B) et (C) sont tous deux faibles,")
println("  le collapse lui-même n'a pas lieu sur ces configurations.")

out = Dict{String,Any}(
    "protocol" => Dict("sweep_seed"=>SEED_SWEEP, "delta_seed"=>SEED_DELTA,
                        "probe_seed_A"=>SEED_PROBE_A, "probe_seed_B"=>SEED_PROBE_B,
                        "n_pairs"=>2*N_PAIRS_FAM),
    "configs" => [Dict(String(k)=>getfield(r, k) for k in keys(r)) for r in ALL],
)
open(joinpath(@__DIR__, "robust_collapse_results.json"), "w") do io
    JSON.print(io, out)
end
println("\nÉcrit -> notebook/robust_collapse_results.json")
