# Reprise du test de suffisance pour la SEULE instance V=16 (seed_9160, dim=128)
# -- séparée du script principal (marker_task_restricted_loss.jl) car celui-ci
# inclut marker_task_experiment.jl avec les défauts V=8/dim=64 ; charger le
# checkpoint V=16/dim=128 sans fixer MARKER_V/MARKER_DIM AVANT l'include
# corrompt la sémantique d'analyse (vocabulaire/sigma faux) même si
# load_graph! recharge les bonnes formes de poids -- bug trouvé et documenté
# (voir RESUME_RESTRICTED manquant pour seed_9160_V16 dans le run principal :
# la garde P1 interne l'a exclu automatiquement, aucune donnée corrompue
# n'a fuité dans les résultats, mais ce point restait incomplet).
ENV["MARKER_V"] = "16"; ENV["MARKER_DIM"] = "128"
ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))
using LinearAlgebra

const CKPT_DIR = joinpath(@__DIR__, "marker_ckpt")
const N_HEADS = 4
const D_HEAD  = DIM ÷ N_HEADS

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
function sample_contrast_A(rng)
    while true
        k = rand(rng, 1:V); ik = SIGMA_INV[k]
        rest = shuffle(rng, collect(setdiff(1:V, (k, ik))))[1:N_PAIRS-2]
        ks = vcat([k, ik], rest); vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS); push!(ctx, ks[i]); push!(ctx, vs[i]); end
        return (; tokA = vcat(ctx, [MARKER_A, k]), ok = vs[1], inv = vs[2])
    end
end
function draw_clean_pair!(g, ns, rng)
    for _ in 1:200
        c = sample_contrast(rng)
        d_out = run_forward!(g, ns, c.tokA); r_out = run_forward!(g, ns, c.tokB)
        sop(o) = Float64(o[1, c.lit] - o[1, c.inv])
        dd, dr = sop(d_out), sop(r_out)
        (argmax(vec(d_out)) == c.lit && argmax(vec(r_out)) == c.inv && dd > 0 > dr) && return c
    end
    error("pas de paire propre")
end
function collect_deltas!(g, ns, sites; n_pairs, seed)
    rng = MersenneTwister(seed)
    rows = Dict{Symbol,Vector{Vector{Float64}}}(s => Vector{Vector{Float64}}() for s in sites)
    for _ in 1:n_pairs
        c = draw_clean_pair!(g, ns, rng)
        run_forward!(g, ns, c.tokA); dcache = NeuroDSL.capture_activations(g, ns)
        run_forward!(g, ns, c.tokB); rcache = NeuroDSL.capture_activations(g, ns)
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
    return F.U[:, 1:k], k, ce[k]
end
function single_site_sweep!(g, ns; n_pairs, seed)
    rng = MersenneTwister(seed)
    sums = Dict{Symbol,Float64}(s => 0.0 for s in CANDIDATES)
    for _ in 1:n_pairs
        c = draw_clean_pair!(g, ns, rng)
        d_out = run_forward!(g, ns, c.tokA); dcache = NeuroDSL.capture_activations(g, ns)
        r_out = run_forward!(g, ns, c.tokB); rcache = NeuroDSL.capture_activations(g, ns)
        sop(o) = Float64(o[1, c.lit] - o[1, c.inv])
        dd, dr = sop(d_out), sop(r_out)
        for s in CANDIDATES
            NeuroDSL.patch_node!(g, s, dcache; namespace=ns)
            out = Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
            sums[s] += (sop(out) - dr) / (dd - dr)
            NeuroDSL.patch_node!(g, s, rcache; namespace=ns)
            NeuroDSL.demand!(g, :final_logits; namespace=ns)
        end
        NeuroDSL.invalidate_all!(g; namespace=ns); NeuroDSL.demand!(g, :final_logits; namespace=ns)
    end
    return Dict{Symbol,Float64}(s => sums[s] / n_pairs for s in CANDIDATES)
end
_current_W(g, ns, wsym) = Matrix{Float32}(Array(NeuroDSL.node(g, wsym; namespace=ns).value))
function subspace_weights(g, ns, sites, subs::Dict; keep::Bool)
    w = Dict{Symbol,Matrix{Float32}}()
    for s in sites
        U = Float32.(subs[s])
        m = match(r"^layer_(\d+)_mha_ao_h(\d+)$", String(s))
        if m !== nothing
            l = parse(Int, m.captures[1]); h = parse(Int, m.captures[2])
            wsym = Symbol("layer_$(l)_mha_output_W")
            W = get!(() -> _current_W(g, ns, wsym), w, wsym)
            cols = (h-1)*D_HEAD+1 : h*D_HEAD
            P = U * U'
            Proj = keep ? P : (Matrix{Float32}(I, D_HEAD, D_HEAD) .- P)
            W[:, cols] = W[:, cols] * Proj
        else
            wsym = Symbol("layer_$(match(r"^layer_(\d+)_mlp_out$", String(s)).captures[1])_mlp_w2")
            P = U * U'
            Proj = keep ? P : (Matrix{Float32}(I, DIM, DIM) .- P)
            w[wsym] = Proj * _current_W(g, ns, wsym)
        end
    end
    return w
end
function eval_condition(g, ns; n, seed)
    rngAu = MersenneTwister(seed); okAu = 0
    for _ in 1:n
        tokens, _, _, _, v = sample_marker_sequence(rngAu)
        okAu += argmax(vec(run_forward!(g, ns, tokens))) == v
    end
    rngA = MersenneTwister(seed + 1); okA = 0; invA = 0
    for _ in 1:n
        c = sample_contrast_A(rngA)
        pred = argmax(vec(run_forward!(g, ns, c.tokA)))
        okA += pred == c.ok; invA += pred == c.inv
    end
    rngB = MersenneTwister(seed + 2); okB = 0; litB = 0
    for _ in 1:n
        c = sample_contrast(rngB)
        pred = argmax(vec(run_forward!(g, ns, c.tokB)))
        okB += pred == c.inv; litB += pred == c.lit
    end
    return (; acc_A_u = okAu / n, acc_A = okA / n, inv_A = invA / n, acc_B = okB / n, lit_B = litB / n)
end

const N_EVAL = 300
name, ckpt = "seed_9160_V16", joinpath(CKPT_DIR, "seed_9160")
println("═"^70); println("INSTANCE $name  ($ckpt)  [V=16, dim=128 -- config corrigée]"); flush(stdout)
NeuroDSL.load_graph!(g, ns, ckpt; overwrite=true)
r1 = evaluate_marker(g, logits, ns; n_eval=200)
println("  P1 : ", r1)
@assert r1.acc_A >= 0.95 && r1.acc_B >= 0.95 "P1 toujours en échec -- vérifier la config"

MEAN_R = single_site_sweep!(g, ns; n_pairs=8, seed=4242)
carriers = sort([s for s in CANDIDATES if MEAN_R[s] >= 0.25]; by=s -> -MEAN_R[s])
println("  Carriers : ", [(String(s), round(MEAN_R[s], digits=3)) for s in carriers])
deltas = collect_deltas!(g, ns, carriers; n_pairs=32, seed=1618)
subs = Dict{Symbol,Any}(); ks = Dict{Symbol,Int}()
for s in carriers
    U, k, en = format_subspace(deltas[s]; kcap = endswith(String(s), "_mlp_out") ? 8 : 4)
    subs[s] = U; ks[s] = k
    println("    $s : k=$k énergie=", round(en, digits=3))
end
base = eval_condition(g, ns; n=N_EVAL, seed=555)
println("  [baseline] acc_Au=$(round(base.acc_A_u,digits=3))  A: ok=$(round(base.acc_A,digits=3)) inv=$(round(base.inv_A,digits=3))  B: ok=$(round(base.acc_B,digits=3)) lit=$(round(base.lit_B,digits=3))")
for (tag, keep) in [("necessity(remove)", false), ("sufficiency(keep_only)", true)]
    w = subspace_weights(g, ns, carriers, subs; keep=keep)
    restore = Dict{Symbol,Matrix{Float32}}(k => _current_W(g, ns, k) for k in keys(w))
    NeuroDSL.set_params!(g, ns, w)
    r = eval_condition(g, ns; n=N_EVAL, seed=555)
    NeuroDSL.set_params!(g, ns, restore)
    NeuroDSL.invalidate_all!(g; namespace=ns); NeuroDSL.demand!(g, :final_logits; namespace=ns)
    println("  [all_$tag] acc_Au=$(round(r.acc_A_u,digits=3))  A: ok=$(round(r.acc_A,digits=3)) inv=$(round(r.inv_A,digits=3))  B: ok=$(round(r.acc_B,digits=3)) lit=$(round(r.lit_B,digits=3))")
    if tag == "necessity(remove)"
        global NEC = r
    else
        global SUFF = r
    end
end
println("RESUME_RESTRICTED inst=$name carriers=", join(String.(carriers), "+"),
        " k_total=", sum(values(ks)), " suff_all_accAu=", round(SUFF.acc_A_u, digits=4),
        " suff_all_accB=", round(SUFF.acc_B, digits=4), " nec_all_accAu=", round(NEC.acc_A_u, digits=4))
