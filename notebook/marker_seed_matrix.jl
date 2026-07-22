# ══════════════════════════════════════════════════════════════════════════════
# Matrice de graines -- UNE graine par invocation (MARKER_MATRIX_SEED), pipeline
# complet : entraînement (config gagnante figée) -> gate P1 -> checkpoint ->
# sweep de carriers -> sous-espaces SVD -> configs d'ablation -> diagnostic
# miroir. Lancé en boucle séquentielle par notebook/run_seed_matrix.cmd.
#
# Graines dérivées de S = MARKER_MATRIX_SEED (documentation du choix) :
#   - init poids : Julia Random.seed!(S) ET CUDA.seed!(S) (Backend.rand32 sur
#     GPU tire de CUDA.rand, pas du RNG Julia -- vérifié src/backend.jl:16) ;
#   - données d'entraînement : seed 1000*S + 123 (MersenneTwister CPU de
#     train_marker!) ;
#   - évaluations/analyses : graines fixes (999/4242/1618/555) identiques pour
#     toutes les graines -> mêmes jeux de test, comparaison directe.
# SIGMA reste FIXE (MersenneTwister(42) dans marker_task_experiment.jl) : même
# tâche pour toutes les graines, seule l'instance de modèle varie.
#
# PHÉNOMÈNES testés par graine (robustesse inter-instances -- les SITES précis
# peuvent varier, ce sont les phénomènes qui doivent se reproduire) :
#   PH1 (redondance)  : >= 2 carriers avec r moyen >= 0.25 au sweep individuel.
#   PH2 (bas-rang)    : somme des k (sous-espaces SVD >= 90% d'énergie) des
#                       carriers de la couche porteuse <= 8.
#   PH3 (polarité par défaut / miroir) : après ablation directionnelle jointe
#                       de la couche porteuse : acc_B(filtré) >= 0.9 ET
#                       inversé_A(filtré) >= 0.8 ET pureté >= 0.9.
#                       HONNÊTETÉ DU SEUIL : 0.8 est fixé APRÈS l'observation
#                       0.86 sur l'instance 2 (marker_task_adiag.jl) -- la
#                       matrice teste la REPRODUCTIBILITÉ de cette observation ;
#                       le seuil strict 0.9 est aussi rapporté.
#   Témoin : tête de couche 4 de |r| minimal, neutre au patch ET à l'ablation
#            (|delta acc| <= 0.05 vs baseline).
# ══════════════════════════════════════════════════════════════════════════════

const SMOKE    = get(ENV, "MARKER_SMOKE", "0") == "1"
const SEED     = parse(Int, get(ENV, "MARKER_MATRIX_SEED", "11"))
const CKPT_DIR = get(ENV, "MARKER_CKPT_DIR", joinpath(@__DIR__, "marker_ckpt"))
const CKPT     = joinpath(CKPT_DIR, "seed_$(SEED)")
const HAVE_CKPT = isfile(CKPT * ".json") && isfile(CKPT * ".bin")

println("SEED $SEED : init_seed=$SEED  train_seed=$(1000*SEED + 123)  ckpt=$CKPT  (checkpoint existant : $HAVE_CKPT)")
flush(stdout)

if HAVE_CKPT
    ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
else
    ENV["MARKER_INIT_SEED"]  = string(SEED)
    ENV["MARKER_TRAIN_SEED"] = string(1000 * SEED + 123)
    ENV["MARKER_SAVE"]       = CKPT
    if SMOKE
        get!(ENV, "MARKER_STEPS", "40"); get!(ENV, "MARKER_BATCH", "8")
    end
end

include(joinpath(@__DIR__, "marker_task_experiment.jl"))
using LinearAlgebra

HAVE_CKPT && (println("Chargement du checkpoint ", CKPT); NeuroDSL.load_graph!(g, ns, CKPT; overwrite=true))

resP1 = evaluate_marker(g, logits, ns; n_eval=400)
println("P1 (seed $SEED) : acc_A = ", resP1.acc_A, "   acc_B = ", resP1.acc_B)
if !SMOKE && !(resP1.acc_A >= 0.95 && resP1.acc_B >= 0.95)
    println("RESUME_SEED seed=$SEED P1_ECHEC accA=$(resP1.acc_A) accB=$(resP1.acc_B) -- graine abandonnée (info de robustesse en soi)")
    exit(0)
end
flush(stdout)

# ── Helpers (repris de marker_task_p4bis.jl / marker_task_adiag.jl, validés) ──
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

function run_and_capture!(g, ns, tokens)
    out = run_forward!(g, ns, tokens)
    return out, NeuroDSL.capture_activations(g, ns)
end

function full_reset!(g, ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, :final_logits; namespace=ns)
    return g
end

function sample_contrast(rng)     # B filtré : k ET sigma(k) en contexte, valeurs distinctes
    while true
        k = rand(rng, 1:V); sk = SIGMA[k]
        others = collect(setdiff(1:V, (k, sk)))
        rest = shuffle(rng, others)[1:N_PAIRS-2]
        ks = vcat([k, sk], rest); vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS); push!(ctx, ks[i]); push!(ctx, vs[i]); end
        return (; ctx, k, sk, vk = vs[1], vsk = vs[2])
    end
end

function sample_contrast_A(rng)   # A filtré : k ET sigma^{-1}(k) en contexte, valeurs distinctes
    while true
        k = rand(rng, 1:V); ik = SIGMA_INV[k]
        others = collect(setdiff(1:V, (k, ik)))
        rest = shuffle(rng, others)[1:N_PAIRS-2]
        ks = vcat([k, ik], rest); vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS); push!(ctx, ks[i]); push!(ctx, vs[i]); end
        return (; ctx, k, ik, vk = vs[1], vik = vs[2])
    end
end

donor_tokens(c)    = vcat(c.ctx, [MARKER_A, c.sk])
receiver_tokens(c) = vcat(c.ctx, [MARKER_B, c.sk])
delta_logit(out, c) = Float64(out[1, c.vsk] - out[1, c.vk])

function draw_clean_pair!(g, ns, rng; max_tries=200)
    for _ in 1:max_tries
        c = sample_contrast(rng)
        d_out = run_forward!(g, ns, donor_tokens(c))
        r_out = run_forward!(g, ns, receiver_tokens(c))
        dd, dr = delta_logit(d_out, c), delta_logit(r_out, c)
        if SMOKE
            abs(dd - dr) > 1e-6 && return c
        else
            (argmax(vec(d_out)) == c.vsk && argmax(vec(r_out)) == c.vk && dd > 0 > dr) && return c
        end
    end
    error("Pas de paire de contraste exploitable")
end

function single_site_sweep!(g, ns; n_pairs, seed)
    rng = MersenneTwister(seed)
    sums = Dict{Symbol,Float64}(s => 0.0 for s in CANDIDATES)
    for _ in 1:n_pairs
        c = draw_clean_pair!(g, ns, rng)
        d_out, dcache = run_and_capture!(g, ns, donor_tokens(c))
        r_out, rcache = run_and_capture!(g, ns, receiver_tokens(c))
        dd, dr = delta_logit(d_out, c), delta_logit(r_out, c)
        for s in CANDIDATES
            NeuroDSL.patch_node!(g, s, dcache; namespace=ns)
            out = Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
            sums[s] += (delta_logit(out, c) - dr) / (dd - dr)
            NeuroDSL.patch_node!(g, s, rcache; namespace=ns)
            NeuroDSL.demand!(g, :final_logits; namespace=ns)
        end
        full_reset!(g, ns)
    end
    return Dict{Symbol,Float64}(s => sums[s] / n_pairs for s in CANDIDATES)
end

function collect_deltas!(g, ns, sites; n_pairs, seed)
    rng = MersenneTwister(seed)
    deltas = Dict{Symbol,Vector{Vector{Float64}}}(s => Vector{Vector{Float64}}() for s in sites)
    for _ in 1:n_pairs
        c = draw_clean_pair!(g, ns, rng)
        _, dcache = run_and_capture!(g, ns, donor_tokens(c))
        _, rcache = run_and_capture!(g, ns, receiver_tokens(c))
        for s in sites
            da = Array(dcache[s]); ra = Array(rcache[s])
            for row in (SEQ_LEN - 1, SEQ_LEN)
                push!(deltas[s], vec(Float64.(da[row, :] .- ra[row, :])))
            end
        end
    end
    return deltas
end

function format_subspace(delta_vecs; energy=0.90, kcap::Int)
    X = reduce(hcat, delta_vecs); F = svd(X)
    e = F.S .^ 2; ce = cumsum(e) ./ sum(e)
    k = min(something(findfirst(>=(energy), ce), length(ce)), kcap)
    align = sum(norm(F.U[:, 1:k]' * v) / (norm(v) + 1e-12) for v in delta_vecs) / length(delta_vecs)
    return F.U[:, 1:k], k, ce[k], align
end

_current_W(wsym) = Matrix{Float32}(Array(NeuroDSL.node(g, wsym; namespace=ns).value))

function directional_weights(sites, subspaces)
    w = Dict{Symbol,Matrix{Float32}}()
    for s in sites
        U = Float32.(subspaces[s])
        m = match(r"^layer_(\d+)_mha_ao_h(\d+)$", String(s))
        if m !== nothing
            l = parse(Int, m.captures[1]); h = parse(Int, m.captures[2])
            wsym = Symbol("layer_$(l)_mha_output_W")
            W = get!(() -> _current_W(wsym), w, wsym)
            cols = (h-1)*D_HEAD+1 : h*D_HEAD
            W[:, cols] = W[:, cols] * (Matrix{Float32}(I, D_HEAD, D_HEAD) .- U * U')
        else
            wsym = Symbol("layer_$(match(r"^layer_(\d+)_mlp_out$", String(s)).captures[1])_mlp_w2")
            w[wsym] = (Matrix{Float32}(I, DIM, DIM) .- U * U') * _current_W(wsym)
        end
    end
    return w
end

function zero_head_weights(site)
    m = match(r"^layer_(\d+)_mha_ao_h(\d+)$", String(site))
    l = parse(Int, m.captures[1]); h = parse(Int, m.captures[2])
    wsym = Symbol("layer_$(l)_mha_output_W")
    W = _current_W(wsym)
    W[:, (h-1)*D_HEAD+1 : h*D_HEAD] .= 0f0
    return Dict{Symbol,Matrix{Float32}}(wsym => W)
end

# acc_A non filtrée + A filtré (correct/inversé/autre) + B filtré (correct/littéral/autre)
function eval_full(g, ns; n, seed)
    rng0 = MersenneTwister(seed)
    okAu = 0
    for _ in 1:n
        tokens, _, _, _, v = sample_marker_sequence(rng0, :A)
        okAu += argmax(vec(run_forward!(g, ns, tokens))) == v
    end
    rngA = MersenneTwister(seed + 1)
    okA = 0; invA = 0
    for _ in 1:n
        c = sample_contrast_A(rngA)
        pred = argmax(vec(run_forward!(g, ns, vcat(c.ctx, [MARKER_A, c.k]))))
        okA += pred == c.vk; invA += pred == c.vik
    end
    rngB = MersenneTwister(seed + 2)
    okB = 0; litB = 0
    for _ in 1:n
        c = sample_contrast(rngB)
        pred = argmax(vec(run_forward!(g, ns, receiver_tokens(c))))
        okB += pred == c.vk; litB += pred == c.vsk
    end
    return (; acc_A_u = okAu/n, acc_A = okA/n, inv_A = invA/n,
             acc_B = okB/n, lit_B = litB/n)
end

const N_EVAL = SMOKE ? 30 : 300
function run_config!(name, desc; weights=nothing)
    r = if weights !== nothing
        restore = Dict{Symbol,Matrix{Float32}}(wsym => _current_W(wsym) for wsym in keys(weights))
        NeuroDSL.set_params!(g, ns, weights)
        rr = eval_full(g, ns; n=N_EVAL, seed=555)
        NeuroDSL.set_params!(g, ns, restore)
        rr
    else
        eval_full(g, ns; n=N_EVAL, seed=555)
    end
    full_reset!(g, ns)
    println("  [$name] $desc")
    println("        acc_A(u)=$(r.acc_A_u)  A_filt: correct=$(r.acc_A) inversé=$(r.inv_A)  B_filt: correct=$(r.acc_B) littéral=$(r.lit_B)")
    flush(stdout)
    return r
end

# ── Pipeline d'analyse ───────────────────────────────────────────────────────
println("\nSweep individuel (8 paires) :")
MEAN_R = single_site_sweep!(g, ns; n_pairs=(SMOKE ? 2 : 8), seed=4242)
for l in 1:N_LAYERS
    println("  layer_$l : ", [(String(s), round(MEAN_R[s], digits=3))
                              for s in vcat([head_site(l,h) for h in 1:N_HEADS], [mlp_site(l)])])
end

CARRIERS = sort([s for s in CANDIDATES if MEAN_R[s] >= 0.25]; by=s -> -MEAN_R[s])
if isempty(CARRIERS)
    println("RESUME_SEED seed=$SEED AUCUN_CARRIER (max r = $(round(maximum(values(MEAN_R)), digits=3))) -- info de robustesse")
    exit(0)
end
MAIN_LAYER    = minimum(site_layer(s) for s in CARRIERS)
CARRIERS_MAIN = [s for s in CARRIERS if site_layer(s) == MAIN_LAYER]
MAX_R_SINGLE  = maximum(MEAN_R[s] for s in CARRIERS)
println("  Carriers : ", [(String(s), round(MEAN_R[s], digits=3)) for s in CARRIERS],
        " | couche porteuse : $MAIN_LAYER -> ", CARRIERS_MAIN)

CONTROL_SITE = let l4 = [head_site(4, h) for h in 1:N_HEADS if !(head_site(4, h) in CARRIERS)]
    l4[argmin([abs(MEAN_R[s]) for s in l4])]
end
println("  Témoin : $CONTROL_SITE (r patch = ", round(MEAN_R[CONTROL_SITE], digits=4), ")")

DELTAS = collect_deltas!(g, ns, CARRIERS; n_pairs=(SMOKE ? 4 : 32), seed=1618)
SUBS = Dict{Symbol,Any}(); KS = Dict{Symbol,Int}()
for s in CARRIERS
    U, k, en, align = format_subspace(DELTAS[s]; kcap = endswith(String(s), "_mlp_out") ? 8 : 4)
    SUBS[s] = U; KS[s] = k
    println("  $s : k=$k énergie=", round(en, digits=3), " alignement=", round(align, digits=3))
end
K_TOTAL_MAIN = sum(KS[s] for s in CARRIERS_MAIN)
flush(stdout)

println("\nConfigs :")
base = run_config!("B",  "baseline")
ctrl = run_config!("C",  "témoin zero-ablation $CONTROL_SITE"; weights=zero_head_weights(CONTROL_SITE))
d1   = run_config!("D1", "directionnelle carrier principal $(CARRIERS_MAIN[1])";
                   weights=directional_weights([CARRIERS_MAIN[1]], SUBS))
dj   = run_config!("DJ", "directionnelle jointe couche $MAIN_LAYER $(CARRIERS_MAIN)";
                   weights=directional_weights(CARRIERS_MAIN, SUBS))
dja  = length(CARRIERS) > length(CARRIERS_MAIN) ?
       run_config!("DJA", "directionnelle jointe tous carriers $(CARRIERS)";
                   weights=directional_weights(CARRIERS, SUBS)) : nothing

# ── Verdicts par phénomène ───────────────────────────────────────────────────
ctrl_neutral = abs(ctrl.acc_A_u - base.acc_A_u) <= 0.05 && abs(ctrl.acc_B - base.acc_B) <= 0.05
best = dja === nothing ? dj : (dja.inv_A > dj.inv_A ? dja : dj)
best_name = best === dj ? "DJ" : "DJA"
purity = best.acc_A < 1.0 ? best.inv_A / (1.0 - best.acc_A) : NaN

PH1 = length(CARRIERS) >= 2
PH2 = K_TOTAL_MAIN <= 8
PH3 = best.acc_B >= 0.9 && best.inv_A >= 0.8 && (isnan(purity) ? false : purity >= 0.9)
PH3_STRICT = PH3 && best.inv_A >= 0.9

println("\nRESUME_SEED seed=$SEED P1=($(resP1.acc_A),$(resP1.acc_B)) ",
        "carriers=", join(String.(CARRIERS), "+"), " r=", join([round(MEAN_R[s], digits=2) for s in CARRIERS], "/"),
        " max_r=$(round(MAX_R_SINGLE, digits=3)) couche=$MAIN_LAYER k_total=$K_TOTAL_MAIN ",
        "témoin=$CONTROL_SITE:", ctrl_neutral ? "neutre" : "NON-NEUTRE",
        " $best_name: accA_filt=$(best.acc_A) invA=$(best.inv_A) pureté=$(round(purity, digits=3)) accB_filt=$(best.acc_B)",
        " PH1=", PH1 ? "OUI" : "NON", " PH2=", PH2 ? "OUI" : "NON",
        " PH3=", PH3 ? "OUI" : "NON", " PH3_strict(inv>=0.9)=", PH3_STRICT ? "OUI" : "NON")
flush(stdout)
