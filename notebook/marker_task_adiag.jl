# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic post-P4-bis : DIRECTION des erreurs de format A après ablation
# directionnelle jointe (config DJ du run marker_task_p4bis.log).
#
# Observation surprenante du P4-bis : toutes les ablations chirurgicales
# laissent acc_B quasi intacte (0.96-1.0, littéral_B ~0) mais dégradent acc_A
# (jusqu'à 0.79). Hypothèse révisée : le sous-espace de contraste
# donneur-receveur encode surtout "l'A-itude" (le traitement B/inversion est
# le défaut du réseau) -- le retirer fait basculer des entrées A vers le
# traitement B.
#
# Prédiction falsifiable de cette hypothèse : en format A ([m_A, k]), les
# erreurs post-DJ doivent être SYSTÉMATIQUEMENT v(sigma^{-1}(k)) (le modèle
# applique l'inversion à une clé littérale), pas du bruit. Tirages filtrés :
# sigma^{-1}(k) présent en contexte comme clé, v(k) != v(sigma^{-1}(k)).
# Baseline attendue : taux "inversé" ~0. Post-DJ attendu si hypothèse vraie :
# erreurs concentrées sur v(sigma^{-1}(k)) (correct+inversé ≈ 1).
#
# Réutilise le checkpoint notebook/marker_ckpt/marker_task (P1 = 1.0/1.0).
# Carriers de couche 1 repris du run P4-bis (log, sweep 8 paires) :
# layer_1_mha_ao_h2 (r=0.71) et layer_1_mlp_out (r=0.699).
# ══════════════════════════════════════════════════════════════════════════════

const CKPT = get(ENV, "MARKER_CKPT", joinpath(@__DIR__, "marker_ckpt", "marker_task"))
(isfile(CKPT * ".json") && isfile(CKPT * ".bin")) || error("Checkpoint introuvable : $CKPT")
ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"

include(joinpath(@__DIR__, "marker_task_experiment.jl"))
using LinearAlgebra

NeuroDSL.load_graph!(g, ns, CKPT; overwrite=true)
resP1 = evaluate_marker(g, logits, ns; n_eval=200)
println("\nP1 (checkpoint) : ", resP1)
(resP1.acc_A >= 0.95 && resP1.acc_B >= 0.95) || error("P1 non satisfait sur le checkpoint")

const N_HEADS = 4
const D_HEAD  = DIM ÷ N_HEADS
const CARRIERS_L1 = [:layer_1_mha_ao_h2, :layer_1_mlp_out]   # du log P4-bis (sweep)

function run_forward!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
end

# Paires de contraste B (pour l'estimation des deltas, comme au P4-bis).
function sample_contrast(rng)
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

# Tirages A filtrés : contexte contient k ET sigma^{-1}(k), valeurs distinctes.
function sample_contrast_A(rng)
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

# ── Sous-espaces de format (identique au P4-bis : SVD des deltas, rows 7-8) ──
function collect_deltas!(g, ns, sites; n_pairs, seed)
    rng = MersenneTwister(seed)
    deltas = Dict{Symbol,Vector{Vector{Float64}}}(s => Vector{Vector{Float64}}() for s in sites)
    for _ in 1:n_pairs
        c = sample_contrast(rng)
        run_forward!(g, ns, vcat(c.ctx, [MARKER_A, c.sk]))
        dcache = NeuroDSL.capture_activations(g, ns)
        run_forward!(g, ns, vcat(c.ctx, [MARKER_B, c.sk]))
        rcache = NeuroDSL.capture_activations(g, ns)
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
    return F.U[:, 1:k], k, ce[k]
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

DELTAS = collect_deltas!(g, ns, CARRIERS_L1; n_pairs=32, seed=1618)
SUBS = Dict{Symbol,Any}()
for s in CARRIERS_L1
    U, k, en = format_subspace(DELTAS[s]; kcap = endswith(String(s), "_mlp_out") ? 8 : 4)
    SUBS[s] = U
    println("  $s : k=$k, énergie=", round(en, digits=3))
end
DJ_W = directional_weights(CARRIERS_L1, SUBS)

# ── Évaluation directionnelle des erreurs A (et recheck B) ───────────────────
function eval_direction(g, ns; n, seed)
    rngA = MersenneTwister(seed)
    okA = 0; invA = 0; othA = 0
    for _ in 1:n
        c = sample_contrast_A(rngA)
        pred = argmax(vec(run_forward!(g, ns, vcat(c.ctx, [MARKER_A, c.k]))))
        if pred == c.vk;      okA += 1
        elseif pred == c.vik; invA += 1
        else;                 othA += 1
        end
    end
    rngB = MersenneTwister(seed + 1)
    okB = 0; litB = 0; othB = 0
    for _ in 1:n
        c = sample_contrast(rngB)
        pred = argmax(vec(run_forward!(g, ns, vcat(c.ctx, [MARKER_B, c.sk]))))
        if pred == c.vk;      okB += 1
        elseif pred == c.vsk; litB += 1
        else;                 othB += 1
        end
    end
    return (; acc_A=okA/n, invert_A=invA/n, other_A=othA/n,
             acc_B=okB/n, literal_B=litB/n, other_B=othB/n)
end

println("\n", "═"^70)
base = eval_direction(g, ns; n=300, seed=555)
println("BASELINE : acc_A=$(base.acc_A)  inversé_A=$(base.invert_A)  autre_A=$(base.other_A)")
println("           acc_B=$(base.acc_B)  littéral_B=$(base.literal_B)  autre_B=$(base.other_B)")

restore = Dict{Symbol,Matrix{Float32}}(wsym => _current_W(wsym) for wsym in keys(DJ_W))
NeuroDSL.set_params!(g, ns, DJ_W)
dj = eval_direction(g, ns; n=300, seed=555)
NeuroDSL.set_params!(g, ns, restore)
println("POST-DJ  : acc_A=$(dj.acc_A)  inversé_A=$(dj.invert_A)  autre_A=$(dj.other_A)")
println("           acc_B=$(dj.acc_B)  littéral_B=$(dj.literal_B)  autre_B=$(dj.other_B)")

err_A = 1.0 - dj.acc_A
frac_inv = err_A > 0 ? dj.invert_A / err_A : NaN
println("\nErreurs A post-DJ : ", round(err_A, digits=3),
        "  dont fraction inversée (v(sigma^{-1}(k))) : ", round(frac_inv, digits=3))
println("VERDICT DIAG : ", (err_A >= 0.05 && frac_inv >= 0.8) ?
        "erreur A SYSTÉMATIQUEMENT INVERSÉE -- hypothèse 'B par défaut / direction = A-itude' CONFIRMÉE" :
        (err_A < 0.05 ? "trop peu d'erreurs A sur la distribution filtrée pour trancher" :
         "erreurs A NON systématiquement inversées -- hypothèse non confirmée"))
