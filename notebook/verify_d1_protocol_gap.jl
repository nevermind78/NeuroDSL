# ══════════════════════════════════════════════════════════════════════════════
# TEST DE L'HYPOTHÈSE "ERREUR D'ESTIMATION" (artilce/code4.tex, §7, paragraphe
# "The interaction term") -- mesure réelle, aucun réentraînement.
#
# CE QUI EST TESTÉ
# ----------------
# Le §7 de code4.tex affirme (i) que sur les 5 configurations mono-carrier (D1)
# la "magnitude d'interaction" -- l'écart entre la route ACTIVATIONS GELÉES et
# la route ÉDITION DE POIDS -- vaut ~0, et (ii) que le résidu C1a < 1 sur ces
# mêmes configurations "est cohérent avec cela : les directions des carriers,
# et donc les valeurs patchées, sont estimées à partir des données plutôt que
# lues sur les poids édités, donc la coïncidence n'est exacte qu'à cette erreur
# d'estimation près".
#
# (ii) est une HYPOTHÈSE non testée sur la numérique du pipeline. Ce script la
# teste directement, sur des sondes NEUVES (graines distinctes de tout run
# existant), en mesurant par entrée :
#
#        gap(x) = | s_op(x | activations D1 projetées)  -  s_op(x | poids D1 édités) |
#
# où s_op(x) = logit_{lecture littérale} - logit_{lecture inversée} en position
# finale -- EXACTEMENT la quantité dont marker_conj1_verify.jl rapporte le max
# sous le nom `interaction_max`, qui est la magnitude d'interaction portée en
# abscisse de fig:interaction. On rapporte ici médiane ET max (le run d'origine
# ne rapportait que le max), plus l'échelle |s_op| pour situer le gap.
#
# DÉCOMPOSITION DU RÉSIDU C1a
# ---------------------------
# On recalcule aussi C1a sur ces sondes neuves, en séparant les deux causes
# possibles de désaccord :
#   - "autre"     : l'argmax post-ablation tombe sur un 3e token, hors des deux
#                   lectures candidates {littérale, inversée} -- la prédiction
#                   binaire signe(s_op) ne peut structurellement pas le prévoir ;
#   - "signe"     : l'argmax EST l'une des deux lectures mais pas celle prédite
#                   -- c'est la seule catégorie qu'une erreur d'estimation ou un
#                   écart patch/poids pourrait produire.
# Si le résidu C1a est intégralement du "autre" et que "signe" = 0, l'hypothèse
# d'erreur d'estimation est réfutée : il n'y a aucun désaccord à lui imputer.
#
# CONTRÔLE DE FALSIFIABILITÉ (le test ne peut pas être vide)
# ---------------------------------------------------------
# Pour montrer qu'un écart d'estimation SERAIT visible s'il opérait, on rejoue
# la même mesure avec un sous-espace RE-ESTIMÉ sur d'autres données (graine de
# deltas 2718 au lieu de 1618) pour la seule route "activations gelées", la
# route poids gardant le sous-espace d'origine. C'est littéralement la situation
# que l'hypothèse décrit ("patch et édition ne coïncident qu'à l'erreur
# d'estimation près"). Si ce gap de contrôle est grand alors que le gap réel est
# à la précision machine, c'est que dans le vrai pipeline les DEUX routes
# partagent le même U estimé et que l'erreur d'estimation s'annule exactement.
#
# Carriers et sous-espaces reconstruits à l'identique du pipeline qui a produit
# les nombres de l'article (marker_conj1_verify.jl : sweep graine 4242, seuil
# 0.25, deltas graine 1618, kcap 4/8, D1 = carrier de plus fort r dans la couche
# porteuse) -- déterministe à poids fixés.
#
# Graines des sondes de CE script (distinctes de 90210/31337 de conj1) :
#   famille A : 20260726   famille B : 20260727   (100 paires par famille)
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
for (nm, ck) in INSTANCES
    (isfile(ck * ".json") && isfile(ck * ".bin")) || error("Checkpoint manquant : $ck")
end

const N_HEADS = 4
const D_HEAD  = DIM ÷ N_HEADS
const N_PAIRS_FAM = SMOKE ? 5 : 100
const N_SWEEP     = SMOKE ? 2 : 8
const N_DELTA     = SMOKE ? 4 : 32
const SEED_SWEEP  = 4242      # identique au pipeline d'origine
const SEED_DELTA  = 1618      # identique au pipeline d'origine
const SEED_DELTA_CONTROL = 2718   # NEUF -- uniquement pour le contrôle de falsifiabilité
const SEED_PROBE_A = 20260726     # NEUF -- distinct de conj1 (90210)
const SEED_PROBE_B = 20260727     # NEUF -- distinct de conj1 (31337)

head_site(l, h) = Symbol("layer_$(l)_mha_ao_h$(h)")
mlp_site(l)     = Symbol("layer_$(l)_mlp_out")
site_layer(s)   = parse(Int, match(r"^layer_(\d+)_", String(s)).captures[1])
is_mlp(s)       = endswith(String(s), "_mlp_out")
const CANDIDATES = Symbol[]
for l in 1:N_LAYERS
    for h in 1:N_HEADS; push!(CANDIDATES, head_site(l, h)); end
    push!(CANDIDATES, mlp_site(l))
end

# ── Helpers repris à l'identique de marker_conj1_verify.jl ───────────────────
function run_forward!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
end
run_and_capture!(g, ns, tokens) = (out = run_forward!(g, ns, tokens); (out, NeuroDSL.capture_activations(g, ns)))
full_reset!(g, ns) = (NeuroDSL.invalidate_all!(g; namespace=ns); NeuroDSL.demand!(g, :final_logits; namespace=ns); g)

function sample_contrast(rng)      # B-filtrée : t = sigma(k)
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
function sample_contrast_A(rng)    # A-filtrée : t = k
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
category(out, p) = (a = argmax(vec(out)); a == p.lit ? :lit : (a == p.inv ? :inv : :other))

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

_current_W(g, ns, wsym) = Matrix{Float32}(Array(NeuroDSL.node(g, wsym; namespace=ns).value))

projector(U) = (d = size(U, 1); Matrix{Float32}(I, d, d) .- Float32.(U) * Float32.(U)')

function directional_weight(g, ns, s::Symbol, U)
    Uf = Float32.(U)
    m = match(r"^layer_(\d+)_mha_ao_h(\d+)$", String(s))
    if m !== nothing
        l = parse(Int, m.captures[1]); h = parse(Int, m.captures[2])
        wsym = Symbol("layer_$(l)_mha_output_W")
        W = _current_W(g, ns, wsym)
        cols = (h-1)*D_HEAD+1 : h*D_HEAD
        W[:, cols] = W[:, cols] * (Matrix{Float32}(I, D_HEAD, D_HEAD) .- Uf * Uf')
        return Dict{Symbol,Matrix{Float32}}(wsym => W)
    else
        l = match(r"^layer_(\d+)_mlp_out$", String(s)).captures[1]
        wsym = Symbol("layer_$(l)_mlp_w2")
        return Dict{Symbol,Matrix{Float32}}(wsym => (Matrix{Float32}(I, DIM, DIM) .- Uf * Uf') * _current_W(g, ns, wsym))
    end
end

# Route ACTIVATIONS GELÉES : patch du seul nœud D1 par sa projection, le reste
# du réseau recalcule naturellement.
function sop_patched!(g, ns, tokens, p, s::Symbol, Pm)
    run_forward!(g, ns, tokens)
    v = Array(NeuroDSL.node(g, s; namespace=ns).value) * Pm
    NeuroDSL.patch_nodes!(g, [s], Dict(s => v); namespace=ns)
    return sop(Array(NeuroDSL.demand!(g, :final_logits; namespace=ns)), p)
end

med(v) = isempty(v) ? NaN : sort(v)[cld(length(v), 2)]

function measure_instance(g, ns, name, ckpt)
    println("\n", "═"^74); println("INSTANCE $name  ($ckpt)"); flush(stdout)
    NeuroDSL.load_graph!(g, ns, ckpt; overwrite=true)
    p1 = evaluate_marker(g, logits, ns; n_eval=200)
    println("  P1 : ", p1)

    # ── carriers + D1, reconstruits à l'identique du pipeline d'origine ──────
    MEAN_R = single_site_sweep!(g, ns; n_pairs=N_SWEEP, seed=SEED_SWEEP)
    carriers = sort([s for s in CANDIDATES if MEAN_R[s] >= 0.25]; by=s -> -MEAN_R[s])
    isempty(carriers) && (println("  AUCUN CARRIER -- sautée"); return nothing)
    main_layer = minimum(site_layer(s) for s in carriers)
    cmain = [s for s in carriers if site_layer(s) == main_layer]
    d1 = cmain[1]
    println("  Carriers : ", [(String(s), round(MEAN_R[s], digits=3)) for s in carriers])
    println("  D1 = $d1  (", is_mlp(d1) ? "MLP" : "tête", ", r=", round(MEAN_R[d1], digits=3), ")")

    kcap = is_mlp(d1) ? 8 : 4
    U  = format_subspace(collect_deltas!(g, ns, [d1]; n_pairs=N_DELTA, seed=SEED_DELTA)[d1]; kcap=kcap)
    Uc = format_subspace(collect_deltas!(g, ns, [d1]; n_pairs=N_DELTA, seed=SEED_DELTA_CONTROL)[d1]; kcap=kcap)
    println("  k=", size(U, 2), "  (contrôle : k'=", size(Uc, 2),
            ", écart de sous-espaces ||UU' - U'U''||_F = ",
            round(norm(U*U' - Uc*Uc'), sigdigits=4), ")")
    Pm  = projector(U)
    Pmc = projector(Uc)

    # ── sondes NEUVES ────────────────────────────────────────────────────────
    rngA = MersenneTwister(SEED_PROBE_A); rngB = MersenneTwister(SEED_PROBE_B)
    pairsA = [sample_contrast_A(rngA) for _ in 1:N_PAIRS_FAM]
    pairsB = [sample_contrast(rngB)   for _ in 1:N_PAIRS_FAM]

    # route activations gelées (poids intacts) : deux sous-espaces
    frozen = NamedTuple[]
    for (fam, plist) in ((:A, pairsA), (:B, pairsB))
        for p in plist
            push!(frozen, (; fam, p,
                            sA  = sop_patched!(g, ns, p.tokA, p, d1, Pm),
                            sB  = sop_patched!(g, ns, p.tokB, p, d1, Pm),
                            cA  = sop_patched!(g, ns, p.tokA, p, d1, Pmc),
                            cB  = sop_patched!(g, ns, p.tokB, p, d1, Pmc)))
        end
    end
    full_reset!(g, ns)

    # route édition de poids (sous-espace d'origine) : une seule édition, puis
    # une passe fraîche par sonde
    w = directional_weight(g, ns, d1, U)
    restore = Dict{Symbol,Matrix{Float32}}(k => _current_W(g, ns, k) for k in keys(w))
    NeuroDSL.set_params!(g, ns, w)
    gaps = Float64[]; gaps_ctrl = Float64[]; sop_scale = Float64[]
    agree = 0; nother = 0; nsign = 0; total = 0
    obs = NamedTuple[]
    for m in frozen
        outA = run_forward!(g, ns, m.p.tokA); outB = run_forward!(g, ns, m.p.tokB)
        tA, tB = sop(outA, m.p), sop(outB, m.p)
        push!(gaps, abs(m.sA - tA), abs(m.sB - tB))
        push!(gaps_ctrl, abs(m.cA - tA), abs(m.cB - tB))
        push!(sop_scale, abs(tA), abs(tB))
        cA, cB = category(outA, m.p), category(outB, m.p)
        push!(obs, (; fam=m.fam, cA, cB))
        for (sv, cv) in ((m.sA, cA), (m.sB, cB))
            pred = sv > 0 ? :lit : :inv
            total += 1
            if cv == :other
                nother += 1
            elseif pred == cv
                agree += 1
            else
                nsign += 1
            end
        end
    end
    NeuroDSL.set_params!(g, ns, restore); full_reset!(g, ns)

    c1a = agree / total
    println("  gap D1  |s_op(gelé) - s_op(poids édités)|  sur $(length(gaps)) sondes neuves :")
    println("      médiane = ", med(gaps), "   max = ", maximum(gaps),
            "   (échelle : médiane |s_op| édité = ", round(med(sop_scale), digits=3), ")")
    println("  CONTRÔLE (sous-espace ré-estimé graine $SEED_DELTA_CONTROL côté patch seulement) :")
    println("      médiane = ", round(med(gaps_ctrl), digits=5), "   max = ", round(maximum(gaps_ctrl), digits=5))
    println("  C1a (sondes neuves) = ", round(c1a, digits=4),
            "   désaccords : 'autre' = ", nother, "/", total, " (", round(nother/total, digits=4), ")",
            "   'signe' = ", nsign, "/", total, " (", round(nsign/total, digits=4), ")")
    println("      -> résidu 1-C1a = ", round(1-c1a, digits=4),
            " ; imputable au protocole patch/poids ('signe') : ", nsign == 0 ? "AUCUN" : "$nsign entrée(s)")
    flush(stdout)

    return (; name, d1=String(d1), d1_kind = is_mlp(d1) ? "mlp" : "head", k=size(U,2),
             r_d1 = MEAN_R[d1], n_probes=length(gaps),
             gap_median=med(gaps), gap_max=maximum(gaps), gap_mean=sum(gaps)/length(gaps),
             sop_scale_median=med(sop_scale),
             ctrl_gap_median=med(gaps_ctrl), ctrl_gap_max=maximum(gaps_ctrl),
             ctrl_subspace_dist=norm(U*U' - Uc*Uc'),
             c1a, other_rate=nother/total, sign_rate=nsign/total,
             n_other=nother, n_sign=nsign, n_total=total)
end

ALL = NamedTuple[]
for (name, ckpt) in INSTANCES
    r = measure_instance(g, ns, name, ckpt)
    r !== nothing && push!(ALL, r)
end

println("\n", "═"^74)
println("SYNTHÈSE -- écart de protocole sur les 5 configurations D1")
println("═"^74)
println(rpad("instance",10), rpad("carrier D1",22), rpad("type",6), rpad("k",3),
        rpad("gap med",11), rpad("gap max",11), rpad("|s_op| med",11),
        rpad("C1a",8), rpad("autre",8), "signe")
for r in ALL
    println(rpad(r.name,10), rpad(r.d1,22), rpad(r.d1_kind,6), rpad(string(r.k),3),
            rpad(string(round(r.gap_median, sigdigits=3)),11),
            rpad(string(round(r.gap_max, sigdigits=3)),11),
            rpad(string(round(r.sop_scale_median, digits=2)),11),
            rpad(string(round(r.c1a, digits=4)),8),
            rpad(string(round(r.other_rate, digits=4)),8),
            r.n_sign)
end
println()
println("  gap max sur les 5 instances            : ", maximum(r.gap_max for r in ALL))
println("  gap max du CONTRÔLE (U ré-estimé)      : ", round(maximum(r.ctrl_gap_max for r in ALL), digits=4))
println("  désaccords de type 'signe' (total)     : ", sum(r.n_sign for r in ALL), " / ", sum(r.n_total for r in ALL))
println("  désaccords de type 'autre'  (total)    : ", sum(r.n_other for r in ALL), " / ", sum(r.n_total for r in ALL))
println()
println("  LECTURE : si gap max ~ précision Float32 ET 'signe' = 0 partout, le résidu")
println("  C1a<1 des configs D1 n'est PAS imputable à une erreur d'estimation des")
println("  directions (les deux routes partagent le même U estimé, l'erreur s'annule")
println("  exactement) ; il est intégralement dû aux réponses 'autre', hors des deux")
println("  lectures candidates, que la prédiction binaire ne peut structurellement pas")
println("  couvrir. Le contrôle borne ce qu'un VRAI désaccord d'estimation produirait.")

out = Dict{String,Any}(
    "protocol" => Dict("sweep_seed"=>SEED_SWEEP, "delta_seed"=>SEED_DELTA,
                        "control_delta_seed"=>SEED_DELTA_CONTROL,
                        "probe_seed_A"=>SEED_PROBE_A, "probe_seed_B"=>SEED_PROBE_B,
                        "n_pairs_per_family"=>N_PAIRS_FAM,
                        "quantity"=>"abs(s_op frozen-activation patch - s_op weight-edited fresh forward)"),
    "per_instance" => [Dict(String(k)=>getfield(r, k) for k in keys(r)) for r in ALL],
)
open(joinpath(@__DIR__, "d1_protocol_gap_results.json"), "w") do io
    JSON.print(io, out)
end
println("\nÉcrit -> notebook/d1_protocol_gap_results.json")
