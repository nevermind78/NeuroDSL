# ══════════════════════════════════════════════════════════════════════════════
# DÉCOMPOSITION MÊME-BLOC / INTER-COUCHES DE LA MAGNITUDE D'INTERACTION
# (artilce/code4.tex, §6 -- nouvelle Proposition prop:multilayer -- et §7,
# extension du test hors-échantillon) -- aucun réentraînement, checkpoints
# existants uniquement (notebook/marker_ckpt/).
#
# CE QUI EST TESTÉ
# ----------------
# Le Théorème 6 (interaction) et la Proposition readout ne couvrent QUE la
# composition à deux carriers d'UN SEUL bloc (une tête et le MLP de sa propre
# couche), et Proposition readout elle-même exige que CE bloc alimente
# directement la lecture (pas de couche non linéaire après lui) -- vrai
# seulement pour la DERNIÈRE couche touchée dans un réseau multi-couches.
# Mais la "magnitude d'interaction" mesurée en §7 sur les 39 configurations
# (verify_config_band_oos.jl) est calculée pour N'IMPORTE QUEL sous-ensemble
# S de carriers, y compris ceux qui s'étendent sur PLUSIEURS couches (ex.
# seed_44 : L1h3+L4mlp). Pour ces configurations, l'écart mesuré mélange
# plusieurs mécanismes :
#   (a) le "Delta effectif" MÊME-BLOC à chaque couche touchée -- identiquement
#       nul si cette couche n'ablate que son MLP (quelle que soit sa
#       position) ; couvert en forme close par la Proposition readout
#       SEULEMENT si cette couche alimente directement la lecture (la
#       dernière couche touchée, dans les checkpoints de ce dépôt) ; sinon
#       une quantité bien définie mais PAS encore donnée en forme close --
#       exactement ce que la Remarque "Propagation across layers" de
#       artilce/code4.tex laissait déjà ouvert ;
#   (b) une PROPAGATION INTER-COUCHES (ablate la couche l change ce qui
#       arrive à la couche l' > l) qu'AUCUN théorème actuel ne couvre.
#
# Ce script prouve puis mesure une IDENTITÉ EXACTE (pas une approximation)
# qui sépare les deux :
#
#   s_S^true(x) - s_S(x)  =  Sum_l [ s_{S_l}^{(l),true}(x) - s_{S_l}(x) ]  +  R_x(x)
#                            \______________ Delta_l effectif, même-bloc ____/  \_ reste _/
#
# où s_{S_l}^{(l),true}(x) est le sélecteur du réseau où SEULE la couche l
# (restreinte à S_l = la partie de S portée par cette couche) est réellement
# éditée en poids -- toutes les autres couches, y compris les autres éléments
# de S à d'autres couches, restent à leurs poids PROPRES (non édités) --
# c'est une VRAIE passe avant sur tout le réseau restant, donc chaque terme
# mesure le Delta_l effectivement propagé jusqu'à la lecture, PAS seulement
# sa valeur brute au niveau du bloc. R_x(x) := ce qui reste -- PAS borné
# analytiquement ici (cela demanderait de borner la courbure de l'attention à
# travers plusieurs couches, hors-scope, voir la discussion de
# artilce/code4.tex) -- mesuré empiriquement, comme le Delta_l effectif
# lui-même dès que sa couche n'alimente pas directement la lecture.
#
# L'identité elle-même est vérifiée en (A) (résidu machine ~0, contrôle
# d'algèbre) ; ce qui est vraiment mesuré est la TAILLE relative de chaque
# terme sur les configurations qui s'étendent sur >= 2 couches parmi les 39
# déjà cataloguées par verify_config_band_oos.jl (mêmes sous-espaces, mêmes
# graines de sweep/delta -- seules les sondes sont nouvelles et disjointes de
# tous les autres scripts).
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
const N_PAIRS_FAM = SMOKE ? 4 : 40
const N_SWEEP     = SMOKE ? 2 : 8
const N_DELTA     = SMOKE ? 4 : 32
const SEED_SWEEP  = 4242
const SEED_DELTA  = 1618
const SEED_PROBE_A = 20260803
const SEED_PROBE_B = 20260804

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

_current_W(g, ns, wsym) = Matrix{Float32}(Array(NeuroDSL.node(g, wsym; namespace=ns).value))

function projectors(subs::Dict{Symbol,<:AbstractMatrix})
    P = Dict{Symbol,Matrix{Float32}}()
    for (s, U) in subs
        d = size(U, 1)
        P[s] = Matrix{Float32}(I, d, d) .- Float32.(U) * Float32.(U)'
    end
    return P
end

function directional_weights(g, ns, subs::Dict{Symbol,<:AbstractMatrix})
    w = Dict{Symbol,Matrix{Float32}}()
    for (s, U) in subs
        Uf = Float32.(U)
        m = match(r"^layer_(\d+)_mha_ao_h(\d+)$", String(s))
        if m !== nothing
            l = parse(Int, m.captures[1]); h = parse(Int, m.captures[2])
            wsym = Symbol("layer_$(l)_mha_output_W")
            W = get!(() -> _current_W(g, ns, wsym), w, wsym)
            cols = (h-1)*D_HEAD+1 : h*D_HEAD
            W[:, cols] = W[:, cols] * (Matrix{Float32}(I, D_HEAD, D_HEAD) .- Uf * Uf')
        else
            wsym = Symbol("layer_$(match(r"^layer_(\d+)_mlp_out$", String(s)).captures[1])_mlp_w2")
            w[wsym] = (Matrix{Float32}(I, DIM, DIM) .- Uf * Uf') * _current_W(g, ns, wsym)
        end
    end
    return w
end

# q(x) idéalisé pour un sous-ensemble de sites (gèle CHAQUE site à sa propre
# activation PROPRE -- calculée sur le run PROPRE, jamais recalculée à partir
# d'un autre site ablaté -- exactement Eq.(ablation), généralisée à n'importe
# quel regroupement de sites).
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

# Sélecteur du réseau où SEULEMENT `subs_l` (une seule couche) est édité en
# poids -- toutes les autres couches restent à leurs poids d'origine (même si
# d'autres carriers de S vivent ailleurs). Restaure les poids avant de rendre
# la main -- aucune trace laissée dans le graphe.
function isolated_true_selector!(g, ns, tokens, p, subs_l::Dict{Symbol,<:AbstractMatrix})
    w = directional_weights(g, ns, subs_l)
    restore = Dict{Symbol,Matrix{Float32}}(k => _current_W(g, ns, k) for k in keys(w))
    NeuroDSL.set_params!(g, ns, w)
    out = run_forward!(g, ns, tokens)
    val = sop(out, p)
    NeuroDSL.set_params!(g, ns, restore); full_reset!(g, ns)
    return val
end

# Sélecteur idéalisé (frozen-activation) d'une config restreinte à UNE seule
# couche `l`, sur `tokens` -- c'est exactement s_{S_l}(x) de l'identité.
function isolated_idealized_selector!(g, ns, tokens, p, P_l)
    gval, q_l = measure_gq!(g, ns, tokens, p, P_l)
    return gval - q_l
end

# Construit, pour la couche `l`, le dictionnaire de projecteurs à utiliser
# pour la version IDÉALISÉE isolée s_{S_l}(x). PIÈGE ÉVITÉ ICI (trouvé en
# relisant ce script après un premier essai) : `measure_gq!` ne gèle QUE les
# sites explicitement présents dans son dictionnaire `P` -- tout site absent
# est laissé au graphe, qui le recalcule NATURELLEMENT après le patch. Si
# S_l = {tête seule} (le MLP de la même couche n'est PAS dans S_l), ce
# recalcul naturel du MLP à partir du résidu déjà ablaté par la tête produit
# EXACTEMENT la sortie du réseau VRAIMENT édité (Proposition patchedit,
# généralisée à plusieurs sites simultanés qui n'interagissent pas par une
# non-linéarité partagée) -- PAS la prédiction idéalisée que le Théorème 6
# borne. La définition de s_{S_l}(x) (Eq.(ablation)) exige que le MLP
# survivant de la même couche reste évalué sur le résidu PROPRE même si S_l
# ne le touche pas -- il faut donc l'épingler explicitement (projecteur
# identité, "ne rien retirer") pour empêcher ce recalcul naturel, EXACTEMENT
# comme le fait déjà `measure_gq!` pour n'importe quel site présent dans `P`.
# Par la Corollaire d'exactitude, cet épinglage n'est nécessaire (et n'a
# d'effet) que si S_l contient une tête ; si S_l est seulement le MLP,
# Delta_l est nul par construction et rien n'a besoin d'être épinglé.
function idealized_projectors_pinned(g, ns, subs_l::Dict{Symbol,<:AbstractMatrix}, l::Int)
    P_l = projectors(subs_l)
    has_head = any(occursin("_mha_ao_h", String(s)) for s in keys(subs_l))
    mlp_sym = mlp_site(l)
    if has_head && !haskey(subs_l, mlp_sym)
        P_l[mlp_sym] = Matrix{Float32}(I, DIM, DIM)  # épingle au propre, ne retire rien
    end
    return P_l
end

# Décompose une paire (tokens, p) unique pour une config multi-couches S =
# union des S_l. Retourne (; g, s_full_ideal, s_full_true, same_block_sum, Rx)
# où :
#   g              = sop(x) sur le réseau propre (clean)
#   s_full_ideal   = s_S(x)      (idéalisé, S entier gelé simultanément)
#   s_full_true    = s_S^true(x) (édition jointe de S entier -- même quantité
#                    que verify_config_band_oos.jl)
#   same_block_sum = Sum_{l in L(S)} [s_{S_l}^{(l),true}(x) - s_{S_l}(x)]
#                    (chaque terme = Delta_l effectif, propagé à la lecture ;
#                    nul si S_l n'ablate que le MLP de sa couche, quelle que
#                    soit sa position ; couvert EXACTEMENT par Proposition
#                    readout seulement si cette couche alimente directement
#                    la lecture -- sinon une quantité mesurée ici pour la
#                    première fois, pas encore donnée en forme close)
#   Rx             = (s_full_true - s_full_ideal) - same_block_sum
#                    (résidu inter-couches, PAS borné analytiquement)
# L'identité s_full_true - s_full_ideal = same_block_sum + Rx est vérifiée à
# la précision machine par construction (voir le résidu d'algèbre imprimé
# plus bas) -- ce n'est pas ce qui est "testé", c'est ce qui rend la
# décomposition possible.
function decompose_pair!(g, ns, tokens, p, subs::Dict{Symbol,<:AbstractMatrix}, P_full, subs_by_layer, P_by_layer_pinned)
    gval, q_full = measure_gq!(g, ns, tokens, p, P_full)
    s_ideal = gval - q_full

    # s_S^true(x) : édition JOINTE de tout S (même quantité que
    # verify_config_band_oos.jl -- réutilise directional_weights sur `subs`
    # entier, restaure ensuite).
    s_true = isolated_true_selector!(g, ns, tokens, p, subs)

    same_block_sum = 0.0
    for l in keys(subs_by_layer)
        s_l_ideal = isolated_idealized_selector!(g, ns, tokens, p, P_by_layer_pinned[l])
        s_l_true  = isolated_true_selector!(g, ns, tokens, p, subs_by_layer[l])
        same_block_sum += s_l_true - s_l_ideal
    end

    total = s_true - s_ideal
    Rx = total - same_block_sum
    return (; g = gval, s_ideal, s_true, total, same_block_sum, Rx)
end

function decompose_config!(g, ns, name, cfgname, subs::Dict{Symbol,<:AbstractMatrix}, pairsA, pairsB)
    layers = sort(unique(site_layer(s) for s in keys(subs)))
    length(layers) < 2 && return nothing  # rien à décomposer, Rx≡0 trivialement
    subs_by_layer = Dict(l => Dict(s => U for (s, U) in subs if site_layer(s) == l) for l in layers)
    P_full = projectors(subs)
    P_by_layer_pinned = Dict(l => idealized_projectors_pinned(g, ns, subs_by_layer[l], l) for l in layers)

    rows = NamedTuple[]
    for (tok_field, plist) in ((:tokA, pairsA), (:tokB, pairsB))
        for p in plist
            tokens = getfield(p, tok_field)
            push!(rows, decompose_pair!(g, ns, tokens, p, subs, P_full, subs_by_layer, P_by_layer_pinned))
        end
    end

    ident_resid = maximum(abs(r.total - (r.same_block_sum + r.Rx)) for r in rows)  # ~0 par construction
    tot_abs = [abs(r.total) for r in rows]
    sb_abs  = [abs(r.same_block_sum) for r in rows]
    rx_abs  = [abs(r.Rx) for r in rows]
    med(v) = isempty(v) ? NaN : sort(v)[cld(length(v), 2)]
    sb_frac = med([abs(r.same_block_sum) / max(abs(r.total), 1e-9) for r in rows])
    rx_frac = med([abs(r.Rx) / max(abs(r.total), 1e-9) for r in rows])
    println("  [$name/$cfgname] couches=", layers, "  |total| med=", round(med(tot_abs), digits=3),
            "  |same-bloc| med=", round(med(sb_abs), digits=3), " (frac med=", round(sb_frac, digits=3), ")",
            "  |R_croisé| med=", round(med(rx_abs), digits=3), " (frac med=", round(rx_frac, digits=3), ")",
            "  résidu identité=", round(ident_resid, sigdigits=3))
    flush(stdout)
    return (; name, cfgname, layers, n = length(rows),
             total_med = med(tot_abs), same_block_med = med(sb_abs), rx_med = med(rx_abs),
             same_block_frac_med = sb_frac, rx_frac_med = rx_frac, identity_residual_max = ident_resid,
             totals = tot_abs, same_blocks = sb_abs, rxs = rx_abs)
end

# ── Boucle principale : mêmes carriers/sous-espaces que verify_config_band_oos.jl,
# décomposition appliquée aux configs à >= 2 couches uniquement ────────────────
ALL = NamedTuple[]
for (name, ckpt) in INSTANCES
    println("\n", "═"^74); println("INSTANCE $name"); flush(stdout)
    NeuroDSL.load_graph!(g, ns, ckpt; overwrite=true)
    println("  P1 : ", evaluate_marker(g, logits, ns; n_eval=200))
    MEAN_R = single_site_sweep!(g, ns; n_pairs=N_SWEEP, seed=SEED_SWEEP)
    carriers = sort([s for s in CANDIDATES if MEAN_R[s] >= 0.25]; by=s -> -MEAN_R[s])
    isempty(carriers) && continue
    deltas = collect_deltas!(g, ns, carriers; n_pairs=N_DELTA, seed=SEED_DELTA)
    subs_all = Dict{Symbol,Matrix{Float64}}(
        s => format_subspace(deltas[s]; kcap = endswith(String(s), "_mlp_out") ? 8 : 4) for s in carriers)

    rngA = MersenneTwister(SEED_PROBE_A); rngB = MersenneTwister(SEED_PROBE_B)
    pairsA = [sample_contrast_A(rngA) for _ in 1:N_PAIRS_FAM]
    pairsB = [sample_contrast(rngB)   for _ in 1:N_PAIRS_FAM]

    nc = length(carriers)
    for mask in 1:(2^nc - 1)
        sites = [carriers[i] for i in 1:nc if (mask >> (i-1)) & 1 == 1]
        length(unique(site_layer(s) for s in sites)) < 2 && continue  # même-couche : Rx≡0 trivial, rien à mesurer
        cfgname = join([replace(String(s), "layer_" => "L", "_mha_ao_h" => "h", "_mlp_out" => "mlp") for s in sites], "+")
        subs = Dict{Symbol,Matrix{Float64}}(s => subs_all[s] for s in sites)
        r = decompose_config!(g, ns, name, cfgname, subs, pairsA, pairsB)
        r !== nothing && push!(ALL, r)
    end
end

println("\n", "═"^74)
println("VERDICT -- décomposition même-bloc / inter-couches")
println("═"^74)
println("  configurations multi-couches décomposées : ", length(ALL))
println("  résidu d'algèbre max (identité exacte, attendu ~0) : ",
         isempty(ALL) ? NaN : maximum(r.identity_residual_max for r in ALL))
if !isempty(ALL)
    all_sb_frac = [r.same_block_frac_med for r in ALL]
    all_rx_frac = [r.rx_frac_med for r in ALL]
    med(v) = sort(v)[cld(length(v), 2)]
    println("  fraction même-bloc (Théorème 6, médiane des médianes)     : ", round(med(all_sb_frac), digits=3))
    println("  fraction inter-couches (résidu non couvert, idem)         : ", round(med(all_rx_frac), digits=3))
    println("  configs où le même-bloc domine (frac_med > 0.5)           : ",
            count(>(0.5), all_sb_frac), " / ", length(ALL))
    println("  configs où l'inter-couches domine (frac_med > 0.5)        : ",
            count(>(0.5), all_rx_frac), " / ", length(ALL))
end

out = Dict{String,Any}(
    "protocol" => Dict("sweep_seed"=>SEED_SWEEP, "delta_seed"=>SEED_DELTA,
                        "probe_seed_A"=>SEED_PROBE_A, "probe_seed_B"=>SEED_PROBE_B,
                        "n_pairs_per_family"=>N_PAIRS_FAM),
    "configs" => [Dict(String(k) => getfield(r, k) for k in keys(r)) for r in ALL],
)
open(joinpath(@__DIR__, "crosslayer_decomposition_results.json"), "w") do io
    JSON.print(io, out)
end
println("\nÉcrit -> notebook/crosslayer_decomposition_results.json")
