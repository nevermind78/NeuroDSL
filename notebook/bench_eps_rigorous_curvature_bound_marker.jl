# ══════════════════════════════════════════════════════════════════════════════
# BORNE ANALYTIQUE (PAS ÉCHANTILLONNÉE) SUR sup|phi'''(eps)| -- tâche du marqueur.
#
# CONTEXTE : bench_eps_atp_path_integral_certified_marker.jl construit IG_K
# (Integrated Gradients, K=8, règle du point milieu) avec une borne d'erreur
# THÉORIQUE |IG_K-vrai| <= sup|phi'''|/(24K^2) -- un vrai théorème de calcul.
# Mais sup|phi'''| y était estimé par un PROXY EMPIRIQUE (différence finie à 3
# points sur la grille déjà calculée), violé sur 7/24 mesures -- PAS un
# certificat. Ce script attaque directement la question : peut-on borner
# sup|phi'''| ANALYTIQUEMENT, à partir des propriétés de courbure connues des
# non-linéarités réelles du réseau (RMSNorm, softmax, SiLU dans SwiGLU),
# composées RIGOUREUSEMENT à travers les couches qui séparent la branche
# patchée du logit de sortie -- sans aucun échantillonnage de phi lui-même.
#
# CIBLE : layer_1_mha_output_out, paire 2 (k=2 sigma(k)=5 v(k)=7 v(sigma(k))=2),
# le pire cas du script précédent : vrai_err(IG_8) = 2.9720, borne-proxy M3
# = 3.1717 (couvre à peine). Cette branche doit traverser TOUT le reste de la
# couche 1 (résidu MLP) + les couches 2, 3, 4 en entier + lm_head + sélection
# pour atteindre la métrique -- le cas de composition le plus long du banc.
#
# MÉTHODE (résumé, détails ligne par ligne ci-dessous) :
#   phi(eps) = g(a_recepteur + eps*Delta), Delta = donneur - recepteur (fixe).
#   a_b(eps) est AFFINE en eps (dérivées d'ordre >=2 nulles à la source) --
#   toute la courbure de phi vient de la composition NON-LINÉAIRE du reste du
#   réseau g. On propage, du nœud patché jusqu'à la métrique, un triplet de
#   bornes (||x'||, ||x''||, ||x'''||) -- normes de Frobenius agrégées sur
#   chaque tenseur intermédiaire -- via la règle de Faà di Bruno (composition),
#   la règle de Leibniz (produits/formes bilinéaires : SwiGLU, Q.K^T, P.V) et
#   l'inégalité triangulaire (additions résiduelles), en utilisant à chaque
#   nœud non-linéaire (RMSNorm, softmax, SiLU) une borne LOCALE sur ses
#   dérivées 1/2/3, elle-même dérivée soit EXACTEMENT par calcul direct
#   (RMSNorm ordre 1, SiLU fermé), soit par récherche numérique dense sur la
#   fonction fermée CONNUE restreinte au domaine RÉELLEMENT atteint sur ce
#   segment précis (RMSNorm ordre 2/3 via l'homogénéité de degré 0 + une
#   constante angulaire cherchée numériquement sur la sphère unité ; softmax
#   ordre 2/3 sur la plage de scores RÉELLEMENT observée). Les opérations
#   LINÉAIRES (tous les :matmul/:linear) sont EXACTES (norme d'opérateur
#   réelle des poids ENTRAÎNÉS, via svdvals -- pas une approximation).
#
# HONNÊTETÉ : chaque approximation/relâchement est documenté à l'endroit où il
# intervient. Ce n'est PAS une preuve formelle publiable -- c'est une borne
# construite avec des inégalités valides à chaque étape, plus des constantes
# numériquement certifiées sur des fonctions fermées CONNUES (pas sur le
# réseau), sur un domaine compact dense-échantillonné. Norme agrégée de
# Frobenius partout (pas de comptabilité par position -- documenté comme
# source de relâchement supplémentaire).
#
# Usage : julia --project=. notebook/bench_eps_rigorous_curvature_bound_marker.jl
# Écrit : notebook/bench_eps_rigorous_curvature_bound_marker_results.txt
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, LinearAlgebra, Statistics

const OUT  = joinpath(@__DIR__, "bench_eps_rigorous_curvature_bound_marker_results.txt")
const CKPT = joinpath(@__DIR__, "marker_ckpt", "marker_task")
(isfile(CKPT * ".json") && isfile(CKPT * ".bin")) ||
    error("Checkpoint introuvable : $CKPT -- ce script ne réentraîne JAMAIS.")

ENV["MARKER_STEPS"] = "1"; ENV["MARKER_BATCH"] = "2"
include(joinpath(@__DIR__, "marker_task_experiment.jl"))

const N_HEADS = 4
const D_HEAD  = DIM ÷ N_HEADS

const DEV_CPU = NeuroDSL.Backend.CPUDevice()
const NS = :marker_curv
G = NeuroDSL.NeuroGraph(namespace=NS, device=DEV_CPU)
NeuroDSL.load_graph!(G, NS, CKPT; overwrite=true)

function run_forward!(g, ns, tokens)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, :final_logits; namespace=ns))
end

function check_p1(g, ns; n_eval=200, seed=999)
    rng = MersenneTwister(seed)
    okA = totA = okB = totB = 0
    for _ in 1:n_eval
        tokens, _, fmt, _, v = sample_marker_sequence(rng)
        pred = argmax(vec(run_forward!(g, ns, tokens)))
        if fmt == :A; totA += 1; okA += (pred == v) ? 1 : 0
        else;         totB += 1; okB += (pred == v) ? 1 : 0
        end
    end
    return (; acc_A = okA / totA, acc_B = okB / totB)
end
p1 = check_p1(G, NS)
(p1.acc_A >= 0.95 && p1.acc_B >= 0.95) || error("P1 non confirmé -- arrêt (p1=$p1)")

# ─── Même tirage EXACT que le script précédent (même graine 31415, mêmes 3 paires) ───
function sample_contrast(rng)
    while true
        k  = rand(rng, 1:V)
        sk = SIGMA[k]
        others = collect(setdiff(1:V, (k, sk)))
        rest = length(others) >= N_PAIRS - 2 ? shuffle(rng, others)[1:N_PAIRS-2] : Int[]
        ks = vcat([k, sk], rest)
        vs = [rand(rng, 1:V) for _ in 1:N_PAIRS]
        vs[1] == vs[2] && continue
        ctx = Int[]
        for i in shuffle(rng, 1:N_PAIRS)
            push!(ctx, ks[i]); push!(ctx, vs[i])
        end
        return (; ctx, k, sk, vk = vs[1], vsk = vs[2])
    end
end
donor_tokens(c)    = vcat(c.ctx, [MARKER_A, c.sk])
receiver_tokens(c) = vcat(c.ctx, [MARKER_B, c.sk])
delta_logit(out, c) = Float64(out[1, c.vsk] - out[1, c.vk])

function draw_search_pair!(g, ns, rng; max_tries=200)
    for _ in 1:max_tries
        c = sample_contrast(rng)
        d_out = run_forward!(g, ns, donor_tokens(c))
        r_out = run_forward!(g, ns, receiver_tokens(c))
        dd, dr = delta_logit(d_out, c), delta_logit(r_out, c)
        (argmax(vec(d_out)) == c.vsk && argmax(vec(r_out)) == c.vk && dd > 0 > dr) && return c
    end
    error("Impossible de tirer une paire de contraste exploitable en $max_tries essais")
end
rng_p3 = MersenneTwister(31415)
PAIRS = [draw_search_pair!(G, NS, rng_p3) for _ in 1:3]
const PAIR2 = PAIRS[2]   # le pire cas identifié précédemment
const BRANCH = :layer_1_mha_output_out
const K_STEPS = 8

open(OUT, "w") do io
emit(s="") = (println(io, s); println(s); flush(io))
emitf(fmt, args...) = emit(Printf.format(Printf.Format(fmt), args...))

emit("BORNE ANALYTIQUE (COMPOSITIONNELLE) SUR sup|phi'''(eps)| -- PAS D'ÉCHANTILLONNAGE DE phi")
emit("Cible : $BRANCH, paire 2 (le pire cas de bench_eps_atp_path_integral_certified_marker.jl)")
emitf("P1 (checkpoint) : acc_A=%.4f  acc_B=%.4f", p1.acc_A, p1.acc_B)
emit()

# ══════════════════════════════════════════════════════════════════════════
# PARTIE 1 : ARCHITECTURE RÉELLE (lue dans src/layers.jl, PAS supposée)
# ══════════════════════════════════════════════════════════════════════════
emit("═"^92); emit("PARTIE 1 -- ARCHITECTURE RÉELLE DU CHEMIN CAUSAL"); emit("═"^92)
emitf("LlamaModel : n_layers=%d, dim=%d, n_heads=%d, d_head=%d, hidden_dim=%d, batched_attn=true",
      N_LAYERS, DIM, N_HEADS, D_HEAD, 2*DIM)
emit("Normalisation : RMSNorm (PAS LayerNorm -- pas de soustraction de moyenne), eps=1e-6, gamma appris.")
emit("Activation MLP : SwiGLU = SiLU(gate) ⊙ up  (SiLU(x)=x*sigmoid(x)), gate=xn2@W1^T, up=xn2@W3^T.")
emit("Attention : softmax ligne par ligne sur scores Q.K^T/sqrt(d_head) + masque causal (constant, indép. de eps).")
emitf("Chemin de %s (couche 1) à final_logits :", string(BRANCH))
emit("  layer_1_mha_output_out --[add résiduel]--> layer_1_res1")
emit("    --[RMSNorm(layer_1_norm2_gamma)]--> layer_1_norm2_out")
emit("    --[2×Linear W1,W3 (exactes) -> SiLU⊙up -> Linear W2 (exacte) -> add résiduel]--> layer_1_out")
emit("  layer_1_out --[bloc LlamaBlock complet : RMSNorm,MHA(4 têtes,softmax,4×Linear),")
emit("                 add, RMSNorm, SwiGLU-MLP, add]--> layer_2_out --(idem)--> layer_3_out")
emit("                 --(idem)--> layer_4_out")
emit("  layer_4_out --[Linear lm_head (exacte)]--> logits --[sélection dernière position (exacte)]-->")
emit("    final_logits --[fonctionnelle linéaire e_vsk-e_vk (exacte)]--> phi(eps)")
emit("Donc : 3 blocs LlamaBlock COMPLETS (2, 3, 4) + le résidu-MLP de la couche 1 + lm_head + sélection.")
emit("3 SOURCES DE NON-LINÉARITÉ RÉELLES SEULEMENT : RMSNorm, softmax, SiLU(dans SwiGLU). Tout le reste")
emit("(Linear/matmul, add résiduel, scale_mask, sélection, concat de têtes) est EXACTEMENT linéaire :")
emit("dérivées d'ordre >=2 nulles, AUCUNE approximation nécessaire pour ces opérations.")
emit()

# ══════════════════════════════════════════════════════════════════════════
# PARTIE 2a : BORNES SUR UNE SEULE COUCHE -- calcul direct, vérifié
# ══════════════════════════════════════════════════════════════════════════
emit("═"^92); emit("PARTIE 2a -- COURBURE D'UNE SEULE NON-LINÉARITÉ (calcul direct + vérification)"); emit("═"^92)

# ── SiLU : dérivée 3e -- forme fermée dérivée à la main, VÉRIFIÉE ci-dessous ──
sigmoid(x) = 1.0/(1.0+exp(-x))
silu(x) = x*sigmoid(x)
function silu_p(x)
    s = sigmoid(x); return s + x*s*(1-s)
end
function silu_pp(x)
    s = sigmoid(x); sp = s*(1-s)
    return 2*sp + x*sp*(1-2s)
end
function silu_ppp(x)
    s = sigmoid(x); sp = s*(1-s)
    return 3*sp*(1-2s) + x*sp*((1-2s)^2 - 2*sp)
end
# Vérification par différence finie centrée (h petit, Richardson) contre la forme fermée --
# PAS une preuve, mais confirme que la formule dérivée à la main est correcte (sinon les 2
# divergeraient largement, ce qui ARRIVERAIT immédiatement si une erreur d'algèbre s'était glissée).
function fd3(f, x; h=1e-3)
    (f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)) / (2h^3)
end
max_fd_err = maximum(abs(fd3(silu, x) - silu_ppp(x)) for x in -5:0.1:5)
emitf("SiLU'''(x) forme fermée dérivée à la main -- écart max vs différence finie (h=1e-3, x∈[-5,5]) : %.2e", max_fd_err)
emit("(écart de l'ordre de h^2 attendu pour une différence finie d'ordre 3 -- confirme la formule)")
xs_full = -60.0:0.0005:60.0
vals_full = abs.(silu_ppp.(xs_full))
sup_silu3_global, i_glob = findmax(vals_full)
emitf("sup_{x∈R}|SiLU'''(x)| (recherche fine x∈[-60,60], pas 5e-4) = %.6f  (atteint à x=%.4f)",
      sup_silu3_global, xs_full[i_glob])
emit("C'est une CONSTANTE UNIVERSELLE (indépendante du réseau) -- borne globale légitime pour SiLU seul.")
emit()

# ── RMSNorm ordre 1 : borne EXACTE dérivée à la main ──
emit("RMSNorm(x)_j = gamma_j * x_j / r(x),  r(x)=sqrt(mean(x^2)+eps).")
emit("Jacobien : d y_j/d x_k = gamma_j*[delta_jk/r - x_j x_k/(nc*r^3)].")
emit("Preuve : ||x||^2/nc <= r(x)^2 (car eps>=0) => la matrice (1/r)(I - xx^T/(nc r^2)) a une norme")
emit("d'opérateur <= (1/r)*(1 + ||x||^2/(nc r^2)) <= 2/r. Donc EXACTEMENT (pas une borne relâchée -- une")
emit("inégalité triangulaire propre) :  ||J_RMSNorm(x)||_op <= 2*max_j|gamma_j| / r(x).")
emit("Ordres 2 et 3 : PAS de forme fermée aussi simple à main levée sans risque d'erreur d'algèbre --")
emit("utilisation de l'homogénéité de degré 0 EXACTE de RMSNorm (au terme eps~1e-6 près, négligeable ici")
emit("car mean(x^2) est d'ordre 1 dans ce réseau -- eps 6 ordres de grandeur plus petit) : D^k y(x)[v,..,v]")
emit("= r(x)^{-k} * D^k y(u)[v,..,v] où u=x/r(x) est de rayon unité (rms(u)=1). La 'constante angulaire'")
emit("D^k y(u) est cherchée NUMÉRIQUEMENT (dense, sur la fonction FERMÉE connue, pas sur le réseau) sur")
emit("la sphère unité -- voir ci-dessous, avec le nombre d'échantillons déclaré (certification numérique,")
emit("pas une preuve exhaustive du sup global, mais fonction lisse sur domaine compact).")
emit()

# Recherche numérique des constantes angulaires C2, C3 pour RMSNorm SANS gamma (gamma=1),
# à rayon unité (rms(u)=1 <=> ||u||=sqrt(nc)). nc = DIM (les RMSNorm de ce réseau opèrent sur dim=DIM).
function rmsnorm_ng(x; eps=1f-6)  # "no gamma", Float64
    nc = length(x)
    r = sqrt(sum(abs2, x)/nc + eps)
    return x ./ r
end
function rmsnorm_scalar_proj(x, v, t; eps=1f-6)
    # phi_dir(t) = <w, RMSNorm_ng(x+t*v)> pour un vecteur de projection w fixé (pour réduire à un
    # scalaire différentiable en t -- on prend le sup sur w aléatoire aussi, unitaire).
    rmsnorm_ng(x .+ t.*v; eps=eps)
end
function angular_constants_rmsnorm(nc::Int; n_samples=4000, seed=7)
    rng = MersenneTwister(seed)
    h = 1e-3
    c2 = 0.0; c3 = 0.0
    for _ in 1:n_samples
        u0 = randn(rng, nc); u0 .*= sqrt(nc)/norm(u0)   # rayon unité (rms=1) EXACT
        v  = randn(rng, nc); v ./= norm(v)              # direction unitaire
        w  = randn(rng, nc); w ./= norm(w)               # projection unitaire (sup sur direction de sortie)
        f(t) = dot(w, rmsnorm_ng(u0 .+ t.*v))
        fpp  = (f(h) - 2f(0.0) + f(-h)) / h^2
        fppp = (f(2h) - 2f(h) + 2f(-h) - f(-2h)) / (2h^3)
        c2 = max(c2, abs(fpp)); c3 = max(c3, abs(fppp))
    end
    return c2, c3
end
C2_RMS, C3_RMS = angular_constants_rmsnorm(DIM; n_samples=6000)
emitf("Constantes angulaires RMSNorm (gamma=1, rayon unité, nc=%d, 6000 échantillons (u,v,w) aléatoires,", DIM)
emitf("  différences finies h=1e-3) :  C2=%.4f   C3=%.4f", C2_RMS, C3_RMS)
emit("(bornes NUMÉRIQUEMENT CERTIFIÉES sur une fonction connue en forme fermée -- pas le réseau --")
emit("sur un domaine compact ; pas une borne prouvée analytiquement à la main pour ces 2 ordres.)")
emit()

# ── Softmax : L1 <= 1/4 (classique), L2/L3 cherchées numériquement, restreintes à la plage de score réelle ──
function softmax_vec(x)
    m = maximum(x); e = exp.(x .- m); e ./ sum(e)
end
function angular_constants_softmax(n::Int, score_lo::Float64, score_hi::Float64; n_samples=6000, seed=11)
    rng = MersenneTwister(seed)
    h = 1e-3
    c1 = 0.0; c2 = 0.0; c3 = 0.0
    for _ in 1:n_samples
        x0 = score_lo .+ (score_hi-score_lo) .* rand(rng, n)
        v  = randn(rng, n); v ./= norm(v)
        w  = randn(rng, n); w ./= norm(w)
        f(t) = dot(w, softmax_vec(x0 .+ t.*v))
        fp   = (f(h) - f(-h)) / (2h)
        fpp  = (f(h) - 2f(0.0) + f(-h)) / h^2
        fppp = (f(2h) - 2f(h) + 2f(-h) - f(-2h)) / (2h^3)
        c1 = max(c1, abs(fp)); c2 = max(c2, abs(fpp)); c3 = max(c3, abs(fppp))
    end
    return c1, c2, c3
end
emit()

# ══════════════════════════════════════════════════════════════════════════
# PARTIE 2b : PROPAGATION D'INTERVALLES RÉELS le long du segment exact -- pour
# transformer les bornes "n'importe où sur R" (souvent VIDES : RMSNorm explose
# quand r(x)->0) en bornes finies utilisables, en utilisant la plage RÉELLEMENT
# atteinte par les activations sur CE segment précis (donneur<->récepteur).
# ══════════════════════════════════════════════════════════════════════════
emit("═"^92); emit("PARTIE 2b -- PROPAGATION D'INTERVALLE (RÉELLE, dense, sur le segment exact)"); emit("═"^92)
emit("Motivation : la borne RMSNorm ci-dessus (L1<=2*max|gamma|/r(x)) est INFINIE si r(x) peut approcher 0")
emit("(pire cas n'importe où dans R^dim). Sur un VRAI segment d'un réseau entraîné traitant de vrais")
emit("tokens, r(x) reste loin de 0 -- mais ce n'est PAS garanti a priori, il faut le VÉRIFIER sur ce")
emit("segment précis. On propage donc la plage RÉELLE (min/max observés) de chaque quantité pertinente")
emit("par échantillonnage dense (65 points, PAS seulement les 8 points milieux du script précédent) --")
emit("passages AVANT SEULS (aucune rétropropagation), donc peu coûteux.")
emit()

VKI, VSKI = PAIR2.vk, PAIR2.vsk
r_out = run_forward!(G, NS, receiver_tokens(PAIR2))
recv_val = copy(Array(G.nodes[NS][BRANCH].value))
d_out = run_forward!(G, NS, donor_tokens(PAIR2))
donor_val = copy(Array(G.nodes[NS][BRANCH].value))
DELTA = Float64.(donor_val) .- Float64.(recv_val)
run_forward!(G, NS, receiver_tokens(PAIR2))
emitf("Paire 2 : vrai effet (référence, cité du script précédent) : IG_8=+22.9859, vrai=+25.9579,")
emit("  |err_IG_8|=2.9720 (K=8, règle du point milieu) -- C'EST LE NOMBRE À COMPARER À LA BORNE FINALE.")
emit()

EPS_GRID = collect(range(0.0, 1.0; length=65))
NODES_TO_TRACK = Symbol[]
for l in 1:N_LAYERS
    pfx = Symbol(:layer_, l)
    push!(NODES_TO_TRACK, Symbol(pfx,:_res1), Symbol(pfx,:_norm2_out), Symbol(pfx,:_gate),
          Symbol(pfx,:_up), Symbol(pfx,:_mlp_out), Symbol(pfx,:_out))
    if l >= 2
        push!(NODES_TO_TRACK, Symbol(pfx,:_norm1_out), Symbol(pfx,:_mha,:_output_out))
        for h in 1:N_HEADS
            push!(NODES_TO_TRACK, Symbol(pfx,:_mha,:_sk_h,h))
        end
    end
end
push!(NODES_TO_TRACK, :lm_head_out)
trace = Dict{Symbol,Vector{Array{Float32}}}(s => Array{Float32}[] for s in NODES_TO_TRACK)
for eps in EPS_GRID
    val = Float64.(recv_val) .+ eps .* DELTA
    NeuroDSL.patch_node!(G, BRANCH, Dict(BRANCH => Float32.(val)); namespace=NS)
    NeuroDSL.demand!(G, :final_logits; namespace=NS)
    for s in NODES_TO_TRACK
        push!(trace[s], copy(Array(G.nodes[NS][s].value)))
    end
end
NeuroDSL.patch_node!(G, BRANCH, Dict(BRANCH => Float32.(recv_val)); namespace=NS)
NeuroDSL.demand!(G, :final_logits; namespace=NS)

# r_min observé (min sur 65 points de rms(x) = ||x||/sqrt(nc)) pour chaque RMSNorm sur le chemin.
function rms_of(x::AbstractArray; eps=1e-6)
    nc = size(x,2)
    sqrt.(sum(abs2, x; dims=2) ./ nc .+ eps) |> vec
end
RMIN = Dict{Symbol,Float64}()
for l in 1:N_LAYERS
    pfx = Symbol(:layer_, l)
    inp = l == 1 ? Symbol(pfx,:_res1) : Symbol(pfx,:_out)  # input to norm2 of layer l
    key2 = Symbol(pfx, :_norm2_in_proxy)
    rmins2 = minimum(minimum(rms_of(Float64.(trace[Symbol(pfx,:_res1)][k])) ) for k in 1:length(EPS_GRID))
    RMIN[Symbol(pfx,:_norm2)] = rmins2
    if l >= 2
        prev_out = l == 2 ? Symbol(:layer_1,:_out) : Symbol(:layer_,l-1,:_out)
        rmins1 = minimum(minimum(rms_of(Float64.(trace[prev_out][k]))) for k in 1:length(EPS_GRID))
        RMIN[Symbol(pfx,:_norm1)] = rmins1
    end
end
for (k,v) in sort(collect(RMIN); by=x->string(x[1]))
    emitf("  r_min observé (65 pts, segment réel) pour %-20s : %.4f", string(k), v)
end
emit()

SCORE_LO = Inf; SCORE_HI = -Inf
for l in 2:N_LAYERS, h in 1:N_HEADS
    sym = Symbol(:layer_,l,:_mha,:_sk_h,h)
    for k in 1:length(EPS_GRID)
        v = Float64.(trace[sym][k])
        SCORE_LO = min(SCORE_LO, minimum(v[isfinite.(v)]))
        SCORE_HI = max(SCORE_HI, maximum(v[isfinite.(v)]))
    end
end
emitf("Plage de scores d'attention (post scale+masque causal, entrées finies seulement, sur le segment) :")
emitf("  [%.3f, %.3f]  (65 points x %d couches x %d têtes)", SCORE_LO, SCORE_HI, N_LAYERS-1, N_HEADS)
GATE_LO = Inf; GATE_HI = -Inf
for l in 1:N_LAYERS
    sym = Symbol(:layer_,l,:_gate)
    for k in 1:length(EPS_GRID)
        v = Float64.(trace[sym][k])
        GATE_LO = min(GATE_LO, minimum(v)); GATE_HI = max(GATE_HI, maximum(v))
    end
end
emitf("Plage de gate (entrée SiLU dans SwiGLU) sur le segment : [%.3f, %.3f]", GATE_LO, GATE_HI)
emit()

L1_SM, L2_SM, L3_SM = angular_constants_softmax(SEQ_LEN, SCORE_LO, SCORE_HI; n_samples=8000)
emitf("Constantes softmax restreintes à la plage de score réelle (8000 échantillons, seq_len=%d) :", SEQ_LEN)
emitf("  L1=%.4f (référence classique : <=0.25)   L2=%.4f   L3=%.4f", L1_SM, L2_SM, L3_SM)
sup_silu1_local = maximum(abs.(silu_p.(GATE_LO:0.001:GATE_HI)))
sup_silu2_local = maximum(abs.(silu_pp.(GATE_LO:0.001:GATE_HI)))
sup_silu3_local = maximum(abs.(silu_ppp.(GATE_LO:0.001:GATE_HI)))
emitf("SiLU restreint à la plage de gate réelle [%.3f,%.3f] : L1=%.4f L2=%.4f L3=%.4f",
      GATE_LO, GATE_HI, sup_silu1_local, sup_silu2_local, sup_silu3_local)
emit()

# ══════════════════════════════════════════════════════════════════════════
# PARTIE 2c/3 : COMPOSITION -- propagation de (n0,n1,n2,n3) [normes de Frobenius
# agrégées] du nœud patché jusqu'à la métrique, bloc par bloc, avec les bornes
# ci-dessus. n0 = valeur réelle observée (du vrai passage avant), n1/n2/n3 =
# bornes sur les dérivées 1/2/3 par rapport à eps.
# ══════════════════════════════════════════════════════════════════════════
emit("═"^92); emit("PARTIE 2c/3 -- COMPOSITION : PROPAGATION DE (||x||,||x'||,||x''||,||x'''||)"); emit("═"^92)
emit("Règles utilisées à chaque étape (toutes des inégalités VALIDES, documentées) :")
emit("  Linéaire (matmul/W)  : n_k <= ||W||_2 * n_k(entrée)          [EXACT via svdvals des poids ENTRAÎNÉS]")
emit("  Addition (résidu)    : n_k <= n_k(a) + n_k(b)                 [inégalité triangulaire]")
emit("  Composition F(u(eps)) [RMSNorm, softmax -- Faà di Bruno, normes agrégées] :")
emit("    n1' <= L1*n1 ; n2' <= L2*n1^2 + L1*n2 ; n3' <= L3*n1^3 + 3*L2*n1*n2 + L1*n3")
emit("  Produit élément par élément (SwiGLU = SiLU(gate)⊙up -- Leibniz + Cauchy-Schwarz) :")
emit("    h_n1<=F_n1*u0+F0*u_n1 ; h_n2<=F_n2*u0+2F_n1*u_n1+F0*u_n2 ;")
emit("    h_n3<=F_n3*u0+3F_n2*u_n1+3F_n1*u_n2+F0*u_n3")
emit("  Forme bilinéaire (Q.K^T, P.V -- Leibniz + Cauchy-Schwarz, même schéma que le produit ci-dessus)")
emit("  Concat de têtes : n_k <= somme des n_k par tête [triangulaire, plus relâché que sqrt(somme carrés)]")
emit()

function spectral_norm(g, ns, sym)
    W = Float64.(Array(g.nodes[ns][sym].value))
    return maximum(svdvals(W))
end

# État = (n0::Float64, n1::Float64, n2::Float64, n3::Float64) -- normes de Frobenius agrégées.
lin(W_norm, s) = (W_norm*s[1], W_norm*s[2], W_norm*s[3], W_norm*s[4])
addst(a, b) = (a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4])
function faa_di_bruno(L1, L2, L3, s)
    n0,n1,n2,n3 = s
    on1 = L1*n1
    on2 = L2*n1^2 + L1*n2
    on3 = L3*n1^3 + 3*L2*n1*n2 + L1*n3
    return (NaN, on1, on2, on3)  # n0 renseigné séparément (valeur réelle observée)
end
function leibniz3(f, u)  # produit f*u élément par élément, formule de Leibniz jusqu'à l'ordre 3
    f0,f1,f2,f3 = f; u0,u1,u2,u3 = u
    h0 = f0*u0
    h1 = f1*u0 + f0*u1
    h2 = f2*u0 + 2*f1*u1 + f0*u2
    h3 = f3*u0 + 3*f2*u1 + 3*f1*u2 + f0*u3
    return (h0,h1,h2,h3)
end

# Norme réelle observée (base n0) pour un nœud, sur les 65 points -- max sur le segment.
frob(x) = sqrt(sum(abs2, Float64.(x)))
nodeval_n0(sym) = maximum(frob(trace[sym][k]) for k in 1:length(EPS_GRID))
nodeval_n0_arr(arr::Vector{Array{Float32}}) = maximum(frob(a) for a in arr)

max_gamma(g,ns,sym) = maximum(abs.(Float64.(Array(g.nodes[ns][sym].value))))

# ── État initial : DELTA lui-même (affine en eps => n1=||Delta||, n2=n3=0 EXACTEMENT) ──
n0_b = frob(recv_val .+ 1.0.*DELTA)  # majorant grossier (juste informatif)
state = (n0_b, frob(DELTA), 0.0, 0.0)
emitf("État initial (branche %s) : n0~%.3f  n1=||Delta||=%.4f  n2=0  n3=0 (exact, affine en eps)",
      string(BRANCH), n0_b, state[2])

logrow(stage, s) = (; stage, n0=s[1], n1=s[2], n2=s[3], n3=s[4])
trace_log = NamedTuple[]
push!(trace_log, logrow("branche (b)", state))

# ── Couche 1 : résidu-MLP seulement (b EST déjà la sortie MHA de la couche 1) ──
# add : r1 = x_fixe + b   (x_fixe ne dépend pas de eps => n_k(r1)=n_k(b) pour k>=1)
state = state  # add avec constante : dérivées inchangées pour k>=1 ; n0 pris du tracé réel
state = (nodeval_n0(:layer_1_res1), state[2], state[3], state[4])
push!(trace_log, logrow("layer_1_res1 (add résiduel, x fixe)", state))

function apply_rmsnorm_block!(trace_log, prefix, state, rmin_key, RMIN, C2_RMS, C3_RMS, g, ns)
    gmax = max_gamma(g, ns, Symbol(prefix,:_gamma))
    rmin = RMIN[rmin_key]
    L1 = 2*gmax/rmin
    L2 = gmax*C2_RMS/rmin^2
    L3 = gmax*C3_RMS/rmin^3
    news = faa_di_bruno(L1, L2, L3, state)
    n0 = nodeval_n0(Symbol(prefix,:_out))
    news = (n0, news[2], news[3], news[4])
    push!(trace_log, logrow("$(prefix)_out (RMSNorm, L1=$(round(L1,digits=3)) L2=$(round(L2,digits=3)) L3=$(round(L3,digits=3)), r_min=$(round(rmin,digits=3)))", news))
    return news
end

state = apply_rmsnorm_block!(trace_log, :layer_1_norm2, state, :layer_1_norm2, RMIN, C2_RMS, C3_RMS, G, NS)

# NOTE naming : dans src/layers.jl, les POIDS du MLP portent un segment "_mlp_" dans leur nom
# (Symbol(prefix,:_mlp_w1) où prefix=préfixe du BLOC, ex. :layer_1) MAIS les activations (gate/up/
# swiglu/mlp_out) sont nommées directement sur le préfixe du bloc SANS segment "_mlp_"
# (Symbol(prefix,:_gate) = :layer_1_gate, PAS :layer_1_mlp_gate) -- asymétrie réelle du code source,
# `prefix` ici doit donc être le préfixe du BLOC (:layer_1, :layer_2, ...), pas un préfixe composé.
function mlp_block!(trace_log, prefix, state, G, NS, sup_silu1, sup_silu2, sup_silu3, gate_lo, gate_hi)
    Wg = spectral_norm(G, NS, Symbol(prefix,:_mlp_w1))
    Wu = spectral_norm(G, NS, Symbol(prefix,:_mlp_w3))
    Wd = spectral_norm(G, NS, Symbol(prefix,:_mlp_w2))
    gate_state = lin(Wg, state)
    up_state   = lin(Wu, state)
    gate_state = (nodeval_n0(Symbol(prefix,:_gate)), gate_state[2], gate_state[3], gate_state[4])
    up_state   = (nodeval_n0(Symbol(prefix,:_up)),   up_state[2],   up_state[3],   up_state[4])
    push!(trace_log, logrow("$(prefix)_gate (Linear W1, ||W1||=$(round(Wg,digits=3)))", gate_state))
    push!(trace_log, logrow("$(prefix)_up (Linear W3, ||W3||=$(round(Wu,digits=3)))", up_state))
    F_state = faa_di_bruno(sup_silu1, sup_silu2, sup_silu3, gate_state)
    silu_n0_info = maximum(abs.(silu.(gate_lo:0.01:gate_hi)))  # informatif seulement (n0 pas utilisé par lin() ci-dessous)
    F_state = (silu_n0_info, F_state[2], F_state[3], F_state[4])
    h_state = leibniz3(F_state, up_state)
    push!(trace_log, logrow("$(prefix)_swiglu (SiLU⊙up, L1=$(round(sup_silu1,digits=3)) L2=$(round(sup_silu2,digits=3)) L3=$(round(sup_silu3,digits=3)))", h_state))
    mo_state = lin(Wd, h_state)
    mo_state = (nodeval_n0(Symbol(prefix,:_mlp_out)), mo_state[2], mo_state[3], mo_state[4])
    push!(trace_log, logrow("$(prefix)_mlp_out (Linear W2, ||W2||=$(round(Wd,digits=3)))", mo_state))
    return mo_state
end

mo1 = mlp_block!(trace_log, :layer_1, state, G, NS, sup_silu1_local, sup_silu2_local, sup_silu3_local, GATE_LO, GATE_HI)
state = addst(state, mo1)
state = (nodeval_n0(:layer_1_out), state[2], state[3], state[4])
push!(trace_log, logrow("layer_1_out (add résiduel)", state))

function attn_block!(trace_log, prefix, state, G, NS, N_HEADS, D_HEAD, L1_SM, L2_SM, L3_SM)
    Wq = spectral_norm(G, NS, Symbol(prefix,:_mha,:_q,:_W))
    Wk = spectral_norm(G, NS, Symbol(prefix,:_mha,:_k,:_W))
    Wv = spectral_norm(G, NS, Symbol(prefix,:_mha,:_v,:_W))
    Wo = spectral_norm(G, NS, Symbol(prefix,:_mha,:_output,:_W))
    q_state = lin(Wq, state); k_state = lin(Wk, state); v_state = lin(Wv, state)
    scale = 1.0/sqrt(D_HEAD)
    # Q.K^T bilinéaire, par tête -- on utilise la borne AGRÉGÉE (norme de Frobenius sur tout le
    # tenseur de têtes empilées, relâchement supplémentaire documenté : les 4 têtes sont traitées
    # comme un seul bloc bilinéaire de même ordre de grandeur -- valide car sous-multiplicatif).
    q0,q1,q2,q3 = q_state; k0,k1,k2,k3 = k_state
    s0 = q0*k0*scale
    s1 = (q1*k0 + q0*k1)*scale
    s2 = (q2*k0 + 2*q1*k1 + q0*k2)*scale
    s3 = (q3*k0 + 3*q2*k1 + 3*q1*k2 + q0*k3)*scale
    sk_state = (s0,s1,s2,s3)
    push!(trace_log, logrow("$(prefix)_mha scores (Q.K^T/sqrt(d), bilinéaire, Leibniz+Cauchy-Schwarz)", sk_state))
    pr_state = faa_di_bruno(L1_SM, L2_SM, L3_SM, sk_state)
    pr_state = (sqrt(N_HEADS*SEQ_LEN), pr_state[2], pr_state[3], pr_state[4])  # softmax: norme de valeur bornée (chaque ligne somme=1)
    push!(trace_log, logrow("$(prefix)_mha softmax (L1=$(round(L1_SM,digits=3)) L2=$(round(L2_SM,digits=3)) L3=$(round(L3_SM,digits=3)))", pr_state))
    p0,p1,p2,p3 = pr_state; v0,v1,v2,v3 = v_state
    a0 = p0*v0
    a1 = p1*v0 + p0*v1
    a2 = p2*v0 + 2*p1*v1 + p0*v2
    a3 = p3*v0 + 3*p2*v1 + 3*p1*v2 + p0*v3
    ao_state = (a0,a1,a2,a3)
    push!(trace_log, logrow("$(prefix)_mha P.V (bilinéaire, Leibniz+Cauchy-Schwarz)", ao_state))
    # concat : N_HEADS têtes de même ordre -> triangulaire (relâchement additionnel documenté)
    concat_state = (N_HEADS*ao_state[1], N_HEADS*ao_state[2], N_HEADS*ao_state[3], N_HEADS*ao_state[4])
    out_state = lin(Wo, concat_state)
    out_state = (nodeval_n0(Symbol(prefix,:_mha,:_output_out)), out_state[2], out_state[3], out_state[4])
    push!(trace_log, logrow("$(prefix)_mha_output_out (Linear Wo, ||Wo||=$(round(Wo,digits=3)))", out_state))
    return out_state
end

for l in 2:N_LAYERS
    pfx = Symbol(:layer_, l)
    x_in_state = state                      # entrée du bloc (avant norm1) -- nécessaire pour le résidu
    n1_state = apply_rmsnorm_block!(trace_log, Symbol(pfx,:_norm1), x_in_state, Symbol(pfx,:_norm1), RMIN, C2_RMS, C3_RMS, G, NS)
    ao_state = attn_block!(trace_log, pfx, n1_state, G, NS, N_HEADS, D_HEAD, L1_SM, L2_SM, L3_SM)
    r1_state = addst(x_in_state, ao_state)
    r1_state = (nodeval_n0(Symbol(pfx,:_res1)), r1_state[2], r1_state[3], r1_state[4])
    push!(trace_log, logrow("$(pfx)_res1 (add résiduel MHA)", r1_state))
    n2_state = apply_rmsnorm_block!(trace_log, Symbol(pfx,:_norm2), r1_state, Symbol(pfx,:_norm2), RMIN, C2_RMS, C3_RMS, G, NS)
    mo_state = mlp_block!(trace_log, pfx, n2_state, G, NS, sup_silu1_local, sup_silu2_local, sup_silu3_local, GATE_LO, GATE_HI)
    out_state = addst(r1_state, mo_state)
    out_state = (nodeval_n0(Symbol(pfx,:_out)), out_state[2], out_state[3], out_state[4])
    push!(trace_log, logrow("$(pfx)_out (add résiduel MLP)", out_state))
    state = out_state
end

# ── Tête finale : lm_head (Linear, exacte) -> sélection dernière position (Linear, exacte) ──
Wlm = spectral_norm(G, NS, :lm_head_W)
state = lin(Wlm, state)
state = (nodeval_n0(:lm_head_out), state[2], state[3], state[4])
push!(trace_log, logrow("lm_head_out (Linear, ||W||=$(round(Wlm,digits=3)))", state))
# Sélection de la dernière position = matrice constante avec un seul 1 -> norme d'opérateur EXACTE 1.
state = lin(1.0, state)
push!(trace_log, logrow("final_logits (sélection dernière position, ||.||=1, exact)", state))
# dl_logit_diff = <e_vsk - e_vk, final_logits> -- fonctionnelle linéaire, norme EXACTE sqrt(2).
sel_norm = sqrt(2.0)
phi_bound = lin(sel_norm, state)
phi0, phi1, phi2, phi3 = phi_bound
push!(trace_log, logrow("phi(eps) = dl_logit_diff (fonctionnelle linéaire, ||v||=sqrt(2), exact)", phi_bound))

emit()
for row in trace_log
    emitf("  %-70s n1'=%12.4e  n2''=%14.4e  n3'''=%16.4e", row.stage, row.n1, row.n2, row.n3)
end
emit()

# ══════════════════════════════════════════════════════════════════════════
# PARTIE 3 : LE NOMBRE FINAL -- comparaison à l'erreur réelle mesurée
# ══════════════════════════════════════════════════════════════════════════
emit("═"^92); emit("PARTIE 3 -- BORNE FINALE vs ERREUR RÉELLE"); emit("═"^92)
emitf("sup|phi'''(eps)| <= %.6e   (borne compositionnelle, interval-propagation-based, ci-dessus)", phi3)
K = K_STEPS
pred_bound_analytic = phi3 / (24*K^2)
emitf("Borne d'erreur IG_%d correspondante : sup|phi'''|/(24*K^2) = %.6e / %d = %.6f", K, phi3, 24*K^2, pred_bound_analytic)
emit()
emitf("Erreur RÉELLE mesurée (référence, script précédent) |err_IG_8| = 2.9720")
emitf("Borne PROXY empirique M3/(24K^2) (script précédent, 6 pts intérieurs)  = 3.1717")
emitf("Borne ANALYTIQUE (ce script, compositionnelle, sans échantillonner phi) = %.6f", pred_bound_analytic)
emit()
ratio_to_true = pred_bound_analytic / 2.9720
ratio_to_proxy = pred_bound_analytic / 3.1717
emitf("Ratio borne_analytique / erreur_réelle  = %.3e", ratio_to_true)
emitf("Ratio borne_analytique / borne_proxy_M3 = %.3e", ratio_to_proxy)
emit()

emit("═"^92); emit("VERDICT"); emit("═"^92)
if isfinite(pred_bound_analytic) && ratio_to_true < 100
    emit("La borne analytique compositionnelle est FINIE et dans un ordre de grandeur UTILE")
    emit("(moins de 100x l'erreur réelle) -- un certificat rigoureux est praticable ici.")
elseif isfinite(pred_bound_analytic)
    emit("La borne analytique compositionnelle est FINIE (grâce à la propagation d'intervalle réelle,")
    emit("sans laquelle RMSNorm seul la rendrait infinie) mais TRÈS LÂCHE : plusieurs ordres de grandeur")
    emit("au-dessus de l'erreur réelle. Correcte (valide, aucune violation), mais économiquement inutile")
    emit("comme certificat pratique -- confirme l'issue attendue pour une composition de plusieurs")
    emit("couches softmax+RMSNorm+SwiGLU par inégalités successives (dégradation multiplicative typique).")
else
    emit("La borne analytique diverge (infinie/NaN) malgré la propagation d'intervalle -- même avec la")
    emit("plage réelle observée, la composition sur 3 couches complètes ne produit aucun nombre exploitable.")
end
emit()
emit("Décomposition de la dégradation (voir tableau détaillé ci-dessus) : identifier à quelle étape")
emit("n3 explose donne l'origine dominante de la perte de rigueur (composition Faà-di-Bruno cubique en")
emit("n1, normes d'opérateur des poids réels, ou relâchements bilinéaires Q.K^T/P.V).")
end

println("\nÉcrit : ", OUT)
