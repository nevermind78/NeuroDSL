# ══════════════════════════════════════════════════════════════════════════════
# verify_attention_jacobian_bound.jl — Vérification numérique (style
# rem:curvature-check : test de réfutation par différences finies, pas une
# confirmation) d'une borne LOCALE (pas globale) sur le jacobien du sous-bloc
# d'attention -- l'ingrédient manquant nommé dans rem:crossterm-gap
# (artilce/code4.tex) pour fermer Gap 1/Gap 2 de prop:crossterm.
#
# Dérivation (voir notebook/code4_step5_run.log pour l'algèbre complète) :
# pour une position clé/valeur j < position de requête tau (tête h fixée),
# en notant u_i := N(x_i) l'entrée normalisée réellement lue par Q/K/V,
# q:=W_Q u_tau, k_i:=W_K u_i, v_i:=W_V u_i, p:=softmax(q^T k_i/sqrt(d_h))_i,
# a:=sum_i p_i v_i (sortie de la tête à la position tau) :
#
#   ∂a/∂u_j = p_j W_V + p_j (v_j-a) ⊗ (W_K^T q / sqrt(d_h))
#
# d'où, EXACTEMENT (pas d'approximation dans cette borne) :
#
#   ||∂a/∂u_j||_op <= p_j ||W_V||_op + p_j ||v_j-a|| ||W_K||_op ||q|| / sqrt(d_h)   [borne "serrée", quantités réelles]
#
# et, en utilisant ||u_i||<=sqrt(d)*||gamma||_inf (même borne de normalisation
# que prop:curvature, ligne ~1307 de code4.tex) pour fermer ||v_j||,||q|| :
#
#   ||∂a/∂u_j||_op <= p_j ||W_V||_op (1 + 2*d*||gamma||_inf^2*||W_Q||_op*||W_K||_op/sqrt(d_h))   [borne "large", weights seuls]
#
# Les DEUX bornes sont LOCALES (évaluées à p_j, v_j, a, q RÉELS au point testé,
# ou à ||gamma||_inf/||W||_op RÉELS du modèle chargé) -- pas des constantes
# universelles sur tout l'espace des entrées, exactement le statut de
# prop:curvature déjà dans l'article. ||W||_op via norme spectrale (SVD, CPU).
#
# Test : sur le graphe Qwen2.5-1.5B réel déjà chargé (recalcul complet),
# plusieurs (couche, tête, position clé j) couvrant des profondeurs variées ;
# pour chacun, M perturbations aléatoires de u_j (patch_node! + demand! sur le
# noeud ao_h, restauration entre essais), estimation par différence finie de
# ||∂a/∂u_j|| le long de chaque direction, comparaison aux deux bornes.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, LinearAlgebra, Statistics, Random

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=1))s] ", msg); flush(stdout))
Random.seed!(20260801)

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6
const D_HEAD = DIM ÷ N_HEADS
const GROUP_SIZE = N_HEADS ÷ N_KV_HEADS

_log("══ Vérification numérique : borne locale sur le jacobien du sous-bloc d'attention (Qwen2.5-1.5B réel) ══")

dev = NeuroDSL.Backend.CUDADevice()
ns = :qwen2_attnjac
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
out_sym = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                               batched_attn=true, n_kv_heads=N_KV_HEADS,
                               qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out_sym, :final_norm; namespace=ns)
logits_sym = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)
_log("Graphe construit. Chargement des poids réels...")
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
_log("Poids chargés.")

# ── Tokenizer minimal (une seule phrase IOI réelle, pas besoin du serveur
# persistant pour ce test -- encode une fois, à la main via les ids déjà
# connus serait fragile ; on relance juste le helper en mode un-coup) ──────
const PYTHON_ENV = raw"C:\Users\Nevermind\anaconda3\envs\neurodsl_llm_check\python.exe"
const TOKENIZER_HELPER = joinpath(@__DIR__, "qwen2_tokenizer_helper.py")
const TOK_PROC = open(`$PYTHON_ENV -u $TOKENIZER_HELPER --serve`, "r+")
using JSON
JSON.parse(readline(TOK_PROC))
function tok_call(req::Dict)
    println(TOK_PROC, JSON.json(req)); flush(TOK_PROC)
    return JSON.parse(readline(TOK_PROC))
end
encode_raw(text::String) = Int.(tok_call(Dict("action"=>"encode", "text"=>text))["ids"])
const PROMPT = "When Mary and John went to the store, John gave the bag to"
const ids1 = encode_raw(PROMPT) .+ 1
const TAU = length(ids1)  # position de requête = dernier token
_log("Prompt : \"$PROMPT\" -- $(TAU) tokens.")

NeuroDSL.set!(g, :token_ids, ids1; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.invalidate_all!(g; namespace=ns)
NeuroDSL.demand!(g, logits_sym; namespace=ns)
_log("Passage avant propre effectué (état de référence capturé ci-dessous).")

opnorm_cpu(A) = opnorm(Array(A), 2)  # norme spectrale, calcul CPU (SVD)

function test_site(layer::Int, head::Int, j::Int; n_trials::Int=24, eps::Float64=0.05)
    mha_prefix = Symbol(:layer_, layer, :_mha)
    kv_head = (head - 1) ÷ GROUP_SIZE + 1
    norm1_sym = Symbol(:layer_, layer, :_norm1_out)
    ao_sym = Symbol(mha_prefix, :_ao_h, head)
    v_sym = Symbol(mha_prefix, :_v_h, kv_head)
    q_sym = Symbol(mha_prefix, :_q_h, head, :_rope)
    pr_sym = Symbol(mha_prefix, :_pr_h, head)
    qW_sym = Symbol(mha_prefix, :_q_W); kW_sym = Symbol(mha_prefix, :_k_W); vW_sym = Symbol(mha_prefix, :_v_W)
    gamma_sym = Symbol(:layer_, layer, :_norm1_gamma)

    u_clean = Array(NeuroDSL.node(g, norm1_sym; namespace=ns).value)        # (T,D) -- forcé sur CPU : patch_node! rappelle Backend.to_device lui-même, éviter toute écriture in-place sur un CuArray depuis un vecteur CPU (échoue à la compilation du kernel GPU broadcast)
    a_clean = copy(Array(NeuroDSL.node(g, ao_sym; namespace=ns).value))     # (T,d_head)
    v_all   = Array(NeuroDSL.node(g, v_sym; namespace=ns).value)           # (T,d_head)
    q_all   = Array(NeuroDSL.node(g, q_sym; namespace=ns).value)           # (T,d_head)
    pr_all  = Array(NeuroDSL.node(g, pr_sym; namespace=ns).value)          # (T,T)
    gamma   = Array(NeuroDSL.node(g, gamma_sym; namespace=ns).value)

    pj = Float64(pr_all[TAU, j])
    vj = Float64.(v_all[j, :]); a_tau = Float64.(a_clean[TAU, :]); qv = Float64.(q_all[TAU, :])
    Wv_op = opnorm_cpu(NeuroDSL.node(g, vW_sym; namespace=ns).value)
    Wk_op = opnorm_cpu(NeuroDSL.node(g, kW_sym; namespace=ns).value)
    Wq_op = opnorm_cpu(NeuroDSL.node(g, qW_sym; namespace=ns).value)
    gamma_inf = maximum(abs.(gamma))

    bound_tight = pj*Wv_op + pj*norm(vj .- a_tau)*Wk_op*norm(qv)/sqrt(D_HEAD)
    bound_loose = pj*Wv_op*(1 + 2*DIM*gamma_inf^2*Wq_op*Wk_op/sqrt(D_HEAD))

    # ── Différences finies : M directions aléatoires unitaires sur u_j ─────
    empirical = Float64[]
    for _ in 1:n_trials
        d = randn(DIM); d ./= norm(d)
        u_pert = copy(u_clean)
        u_pert[j, :] .+= Float32(eps) .* Float32.(d)
        NeuroDSL.patch_node!(g, norm1_sym, Dict(norm1_sym=>u_pert); namespace=ns)
        a_pert = Array(NeuroDSL.demand!(g, ao_sym; namespace=ns))
        da = norm(Float64.(a_pert[TAU, :]) .- a_tau) / eps
        push!(empirical, da)
        # restauration exacte avant l'essai suivant
        NeuroDSL.patch_node!(g, norm1_sym, Dict(norm1_sym=>u_clean); namespace=ns)
        NeuroDSL.demand!(g, ao_sym; namespace=ns)
    end
    emp_max = maximum(empirical)
    violated_tight = emp_max > bound_tight
    violated_loose = emp_max > bound_loose
    _log("  [layer=$layer head=$head j=$j] p_j=$(round(pj,digits=4))  emp_max=$(round(emp_max,digits=5))  " *
         "borne_serrée=$(round(bound_tight,digits=5))  borne_large=$(round(bound_loose,digits=5))  " *
         "violation_serrée=$violated_tight  violation_large=$violated_loose  ratio_serrée=$(round(bound_tight/max(emp_max,1e-12),digits=2))")
    return (; layer, head, j, pj, emp_max, bound_tight, bound_loose, violated_tight, violated_loose)
end

_log("")
_log("═"^78)
_log("Balayage (couche, tête, position clé) -- profondeurs variées")
_log("═"^78)
sites = [(1,1), (1,9), (10,6), (19,6), (25,9), (28,3)]
results = NamedTuple[]
for (layer, head) in sites
    for j in (2, max(2, TAU-2))
        j == TAU && continue
        r = test_site(layer, head, j)
        push!(results, r)
    end
end

close(TOK_PROC)

_log("")
_log("═"^78); _log("VERDICT"); _log("═"^78)
n_viol_tight = count(r->r.violated_tight, results)
n_viol_loose = count(r->r.violated_loose, results)
_log("Sites testés : $(length(results))")
_log("Violations de la borne serrée (quantités réelles) : $n_viol_tight / $(length(results))")
_log("Violations de la borne large (poids seuls, fermée) : $n_viol_loose / $(length(results))")
ratios_tight = [r.bound_tight/max(r.emp_max,1e-12) for r in results]
ratios_loose = [r.bound_loose/max(r.emp_max,1e-12) for r in results]
_log("Ratio borne/empirique (serrée) : min=$(round(minimum(ratios_tight),digits=2)) max=$(round(maximum(ratios_tight),digits=2)) médiane=$(round(median(ratios_tight),digits=2))")
_log("Ratio borne/empirique (large)  : min=$(round(minimum(ratios_loose),digits=2)) max=$(round(maximum(ratios_loose),digits=2)) médiane=$(round(median(ratios_loose),digits=2))")
flush(stdout)
