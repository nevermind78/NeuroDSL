# ══════════════════════════════════════════════════════════════════════════════
# graft_qwen_check1_freeze.jl — Check 1 du pré-enregistrement
# (graft_qwen_correctness_preregistration.md) : correction du gel à l'échelle
# réelle de Qwen2.5-1.5B-Instruct.
#
# Charge Qwen (poids réels), greffe un bloc Gradient Shadowing à :layer_25_out,
# gèle explicitement TOUS les poids Qwen d'origine (is_param=false), lance
# backward_graph!(...; prune_frozen=true) sur une perte cross-entropie réelle
# (prompt réel tokenisé par le vrai tokenizer Qwen), puis inspecte DIRECTEMENT
# .gradient et .is_param sur LA TOTALITÉ des nœuds de poids Qwen d'origine.
#
# Processus isolé, dédié à ce seul check.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, Printf

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6
const GRAFT_SITE = :layer_25_out
const GRAFT_PREFIX = :qwen_shadow_l25
const GRAFT_HEADS, GRAFT_HIDDEN = 4, 384

println("── Check 1 : correction du gel à l'échelle Qwen ──"); flush(stdout)

dev = NeuroDSL.Backend.CUDADevice()
ns = :qwen2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:8); atom_type=NeuroDSL.Datom, namespace=ns)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
out = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                           batched_attn=true, n_kv_heads=N_KV_HEADS,
                           qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out, :final_norm; namespace=ns)
logits_sym = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)
println("Graphe construit. Chargement des poids réels..."); flush(stdout)
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
println("Poids chargés."); flush(stdout)

# Snapshot des symboles de poids Qwen D'ORIGINE, AVANT toute greffe (pour ne
# jamais confondre un poids Qwen avec un tenseur de la greffe plus bas).
qwen_weight_syms = sort(collect(k for (k, nd) in g.nodes[ns] if nd.is_param))
n_qwen_params_before = length(qwen_weight_syms)
@printf("Nombre de tenseurs de poids Qwen d'origine (is_param=true avant greffe) : %d\n", n_qwen_params_before)
flush(stdout)

println("Insertion de la greffe Gradient Shadowing à :$(GRAFT_SITE)..."); flush(stdout)
new_out, handle = NeuroDSL.graft_shadow_block!(g, ns, GRAFT_SITE, DIM, GRAFT_HEADS, GRAFT_HIDDEN;
                                                alpha0=0f0, zero_out_proj=false, prefix=GRAFT_PREFIX)
println("Greffe insérée -> nouvelle sortie : $new_out"); flush(stdout)

graft_param_syms = sort(collect(k for (k, nd) in g.nodes[ns] if nd.is_param && startswith(String(k), String(GRAFT_PREFIX))))
@printf("Nombre de tenseurs de la greffe (is_param=true, préfixe %s) : %d\n", GRAFT_PREFIX, length(graft_param_syms))
flush(stdout)

# ── Gel explicite du backbone -- copié à l'identique du helper local déjà
# utilisé et correct dans real_llm_graft_experiment.jl (aucune modification
# de src/). ──
function freeze_backbone!(g::NeuroDSL.NeuroGraph, ns::Symbol, keep_prefix::Symbol)
    n_frozen = 0
    for (sym, nd) in g.nodes[ns]
        nd.is_param || continue
        startswith(String(sym), String(keep_prefix)) && continue
        nd.is_param = false
        n_frozen += 1
    end
    return n_frozen
end

n_frozen = freeze_backbone!(g, ns, GRAFT_PREFIX)
@printf("Gel effectué : %d nœuds passés à is_param=false.\n", n_frozen)
flush(stdout)

# Vérification directe : est-ce que TOUS les symboles de poids Qwen d'origine
# ont bien is_param=false maintenant, et AUCUN qui ne le devrait pas ?
still_true_qwen = [s for s in qwen_weight_syms if g.nodes[ns][s].is_param]
still_true_graft = [s for s in graft_param_syms if g.nodes[ns][s].is_param]
@printf("Nœuds de poids Qwen encore is_param=true après gel (devrait être 0) : %d\n", length(still_true_qwen))
@printf("Nœuds de la greffe toujours is_param=true après gel (devrait être %d) : %d\n", length(graft_param_syms), length(still_true_graft))
if !isempty(still_true_qwen)
    println("  RATÉS : ", still_true_qwen)
end
flush(stdout)

ps_after_freeze = NeuroDSL.params(g; namespace=ns)
@printf("params(g;ns) après gel : %d tenseurs entraînables (attendu = %d, taille de la greffe)\n",
        length(ps_after_freeze), length(graft_param_syms))
flush(stdout)

# ── Perte réelle : cross-entropie next-token sur un prompt réel tokenisé par
# le vrai tokenizer Qwen (adhoc_prompts.json). ──
adhoc = JSON.parsefile(joinpath(MODEL_DIR, "adhoc_prompts.json"))
prompt_entry = adhoc[1]
prompt_text = prompt_entry["prompt"]
tokens0 = Int.(prompt_entry["token_ids"]) .+ 1   # 1-indexé pour NeuroDSL
@printf("Prompt réel : %s (%d tokens)\n", repr(prompt_text), length(tokens0))
input_tokens = tokens0[1:end-1]
label_tokens = tokens0[2:end]
flush(stdout)

NeuroDSL.set!(g, :token_ids, input_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:length(input_tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :labels, label_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
# logits_sym reste le nœud de sortie final -- graft_shadow_block! a déjà
# rebranché tous les consommateurs préexistants de :layer_25_out (donc toute
# la suite layer_26..28 + final_norm + lm_head) vers new_out automatiquement.
NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [logits_sym, :labels], :cross_entropy; namespace=ns))
NeuroDSL.invalidate_all!(g; namespace=ns)

println("Passe avant (loss)..."); flush(stdout)
loss_val = NeuroDSL.demand!(g, :loss; namespace=ns)
@printf("loss = %.6f\n", Float64(sum(Array(loss_val))))
flush(stdout)

println("Passe arrière (backward_graph!, prune_frozen=true)..."); flush(stdout)
NeuroDSL.backward_graph!(g, :loss; namespace=ns, prune_frozen=true)
println("Backward terminé."); flush(stdout)

# ── VÉRIFICATION PRINCIPALE : .gradient sur TOUS les nœuds de poids Qwen d'origine ──
# Fonction (scope dur, sans ambiguïté soft-scope de script) -- évite le piège
# de l'avertissement "Assignment... is ambiguous" observé au premier run
# (une boucle for top-level de script peut traiter une réaffectation comme
# une nouvelle locale, laissant le compteur externe figé à sa valeur initiale
# même si la vraie boucle interne comptait correctement).
function _count_with_gradient(g, ns, syms)
    n = 0
    off = Symbol[]
    for s in syms
        nd = g.nodes[ns][s]
        if nd.gradient !== nothing
            n += 1
            push!(off, s)
        end
    end
    return n, off
end
n_with_gradient, offending = _count_with_gradient(g, ns, qwen_weight_syms)
@printf("\nNœuds de poids Qwen d'origine avec .gradient != nothing après backward (devrait être 0) : %d / %d\n",
        n_with_gradient, length(qwen_weight_syms))
if !isempty(offending)
    println("  RATÉS : ", offending[1:min(end,20)])
end
flush(stdout)

# Vérification secondaire : le gate/branche de la greffe A bien un gradient
# (sanity -- sinon le backward n'a rien calculé du tout, pas un vrai succès).
graft_with_grad = count(s -> g.nodes[ns][s].gradient !== nothing, graft_param_syms)
@printf("Nœuds de la greffe avec .gradient != nothing après backward (sanity, attendu > 0) : %d / %d\n",
        graft_with_grad, length(graft_param_syms))
flush(stdout)

pass1a = length(still_true_qwen) == 0
pass1b = n_with_gradient == 0
pass1c = graft_with_grad > 0
verdict = pass1a && pass1b && pass1c
println("\n", "="^80)
println("CHECK 1 VERDICT")
println("  (a) 100% des nœuds Qwen d'origine gelés (is_param=false) : ", pass1a)
println("  (b) .gradient === nothing sur 100% des nœuds Qwen d'origine : ", pass1b)
println("  (c) sanity -- la greffe a bien reçu un gradient : ", pass1c)
println("  PASS global : ", verdict)
println("="^80)

open(joinpath(@__DIR__, "graft_qwen_check1_results.json"), "w") do io
    JSON.print(io, Dict(
        "n_qwen_params_before" => n_qwen_params_before,
        "n_graft_params" => length(graft_param_syms),
        "n_frozen" => n_frozen,
        "still_true_qwen" => String.(still_true_qwen),
        "n_with_gradient" => n_with_gradient,
        "offending" => String.(offending),
        "graft_with_grad" => graft_with_grad,
        "n_graft_total" => length(graft_param_syms),
        "loss" => Float64(sum(Array(loss_val))),
        "pass1a" => pass1a, "pass1b" => pass1b, "pass1c" => pass1c, "verdict" => verdict,
    ))
end
println("Résultats écrits -> notebook/graft_qwen_check1_results.json")
