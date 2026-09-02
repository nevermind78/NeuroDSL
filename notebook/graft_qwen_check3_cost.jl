# ══════════════════════════════════════════════════════════════════════════════
# graft_qwen_check3_cost.jl — Check 3 du pré-enregistrement
# (graft_qwen_correctness_preregistration.md) : coût réel du backward gelé
# (prune_frozen=true, cône aval seulement) vs backward complet
# (prune_frozen=false, gaspillage volontaire de référence) sur Qwen réel
# greffé/gelé.
#
# `MODE` (variable d'environnement GRAFT_QWEN_MODE, "pruned" ou "full")
# sélectionne LEQUEL des deux est mesuré -- appelé deux fois, dans deux
# processus julia FRAIS séparés (isolation exigée par la tâche), jamais dans
# le même run, pour éviter toute contamination (état CUDA compilé, cache de
# topologie, etc.) entre les deux conditions comparées.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, Printf, Statistics

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6
const GRAFT_SITE = :layer_25_out
const GRAFT_PREFIX = :qwen_shadow_l25
const GRAFT_HEADS, GRAFT_HIDDEN = 4, 384
const N_REPS = 5

MODE = get(ENV, "GRAFT_QWEN_MODE", "pruned")
@assert MODE in ("pruned", "full") "GRAFT_QWEN_MODE doit être 'pruned' ou 'full'"
PRUNE_FROZEN = MODE == "pruned"
println("── Check 3 : coût backward -- mode = $MODE (prune_frozen=$PRUNE_FROZEN) ──"); flush(stdout)

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
println("Chargement des poids réels..."); flush(stdout)
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
println("Poids chargés."); flush(stdout)

new_out, handle = NeuroDSL.graft_shadow_block!(g, ns, GRAFT_SITE, DIM, GRAFT_HEADS, GRAFT_HIDDEN;
                                                alpha0=0f0, zero_out_proj=false, prefix=GRAFT_PREFIX)
println("Greffe insérée -> $new_out"); flush(stdout)

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
@printf("Backbone gelé : %d nœuds.\n", n_frozen)
flush(stdout)

n_total_nodes = length(g.nodes[ns])
@printf("Nombre total de nœuds du graphe : %d\n", n_total_nodes)
flush(stdout)

adhoc = JSON.parsefile(joinpath(MODEL_DIR, "adhoc_prompts.json"))
prompt_entry = adhoc[1]
tokens0 = Int.(prompt_entry["token_ids"]) .+ 1
input_tokens = tokens0[1:end-1]
label_tokens = tokens0[2:end]
NeuroDSL.set!(g, :token_ids, input_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:length(input_tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :labels, label_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [logits_sym, :labels], :cross_entropy; namespace=ns))

function run_once!(g, ns, prune_frozen)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, :loss; namespace=ns)
    NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.synchronize()
    t0 = time()
    NeuroDSL.backward_graph!(g, :loss; namespace=ns, prune_frozen=prune_frozen)
    NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.synchronize()
    return time() - t0
end

println("Chauffe (compilation CUDA)..."); flush(stdout)
t_warm = run_once!(g, ns, PRUNE_FROZEN)
@printf("  chauffe : %.4f s\n", t_warm)
flush(stdout)

times = Float64[]
for r in 1:N_REPS
    t = run_once!(g, ns, PRUNE_FROZEN)
    push!(times, t)
    @printf("  rep %d/%d : %.4f s\n", r, N_REPS, t)
    flush(stdout)
end
t_median = median(times)
@printf("Temps médian (%d reps, mode=%s) : %.4f s\n", N_REPS, MODE, t_median)
flush(stdout)

# Compte des nœuds réellement touchés par CETTE dernière passe backward --
# `.backwarded` est remis à false en tête de CHAQUE appel à backward_graph!
# puis mis à true pour tout nœud visité, et N'EST PAS nettoyé en fin de passe
# (contrairement à `.gradient`) -- instrument fiable même après coup.
n_backwarded = count(nd -> nd.backwarded, values(g.nodes[ns]))
@printf("Nœuds avec .backwarded=true après la dernière passe (mode=%s) : %d / %d\n",
        MODE, n_backwarded, n_total_nodes)
flush(stdout)

out_path = joinpath(@__DIR__, "graft_qwen_check3_$(MODE)_results.json")
open(out_path, "w") do io
    JSON.print(io, Dict(
        "mode" => MODE, "prune_frozen" => PRUNE_FROZEN,
        "n_total_nodes" => n_total_nodes, "n_frozen" => n_frozen,
        "t_warm" => t_warm, "times" => times, "t_median" => t_median,
        "n_backwarded" => n_backwarded,
    ))
end
println("Résultats écrits -> ", out_path)
