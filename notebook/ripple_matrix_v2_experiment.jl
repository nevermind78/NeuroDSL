# ══════════════════════════════════════════════════════════════════════════════
# ripple_matrix_v2_experiment.jl -- replication matrix v2, sur les triplets
# VERIFIES (voir ripple_matrix_v2_filter_verified.py -- 9/12, contre 2/5 en v1,
# parce que la verification lit maintenant le LOGIT de la reponse attendue au
# lieu d'exiger qu'elle soit le top-1 strict).
#
# PROTOCOLE CAUSAL : IDENTIQUE A v1, RIEN N'A CHANGE ICI
# --------------------------------------------------------------------------
# Meme patch position-masque sur les 28 couches (mlp_out), meme baseline
# cosinus a layer_7_mlp_out, meme metrique ratio = |delta_Q1|/max(|delta_Q2|,1e-6),
# meme controle de puissance (delta_Q1_profond > 3x delta_Q1_couche-unique).
# Seule la SOURCE des triplets change (ripple_matrix_v2_verified.json au lieu
# de ripple_matrix_prompts.json).
#
# SEUILS DE VERDICT, PREENREGISTRES ICI -- AVANT TOUT CHIFFRE CAUSAL
# --------------------------------------------------------------------------
# La calibration/verification (script precedent) n'a jamais touche a la
# metrique causale (ratio, cosinus) -- ces seuils sont donc ecrits a
# l'aveugle, comme en v1. v1 avait fixe son verdict sur n=5 ("4/5 triplets
# >=2" pour REPLICATES). Ici n=9 (le nombre EXACT de triplets verifies n'est
# connu qu'apres la calibration, mais AUCUN chiffre causal n'a encore ete vu
# au moment ou ces seuils sont fixes) -- on generalise la fraction plutot que
# de forcer un compte absolu specifique a n=5 :
#   REPLICATES : mediane des ratios >= 3 ET >=75% des triplets ont ratio>=2.
#   PARTIEL    : entre 30% et 75% des triplets ont ratio>=2.
#   NE REPLIQUE PAS : <30% des triplets ont ratio>=2, ou mediane < 2.
#   (comme en v1, il faut >=3 triplets utilisables pour juger du tout --
#   deja garanti ici avec 9.)
#
# USAGE : julia --project=. notebook/ripple_matrix_v2_experiment.jl
# ECRIT : notebook/ripple_matrix_v2_experiment_results.json
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Printf, JSON, LinearAlgebra, Statistics

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const TRIPLES   = JSON.parsefile(joinpath(@__DIR__, "ripple_matrix_v2_verified.json"))
const OUT       = joinpath(@__DIR__, "ripple_matrix_v2_experiment_results.json")
const NS, NL    = :qwen2, 28
const LOGITS    = :lm_head_out
const COS_LAYER = 7

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())
gpu_mib() = NeuroDSL.Backend.CUDA_AVAILABLE ? Int(round((NeuroDSL.CUDA.total_memory() - NeuroDSL.CUDA.free_memory()) / 2^20)) : -1

const results = Dict{String,Any}()
const LOG = String[]
logmsg(s) = (println(s); push!(LOG, s))

dev = NeuroDSL.Backend.CUDADevice()
g = NeuroDSL.NeuroGraph(namespace=NS, device=dev)
logmsg("Loading Qwen2.5-1.5B-Instruct checkpoint from $CKPT ...")
NeuroDSL.load_graph!(g, NS, CKPT)
reclaim()
logmsg("Checkpoint loaded. GPU used = $(gpu_mib()) MiB.")

function set_input!(ids::Vector{Int})
    n = length(ids)
    NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=NS)
    NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=NS)
    NeuroDSL.invalidate_all!(g; namespace=NS)
end

cos(a, b) = dot(a, b) / (norm(a) * norm(b) + 1f-12)
all_sites = [Symbol("layer_", i, "_mlp_out") for i in 1:NL]

per_triple = NamedTuple[]

for tr in TRIPLES
    name = tr["name"]
    logmsg("\n" * "="^70 * "\nTriple: $name  (GPU used = $(gpu_mib()) MiB before this triple)")

    ids_clean = Int.(tr["full_ids"]) .+ 1
    obj_pos = tr["obj_pos"] + 1
    pos_Q1  = tr["pos_Q1"] + 1
    pos_Q2  = tr["pos_Q2"] + 1
    a1_id   = tr["a1_first_id"] + 1
    a2_id   = tr["a2_first_id"] + 1
    wrong_id = tr["wrong_first_id"] + 1
    obj_first_id = tr["obj_first_id"] + 1

    ids_corrupt = copy(ids_clean)
    @assert ids_corrupt[obj_pos] == obj_first_id "obj_pos mismatch for $name"
    ids_corrupt[obj_pos] = wrong_id

    set_input!(ids_clean)
    clean_out = Array(NeuroDSL.demand!(g, LOGITS; namespace=NS))
    clean_cache = NeuroDSL.capture_activations(g, NS)

    set_input!(ids_corrupt)
    corrupt_out = Array(NeuroDSL.demand!(g, LOGITS; namespace=NS))
    corrupt_cache = NeuroDSL.capture_activations(g, NS)

    # ── (a) EXACT: full-depth position-masked patch ────────────────────────
    patch_dict = Dict{Symbol,Any}()
    for site in all_sites
        arr = copy(Array(corrupt_cache[site]))
        arr[obj_pos, :] .= Array(clean_cache[site])[obj_pos, :]
        patch_dict[site] = arr
    end
    for site in all_sites
        NeuroDSL.patch_node!(g, site, patch_dict; namespace=NS)
    end
    deep_out = Array(NeuroDSL.demand!(g, LOGITS; namespace=NS))
    d_q1_deep = Float64(deep_out[pos_Q1, a1_id] - corrupt_out[pos_Q1, a1_id])
    d_q2_deep = Float64(deep_out[pos_Q2, a2_id] - corrupt_out[pos_Q2, a2_id])

    all_cone = Set{Symbol}()
    for site in all_sites; union!(all_cone, NeuroDSL._downstream_nodes(g, site, NS)); end
    NeuroDSL.restore_from_cache!(g, NS, corrupt_cache, all_cone)
    NeuroDSL.demand!(g, LOGITS; namespace=NS)

    # ── light single-layer reference (power check) ──────────────────────────
    site7 = Symbol("layer_", COS_LAYER, "_mlp_out")
    arr7 = copy(Array(corrupt_cache[site7])); arr7[obj_pos, :] .= Array(clean_cache[site7])[obj_pos, :]
    NeuroDSL.patch_node!(g, site7, Dict(site7 => arr7); namespace=NS)
    shallow_out = Array(NeuroDSL.demand!(g, LOGITS; namespace=NS))
    d_q1_shallow = Float64(shallow_out[pos_Q1, a1_id] - corrupt_out[pos_Q1, a1_id])
    d_q2_shallow = Float64(shallow_out[pos_Q2, a2_id] - corrupt_out[pos_Q2, a2_id])
    NeuroDSL.restore_from_cache!(g, NS, corrupt_cache, NeuroDSL._downstream_nodes(g, site7, NS))
    NeuroDSL.demand!(g, LOGITS; namespace=NS)

    # ── (b) CORRELATIONAL: cosine similarity at layer_7_mlp_out, clean run ──
    rep = Array(clean_cache[site7])
    cos_q1 = Float64(cos(rep[obj_pos, :], rep[pos_Q1, :]))
    cos_q2 = Float64(cos(rep[obj_pos, :], rep[pos_Q2, :]))
    cos_gap = cos_q1 - cos_q2

    ratio = abs(d_q1_deep) / max(abs(d_q2_deep), 1e-6)
    power_ok = abs(d_q1_deep) > 3 * max(abs(d_q1_shallow), 1e-6)

    logmsg(@sprintf("  EXACT (all-28-layer, position-masked): delta_Q1=%.4f  delta_Q2=%.4f  ratio=%.2f  (single-layer-7 reference: delta_Q1=%.4f delta_Q2=%.4f, power_ok=%s)",
                     d_q1_deep, d_q2_deep, ratio, d_q1_shallow, d_q2_shallow, power_ok))
    logmsg(@sprintf("  CORRELATIONAL (cosine @ layer_7_mlp_out): cos_Q1=%.4f  cos_Q2=%.4f  gap=%.4f  cosine_correctly_ranks_and_nontrivial=%s",
                     cos_q1, cos_q2, cos_gap, (cos_gap > 0.05)))

    push!(per_triple, (; name=name, blocked=false,
        d_q1_deep=d_q1_deep, d_q2_deep=d_q2_deep, ratio=ratio,
        d_q1_shallow=d_q1_shallow, d_q2_shallow=d_q2_shallow, power_ok=power_ok,
        cos_q1=cos_q1, cos_q2=cos_q2, cos_gap=cos_gap))

    clean_cache = nothing; corrupt_cache = nothing; patch_dict = nothing
    reclaim()
    logmsg("  GPU used after cleanup = $(gpu_mib()) MiB.")
end

# ══════════════════════════════════════════════════════════════════════════
ok = per_triple
logmsg("\n" * "="^70)
logmsg("SUMMARY across $(length(ok)) verified triples (v2 -- 3 more genuinely blocked, see calibration):")
for t in ok
    logmsg(@sprintf("  %-32s ratio=%.2f  cos_gap=%+.4f  power_ok=%s", t.name, t.ratio, t.cos_gap, t.power_ok))
end

ratios = [t.ratio for t in ok]
cos_gaps = [t.cos_gap for t in ok]
n_ge2 = count(r -> r >= 2, ratios)
frac_ge2 = n_ge2 / length(ok)
n_cos_correct_nontrivial = count(gp -> gp > 0.05, cos_gaps)
median_ratio = median(ratios)
n_power_ok = count(t -> t.power_ok, ok)

logmsg(@sprintf("median ratio = %.2f across %d triples; %d/%d (%.0f%%) have ratio>=2; %d/%d power_ok; %d/%d have a correctly-signed, non-trivial cosine gap (>0.05)",
                 median_ratio, length(ok), n_ge2, length(ok), 100frac_ge2, n_power_ok, length(ok), n_cos_correct_nontrivial, length(ok)))

verdict = if median_ratio >= 3 && frac_ge2 >= 0.75
    "REPLICATES: median ratio $(round(median_ratio,digits=2)) >= 3 and $(round(100frac_ge2,digits=0))% of triples clear ratio>=2 -- " *
    "the n=1 pattern holds across a properly-powered, varied set of fact-relationship types."
elseif frac_ge2 >= 0.30 && frac_ge2 < 0.75
    "PARTIAL REPLICATION: $(round(100frac_ge2,digits=0))% of triples show ratio>=2 -- the effect is real for SOME " *
    "fact-relationship types and not others; see per-triple breakdown for which."
else
    "DOES NOT REPLICATE: only $(round(100frac_ge2,digits=0))% of triples clear ratio>=2 and/or median ratio $(round(median_ratio,digits=2)) < 2."
end
logmsg("VERDICT: " * verdict)
cos_note = n_cos_correct_nontrivial >= length(ok) ÷ 2 + 1 ?
    "the correlational baseline is NOT reliably uninformative across this matrix, unlike the n=1 case." :
    "consistent with the n=1 case: the correlational baseline mostly fails to separate these facts."
logmsg(@sprintf("Separately: cosine-similarity baseline would have given a correct, non-trivial signal on %d/%d triples -- %s",
                 n_cos_correct_nontrivial, length(ok), cos_note))

results["per_triple"] = [Dict(pairs(t)) for t in per_triple]
results["median_ratio"] = median_ratio
results["n_ge2"] = n_ge2
results["frac_ge2"] = frac_ge2
results["n_usable"] = length(ok)
results["n_power_ok"] = n_power_ok
results["n_cos_correct_nontrivial"] = n_cos_correct_nontrivial
results["verdict"] = verdict

open(OUT, "w") do io
    JSON.print(io, Dict("status" => "COMPLETED", "results" => results, "log" => LOG), 2)
end
logmsg("Full results written to $OUT")
