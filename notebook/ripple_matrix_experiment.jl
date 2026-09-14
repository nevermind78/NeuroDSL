# ══════════════════════════════════════════════════════════════════════════════
# ripple_matrix_experiment.jl -- replication matrix for the n=1 positive result
# found in ripple_incontext_experiment.jl (Eiffel Tower/Louvre/gold triple):
# a full-depth, position-masked activation patch cleanly separated an
# entangled fact from a disjoint one by ~10x, while a CLaRE-style
# cosine-similarity baseline at the same site gave no usable signal.
#
# 5 fact triples, deliberately varying the RELATIONSHIP TYPE so this isn't
# just the same relationship tested five times:
#   1. eiffel_louvre_gold          shared-object (city=Paris)        vs chemistry
#   2. colosseum_trevi_jupiter     shared-object (city=Rome)         vs astronomy
#   3. silver_iron_tokyo           shared-CATEGORY (element symbols,
#                                   different specific element)      vs geography
#   4. everest_kathmandu_sodium    shared-object (country=Nepal)     vs chemistry
#   5. shakespeare_hamlet_pacific  shared-object (author=Shakespeare) vs geography
#
# PRE-REGISTERED, WRITTEN BEFORE LOOKING AT ANY RESULT BELOW
# ------------------------------------------------------------------------------
# Per triple, two metrics, same site (layer_7_mlp_out for cosine, all 28
# layers' mlp_out for the exact patch -- same choices as the n=1 experiment):
#   (a) EXACT: delta logit(entangled answer)@Q1 vs delta logit(disjoint
#       answer)@Q2 from the full-depth position-masked patch (only obj_pos's
#       row restored to clean at every layer, everything else left at the
#       corrupted run's values). ratio = |delta_Q1| / max(|delta_Q2|, 1e-6).
#   (b) CORRELATIONAL: cosine(obj_pos, Q1) - cosine(obj_pos, Q2) at
#       layer_7_mlp_out, clean run -- positive means cosine ALSO ranks the
#       entangled fact higher (agrees with (a)'s direction); near zero or
#       negative means the correlational baseline gives no useful/wrong signal.
#
# REPLICATES   : median ratio across the 5 triples >= 3 AND at least 4/5
#                triples individually have ratio >= 2.
# PARTIAL      : 2-3/5 triples have ratio >= 2 (some fact types show the
#                effect, others don't) -- itself a real, reportable finding,
#                not a failure to hide.
# DOES NOT REPLICATE : <=1/5 triples have ratio >= 2, or the median ratio < 2,
#                or the direction flips (disjoint effect bigger than
#                entangled's on most triples).
# Reported SEPARATELY (not folded into the verdict above): the fraction of
# triples where the cosine baseline's gap is both correctly-signed AND
# non-trivial (> 0.05) -- i.e. where CLaRE-style similarity would actually
# have worked fine, which the n=1 case did not show.
#
# Each fact triple is verified known top-1 by Qwen2.5-1.5B-Instruct IN THIS
# EXACT MULTI-FACT CONTEXT before being used, exactly as in the two prior
# scripts -- no fact is assumed.
#
# VRAM: per the coordinator's explicit request after the prior two runs
# (15.7GB peak observed once), this script calls reclaim() between every
# triple and explicitly drops each triple's activation caches before moving
# to the next one, rather than accumulating 5 caches' worth of a 1.5B-param
# model's per-layer activations at once.
#
# USAGE: julia --project=. notebook/ripple_matrix_experiment.jl
# WRITES: notebook/ripple_matrix_experiment_results.json
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, LinearAlgebra, Statistics

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const TRIPLES   = JSON.parsefile(joinpath(@__DIR__, "ripple_matrix_prompts.json"))
const OUT       = joinpath(@__DIR__, "ripple_matrix_experiment_results.json")
const NS, NL    = :qwen2, 28
const LOGITS    = :lm_head_out
const COS_LAYER = 7   # same site as the n=1 experiment, for direct comparability

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

    q1_top1 = argmax(clean_out[pos_Q1, :]) - 1
    q2_top1 = argmax(clean_out[pos_Q2, :]) - 1
    q1_known = q1_top1 == (a1_id - 1)
    q2_known = q2_top1 == (a2_id - 1)
    logmsg("  Fact check: Q1 top1_hf=$q1_top1 (expect $(a1_id-1)) known=$q1_known logit=$(round(clean_out[pos_Q1,a1_id],digits=3)); " *
           "Q2 top1_hf=$q2_top1 (expect $(a2_id-1)) known=$q2_known logit=$(round(clean_out[pos_Q2,a2_id],digits=3))")

    if !(q1_known && q2_known)
        logmsg("  !!! BLOCKED: fact(s) not known top-1 in this context for triple '$name' -- skipping, not silently substituting.")
        push!(per_triple, (; name=name, blocked=true))
        clean_cache = nothing; reclaim()
        continue
    end

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

    # ── light single-layer reference (same power-check pattern as n=1) ──────
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
    power_ok = abs(d_q1_deep) > 3 * max(abs(d_q1_shallow), 1e-6)   # same qualitative power check as n=1,
                                                                     # relaxed to 3x (was 5x) since this is now
                                                                     # a secondary/reference check, not the
                                                                     # main pre-registered metric.

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
# Aggregate, against the criteria pre-registered above.
# ══════════════════════════════════════════════════════════════════════════
ok = filter(t -> !t.blocked, per_triple)
blocked = filter(t -> t.blocked, per_triple)
logmsg("\n" * "="^70)
logmsg("SUMMARY across $(length(per_triple)) triples ($(length(blocked)) blocked on fact-verification, $(length(ok)) usable):")
for t in ok
    logmsg(@sprintf("  %-28s ratio=%.2f  cos_gap=%+.4f  power_ok=%s", t.name, t.ratio, t.cos_gap, t.power_ok))
end
for t in blocked
    logmsg("  $(t.name)  -- BLOCKED (fact not known top-1 in context)")
end

ratios = [t.ratio for t in ok]
cos_gaps = [t.cos_gap for t in ok]
n_ge2 = count(r -> r >= 2, ratios)
n_cos_correct_nontrivial = count(g -> g > 0.05, cos_gaps)
median_ratio = isempty(ratios) ? NaN : median(ratios)

logmsg(@sprintf("median ratio = %.2f across %d usable triples; %d/%d have ratio>=2; %d/%d triples have a correctly-signed, non-trivial cosine gap (>0.05)",
                 median_ratio, length(ok), n_ge2, length(ok), n_cos_correct_nontrivial, length(ok)))

verdict = if length(ok) < 3
    "INCONCLUSIVE: too few triples passed fact-verification ($(length(ok))/$(length(per_triple))) to judge replication at all."
elseif median_ratio >= 3 && n_ge2 >= 4
    "REPLICATES: median ratio $(round(median_ratio,digits=2)) >= 3 and $n_ge2/$(length(ok)) triples individually clear ratio>=2 -- " *
    "the n=1 pattern (exact patch cleanly separates entangled from disjoint) holds across varied relationship types, " *
    "not a lucky single case."
elseif n_ge2 >= 2 && n_ge2 <= 3
    "PARTIAL REPLICATION: $n_ge2/$(length(ok)) triples show ratio>=2, the rest do not -- the effect is real for SOME " *
    "fact-relationship types and not others; see per-triple breakdown for which."
else
    "DOES NOT REPLICATE: only $n_ge2/$(length(ok)) triples clear ratio>=2 and/or median ratio $(round(median_ratio,digits=2)) < 2 -- " *
    "the n=1 Eiffel/Louvre/gold result does not generalize as a reliable pattern across this small matrix."
end
logmsg("VERDICT: " * verdict)
cos_note = n_cos_correct_nontrivial >= length(ok) ÷ 2 + 1 ?
    "the correlational baseline is NOT reliably uninformative across this matrix, unlike the n=1 case." :
    "consistent with the n=1 case: the correlational baseline mostly fails to separate these facts."
logmsg(@sprintf("Separately: cosine-similarity baseline would have given a correct, non-trivial signal on %d/%d usable triples -- %s",
                 n_cos_correct_nontrivial, length(ok), cos_note))

results["per_triple"] = [Dict(pairs(t)) for t in per_triple]
results["median_ratio"] = median_ratio
results["n_ge2"] = n_ge2
results["n_usable"] = length(ok)
results["n_blocked"] = length(blocked)
results["n_cos_correct_nontrivial"] = n_cos_correct_nontrivial
results["verdict"] = verdict

open(OUT, "w") do io
    JSON.print(io, Dict("status" => "COMPLETED", "results" => results, "log" => LOG), 2)
end
logmsg("Full results written to $OUT")
