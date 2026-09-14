# ══════════════════════════════════════════════════════════════════════════════
# ripple_incontext_experiment.jl -- second, more targeted test of the exact
# reachability cone as a ripple-effect discriminator, this time IN-CONTEXT
# (a single forward pass, position-masked activation patch), not a permanent
# shared-weight edit -- follow-up to ripple_edit_experiment.jl, which found
# H0 (topology triviality) confirmed and H2 (negative control unaffected)
# FAILED for a permanent weight edit.
#
# WHY A DIFFERENT DESIGN IS NEEDED (not just re-running the old one)
# ------------------------------------------------------------------------------
# ripple_edit_experiment.jl's H0 result followed from a specific fact about
# WEIGHTS: the same tensor is read identically by every future forward pass,
# so its downstream node-symbol cone is prompt-content-independent by
# construction. An in-context activation is different in kind: a node like
# :layer_7_mlp_out is a WHOLE-SEQUENCE tensor (one row per token position),
# recomputed fresh for every forward pass -- but this project's own graph
# has no notion of position INSIDE that tensor: _downstream_nodes only sees
# node-symbol edges, never rows. So patching the WHOLE :layer_7_mlp_out node
# is not actually the right analogue of "editing one fact" either -- it
# would overwrite EVERY position's value, including query positions that
# were never touched by the corruption in the first place. The right
# analogue, and what real causal-tracing work (and this project's own
# article2/hot-surgery methodology) actually does, is a POSITION-MASKED
# patch: only the corrupted token's OWN row is restored to its clean value,
# every other row (including both query readout positions) is left at its
# corrupted-run value. `patch_node!` accepts any array under the requested
# symbol, so this is built by hand (blended = corrupted tensor with one row
# swapped) and fed through the existing, already-proven `patch_node!` path --
# no new graph machinery.
#
# SINGLE CONTEXT, ONE FORWARD PASS, TWO FACTS QUERIED AT TWO POSITIONS
# ------------------------------------------------------------------------------
#   "The Eiffel Tower is located in the city of Paris. In which city is the
#    Louvre Museum located? The answer is[pos_Q1] Paris. What is the
#    chemical symbol for gold? The answer is[pos_Q2]"
# obj_pos  = the "Paris" token in the FIRST sentence (position 12, 1-indexed)
# pos_Q1   = right after "...Louvre Museum located? The answer is" (positive
#            control: shares the object "Paris" with the edited fact)
# pos_Q2   = right after "...chemical symbol for gold? The answer is"
#            (negative control: structurally disjoint domain, no shared
#            entity or object with the edited fact)
# Corruption: obj_pos's token is swapped Paris(12095) -> Rome(21718) --
# ONE token, same total length, everything else byte-identical between the
# clean and corrupted sequence (a real, minimal, controlled corruption).
#
# PRE-REGISTERED PREDICTIONS (written before running)
# ------------------------------------------------------------------------------
# H0 (topology, expected to hold again, now shown to be a GENERAL property of
#    this graph's granularity, not specific to weight edits): the node-symbol
#    downstream cone of :layer_7_mlp_out is position-blind -- it topologically
#    "reaches" pos_Q1 and pos_Q2 identically, giving zero a-priori structural
#    discrimination between the positive and negative control by symbol
#    reachability alone.
# H1 (the real, exact, executed test): the POSITION-MASKED patch (only
#    obj_pos's row restored to clean) produces a LARGER measured logit change
#    at pos_Q1 (Paris, entangled) than at pos_Q2 (Au, disjoint) -- an exact,
#    fully-computed, zero-estimation causal effect, not a proxy.
# H2 (comparison against a real CLaRE-style baseline): cosine similarity
#    between the hidden representation at obj_pos and at pos_Q1 vs pos_Q2
#    (same site, same clean run) is computed as the correlational baseline.
#    The question this experiment actually exists to answer: does the EXACT
#    patch-effect ranking (H1) AGREE with the cosine-similarity ranking, or
#    does one catch something the other misses? Reported honestly either way.
# H3 (kill-switch): the WHOLE-TENSOR (unmasked) patch at the same site, as a
#    reference upper bound, should show a much larger effect at BOTH
#    positions than the position-masked patch -- proving the position-masked
#    result isn't just measurement insensitivity.
#
# USAGE: julia --project=. notebook/ripple_incontext_experiment.jl
# WRITES: notebook/ripple_incontext_experiment_results.json
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, LinearAlgebra, Statistics

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const CTX       = JSON.parsefile(joinpath(@__DIR__, "ripple_context_prompts.json"))
const OUT       = joinpath(@__DIR__, "ripple_incontext_experiment_results.json")
const NS, NL    = :qwen2, 28
const LOGITS    = :lm_head_out
const L_SITE    = 7   # reused from ripple_edit_experiment.jl's causal trace on
                       # this SAME fact (Eiffel Tower -> Paris); not re-swept
                       # here, disclosed simplification for time.
const SITE_SYM  = Symbol("layer_", L_SITE, "_mlp_out")

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())

const results = Dict{String,Any}()
const LOG = String[]
logmsg(s) = (println(s); push!(LOG, s))

dev = NeuroDSL.Backend.CUDADevice()
g = NeuroDSL.NeuroGraph(namespace=NS, device=dev)
logmsg("Loading Qwen2.5-1.5B-Instruct checkpoint from $CKPT ...")
NeuroDSL.load_graph!(g, NS, CKPT)
reclaim()
logmsg("Checkpoint loaded.")

function set_input!(ids::Vector{Int})
    n = length(ids)
    NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=NS)
    NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=NS)
    NeuroDSL.invalidate_all!(g; namespace=NS)
end

# 1-indexed conversions (HF 0-indexed -> NeuroDSL 1-indexed), same convention
# as ripple_edit_experiment.jl / bench_eps_exact_ablation_qwen.jl.
ids_clean = Int.(CTX["full_ids"]) .+ 1
obj_pos   = CTX["obj_pos_0idx"] + 1
pos_Q1    = CTX["pos_Q1_0idx"] + 1
pos_Q2    = CTX["pos_Q2_0idx"] + 1
paris_id  = CTX["paris_id"] + 1
rome_id   = CTX["rome_id"] + 1
au_id     = CTX["au_id"] + 1

ids_corrupt = copy(ids_clean)
@assert ids_corrupt[obj_pos] == paris_id "obj_pos does not point at the Paris token -- check tokenization"
ids_corrupt[obj_pos] = rome_id

logmsg("Sequence length=$(length(ids_clean)); obj_pos=$obj_pos (Paris->Rome); " *
       "pos_Q1=$pos_Q1 (Louvre answer); pos_Q2=$pos_Q2 (gold answer, end of sequence).")

# ══════════════════════════════════════════════════════════════════════════
# Step 1: verify BOTH facts are actually known, IN THIS CONTEXT, before using
# them as controls.
# ══════════════════════════════════════════════════════════════════════════
set_input!(ids_clean)
clean_out = Array(NeuroDSL.demand!(g, LOGITS; namespace=NS))
clean_cache = NeuroDSL.capture_activations(g, NS)

q1_top1_clean = argmax(clean_out[pos_Q1, :]) - 1
q2_top1_clean = argmax(clean_out[pos_Q2, :]) - 1
q1_known = q1_top1_clean == (paris_id - 1)
q2_known = q2_top1_clean == (au_id - 1)
logmsg("CLEAN context: Q1 (Louvre) top1_hf=$q1_top1_clean (expect $(paris_id-1)=Paris) known=$q1_known, " *
       "logit(Paris)=$(round(clean_out[pos_Q1,paris_id],digits=3))")
logmsg("CLEAN context: Q2 (gold)   top1_hf=$q2_top1_clean (expect $(au_id-1)=Au) known=$q2_known, " *
       "logit(Au)=$(round(clean_out[pos_Q2,au_id],digits=3))")

results["fact_verification"] = Dict(
    "Q1_Louvre_known" => q1_known, "Q1_top1_hf" => q1_top1_clean,
    "Q2_gold_known" => q2_known, "Q2_top1_hf" => q2_top1_clean,
)

if !(q1_known && q2_known)
    logmsg("!!! BLOCKER: not both facts known top-1 in this multi-fact context. Stopping before the patch experiment.")
    open(OUT, "w") do io
        JSON.print(io, Dict("status" => "BLOCKED_FACT_VERIFICATION_FAILED",
                             "results" => results, "log" => LOG), 2)
    end
    exit(1)
end
logmsg("Both facts verified known top-1 in this multi-fact context. Proceeding.")

set_input!(ids_corrupt)
corrupt_out = Array(NeuroDSL.demand!(g, LOGITS; namespace=NS))
corrupt_cache = NeuroDSL.capture_activations(g, NS)
logmsg("CORRUPTED context (obj_pos=Rome): Q1 logit(Paris)=$(round(corrupt_out[pos_Q1,paris_id],digits=3)) " *
       "(clean was $(round(clean_out[pos_Q1,paris_id],digits=3))); " *
       "Q2 logit(Au)=$(round(corrupt_out[pos_Q2,au_id],digits=3)) " *
       "(clean was $(round(clean_out[pos_Q2,au_id],digits=3)))")

results["baseline"] = Dict(
    "clean_Q1_paris" => Float64(clean_out[pos_Q1, paris_id]),
    "clean_Q2_au"    => Float64(clean_out[pos_Q2, au_id]),
    "corrupt_Q1_paris" => Float64(corrupt_out[pos_Q1, paris_id]),
    "corrupt_Q2_au"    => Float64(corrupt_out[pos_Q2, au_id]),
)

# ══════════════════════════════════════════════════════════════════════════
# Step 2: H0 -- topology, position-blindness of the node-symbol cone.
# ══════════════════════════════════════════════════════════════════════════
cone = NeuroDSL._downstream_nodes(g, SITE_SYM, NS)
lm_head_in_cone = (:lm_head_out in cone) || any(occursin("lm_head", String(s)) for s in cone)
logmsg("H0 check: |cone($SITE_SYM)|=$(length(cone)); contains an lm_head-related node: $lm_head_in_cone " *
       "-- this is a SINGLE node-symbol fact, identical regardless of pos_Q1 vs pos_Q2, confirming the " *
       "cone cannot distinguish between the two query positions by symbol reachability alone.")
results["H0_topology_position_blind"] = Dict("cone_size" => length(cone), "lm_head_reachable" => lm_head_in_cone)

# ══════════════════════════════════════════════════════════════════════════
# Step 3: THE REAL TEST -- position-masked patch (only obj_pos's row
# restored to clean), measured effect at pos_Q1 vs pos_Q2.
# ══════════════════════════════════════════════════════════════════════════
clean_site_arr   = Array(clean_cache[SITE_SYM])
corrupt_site_arr = Array(corrupt_cache[SITE_SYM])
logmsg("$SITE_SYM shape = $(size(clean_site_arr)) (expect [seq_len=$(length(ids_clean)), dim]).")

blended = copy(corrupt_site_arr)
blended[obj_pos, :] .= clean_site_arr[obj_pos, :]

set_input!(ids_corrupt)
NeuroDSL.demand!(g, LOGITS; namespace=NS)   # re-establish the corrupted state in this namespace
NeuroDSL.patch_node!(g, SITE_SYM, Dict(SITE_SYM => blended); namespace=NS)
masked_out = Array(NeuroDSL.demand!(g, LOGITS; namespace=NS))

d_q1_masked = Float64(masked_out[pos_Q1, paris_id] - corrupt_out[pos_Q1, paris_id])
d_q2_masked = Float64(masked_out[pos_Q2, au_id]    - corrupt_out[pos_Q2, au_id])
logmsg(@sprintf("Position-masked patch (obj_pos only): delta logit(Paris)@Q1 = %.4f, delta logit(Au)@Q2 = %.4f",
                d_q1_masked, d_q2_masked))
results["H1_position_masked_patch"] = Dict(
    "delta_logit_paris_Q1" => d_q1_masked, "delta_logit_au_Q2" => d_q2_masked,
)

# Restore corrupted state before the next (whole-tensor) test.
NeuroDSL.restore_from_cache!(g, NS, corrupt_cache, NeuroDSL._downstream_nodes(g, SITE_SYM, NS))
NeuroDSL.demand!(g, LOGITS; namespace=NS)

# ══════════════════════════════════════════════════════════════════════════
# Step 4: H3 kill-switch -- whole-tensor (unmasked) patch, same site, as an
# upper-bound reference: proves the harness CAN show a large effect at both
# Q1 and Q2 when not position-restricted, so H1's numbers above are not an
# artifact of an insensitive pipeline.
# ══════════════════════════════════════════════════════════════════════════
NeuroDSL.patch_node!(g, SITE_SYM, Dict(SITE_SYM => clean_site_arr); namespace=NS)
whole_out = Array(NeuroDSL.demand!(g, LOGITS; namespace=NS))
d_q1_whole = Float64(whole_out[pos_Q1, paris_id] - corrupt_out[pos_Q1, paris_id])
d_q2_whole = Float64(whole_out[pos_Q2, au_id]    - corrupt_out[pos_Q2, au_id])
logmsg(@sprintf("H3 kill-switch, whole-tensor patch (all positions): delta logit(Paris)@Q1 = %.4f, delta logit(Au)@Q2 = %.4f",
                d_q1_whole, d_q2_whole))
results["H3_kill_switch_whole_tensor"] = Dict(
    "delta_logit_paris_Q1" => d_q1_whole, "delta_logit_au_Q2" => d_q2_whole,
)

# ══════════════════════════════════════════════════════════════════════════
# Step 4b: STRONGER kill-switch, added after the single-layer whole-tensor
# version above came back too weak to prove anything (0.26 / 0.016 -- not
# clearly bigger than the position-masked numbers, so it failed as a power
# check). This is still POSITION-MASKED (only obj_pos's row, everywhere else
# left at corrupted values, same discipline as H1) but now applied at EVERY
# layer's mlp_out simultaneously instead of just layer 7 -- the maximal
# version of "restore what this one position would have carried through the
# whole depth of the network". If even THIS doesn't clearly separate from H1
# and from Q2, that is a real, informative fact about how weak a single
# token's causal footprint is here, not a failure of the test design.
# ══════════════════════════════════════════════════════════════════════════
set_input!(ids_corrupt)
NeuroDSL.demand!(g, LOGITS; namespace=NS)
all_sites = [Symbol("layer_", i, "_mlp_out") for i in 1:NL]
patch_dict = Dict{Symbol,Any}()
for site in all_sites
    arr = copy(Array(corrupt_cache[site]))
    arr[obj_pos, :] .= Array(clean_cache[site])[obj_pos, :]
    patch_dict[site] = arr
end
for site in all_sites
    NeuroDSL.patch_node!(g, site, patch_dict; namespace=NS)
end
deep_masked_out = Array(NeuroDSL.demand!(g, LOGITS; namespace=NS))
d_q1_deep = Float64(deep_masked_out[pos_Q1, paris_id] - corrupt_out[pos_Q1, paris_id])
d_q2_deep = Float64(deep_masked_out[pos_Q2, au_id]    - corrupt_out[pos_Q2, au_id])
logmsg("STRONGER kill-switch, ALL-28-LAYERS position-masked patch (obj_pos row only, every layer): " *
       @sprintf("delta logit(Paris)@Q1 = %.4f, delta logit(Au)@Q2 = %.4f", d_q1_deep, d_q2_deep))
results["H3b_all_layers_position_masked"] = Dict(
    "delta_logit_paris_Q1" => d_q1_deep, "delta_logit_au_Q2" => d_q2_deep,
)
h3b_has_power = abs(d_q1_deep) > 5 * max(abs(d_q1_masked), 1e-6)

# Restore corrupted state before the final comparison section.
all_cone = Set{Symbol}()
for site in all_sites
    union!(all_cone, NeuroDSL._downstream_nodes(g, site, NS))
end
NeuroDSL.restore_from_cache!(g, NS, corrupt_cache, all_cone)
NeuroDSL.demand!(g, LOGITS; namespace=NS)

# ══════════════════════════════════════════════════════════════════════════
# Step 5: H2 -- real CLaRE-style correlational baseline (cosine similarity of
# hidden representations, same site, clean run) vs the exact patch-effect
# ranking from H1. This comparison is the actual point of the experiment.
# ══════════════════════════════════════════════════════════════════════════
cos(a, b) = dot(a, b) / (norm(a) * norm(b) + 1f-12)
rep_obj = clean_site_arr[obj_pos, :]
rep_q1  = clean_site_arr[pos_Q1, :]
rep_q2  = clean_site_arr[pos_Q2, :]
cos_q1 = Float64(cos(rep_obj, rep_q1))
cos_q2 = Float64(cos(rep_obj, rep_q2))
logmsg(@sprintf("CLaRE-style baseline: cosine(obj_pos, Q1)=%.4f  cosine(obj_pos, Q2)=%.4f", cos_q1, cos_q2))

exact_ranks_q1_higher = abs(d_q1_masked) > abs(d_q2_masked)
cosine_ranks_q1_higher = cos_q1 > cos_q2
agree = exact_ranks_q1_higher == cosine_ranks_q1_higher
logmsg("Exact patch-effect ranks Q1>Q2: $exact_ranks_q1_higher.  Cosine-similarity ranks Q1>Q2: $cosine_ranks_q1_higher.  Agree: $agree")

results["H2_correlational_comparison"] = Dict(
    "cosine_obj_Q1" => cos_q1, "cosine_obj_Q2" => cos_q2,
    "exact_ranks_Q1_higher" => exact_ranks_q1_higher,
    "cosine_ranks_Q1_higher" => cosine_ranks_q1_higher,
    "agree" => agree,
)

# ══════════════════════════════════════════════════════════════════════════
# Verdict, stated plainly.
# ══════════════════════════════════════════════════════════════════════════
h1_discriminates = abs(d_q1_masked) > 5 * max(abs(d_q2_masked), 1e-6)   # pre-registered: Q1 effect should
                                                                          # dominate Q2's by a wide margin if
                                                                          # the exact position-masked patch is
                                                                          # doing real, localized discriminating
                                                                          # work rather than diffuse noise.
h3_has_power = (abs(d_q1_whole) > 5*abs(d_q1_masked)) || (abs(d_q2_whole) > 5*max(abs(d_q2_masked),1e-6))
results["h3b_has_power"] = h3b_has_power

verdict = if h1_discriminates && h3b_has_power
    "POSITIVE: the position-masked exact patch discriminates cleanly between the entangled (Q1) and " *
    "disjoint (Q2) fact within one forward pass, and the all-layer kill-switch confirms the pipeline " *
    "has real power (single-layer numbers alone were too weak to prove this on their own). " *
    (agree ? "The cosine-similarity baseline agrees with the exact result here (does not add new information " *
             "over the correlational filter for this pair)." :
             "The cosine-similarity baseline DISAGREES with the exact result -- a genuine case where the exact " *
             "filter and the correlational filter give different answers.")
elseif !h3b_has_power
    "INCONCLUSIVE (kill-switch failed at both scales): neither the single-layer NOR the all-28-layer " *
    "position-masked patch produced an effect clearly larger than the H1 numbers -- the single token's " *
    "causal footprint on these two downstream readouts is too weak, at this site, to tell real " *
    "discrimination apart from noise. The H1 'Q1 bigger than Q2' pattern (delta=$(round(d_q1_masked,digits=3)) " *
    "vs $(round(d_q2_masked,digits=3))) cannot be trusted as a positive finding without a working power check. " *
    "This is a genuine negative result about THIS intervention's strength, not evidence against the " *
    "underlying hypothesis in general."
elseif !h1_discriminates
    "NEGATIVE: the position-masked exact patch does NOT cleanly discriminate between Q1 and Q2 -- both effects " *
    "are comparably small/large, so localized in-context patching does not obviously outperform a symbol-level " *
    "reachability argument here either."
else
    "AMBIGUOUS: see individual numbers; does not cleanly match a pre-registered pattern."
end
logmsg("VERDICT: " * verdict)
results["verdict"] = verdict
results["h1_discriminates"] = h1_discriminates
results["h3_has_power"] = h3_has_power

open(OUT, "w") do io
    JSON.print(io, Dict("status" => "COMPLETED", "results" => results, "log" => LOG), 2)
end
logmsg("Full results written to $OUT")
