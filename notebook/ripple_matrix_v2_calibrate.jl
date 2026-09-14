# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION (phase 1/2) -- ripple-effect replication matrix v2
#
# POURQUOI CETTE PHASE EXISTE, SEPAREE DE LA MESURE CAUSALE
# -----------------------------------------------------------------------------
# v1 (ripple_matrix_experiment.jl) bloquait un triplet des qu'un token de
# reponse DEVINE A L'AVANCE n'etait pas le top-1 du modele -- et a perdu 3/5
# triplets ainsi, dont au moins un cas ou la reponse etait probablement
# correcte sous une autre forme de surface (voir l'historique de session :
# "Shakespeare" attendu, le modele repond vraisemblablement "William", le
# prenom, avant le nom de famille).
#
# Cette phase ne calcule RIEN de causal. Elle charge le modele UNE fois, fait
# UN forward propre par candidat, et enregistre ce que le modele repond
# REELLEMENT (id + logit) aux deux positions de lecture -- sans jamais le
# comparer a une hypothese. La verification semantique (est-ce que cette
# reponse est correcte) se fait ENSUITE, par un humain lisant le texte
# decode, pas ici. Aucun triplet n'est ecarte a ce stade.
#
# USAGE : julia --project=. notebook/ripple_matrix_v2_calibrate.jl
# ECRIT  : notebook/ripple_matrix_v2_calibration.json
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, Printf

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const CANDS     = JSON.parsefile(joinpath(@__DIR__, "ripple_matrix_v2_candidates.json"))
const OUT       = joinpath(@__DIR__, "ripple_matrix_v2_calibration.json")
const NS, NL    = :qwen2, 28
const LOGITS    = :lm_head_out

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())

dev = NeuroDSL.Backend.CUDADevice()
g = NeuroDSL.NeuroGraph(namespace=NS, device=dev)
println("Chargement du checkpoint natif...")
NeuroDSL.load_graph!(g, NS, CKPT)
reclaim()
println("Checkpoint chargé.")

function set_input!(ids::Vector{Int})
    n = length(ids)
    NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=NS)
    NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=NS)
    NeuroDSL.invalidate_all!(g; namespace=NS)
end

rows = []
for tr in CANDS
    name = tr["name"]
    ids = Int.(tr["full_ids"]) .+ 1
    pos_Q1 = tr["pos_Q1"] + 1
    pos_Q2 = tr["pos_Q2"] + 1

    set_input!(ids)
    out = Array(NeuroDSL.demand!(g, LOGITS; namespace=NS))

    q1_top1 = argmax(out[pos_Q1, :]) - 1     # 0-indexé HF
    q2_top1 = argmax(out[pos_Q2, :]) - 1
    q1_logit = Float64(out[pos_Q1, q1_top1 + 1])
    q2_logit = Float64(out[pos_Q2, q2_top1 + 1])
    hyp1 = tr["a1_hypothesis_id"]
    hyp2 = tr["a2_hypothesis_id"]
    hyp1_logit = Float64(out[pos_Q1, hyp1 + 1])
    hyp2_logit = Float64(out[pos_Q2, hyp2 + 1])
    matches_hyp1 = q1_top1 == hyp1
    matches_hyp2 = q2_top1 == hyp2

    @printf("%-32s Q1: top1=%-7d (hyp=%-7d match=%-5s logit=%.3f vs hyp_logit=%.3f) | Q2: top1=%-7d (hyp=%-7d match=%-5s logit=%.3f vs hyp_logit=%.3f)\n",
            name, q1_top1, hyp1, matches_hyp1, q1_logit, hyp1_logit,
            q2_top1, hyp2, matches_hyp2, q2_logit, hyp2_logit)

    push!(rows, Dict(
        "name" => name,
        "q1_top1_hf" => q1_top1, "q1_top1_logit" => q1_logit,
        "q2_top1_hf" => q2_top1, "q2_top1_logit" => q2_logit,
        "hyp1_id" => hyp1, "hyp1_logit" => hyp1_logit, "matches_hyp1" => matches_hyp1,
        "hyp2_id" => hyp2, "hyp2_logit" => hyp2_logit, "matches_hyp2" => matches_hyp2,
    ))
    reclaim()
end

open(OUT, "w") do io
    JSON.print(io, rows, 2)
end
println("\nÉcrit : ", OUT)
println("Prochaine étape : décoder les top1 réels en texte (Python) et vérifier")
println("sémantiquement chaque réponse avant de lancer la mesure causale.")
