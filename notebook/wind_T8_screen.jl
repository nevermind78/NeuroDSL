# WIND-T8 -- criblage des prompts candidats : top-1 = réponse attendue ET marge logit(réponse) − logit(contrefactuel) ≥ 2.
# USAGE : julia --project=. notebook/wind_T8_screen.jl
using NeuroDSL, JSON, Printf
const CKPT = joinpath(@__DIR__, "qwen2.5-1.5b-instruct", "qwen2_neurodsl")
cands = JSON.parsefile(joinpath(@__DIR__, "wind_T8_candidates.json"))
dev = NeuroDSL.Backend.CUDADevice(); ns = :qwen2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev); NeuroDSL.load_graph!(g, ns, CKPT)
out = Any[]
for (i, c) in enumerate(cands)
    ids = Int.(c["token_ids"]) .+ 1; n = length(ids)
    NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    lg = Float64.(Array(NeuroDSL.demand!(g, :lm_head_out; namespace=ns))[n, :])
    top1 = argmax(lg) - 1; a = c["answer_id"]; cf = c["cf_id"]
    margin = lg[a+1] - lg[cf+1]; ok = top1 == a && margin >= 2
    push!(out, merge(c, Dict("cand_idx" => i, "top1_id" => top1, "margin" => margin, "pass" => ok)))
    @printf("%2d %-9s %-5s marge %6.2f  %s\n", i, c["task"], ok ? "OK" : "rejet", margin, repr(c["prompt"]))
end
open(joinpath(@__DIR__, "wind_T8_screen_results.json"), "w") do f; JSON.print(f, out); end
println("passent : ", count(x -> x["pass"], out), " / ", length(out))
