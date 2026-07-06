include(joinpath(@__DIR__, "surgery_f3_common.jl"))

dev = NeuroDSL.Backend.CPUDevice()
ns = :surgery_f3
graph_prefix = ARGS[1]
losses_path = ARGS[2]
broken_resume = length(ARGS) >= 3 && ARGS[3] == "broken"

# Nouveau processus Julia, graphe totalement vierge -- ne rappelle PAS build_induction_graph,
# preuve que la structure ET l'état d'optimiseur viennent bien du fichier sauvegardé.
g = NeuroDSL.NeuroGraph(device=dev)
opt_state = NeuroDSL.load_graph!(g, ns, graph_prefix)
logits = :lm_head_out

@assert opt_state !== nothing "AdamWState absent du fichier chargé"
@assert opt_state.t == 400

ps = NeuroDSL.params(g; namespace=ns)
if broken_resume
    # Variante DÉLIBÉRÉMENT CASSÉE pour vérifier que la comparaison F3 n'est pas vide :
    # ignore l'AdamWState chargé, repart de moments à zéro.
    m1 = Dict(p.name => zeros(Float32, size(p.value)...) for p in ps)
    m2 = Dict(p.name => zeros(Float32, size(p.value)...) for p in ps)
else
    m1, m2 = opt_state.m1, opt_state.m2
end

losses = Float64[]
for t in 401:500
    push!(losses, train_step!(g, ns, logits, m1, m2, t, ps))
end

open(losses_path, "w") do io
    for l in losses
        println(io, l)
    end
end
println(broken_resume ? "SCRIPT_B_BROKEN_OK" : "SCRIPT_B_OK", "  t=500")
