#=
Vérification structurelle de la conjecture "plafond ≈2" sur de nombreuses
configurations, SANS chronométrage GPU -- seule la structure du graphe
(taille du cône aval de chaque site) entre en jeu, via _downstream_nodes
(BFS pur, déjà prouvé correct). But : confirmer que N_total/moyenne(cône)
-> 2 en grandissant la PROFONDEUR (n_layers), et est INDÉPENDANT de la
LARGEUR (dim/hidden_dim/n_heads) -- exactement ce que prédit la dérivation
(la moyenne d'une suite arithmétique décroissante de N_total à 0 vaut
N_total/2, quelle que soit la largeur qui détermine N_total lui-même).
=#
using NeuroDSL, Random

dev = NeuroDSL.Backend.CPUDevice()

function structural_ratio(n_layers, dim, n_heads, hidden_dim, seq_len)
    ns = Symbol(:struct_test_, n_layers, :_, dim)
    Random.seed!(1)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.set!(g, :input, (NeuroDSL.Backend.rand32(dev, seq_len, dim) .- 0.5f0); namespace=ns)
    out = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim; batched_attn=true)(g, :input; namespace=ns)
    NeuroDSL.demand!(g, out; namespace=ns)  # nécessaire pour peupler les règles/consommateurs

    layer_prefixes = ["layer_$i" for i in 1:n_layers]
    sites = vcat(
        [Symbol(lp, "_mha_ao_h", h) for lp in layer_prefixes for h in 1:n_heads],
        [Symbol(lp, "_mlp_out") for lp in layer_prefixes],
    )
    cones = [length(NeuroDSL._downstream_nodes(g, s, ns)) for s in sites]
    n_total = length(g.nodes[ns])
    return n_total, sum(cones)/length(cones), n_total / (sum(cones)/length(cones))
end

println("── Balayage en PROFONDEUR, largeur fixe (dim=256, n_heads=4, hidden=683) ──")
println(rpad("n_layers",10), rpad("N_total",10), rpad("moy(cône)",12), "N_total/moy(cône)")
for n_layers in [3, 6, 12, 24, 48, 96, 192, 384]
    n_total, mean_cone, ratio_pred = structural_ratio(n_layers, 256, 4, 683, 20)
    println(rpad(n_layers,10), rpad(n_total,10), rpad(round(mean_cone,digits=1),12), round(ratio_pred,digits=4))
end

println("\n── Balayage en PROFONDEUR, largeur fixe (dim=768, n_heads=12, hidden=3072, config jalon 0) ──")
println(rpad("n_layers",10), rpad("N_total",10), rpad("moy(cône)",12), "N_total/moy(cône)")
for n_layers in [3, 6, 12, 24, 48, 96, 192]
    n_total, mean_cone, ratio_pred = structural_ratio(n_layers, 768, 12, 3072, 20)
    println(rpad(n_layers,10), rpad(n_total,10), rpad(round(mean_cone,digits=1),12), round(ratio_pred,digits=4))
end

println("\n── Balayage en LARGEUR, profondeur fixe (n_layers=12) -- doit rester ≈ constant ──")
println(rpad("dim",8), rpad("hidden",8), rpad("N_total",10), rpad("moy(cône)",12), "N_total/moy(cône)")
for (dim, n_heads, hidden) in [(64,4,171), (256,4,683), (768,12,3072), (2048,16,5504)]
    n_total, mean_cone, ratio_pred = structural_ratio(12, dim, n_heads, hidden, 20)
    println(rpad(dim,8), rpad(hidden,8), rpad(n_total,10), rpad(round(mean_cone,digits=1),12), round(ratio_pred,digits=4))
end

println("\n── Balayage en NOMBRE DE TÊTES, dim et profondeur fixes (n_layers=12, dim=768) ──")
println(rpad("n_heads",10), rpad("N_total",10), rpad("moy(cône)",12), "N_total/moy(cône)")
for n_heads in [2, 4, 8, 12, 24]
    n_total, mean_cone, ratio_pred = structural_ratio(12, 768, n_heads, 3072, 20)
    println(rpad(n_heads,10), rpad(n_total,10), rpad(round(mean_cone,digits=1),12), round(ratio_pred,digits=4))
end
