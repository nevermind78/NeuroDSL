# ══════════════════════════════════════════════════════════════════════════════
# Piste 4.1 : vérification numérique du Théorème "Universality of the exhaustive
# per-node ratio" -- ajout proposé à math2.tex après la Remark "Why 2 is not
# universal" (ligne 193). Contraste avec le Théorème "Aggregate sweep ratio"
# (thm:aggregate, déjà dans math2.tex, vérifié par verify_aggregate_sweep_theorem.jl) :
# celui-ci échantillonne S sites PAR COUCHE (uniforme en profondeur) ; celui-ci
# échantillonne CHAQUE NŒUD comme site (biaisé par la taille des couches).
#
# Trois bras, tous à tolérance ZÉRO (identités entières, pas de convergence
# asymptotique à vérifier par la mesure -- seule la LIMITE L->infini est
# asymptotique, l'identité de comptage à L fini est exacte) :
#   (A) Chaîne totalement ordonnée : rho_node = 2N/(N-1) exactement, à tout L,
#       pour tout profil -- aucune hypothèse de variation régulière.
#   (B) Cônes réels par couche (comme l'existant), formule pondérée :
#       2*D_B = N^2 - sum(w_i^2) exactement ; rho_B -> 2 sous Karamata, taux
#       mesuré contre la prédiction (q+1)^2/((2q+1)L).
#   (C) Nécessité de l'hypothèse "pas de couche macroscopique" : un profil
#       :atom (une couche massive) donne rho_B -> 8/3, pas 2 -- même mécanique
#       de comptage, hypothèse violée, limite différente.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL

const TARGET_N = 4000

function build_layered_graph(L::Int, q::Float64, profile::Symbol)
    ns = Symbol(:pernode_, L, :_, replace(string(q), "."=>"p"), :_, profile)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=NeuroDSL.Backend.CPUDevice())
    NeuroDSL.set!(g, :input, zeros(Float32, 4); atom_type=NeuroDSL.Datom, namespace=ns)
    cur = :input
    layer_outputs = Symbol[]
    weights = Int[]
    all_nodes = Symbol[]
    if profile == :atom
        # Une couche massive au milieu (~moitié du poids total), le reste fin.
        mid = L ÷ 2
        thin = max(1, round(Int, (TARGET_N / 2) / max(1, L - 1)))
        raw_weights = [j == mid ? TARGET_N ÷ 2 : thin for j in 1:L]
    else
        raw = profile == :output_heavy ? [Float64(j)^q for j in 1:L] : [Float64(L-j+1)^q for j in 1:L]
        scale = TARGET_N / sum(raw)
        raw_weights = [max(1, round(Int, r * scale)) for r in raw]
    end
    for j in 1:L
        wj = raw_weights[j]
        push!(weights, wj)
        for k in 1:wj
            nxt = Symbol(:L, j, :_, k)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(nxt, [cur], :relu; namespace=ns))
            cur = nxt
            push!(all_nodes, nxt)
        end
        push!(layer_outputs, cur)
    end
    NeuroDSL.demand!(g, cur; namespace=ns)
    return g, ns, layer_outputs, weights, all_nodes
end

# ── Arm A : chaîne totalement ordonnée, balayage exhaustif par nœud ───────────
function exhaustive_ratio(L::Int, q::Float64, profile::Symbol)
    g, ns, louts, weights, all_nodes = build_layered_graph(L, q, profile)
    Nc = sum(weights)
    D = sum(length(NeuroDSL._downstream_nodes(g, s, ns)) - 1 for s in all_nodes)
    predicted_D = Nc * (Nc - 1) ÷ 2
    exact = D == predicted_D
    rho = Nc^2 / D
    return rho, exact, Nc
end

println("="^70)
println("(A) Chaîne totalement ordonnée -- balayage EXHAUSTIF PAR NŒUD")
println("Prédiction : D == C(Nc,2) EXACTEMENT ; rho_node == 2*Nc/(Nc-1)")
println("(contraste : le théorème existant prédirait (q+2)/(q+1) ou q+2 sur le MÊME graphe)")
println("="^70)
for (q, profile) in [(0.0, :output_heavy), (1.0, :output_heavy), (2.0, :input_heavy)]
    for L in [20, 80, 320]
        rho, exact, Nc = exhaustive_ratio(L, q, profile)
        thmK_pred = profile == :output_heavy ? (q+2)/(q+1) : q+2
        pred_exact = 2*Nc/(Nc-1)
        println("q=$q $profile L=$L Nc=$Nc  D==C(Nc,2) exact? $exact   rho_node=$(round(rho,digits=5)) (prédit $(round(pred_exact,digits=5)))   [thm existant prédirait $(round(thmK_pred,digits=4))]")
    end
end

# ── Arm B : cônes réels par couche, formule pondérée ─────────────────────────
function weighted_ratio(L::Int, q::Float64, profile::Symbol)
    g, ns, louts, weights, _ = build_layered_graph(L, q, profile)
    Nc = sum(weights)
    D_B = sum(weights[i] * (length(NeuroDSL._downstream_nodes(g, louts[i], ns)) - 1) for i in 1:L)
    predicted_2DB = Nc^2 - sum(abs2, weights)
    exact = 2*D_B == predicted_2DB
    rho_B = Nc^2 / D_B
    rate_measured = L * (rho_B - 2) / 2
    return rho_B, exact, rate_measured, Nc
end

println()
println("="^70)
println("(B) Cônes réels par couche -- identité 2*D_B == Nc^2 - sum(w_i^2)")
println("Prédiction : rho_B -> 2 ; taux L*(rho_B-2)/2 -> (q+1)^2/(2q+1)")
println("="^70)
for (q, profile) in [(0.0,:output_heavy),(0.5,:output_heavy),(1.0,:output_heavy),(2.0,:output_heavy),
                     (0.0,:input_heavy),(0.5,:input_heavy),(1.0,:input_heavy),(2.0,:input_heavy)]
    pred_rate = (q+1)^2/(2q+1)
    print("q=$q $profile  (taux prédit $(round(pred_rate,digits=4)))  :  ")
    for L in [20, 40, 80, 160, 320]
        rho_B, exact, rate, Nc = weighted_ratio(L, q, profile)
        print("L=$L:rho=$(round(rho_B,digits=5))[id=$exact,taux=$(round(rate,digits=3))]  ")
    end
    println()
end

println()
println("="^70)
println("(C) Nécessité de l'hypothèse -- profil :atom (une couche massive)")
println("Prédiction : rho_B -> 8/3 ~= 2.667 (PAS 2), via 2/(1 - ||w||^2/N^2)")
println("="^70)
for L in [10, 20, 40, 80]
    rho_B, exact, rate, Nc = weighted_ratio(L, 0.0, :atom)
    println("L=$L Nc=$Nc  rho_B=$(round(rho_B,digits=4))  identité exacte=$exact")
end
