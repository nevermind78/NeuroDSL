# ══════════════════════════════════════════════════════════════════════════════
# backward_sparse.jl — Backward sparse (ignore les gradients des paramètres gelés)
#
# Ce fichier AJOUTE la capacité de ne pas calculer les gradients
# des paramètres marqués is_param=false, sans toucher au
# backward standard existant.
# ══════════════════════════════════════════════════════════════════════════════

"""
    backward_graph_sparse!(g, loss_sym; ctx_store, namespace)

Effectue un backward qui NE CALCULE PAS les gradients des paramètres
marqués `is_param=false`. Les gradients traversent ces paramètres
pour la rétropropagation mais ne sont pas stockés.

Cette fonction est utile pour :
- Fine-tuning partiel (certaines couches gelées)
- Architecture évolutive (couches pas encore activées)
- Économie de calcul dans les grands réseaux
"""
function backward_graph_sparse!(g::NeuroGraph, loss_sym::Symbol;
                                 ctx_store::CtxStore = CtxStore(),
                                 namespace::Symbol = g.active_ns,
                                 prune_frozen::Bool = false)
    ns = namespace

    # 1. Réinitialiser les gradients -- même logique que backward_graph!, voir
    # ce fichier pour l'explication complète : un paramètre réutilise son
    # buffer (fill! à zéro) au lieu d'être nullifié, pour laisser accum_grad!
    # accumuler in-place plutôt que réallouer.
    for (_, nd) in g.nodes[ns]
        if nd.is_param && nd.gradient !== nothing
            fill!(nd.gradient, 0f0)
        else
            nd.gradient = nothing
        end
        nd.backwarded = false
    end

    # 2. Initialiser le gradient de la loss
    ln = g.nodes[ns][loss_sym]
    @assert length(ln.value) == 1 "Loss must be scalar"
    ln.gradient = Backend.ones32(g.device, size(ln.value)...)
    ln.backwarded = false

    needs_bwd = prune_frozen ? _compute_needs_bwd(g, ns) : nothing

    # 3. Backward : parcours topologique inverse
    for out_sym in reverse(topo_order!(g; namespace=ns))
        !haskey(g.rules[ns], out_sym) && continue
        rule = g.rules[ns][out_sym]
        nd_out = g.nodes[ns][out_sym]

        # Sauter si pas de gradient entrant
        nd_out.gradient === nothing && continue

        # Sauter si pas de règle backward
        !haskey(GRAD_RULES, rule.op) && continue

        # Garde défensive (voir backward_graph! pour l'explication complète) :
        # la garde symétrique côté entrées, plus bas, empêche déjà nd_out
        # d'être jamais atteint ici pour un sous-arbre entièrement gelé.
        needs_bwd !== nothing && !needs_bwd[out_sym] && continue
        
        # Récupérer le contexte forward -- voir backward_graph! pour
        # l'explication complète (priorité au ctx_store fourni, sinon
        # reconstruction sans ré-exécution via _ctx_for_backward! avec repli
        # automatique sur l'ancien comportement).
        ctx = get(ctx_store, out_sym, nothing)
        if ctx === nothing
            ctx = _ctx_for_backward!(g, rule, ns)
        end
        
        # Calculer les gradients des entrées
        inputs_vals = [g.nodes[ns][s].value for s in rule.inputs]
        # Voir backward_graph! pour l'explication : permet à :matmul/:linear
        # d'accumuler directement dans le buffer de gradient d'un paramètre.
        ctx[:_in_nodes] = [g.nodes[ns][s] for s in rule.inputs]
        grads = GRAD_RULES[rule.op](g.device, nd_out.gradient, ctx, inputs_vals)
        
        # Propager les gradients vers les entrées
        for (i, in_sym) in enumerate(rule.inputs)
            in_nd = g.nodes[ns][in_sym]

            if needs_bwd !== nothing && !needs_bwd[in_sym]
                # Sous-arbre prouvé sans paramètre entraînable en amont --
                # même logique que backward_graph!, voir ce fichier.
                continue
            end

            if in_nd.is_param
                # 🔑 Paramètre entraînable → accumuler le gradient
                accum_grad!(in_nd, grads[i])
            else
                # 🔑 Paramètre gelé → passer le gradient pour la propagation,
                # il sera effacé à la fin. Sommer (pas juste affecter) si une
                # deuxième contribution arrive : un nœud gelé à plusieurs
                # consommateurs (connexion résiduelle, poids partagé) doit
                # recevoir la somme exacte, sinon seule la dernière
                # contribution survivrait et le gradient propagé plus en
                # amont serait faux.
                if in_nd.gradient === nothing
                    in_nd.gradient = grads[i]
                else
                    in_nd.gradient .+= grads[i]
                end
            end
            in_nd.backwarded = true
        end
        
        # Nettoyer le contexte
        delete!(ctx_store, out_sym)
        nd_out.backwarded = true
        
        # Libérer le gradient des nœuds intermédiaires
        if !nd_out.is_param
            nd_out.gradient = nothing
        end
    end
    
    # 4. NETTOYAGE FINAL : effacer les gradients des paramètres gelés
    for (_, nd) in g.nodes[ns]
        if !nd.is_param
            nd.gradient = nothing
        end
    end
    
    return g
end