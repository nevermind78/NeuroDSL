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
                                 namespace::Symbol = g.active_ns)
    ns = namespace
    
    # 1. Réinitialiser TOUS les gradients
    for (_, nd) in g.nodes[ns]
        nd.gradient = nothing
        nd.backwarded = false
    end
    
    # 2. Initialiser le gradient de la loss
    ln = g.nodes[ns][loss_sym]
    @assert length(ln.value) == 1 "Loss must be scalar"
    ln.gradient = Backend.ones32(g.device, size(ln.value)...)
    ln.backwarded = false
    
    # 3. Backward : parcours topologique inverse
    for out_sym in reverse(topo_order!(g; namespace=ns))
        !haskey(g.rules[ns], out_sym) && continue
        rule = g.rules[ns][out_sym]
        nd_out = g.nodes[ns][out_sym]
        
        # Sauter si pas de gradient entrant
        nd_out.gradient === nothing && continue
        
        # Sauter si pas de règle backward
        !haskey(GRAD_RULES, rule.op) && continue
        
        # Récupérer le contexte forward
        ctx = get(ctx_store, out_sym, nothing)
        if ctx === nothing
            ctx_tmp = CtxStore()
            execute_rule!(g, rule; ctx_store=ctx_tmp, namespace=ns)
            ctx = get(ctx_tmp, out_sym, Dict{Symbol,Any}())
        end
        
        # Calculer les gradients des entrées
        inputs_vals = [g.nodes[ns][s].value for s in rule.inputs]
        grads = GRAD_RULES[rule.op](g.device, nd_out.gradient, ctx, inputs_vals)
        
        # Propager les gradients vers les entrées
        for (i, in_sym) in enumerate(rule.inputs)
            in_nd = g.nodes[ns][in_sym]
            
            if in_nd.is_param
                # 🔑 Paramètre entraînable → accumuler le gradient
                accum_grad!(in_nd, grads[i])
            else
                # 🔑 Paramètre gelé → passer le gradient pour la propagation
                # mais il sera effacé à la fin
                in_nd.gradient = grads[i]
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