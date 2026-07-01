# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    NeuroDSL — compiler.jl (NO METATHEORY)                 ║
# ║  Version légère qui applique directement les fusions détectées            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── 1. Fonction interne : appliquer une fusion simple ────────────────────────
#
# NB : cette fonction honore désormais réellement `fused_op` (le `RewriteRule.result`
# du match). Avant ce fix, elle délégait à `_fuse!`/`FUSION_TABLE` (graph_api.jl), qui
# ne connaît que deux patterns câblés en dur (matmul+relu, relu+matmul) — toute autre
# règle déclarée dans compiler_config.jl/compiler_rules.jl était détectée par
# `scan_graph` mais jamais réellement appliquée. `_fuse!`/`FUSION_TABLE` restent en
# place dans graph_api.jl (dépréciés) au cas où du code externe y ferait référence,
# mais ne sont plus appelés par ce chemin.

function _apply_fusion!(g::NeuroGraph, ns::Symbol, chain::Vector{Symbol}, fused_op::Symbol;
                        training::Bool = true)
    # Garde de sécurité training : refuser une fusion vers un op sans règle de
    # gradient tant que training=true, plutôt que de laisser backward_graph!
    # planter plus tard. :identity est exempté (élimination algébrique pure,
    # a sa propre règle de gradient passe-plat dans GRAD_RULES).
    if training && fused_op !== :identity && !_has_backward_support(fused_op)
        return false
    end

    # Vérifier que la chaîne est encore fuseable
    for i in 1:length(chain)-1
        sym = chain[i]
        users = [out for (out, rule) in g.rules[ns] if sym in rule.inputs]
        if length(users) != 1 || users[1] != chain[i+1]
            return false
        end
    end

    rules = [g.rules[ns][sym] for sym in chain]
    merged_attrs = Dict{Symbol, Any}()
    for r in rules
        merge!(merged_attrs, r.attrs)
    end

    # Collecter les inputs externes
    fused_inputs = Symbol[]
    intermediate_outputs = [r.output for r in rules[1:end-1]]
    for r in rules
        for inp in r.inputs
            if !(inp in intermediate_outputs) && !(inp in fused_inputs)
                push!(fused_inputs, inp)
            end
        end
    end

    fused_output = rules[end].output

    # Nettoyer les nœuds intermédiaires
    for sym in chain[1:end-1]
        delete!(g.rules[ns], sym)
        delete!(g.nodes[ns], sym)
    end
    delete!(g.rules[ns], chain[end])

    # Ajouter la règle fusionnée
    addrule!(g, GraphRule(fused_output, fused_inputs, fused_op;
                          attrs=merged_attrs, namespace=ns))
    _invalidate_downstream!(g, fused_output, ns)
    return true
end

# ── 2. Compilation principale (utilise scan_graph) ───────────────────────────

function compile(g::NeuroGraph, config::CompilerConfig = CompilerConfig();
                 namespace::Symbol = g.active_ns)
    # 1. Scanner les opportunités (puis, pour les paires alternatives connues
    #    comme sdpa_fusion vs flash_attention, ne garder que la moins coûteuse
    #    pour l'instance de pattern concernée — voir compiler_rules.jl)
    matches = scan_graph(g, config.rules; namespace=namespace)
    matches = _choose_cheapest_alternative(g, matches, config.cost_fn; namespace=namespace)
    isempty(matches) && @info "Aucune fusion applicable."

    # 2. Appliquer les fusions dans l'ordre décroissant de score
    fused_count = 0
    fused_ops = Dict{Symbol, Symbol}()
    for m in matches
        if _apply_fusion!(g, namespace, m.nodes, m.rule.result; training=config.training)
            fused_count += 1
            fused_ops[m.nodes[end]] = m.rule.result
        end
    end

    # 3. Après fusions, recalculer l'ordre topologique et le plan mémoire
    order = topo_order!(g; namespace=namespace)
    liveness = compute_liveness(g; namespace=namespace)
    slot_of  = greedy_interval_coloring(liveness, order)
    n_slots  = isempty(slot_of) ? 0 : maximum(values(slot_of))
    memory_plan = MemoryPlan(slot_of, liveness, n_slots, order)
    pool = BufferPool(g.device)

    # 4. Construire un CompiledPlan (sans e-graph pour l'instant)
    plan = CompiledPlan(
        namespace,
        order,
        fused_ops,
        memory_plan,
        pool,
        config
    )
    plan.compiled_at = time()
    plan.n_recompiles = 0

    # 5. Câbler la recompilation incrémentale : chaque nœud du plan reçoit un
    #    callback on_change qui alimente plan.dirty_nodes (voir _invalidate_downstream!
    #    dans graph_api.jl, qui appelle déjà ce callback s'il existe).
    _register_dirty_tracking!(plan, g, namespace)

    return plan
end

# ── 3. Recompilation incrémentale ─────────────────────────────────────────────
#
# `GraphNode.on_change`, `CompiledPlan.dirty_nodes` et `n_recompiles` existaient
# déjà comme champs mais n'étaient alimentés par rien. `_register_dirty_tracking!`
# attache un callback à chaque nœud du plan ; `_invalidate_downstream!` (déjà
# existant dans graph_api.jl) l'appellera automatiquement à chaque `set!` qui
# invalide ce nœud. `recompile!` ne rescanne/refuse que la région du graphe
# réellement atteignable depuis les nœuds marqués sales — pas tout le graphe.

function _register_dirty_tracking!(plan::CompiledPlan, g::NeuroGraph, ns::Symbol)
    for sym in plan.exec_order
        haskey(g.nodes[ns], sym) || continue
        g.nodes[ns][sym].on_change = (gg, changed_sym, changed_ns) -> push!(plan.dirty_nodes, changed_sym)
    end
    return plan
end

"""
    recompile!(plan, g) → CompiledPlan

No-op si `plan` n'est pas sale (`is_dirty(plan) == false`). Sinon, ne retraite que la
région du graphe atteignable depuis `plan.dirty_nodes` (pas de rescan complet), sauf si
la forme du graphe a changé depuis la dernière compilation (nœuds ajoutés après coup),
auquel cas on retombe sur une compilation complète par sécurité.
"""
function recompile!(plan::CompiledPlan, g::NeuroGraph)
    is_dirty(plan) || return plan
    ns = plan.namespace

    if length(g.nodes[ns]) != length(plan.exec_order)
        # Le graphe a grandi depuis la dernière compilation : le suivi incrémental
        # ne couvre pas les nœuds jamais enregistrés — repli sur une compilation complète.
        fresh = compile(g, plan.config; namespace=ns)
        plan.exec_order  = fresh.exec_order
        plan.memory_plan = fresh.memory_plan
        plan.pool        = fresh.pool
        plan.fused_ops   = fresh.fused_ops
        empty!(plan.dirty_nodes)
        plan.n_recompiles += 1
        plan.compiled_at = time()
        return plan
    end

    # Région affectée = tout ce qui est atteignable en aval des nœuds sales.
    affected = Set{Symbol}()
    queue = collect(plan.dirty_nodes)
    while !isempty(queue)
        sym = pop!(queue)
        sym in affected && continue
        push!(affected, sym)
        for c in consumers(g, sym; ns=ns)
            push!(queue, c)
        end
    end

    matches = scan_graph(g, plan.config.rules; namespace=ns)
    matches = _choose_cheapest_alternative(g, matches, plan.config.cost_fn; namespace=ns)
    for m in matches
        isempty(intersect(Set(m.nodes), affected)) && continue
        if _apply_fusion!(g, ns, m.nodes, m.rule.result; training=plan.config.training)
            plan.fused_ops[m.nodes[end]] = m.rule.result
        end
    end

    order = topo_order!(g; namespace=ns)
    liveness = compute_liveness(g; namespace=ns)
    slot_of  = greedy_interval_coloring(liveness, order)
    n_slots  = isempty(slot_of) ? 0 : maximum(values(slot_of))
    plan.exec_order  = order
    plan.memory_plan = MemoryPlan(slot_of, liveness, n_slots, order)

    _register_dirty_tracking!(plan, g, ns)  # les nœuds fusionnés sont neufs, besoin d'un callback
    empty!(plan.dirty_nodes)
    plan.n_recompiles += 1
    plan.compiled_at = time()
    return plan
end

# ── 4. Exécution du plan ──────────────────────────────────────────────────────
#
# Utilise execute_rule_pooled! (dispatch.jl) au lieu d'execute_rule! : les buffers
# intermédiaires sont emprunté/rendus via plan.pool plutôt qu'alloués frais à chaque
# nœud. Garde de sécurité : en mode training (défaut), un nœud backpropable (Quantom)
# n'est jamais libéré avant la fin — backward_graph! a besoin de sa valeur forward.
# Seuls les intermédiaires non backpropables (Datom), ou tout intermédiaire en mode
# training=false, sont effectivement recyclés.

function (plan::CompiledPlan)(g::NeuroGraph, output_sym::Symbol;
                              ctx_store=CtxStore())
    is_dirty(plan) && recompile!(plan, g)
    ns = plan.namespace
    training = plan.config.training

    # En mode training, seul un intermédiaire non backpropable (Datom) peut jamais
    # être libéré (voir la garde plus bas) — les params et les Quantom ne le sont
    # jamais. Si le graphe n'en contient aucun, la boucle de libération ci-dessous
    # ne ferait jamais rien : on le détecte une seule fois par appel (O(n)) plutôt
    # que de repayer un balayage O(n) à chaque étape pour un résultat garanti nul.
    any_releasable = !training || any(
        !g.nodes[ns][s].is_param && !is_backpropable(g.nodes[ns][s])
        for s in plan.exec_order
    )

    for (step, sym) in enumerate(plan.exec_order)
        nd = g.nodes[ns][sym]
        if !(nd.valid && nd.value !== nothing) && haskey(g.rules[ns], sym)
            rule = g.rules[ns][sym]
            execute_rule_pooled!(g, rule, plan.pool; ctx_store=ctx_store)
        end

        any_releasable || continue
        for prev_sym in plan.exec_order[1:step]
            prev_sym == output_sym && continue
            prev_nd = g.nodes[ns][prev_sym]
            prev_nd.value === nothing && continue
            prev_nd.is_param && continue
            training && is_backpropable(prev_nd) && continue

            iv = get(plan.memory_plan.liveness, prev_sym, nothing)
            iv === nothing && continue
            if iv.last_use <= step
                release!(plan.pool, prev_nd.value)
                prev_nd.value = nothing
                prev_nd.valid = false
            end
        end
    end
    return g.nodes[ns][output_sym].value
end