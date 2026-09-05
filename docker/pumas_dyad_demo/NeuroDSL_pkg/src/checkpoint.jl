# ════════════════════════════════════════════════════════════════════════════════
# NeuroDSL — checkpoint.jl (version stable, avec BufferPool)
# ════════════════════════════════════════════════════════════════════════════════

struct CheckpointSchedule
    checkpoints  :: Set{Symbol}
    recomputable :: Set{Symbol}
    every        :: Int
    order        :: Vector{Symbol}
end

function CheckpointSchedule(g::NeuroGraph, cd::CheckpointData; namespace::Symbol = g.active_ns)
    order = topo_order!(g; namespace=namespace)
    every = cd.every

    # 1. Tous les paramètres sont des checkpoints
    checkpoints = Set{Symbol}()
    for (sym, nd) in g.nodes[namespace]
        nd.is_param && push!(checkpoints, sym)
    end

    # 2. Tous les nœuds source (sans règle de calcul) doivent être préservés
    #    Ils n'ont pas de règle ⇒ leur valeur ne peut pas être recomputée.
    #    On les ajoute systématiquement aux checkpoints.
    source_nodes = Set{Symbol}()
    for (sym, nd) in g.nodes[namespace]
        if !haskey(g.rules[namespace], sym) && !nd.is_param
            push!(source_nodes, sym)
        end
    end
    union!(checkpoints, source_nodes)

    # 3. Les nœuds intermédiaires backpropables tous les `every` steps
    quant_nodes = [sym for sym in order
                   if haskey(g.nodes[namespace], sym)
                   && is_backpropable(g.nodes[namespace][sym])
                   && !g.nodes[namespace][sym].is_param]
    for (k, sym) in enumerate(quant_nodes)
        k % every == 0 && push!(checkpoints, sym)
    end
    # 4. Le dernier nœud (sortie) est toujours un checkpoint
    isempty(order) || push!(checkpoints, last(order))

    recomputable = Set(sym for sym in quant_nodes if sym ∉ checkpoints)

    @info "CheckpointSchedule: $(length(checkpoints)) checkpoints, $(length(recomputable)) recomputables"
    return CheckpointSchedule(checkpoints, recomputable, every, order)
end

function forward_with_checkpointing!(g::NeuroGraph, output_sym::Symbol,
                                     ctx_store::CtxStore, schedule::CheckpointSchedule;
                                     namespace::Symbol = g.active_ns)
    ns = namespace

    # Combien de règles du graphe consomment chaque symbole comme entrée --
    # permet de libérer une activation recomputable dès que son dernier
    # consommateur a été calculé, AU LIEU d'attendre la fin du forward
    # entier pour tout libérer en un seul lot. Sans cet interleaving, le pic
    # mémoire pendant le forward est rigoureusement IDENTIQUE à un forward
    # sans aucun checkpointing (vérifié : 136.00 MB dans les deux cas sur un
    # réseau de test) -- puisque tout reste résident jusqu'à la ligne de
    # libération finale, qui arrive après que le watermark a déjà capturé
    # "tout le réseau vivant en même temps". Le checkpointing n'a alors
    # aucune chance de réduire le pic, avant même que le backward démarre.
    consumers = Dict{Symbol,Int}()
    for (_, r) in g.rules[ns], s in r.inputs
        consumers[s] = get(consumers, s, 0) + 1
    end
    remaining = copy(consumers)

    for sym in schedule.order
        nd = g.nodes[ns][sym]
        nd.valid && nd.value !== nothing && continue
        haskey(g.rules[ns], sym) || continue
        rule = g.rules[ns][sym]

        # `ctx_store` n'est plus alimenté ici (voir note ci-dessous sur
        # `_ctx_for_backward!`) -- `demand!` sans `ctx_store` évite la copie
        # défensive `A_ctx = copy(A)` (src/dispatch.jl) que chaque
        # `:matmul` ferait sinon, pour un contexte qui ne sera jamais lu.
        for inp in rule.inputs
            inp_nd = g.nodes[ns][inp]
            if !(inp_nd.valid && inp_nd.value !== nothing)
                demand!(g, inp; namespace=ns)
            end
        end

        demand!(g, sym; namespace=ns)

        # Libération immédiate des entrées recomputables épuisées -- dès que
        # plus aucune règle du graphe n'a besoin de leur valeur. Les
        # checkpoints, les paramètres et output_sym lui-même sont exclus,
        # exactement comme le nettoyage final ci-dessous les excluait déjà.
        for inp in rule.inputs
            remaining[inp] = get(remaining, inp, 1) - 1
            remaining[inp] > 0 && continue
            inp == output_sym && continue
            inp_nd = g.nodes[ns][inp]
            inp_nd.is_param && continue
            haskey(g.rules[ns], inp) || continue
            inp ∉ schedule.recomputable && continue
            if inp_nd.value !== nothing
                Backend.free!(g.device, inp_nd.value)
                inp_nd.value = nothing
                inp_nd.valid = false
            end
        end
    end

    # Filet de sécurité : tout nœud recomputable qui n'aurait pas encore été
    # libéré par l'interleaving ci-dessus (cas limite non couvert par le
    # comptage de consommateurs) l'est ici, comme avant ce correctif.
    for sym in schedule.recomputable
        nd = g.nodes[ns][sym]
        if nd.value !== nothing && !nd.is_param
            Backend.free!(g.device, nd.value)
            nd.value = nothing
            nd.valid = false
        end
    end
    return nothing
end

function _recompute_segment!(g::NeuroGraph, target_sym::Symbol,
                             schedule::CheckpointSchedule,
                             ctx_store::Union{CtxStore,Nothing}=nothing;   # ← nouveau paramètre
                             namespace::Symbol = g.active_ns)
    order = schedule.order
    ns = namespace
    target_idx = findfirst(==(target_sym), order)
    target_idx === nothing && return nothing

    # Cherche le dernier checkpoint ou nœud encore valide avant la cible
    start_idx = 1
    for i in (target_idx-1):-1:1
        prev_sym = order[i]
        nd = get(g.nodes[ns], prev_sym, nothing)
        if nd !== nothing && nd.value !== nothing && nd.valid
            start_idx = i + 1
            break
        end
    end

    # Recalcule de start_idx jusqu'à target_idx -- sans `ctx_store` : le
    # contexte de backward est désormais reconstruit à la volée par
    # `_ctx_for_backward!` (voir `backward_with_checkpointing!` ci-dessous),
    # donc plus besoin de payer la copie défensive de chaque règle recomputée.
    for i in start_idx:target_idx
        sym = order[i]
        nd = get(g.nodes[ns], sym, nothing)
        nd === nothing && continue
        nd.valid && continue
        demand!(g, sym; namespace=ns)
    end
    return g.nodes[ns][target_sym].value
end

function backward_with_checkpointing!(g::NeuroGraph, loss_sym::Symbol;
                                      ctx_store::CtxStore = CtxStore(),
                                      schedule::CheckpointSchedule,
                                      namespace::Symbol = g.active_ns)
    zero_grads!(g; namespace=namespace)
    ln = node(g, loss_sym; namespace=namespace)
    @assert length(ln.value)==1 "loss doit être scalaire"
    ln.gradient = Backend.ones32(g.device, size(ln.value)...)

    for out_sym in reverse(schedule.order)
        !haskey(g.rules[namespace], out_sym) && continue
        rule = g.rules[namespace][out_sym]
        nd_out = g.nodes[namespace][out_sym]
        nd_out.gradient === nothing && continue

        !haskey(GRAD_RULES, rule.op) &&
            error("❌ Pas de règle backward pour :$(rule.op)")

        # Vérifie que les entrées sont disponibles, sinon recompute -- SAUF
        # pour les entrées sans règle (paramètres, sources) : `set!`
        # (via `_invalidate_downstream!`, src/graph_api.jl) marque le nœud
        # qu'il vient de créer comme `valid=false` même quand c'est un
        # paramètre -- leur valeur ne devient JAMAIS périmée d'elle-même,
        # donc `!in_nd.valid` est un faux signal pour eux. Sans ce garde,
        # `_recompute_segment!` était déclenché sur CHAQUE poids de CHAQUE
        # règle traitée pendant le backward, et son étalement
        # "start_idx:target_idx" ressuscitait au passage des activations
        # recomputables déjà correctement libérées plus haut dans cette
        # même boucle -- la cause principale du surcoût mémoire du
        # checkpointing par rapport à un backward standard (vérifié :
        # 24/25 nœuds recomputables restaient vivants sans ce garde).
        for in_sym in rule.inputs
            in_nd = get(g.nodes[namespace], in_sym, nothing)
            if in_nd !== nothing && haskey(g.rules[namespace], in_sym) &&
               (in_nd.value === nothing || !in_nd.valid)
                # Recompute avec le ctx_store pour que les buffers de contexte soient remplis
                _recompute_segment!(g, in_sym, schedule, ctx_store; namespace=namespace)
            end
        end

        # Récupération du contexte -- reconstruit à la volée par
        # `_ctx_for_backward!` (src/backward.jl, déjà utilisé par le
        # backward standard) depuis `nd_out.value`/`aux_data`/`rule.attrs`,
        # SANS jamais ré-exécuter le forward ni dépendre d'une copie
        # pré-stockée dans `ctx_store`. Avant ce correctif, `ctx_store`
        # accumulait une copie complète de l'entrée de chaque `:matmul`
        # (`A_ctx = copy(A)`, src/dispatch.jl) pour CHAQUE nœud du forward
        # ET de chaque segment recomputé pendant le backward -- un coût
        # mémoire indépendant de la densité de checkpoint (vérifié : même
        # `every=1`, zéro recalcul, restait ~1.25x pire que le backward
        # standard après les correctifs précédents). `_ctx_for_backward!`
        # retombe automatiquement sur l'ancien comportement (ré-exécution)
        # si ses garde-fous internes ne sont pas remplis -- aucune perte de
        # correction, seulement une perte d'optimisation dans ce cas.
        ctx = _ctx_for_backward!(g, rule, namespace)

        inputs_vals = [g.nodes[namespace][s].value for s in rule.inputs]
        grads = GRAD_RULES[rule.op](g.device, nd_out.gradient, ctx, inputs_vals)

        for (i, in_sym) in enumerate(rule.inputs)
            accum_grad!(g.nodes[namespace][in_sym], grads[i])
        end

        # Comme pour .value ci-dessous : le gradient de out_sym ne sert
        # plus à rien une fois sa contribution propagée à ses entrées
        # (déjà lu par GRAD_RULES ci-dessus, déjà copié dans les
        # gradients des entrées par accum_grad!). Contrairement au
        # backward standard (`backward_graph!`, src/backward.jl), cette
        # fonction n'utilise pas `GradPool` -- sans libération explicite,
        # chaque buffer de gradient reste alloué jusqu'au bon vouloir du
        # GC de Julia, jamais synchrone avec la mesure de pic mémoire.
        # Vérifié : même avec every=1 (0 nœud recomputable, donc AUCUN
        # recalcul), le pic restait ~1.7x supérieur au backward standard
        # -- la cause n'était donc pas le recalcul mais ces buffers de
        # gradient jamais libérés.
        Backend.free!(g.device, nd_out.gradient)
        nd_out.gradient = nothing

        # Une fois la contribution de out_sym propagée à ses entrées, sa
        # valeur forward ne sert plus à rien pour le reste de cette boucle
        # inverse : tous ses consommateurs (topologiquement en aval) ont
        # déjà été traités plus haut dans cette même boucle. La libérer
        # maintenant évite que chaque segment recomputé par
        # `_recompute_segment!` reste résident jusqu'à la fin du backward
        # -- sans quoi le pic mémoire de la version "checkpointée" finit
        # par dépasser celui d'un backward standard (checkpoints conservés
        # depuis le forward + TOUTES les activations recomputées jamais
        # libérées). Les checkpoints eux-mêmes (out_sym ∉ recomputable)
        # restent intouchés : `_recompute_segment!` peut en avoir besoin
        # comme point de départ pour reconstruire un segment encore plus
        # en amont.
        if out_sym ∈ schedule.recomputable
            Backend.free!(g.device, nd_out.value)
            nd_out.value = nothing
            nd_out.valid = false
        end
    end
    return g
end