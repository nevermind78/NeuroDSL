using CUDA, Printf
using NeuroDSL

# ═══════════════════════════════════════════════════════════════════
# À charger après bench_mem_neurodsl.jl et fix_checkpointing.jl
# (réutilise peak_mem, _ptr, _free_temp!, UNSAFE_FREE, build_deep_network)
#
# CORRECTIFS APPLIQUÉS (session diagnostic) :
#   1. src/checkpoint.jl, backward_with_checkpointing! : libère désormais
#      nd_out.value pour chaque nœud RECOMPUTABLE juste après que sa
#      contribution a été propagée -- avant ce correctif, chaque segment
#      recomputé pendant le backward restait résident jusqu'à la fin de la
#      passe, ce qui faisait dépasser le pic mémoire du checkpointing par
#      rapport à un backward standard sans checkpointing (le comble).
#   2. Ce fichier, forward_checkpointed! : le nettoyage du `tmp` (CtxStore
#      par appel de règle) était asymétrique -- les nœuds RECOMPUTABLES
#      avaient leur scratch temporaire libéré explicitement via
#      _free_temp!, mais les nœuds CHECKPOINT n'avaient QUE tmp[sym]
#      conservé, tout le reste de `tmp` fuitait jusqu'au bon vouloir du GC.
#      Corrigé pour libérer aussi ce scratch annexe, en protégeant
#      uniquement ce qui est réellement conservé (ctx_store[sym] + les
#      valeurs des entrées de la règle).
# ═══════════════════════════════════════════════════════════════════

# ── 1. DIAGNOSTIC : ensemble vivant réel vs mémoire du pool ────────
# Si `pool used` ≫ `live set`, la différence est du garbage non libéré :
# des buffers déréférencés côté Julia mais toujours détenus côté CUDA.

_nb(x) = x isa CUDA.CuArray ? sizeof(x) : 0

function live_report(g, ns, ctx_store, sch; label="")
    seen = Set{UInt}()
    bytes_ckpt = 0; bytes_recomp = 0; bytes_param = 0; bytes_grad = 0; bytes_ctx = 0

    for (sym, nd) in g.nodes[ns]
        if nd.value isa CUDA.CuArray
            p = _ptr(nd.value)
            if p ∉ seen
                push!(seen, p)
                n = _nb(nd.value)
                if nd.is_param
                    bytes_param += n
                elseif sym ∈ sch.recomputable
                    bytes_recomp += n
                else
                    bytes_ckpt += n
                end
            end
        end
        if nd.gradient isa CUDA.CuArray
            p = _ptr(nd.gradient)
            p ∉ seen && (push!(seen, p); bytes_grad += _nb(nd.gradient))
        end
    end

    for (_, ctx) in ctx_store, (_, v) in ctx
        v isa CUDA.CuArray || continue
        p = _ptr(v)
        p ∉ seen && (push!(seen, p); bytes_ctx += _nb(v))
    end

    live = bytes_ckpt + bytes_recomp + bytes_param + bytes_grad + bytes_ctx
    pool = pool_used_bytes()
    mb(x) = x / 1024^2

    println("  ── ensemble vivant $(label) ──")
    @printf "     params        %9.2f MB\n" mb(bytes_param)
    @printf "     gradients     %9.2f MB\n" mb(bytes_grad)
    @printf "     checkpoints   %9.2f MB\n" mb(bytes_ckpt)
    @printf "     recomputables %9.2f MB  ← devrait être ≈ 0 après le forward\n" mb(bytes_recomp)
    @printf "     ctx_store     %9.2f MB\n" mb(bytes_ctx)
    @printf "     ────────────────────────\n"
    @printf "     LIVE          %9.2f MB\n" mb(live)
    @printf "     POOL USED     %9.2f MB\n" mb(pool)
    @printf "     GARBAGE       %9.2f MB  ← mémoire déréférencée mais non libérée\n" mb(pool - live)
    return (live=mb(live), pool=mb(pool), garbage=mb(pool - live))
end

# ── 2. FORWARD CHECKPOINTÉ AVEC LIBÉRATION PHYSIQUE (corrigé) ──────
# Parcourt l'ordre topologique, exécute chaque règle, et dès qu'une
# activation recomputable a été consommée par tous ses consommateurs,
# la détruit réellement (unsafe_free!) au lieu de la déréférencer.
# CORRECTIF 2 : le scratch temporaire annexe des nœuds CHECKPOINT est
# maintenant lui aussi explicitement libéré (sauf ce qui est réellement
# protégé), au lieu de fuiter jusqu'au GC.

function forward_checkpointed!(g::NeuroDSL.NeuroGraph, loss_sym::Symbol,
                               ctx_store::NeuroDSL.CtxStore,
                               sch::NeuroDSL.CheckpointSchedule;
                               namespace::Symbol=g.active_ns)
    ns = namespace

    # combien de règles consomment chaque nœud
    consumers = Dict{Symbol,Int}()
    for (_, r) in g.rules[ns], s in r.inputs
        consumers[s] = get(consumers, s, 0) + 1
    end
    remaining = copy(consumers)

    # tout ce qui traîne d'un run précédent : on repart d'un état propre
    for (sym, nd) in g.nodes[ns]
        (nd.is_param || !haskey(g.rules[ns], sym)) && continue
        if nd.value isa CUDA.CuArray
            UNSAFE_FREE[] && CUDA.unsafe_free!(nd.value)
        end
        nd.value = nothing
        nd.valid = false
    end
    empty!(ctx_store)

    for sym in sch.order
        haskey(g.rules[ns], sym) || continue
        rule = g.rules[ns][sym]

        tmp = NeuroDSL.CtxStore()
        NeuroDSL.execute_rule!(g, rule; ctx_store=tmp)
        nd = g.nodes[ns][sym]
        nd.valid = true

        # protégé dans tous les cas : la valeur propre du nœud + les
        # valeurs de ses entrées (pourraient être partagées avec le tmp
        # d'un op qui les réutilise comme buffer de sortie).
        protected = Set{UInt}()
        push!(protected, _ptr(nd.value))
        for s in rule.inputs
            v = g.nodes[ns][s].value
            v !== nothing && push!(protected, _ptr(v))
        end

        if sym ∉ sch.recomputable
            # checkpoint : le ctx propre au nœud est conservé pour la
            # reconstruction pendant le backward -- on protège ses
            # tenseurs, puis on libère tout le reste du scratch de `tmp`
            # (avant ce correctif, ce "reste" n'était jamais touché ici et
            # fuitait jusqu'au GC : c'est le second bug diagnostiqué).
            if haskey(tmp, sym)
                ctx_store[sym] = tmp[sym]
                for (_, v) in tmp[sym]
                    v isa CUDA.CuArray && push!(protected, _ptr(v))
                end
            end
            for (k, ctx) in tmp
                k == sym && continue
                for (_, v) in ctx
                    _free_temp!(v, protected)
                end
            end
        else
            for (_, ctx) in tmp, (_, v) in ctx
                _free_temp!(v, protected)
            end
        end

        # largage physique des entrées épuisées
        for s in rule.inputs
            remaining[s] = get(remaining, s, 1) - 1
            remaining[s] > 0 && continue
            s == loss_sym && continue
            in_nd = g.nodes[ns][s]
            in_nd.is_param && continue                 # poids : gardés
            haskey(g.rules[ns], s) || continue          # sources : gardées
            s ∉ sch.recomputable && continue            # checkpoints : gardés
            if in_nd.value isa CUDA.CuArray
                # ne pas détruire un buffer encore détenu par un ctx conservé
                aliased = any(_ptr(v) == _ptr(in_nd.value)
                              for (_, c) in ctx_store for (_, v) in c
                              if v isa CUDA.CuArray)
                if !aliased && UNSAFE_FREE[]
                    CUDA.unsafe_free!(in_nd.value)
                end
            end
            in_nd.value = nothing
            in_nd.valid = false
        end
    end
    return g.nodes[ns][loss_sym].value
end

# ── 3. BENCHMARK COMPARATIF À TROIS BRANCHES ───────────────────────
function bench3(depth, dim, every, batch)
    w_mb = dim * dim * 4 / 1024^2
    a_mb = batch * dim * 4 / 1024^2
    @printf "\n  depth=%d dim=%d batch=%d every=%d  |  poids %.1f MB, activation %.1f MB (×%.1f)\n" depth dim batch every w_mb a_mb (a_mb/w_mb)

    ns = Symbol(:c3_, depth, :_, dim, :_, batch)
    g  = NeuroDSL.NeuroGraph(device=NeuroDSL.Backend.CUDADevice(), namespace=ns)
    loss_sym = build_deep_network(g, depth, dim, batch; ns=ns)
    ctx = NeuroDSL.CtxStore()
    warmup!(g, loss_sym, ctx, ns)

    cd  = NeuroDSL.CheckpointData(every=every)
    sch = NeuroDSL.CheckpointSchedule(g, cd; namespace=ns)

    # (a) référence : backward standard
    a = peak_mem() do
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.zero_grads!(g; namespace=ns)
        NeuroDSL.demand!(g, loss_sym; ctx_store=ctx, namespace=ns)
        NeuroDSL.backward_graph!(g, loss_sym; ctx_store=ctx, namespace=ns, full=true)
        CUDA.synchronize()
    end

    # (b) checkpointing NeuroDSL actuel (bénéficie déjà du correctif 1,
    # backward_with_checkpointing! libère maintenant les segments recomputés)
    ctx_b = NeuroDSL.CtxStore()
    b = peak_mem() do
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.zero_grads!(g; namespace=ns)
        NeuroDSL.forward_with_checkpointing!(g, loss_sym, ctx_b, sch; namespace=ns)
        NeuroDSL.backward_with_checkpointing!(g, loss_sym;
            ctx_store=ctx_b, schedule=sch, namespace=ns)
        CUDA.synchronize()
    end

    # (c) checkpointing avec libération physique forward (correctif 2) +
    # backward corrigé (correctif 1)
    ctx_c = NeuroDSL.CtxStore()
    c = peak_mem() do
        NeuroDSL.zero_grads!(g; namespace=ns)
        forward_checkpointed!(g, loss_sym, ctx_c, sch; namespace=ns)
        NeuroDSL.backward_with_checkpointing!(g, loss_sym;
            ctx_store=ctx_c, schedule=sch, namespace=ns)
        CUDA.synchronize()
    end

    @printf "    (a) backward standard          : activations %9.2f MB\n" a.delta
    @printf "    (b) ckpt (backward corrigé)     : activations %9.2f MB  (%+.1f%%)\n" b.delta (b.delta-a.delta)/a.delta*100
    @printf "    (c) ckpt + unsafe_free! (2 fixs): activations %9.2f MB  (%+.1f%%)\n" c.delta (c.delta-a.delta)/a.delta*100

    g = nothing; GC.gc(true); CUDA.reclaim()
end

# ── 4. INSPECTION D'UN FORWARD ISOLÉ ───────────────────────────────
function inspect_forward(depth, dim, every, batch)
    ns = Symbol(:insp_, depth, :_, dim)
    g  = NeuroDSL.NeuroGraph(device=NeuroDSL.Backend.CUDADevice(), namespace=ns)
    loss_sym = build_deep_network(g, depth, dim, batch; ns=ns)
    cd  = NeuroDSL.CheckpointData(every=every)
    sch = NeuroDSL.CheckpointSchedule(g, cd; namespace=ns)

    GC.gc(true); CUDA.reclaim()
    ctx_b = NeuroDSL.CtxStore()
    NeuroDSL.forward_with_checkpointing!(g, loss_sym, ctx_b, sch; namespace=ns)
    CUDA.synchronize()
    println("\n  FORWARD ACTUEL (déréférencement seul)")
    live_report(g, ns, ctx_b, sch)

    GC.gc(true); CUDA.reclaim()
    ctx_c = NeuroDSL.CtxStore()
    forward_checkpointed!(g, loss_sym, ctx_c, sch; namespace=ns)
    CUDA.synchronize()
    println("\n  FORWARD AVEC LIBÉRATION PHYSIQUE (corrigé)")
    live_report(g, ns, ctx_c, sch)

    g = nothing; GC.gc(true); CUDA.reclaim()
end

println("="^78)
println("   Diagnostic : où passe la mémoire après le forward checkpointé ?")
println("="^78)
inspect_forward(16, 512, 4, 4096)

println()
println("="^78)
println("   Comparaison à trois branches (backward_with_checkpointing! corrigé)")
println("="^78)
for (d, dim, e, b) in [(16, 512, 4, 4096), (32, 512, 4, 8192)]
    bench3(d, dim, e, b)
end
