
# ─────────────────────────────────────────────────────────────────────────────
# CTX_REBUILD -- reconstruction du contexte de backward sans ré-exécuter le
# forward (src/backward.jl). `grad_check` (test_backward.jl) thread un
# ctx_store PARTAGÉ entre demand! et backward_graph!, donc n'exerce JAMAIS
# ce nouveau chemin (le ctx est toujours trouvé via ctx_store, jamais
# `nothing`). `grad_check_rebuild` ci-dessous force le chemin par défaut
# (`ctx_store=CtxStore()` frais), qui appelle `_ctx_for_backward!`.
# ─────────────────────────────────────────────────────────────────────────────

function grad_check_rebuild(g, param_sym, loss_sym;
                             eps=Float32(1e-4), tol=Float32(5e-3), verbose=true)
    NeuroDSL.invalidate_all!(g)
    NeuroDSL.demand!(g, loss_sym)                 # forward SEUL, sans ctx_store
    NeuroDSL.backward_graph!(g, loss_sym)          # ctx_store frais par défaut -> rebuild
    grad_a = Array(NeuroDSL.node(g, param_sym).gradient)

    pn = NeuroDSL.node(g, param_sym)
    orig = copy(pn.value); orig_cpu = Array(orig)
    grad_n = zeros(Float32, size(orig_cpu))
    for i in eachindex(orig_cpu)
        v⁺ = copy(orig_cpu); v⁺[i] += eps
        pn.value = NeuroDSL.Backend.to_device(g.device, v⁺); NeuroDSL.invalidate_all!(g)
        l⁺ = sum(Array(NeuroDSL.demand!(g, loss_sym)))
        v⁻ = copy(orig_cpu); v⁻[i] -= eps
        pn.value = NeuroDSL.Backend.to_device(g.device, v⁻); NeuroDSL.invalidate_all!(g)
        l⁻ = sum(Array(NeuroDSL.demand!(g, loss_sym)))
        grad_n[i] = (l⁺ - l⁻) / (2f0 * eps)
    end
    pn.value = orig; NeuroDSL.invalidate_all!(g)

    diff     = abs.(grad_a .- grad_n)
    max_err  = maximum(diff)
    ok       = max_err < tol
    if verbose
        status = ok ? "✅" : "❌"
        @printf "  [rebuild :%s] max_err=%.2e (tol=%.0e) %s\n" param_sym max_err tol status
    end
    return ok, max_err
end

@testset "CTX_REBUILD -- différences finies (chemin sans ré-exécution)" begin
    dev = NeuroDSL.Backend.CPUDevice()

    @testset ":matmul (trans_b=true, via Linear)" begin
        g = NeuroDSL.NeuroGraph(device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 5, 6); atom_type=NeuroDSL.Datom)
        out = NeuroDSL.Linear(6, 4)(g, :x, :fc)
        NeuroDSL.set!(g, :y, randn(Float32, 5, 4); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :y], :mse_loss))
        ok1, _ = grad_check_rebuild(g, :fc_W, :loss)
        ok2, _ = grad_check_rebuild(g, :fc_b, :loss)
        @test ok1
        @test ok2
    end

    @testset ":rmsnorm + :swiglu + :softmax + :scale_mask (LlamaModel complet)" begin
        g = NeuroDSL.NeuroGraph(device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom)
        out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, :x)
        NeuroDSL.set!(g, :target, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :target], :mse_loss))
        worst = 0.0f0; worst_sym = nothing
        for p in NeuroDSL.params(g)
            ok, err = grad_check_rebuild(g, p.name, :loss; tol=Float32(1.5e-2), verbose=false)
            @test ok
            if err > worst; worst = err; worst_sym = p.name; end
        end
        @printf "  [rebuild LlamaModel] pire erreur sur tous les params : %.2e (%s)\n" worst worst_sym
    end

    @testset ":embedding + :cross_entropy" begin
        g = NeuroDSL.NeuroGraph(device=dev)
        NeuroDSL.set!(g, :ids, [1,3,2,4]; atom_type=NeuroDSL.Datom)
        tok = NeuroDSL.Embedding(5, 6)(g, :ids, :tok)
        logits = NeuroDSL.Linear(6, 5)(g, tok, :head)
        NeuroDSL.set!(g, :labels, [2,1,4,3]; atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [logits, :labels], :cross_entropy))
        ok1, _ = grad_check_rebuild(g, :tok_E, :loss)
        ok2, _ = grad_check_rebuild(g, :head_W, :loss)
        @test ok1
        @test ok2
    end
end

@testset "CTX_REBUILD -- identité bit-à-bit entre les 3 chemins (ctx threadé / rebuild / repli forcé)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    Random.seed!(2024)
    dim, seq = 16, 6

    function build_ref(ns)
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, seq, dim); atom_type=NeuroDSL.Datom, namespace=ns)
        out = NeuroDSL.LlamaModel(2, dim, 4, 32)(g, :x; namespace=ns)
        NeuroDSL.set!(g, :target, randn(Float32, seq, dim); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :target], :mse_loss; namespace=ns))
        return g
    end

    g_a = build_ref(:ctxr_bit_a)
    g_b = build_ref(:ctxr_bit_b)
    g_c = build_ref(:ctxr_bit_c)
    # Synchroniser les poids des trois graphes (LlamaModel tire des poids
    # aléatoires indépendamment à chaque construction).
    for (p_a, p_b, p_c) in zip(NeuroDSL.params(g_a; namespace=:ctxr_bit_a),
                                NeuroDSL.params(g_b; namespace=:ctxr_bit_b),
                                NeuroDSL.params(g_c; namespace=:ctxr_bit_c))
        NeuroDSL.node(g_b, p_b.name; namespace=:ctxr_bit_b).value = copy(p_a.value)
        NeuroDSL.node(g_c, p_c.name; namespace=:ctxr_bit_c).value = copy(p_a.value)
    end
    # :x ET :target sont tirés indépendamment par build_ref (RNG global qui
    # avance entre les 3 constructions) -- synchroniser les DEUX, sinon la
    # perte (donc les gradients) diffère légitimement pour une raison qui
    # n'a rien à voir avec CTX_REBUILD.
    x0 = randn(Float32, seq, dim)
    target0 = randn(Float32, seq, dim)
    for (g, ns) in ((g_a, :ctxr_bit_a), (g_b, :ctxr_bit_b), (g_c, :ctxr_bit_c))
        NeuroDSL.node(g, :x; namespace=ns).value = copy(x0)
        NeuroDSL.node(g, :target; namespace=ns).value = copy(target0)
        NeuroDSL.invalidate_all!(g; namespace=ns)
    end

    # (A) chemin historique exact : ctx_store threadé entre forward et backward.
    ctx_a = NeuroDSL.CtxStore()
    NeuroDSL.demand!(g_a, :loss; ctx_store=ctx_a, namespace=:ctxr_bit_a)
    NeuroDSL.backward_graph!(g_a, :loss; ctx_store=ctx_a, namespace=:ctxr_bit_a)

    # (B) chemin rebuild : ctx_store frais par défaut.
    NeuroDSL.demand!(g_b, :loss; namespace=:ctxr_bit_b)
    NeuroDSL.backward_graph!(g_b, :loss; namespace=:ctxr_bit_b)

    # (C) repli forcé : on retire temporairement les entrées CTX_REBUILD des
    # ops réellement présents dans ce graphe, pour forcer la ré-exécution.
    saved = Dict(op => NeuroDSL.CTX_REBUILD[op] for op in keys(NeuroDSL.CTX_REBUILD))
    empty!(NeuroDSL.CTX_REBUILD)
    try
        NeuroDSL.demand!(g_c, :loss; namespace=:ctxr_bit_c)
        NeuroDSL.backward_graph!(g_c, :loss; namespace=:ctxr_bit_c)
    finally
        merge!(NeuroDSL.CTX_REBUILD, saved)
    end

    @testset "gradients identiques (erreur=0) sur tous les paramètres" begin
        for (p_a, p_b, p_c) in zip(NeuroDSL.params(g_a; namespace=:ctxr_bit_a),
                                    NeuroDSL.params(g_b; namespace=:ctxr_bit_b),
                                    NeuroDSL.params(g_c; namespace=:ctxr_bit_c))
            @test Array(p_a.gradient) == Array(p_b.gradient)
            @test Array(p_a.gradient) == Array(p_c.gradient)
        end
    end
end

@testset "CTX_REBUILD -- :dropout (masque persistant, cohérent entre forward et backward)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    g = NeuroDSL.NeuroGraph(device=dev)
    # :x déclaré is_param=true (uniquement pour que son gradient survive au
    # nettoyage final de backward_graph! et reste inspectable ici -- un
    # Datom n'accumule JAMAIS de gradient, accum_grad!/backward.jl, et un
    # Quantom non-paramètre est nullifié par le nettoyage final ; ce n'est
    # pas une déclaration sémantique réelle, juste un artifice de test).
    NeuroDSL.set!(g, :x, ones(Float32, 8, 8); is_param=true)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:drop, [:x], :dropout;
                       attrs=Dict(:rate=>0.5, :training=>true)))
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:drop], :sum_matrix))

    NeuroDSL.invalidate_all!(g)
    NeuroDSL.demand!(g, :loss)   # calcule :drop ET :loss (backward_graph! suppose :loss déjà demandé)
    y_fwd = copy(Array(NeuroDSL.node(g, :drop).value))
    NeuroDSL.backward_graph!(g, :loss)   # ctx_store frais -> rebuild pour :dropout
    # Le nœud :x reçoit le VRAI gradient du dropout : dx_x = dy .* mask ./ (1-rate).
    # dy = 1 partout (sum_matrix), donc dx_x .* x doit reproduire EXACTEMENT y_fwd
    # (identité : y_fwd = x .* mask ./ (1-rate) = dx_x .* x puisque dy=1).
    dx_x = Array(NeuroDSL.node(g, :x).gradient)
    @test dx_x .* Array(NeuroDSL.node(g, :x).value) == y_fwd
    @printf "  [dropout] masque backward == masque forward : %s\n" (dx_x .* Array(NeuroDSL.node(g, :x).value) == y_fwd)

    # Nœud toujours valide et non corrompu après le backward (contrairement à
    # la ré-exécution historique, qui écrasait out_node.value avec un second
    # tirage aléatoire).
    @test Array(NeuroDSL.node(g, :drop).value) == y_fwd

    # training=false : le gradient doit passer inchangé (identité), pas être
    # filtré par un masque -- bug latent corrigé au passage.
    g2 = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g2, :x, ones(Float32, 4, 4); is_param=true)   # is_param=true : artifice de test, voir plus haut
    NeuroDSL.addrule!(g2, NeuroDSL.GraphRule(:drop, [:x], :dropout;
                       attrs=Dict(:rate=>0.5, :training=>false)))
    NeuroDSL.addrule!(g2, NeuroDSL.GraphRule(:loss, [:drop], :sum_matrix))
    NeuroDSL.invalidate_all!(g2)
    NeuroDSL.demand!(g2, :loss)
    NeuroDSL.backward_graph!(g2, :loss)
    @test all(Array(NeuroDSL.node(g2, :x).gradient) .== 1f0)
end

@testset "CTX_REBUILD -- garde-fous (kill-switch pool, garde d'alias)" begin
    dev = NeuroDSL.Backend.CPUDevice()

    @testset "kill-switch : usage du BufferPool désarme CTX_REBUILD définitivement" begin
        g = NeuroDSL.NeuroGraph(device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 4, 8); atom_type=NeuroDSL.Datom)
        out = NeuroDSL.Linear(8, 6)(g, :x, :fc)
        NeuroDSL.set!(g, :y, randn(Float32, 4, 6); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :y], :mse_loss))
        NeuroDSL.invalidate_all!(g)

        was_armed = NeuroDSL._POOLED_EXECUTION_SEEN[]
        plan, pool = NeuroDSL.plan_memory!(g)
        NeuroDSL.demand_planned!(g, :loss, plan, pool)
        @test NeuroDSL._POOLED_EXECUTION_SEEN[]   # armé par acquire! (liveness.jl)

        # Le backward doit rester correct (replié sur la ré-exécution) même
        # avec le kill-switch armé -- sur un graphe NEUF : `demand_planned!`
        # libère au pool (donc met .value=nothing) les nœuds non-paramètres
        # dont la durée de vie est expirée, y compris des feuilles -- une
        # feuille n'a pas de règle pour être recalculée, donc réutiliser LE
        # MÊME graphe après `demand_planned!` casserait pour une raison sans
        # rapport avec CTX_REBUILD (limitation déjà existante, pas introduite
        # par ce fix). Le kill-switch lui-même est global au processus
        # (_POOLED_EXECUTION_SEEN[]), donc un graphe neuf le vérifie tout autant.
        g_fresh = NeuroDSL.NeuroGraph(device=dev)
        NeuroDSL.set!(g_fresh, :x, randn(Float32, 4, 8); atom_type=NeuroDSL.Datom)
        out_fresh = NeuroDSL.Linear(8, 6)(g_fresh, :x, :fc)
        NeuroDSL.set!(g_fresh, :y, randn(Float32, 4, 6); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g_fresh, NeuroDSL.GraphRule(:loss, [out_fresh, :y], :mse_loss))
        NeuroDSL.invalidate_all!(g_fresh)
        NeuroDSL.demand!(g_fresh, :loss)
        NeuroDSL.backward_graph!(g_fresh, :loss)
        ok, err = grad_check_rebuild(g_fresh, :fc_W, :loss; verbose=false)
        @test ok
        if !was_armed
            @printf "  [kill-switch] armé avec succès par ce test, gradients toujours corrects sur un graphe neuf (err=%.2e)\n" err
        end
    end

    @testset "garde d'alias : nœud dont .value est partagé avec une entrée -> repli forcé" begin
        g = NeuroDSL.NeuroGraph(device=dev)
        NeuroDSL.set!(g, :a, randn(Float32, 4, 4); atom_type=NeuroDSL.Datom)
        NeuroDSL.set!(g, :W, randn(Float32, 4, 4); is_param=true)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:out, [:a, :W], :matmul; attrs=Dict(:trans_b=>true)))
        NeuroDSL.invalidate_all!(g)
        NeuroDSL.demand!(g, :out)
        # Forcer artificiellement l'alias : :out et :a partagent maintenant le même tableau.
        nd_out = NeuroDSL.node(g, :out); nd_a = NeuroDSL.node(g, :a)
        nd_out.value = nd_a.value
        rule = g.rules[g.active_ns][:out]
        ctx = NeuroDSL._ctx_for_backward!(g, rule, g.active_ns)
        @test haskey(ctx, :A)   # seul le chemin de repli (ré-exécution) peuple :A
    end
end
