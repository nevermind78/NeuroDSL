
# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD_RELEASE_VALUES (src/backward.jl) -- libération des activations
# forward PENDANT le backward, à la fin de la propre visite d'un nœud dans
# la boucle inverse (pas à son dernier consommateur -- voir la docstring de
# BACKWARD_RELEASE_VALUES pour la preuve de sûreté). Cible le pic de
# `train_step` (les activations restent résidentes de la transition
# forward->backward jusqu'à la fin du backward sinon).
# ─────────────────────────────────────────────────────────────────────────────

@testset "BACKWARD_RELEASE_VALUES -- identité bit-à-bit sur plusieurs pas d'AdamW" begin
    dev = NeuroDSL.Backend.CPUDevice()
    Random.seed!(3030)

    function build(ns)
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom, namespace=ns)
        out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, :x; namespace=ns)
        NeuroDSL.set!(g, :target, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :target], :mse_loss; namespace=ns))
        return g
    end

    g_off = build(:brv_off)
    g_on  = build(:brv_on)
    for (p_off, p_on) in zip(NeuroDSL.params(g_off; namespace=:brv_off),
                              NeuroDSL.params(g_on; namespace=:brv_on))
        NeuroDSL.node(g_on, p_on.name; namespace=:brv_on).value = copy(p_off.value)
    end
    x0 = randn(Float32, 6, 16)
    NeuroDSL.node(g_off, :x; namespace=:brv_off).value = copy(x0)
    NeuroDSL.node(g_on, :x; namespace=:brv_on).value = copy(x0)

    ps_off = NeuroDSL.params(g_off; namespace=:brv_off)
    ps_on  = NeuroDSL.params(g_on; namespace=:brv_on)
    m1_off = [zeros(Float32, size(p.value)...) for p in ps_off]
    m2_off = [zeros(Float32, size(p.value)...) for p in ps_off]
    m1_on  = [zeros(Float32, size(p.value)...) for p in ps_on]
    m2_on  = [zeros(Float32, size(p.value)...) for p in ps_on]

    for t in 1:4
        tgt = randn(Float32, 6, 16)
        NeuroDSL.node(g_off, :target; namespace=:brv_off).value = copy(tgt)
        NeuroDSL.node(g_on, :target; namespace=:brv_on).value = copy(tgt)
        NeuroDSL.invalidate_all!(g_off; namespace=:brv_off)
        NeuroDSL.invalidate_all!(g_on; namespace=:brv_on)
        NeuroDSL.demand!(g_off, :loss; namespace=:brv_off)
        NeuroDSL.demand!(g_on, :loss; namespace=:brv_on)
        NeuroDSL.backward_graph!(g_off, :loss; namespace=:brv_off, release_values=false)
        NeuroDSL.backward_graph!(g_on, :loss; namespace=:brv_on, release_values=true)
        for (i, (p_off, p_on)) in enumerate(zip(ps_off, ps_on))
            NeuroDSL.adamw_step!(dev, p_off.value, p_off.gradient, m1_off[i], m2_off[i], 1f-3, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
            NeuroDSL.adamw_step!(dev, p_on.value, p_on.gradient, m1_on[i], m2_on[i], 1f-3, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
        end
    end

    for (p_off, p_on) in zip(ps_off, ps_on)
        @test Array(p_off.value) == Array(p_on.value)
    end
end

@testset "BACKWARD_RELEASE_VALUES -- différences finies (release actif)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    g = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g, :x, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom)
    out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, :x)
    NeuroDSL.set!(g, :target, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :target], :mse_loss))

    function grad_check_release(g, param_sym, loss_sym; eps=Float32(1e-4), tol=Float32(1.5e-2))
        NeuroDSL.invalidate_all!(g)
        NeuroDSL.demand!(g, loss_sym)
        NeuroDSL.backward_graph!(g, loss_sym; release_values=true)
        grad_a = Array(NeuroDSL.node(g, param_sym).gradient)
        pn = NeuroDSL.node(g, param_sym)
        orig = copy(pn.value); orig_cpu = Array(orig)
        grad_n = zeros(Float32, size(orig_cpu))
        for i in eachindex(orig_cpu)
            v⁺ = copy(orig_cpu); v⁺[i] += eps
            pn.value = copy(v⁺); NeuroDSL.invalidate_all!(g)
            l⁺ = sum(Array(NeuroDSL.demand!(g, loss_sym)))
            v⁻ = copy(orig_cpu); v⁻[i] -= eps
            pn.value = copy(v⁻); NeuroDSL.invalidate_all!(g)
            l⁻ = sum(Array(NeuroDSL.demand!(g, loss_sym)))
            grad_n[i] = (l⁺ - l⁻) / (2f0 * eps)
        end
        pn.value = orig; NeuroDSL.invalidate_all!(g)
        return maximum(abs.(grad_a .- grad_n)) < tol, maximum(abs.(grad_a .- grad_n))
    end

    worst = 0.0f0; worst_sym = nothing
    for p in NeuroDSL.params(g)
        ok, err = grad_check_release(g, p.name, :loss)
        @test ok
        if err > worst; worst = err; worst_sym = p.name; end
    end
    @printf "  [release_values] pire erreur différences finies : %.2e (%s)\n" worst worst_sym
end

@testset "BACKWARD_RELEASE_VALUES -- nœud à plusieurs consommateurs (résidu :add)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    g = NeuroDSL.NeuroGraph(device=dev)
    # a a DEUX consommateurs directs : :mid (via matmul) ET :out (via add,
    # motif résiduel) -- exactement le cas qui a produit de vrais bugs cette
    # session (:add aliasing, fuite fan-out du GradPool).
    NeuroDSL.set!(g, :a, randn(Float32, 4, 8); atom_type=NeuroDSL.Datom)
    NeuroDSL.set!(g, :W, randn(Float32, 8, 8); is_param=true)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:mid, [:a, :W], :matmul; attrs=Dict(:trans_b=>true)))
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:out, [:a, :mid], :add))
    NeuroDSL.set!(g, :target, randn(Float32, 4, 8); atom_type=NeuroDSL.Datom)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:out, :target], :mse_loss))

    NeuroDSL.invalidate_all!(g)
    NeuroDSL.demand!(g, :loss)
    NeuroDSL.backward_graph!(g, :loss; release_values=true)

    # Vérité terrain indépendante par différences finies sur :a (is_param
    # temporairement pour inspecter son gradient -- artifice de test).
    g2 = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g2, :a, Array(NeuroDSL.node(g, :a).value); is_param=true)
    NeuroDSL.set!(g2, :W, Array(NeuroDSL.node(g, :W).value); is_param=true)
    NeuroDSL.addrule!(g2, NeuroDSL.GraphRule(:mid, [:a, :W], :matmul; attrs=Dict(:trans_b=>true)))
    NeuroDSL.addrule!(g2, NeuroDSL.GraphRule(:out, [:a, :mid], :add))
    NeuroDSL.set!(g2, :target, Array(NeuroDSL.node(g, :target).value); atom_type=NeuroDSL.Datom)
    NeuroDSL.addrule!(g2, NeuroDSL.GraphRule(:loss, [:out, :target], :mse_loss))
    NeuroDSL.invalidate_all!(g2)
    NeuroDSL.demand!(g2, :loss)
    NeuroDSL.backward_graph!(g2, :loss; release_values=false)   # référence classique

    W_grad_g = Array(NeuroDSL.node(g, :W).gradient)
    W_grad_ref = Array(NeuroDSL.node(g2, :W).gradient)
    @test W_grad_g == W_grad_ref

    # Après la passe, toutes les valeurs intermédiaires libérées sauf loss.
    for (sym, nd) in g.nodes[g.active_ns]
        if sym == :loss
            @test nd.value !== nothing
        elseif nd.is_param || !haskey(g.rules[g.active_ns], sym)
            @test nd.value !== nothing   # paramètres et feuilles jamais libérés
        else
            @test nd.value === nothing
        end
    end
end

@testset "BACKWARD_RELEASE_VALUES -- repli CTX_REBUILD reste sûr (op custom sans entrée)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    g = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g, :x, randn(Float32, 4, 8); atom_type=NeuroDSL.Datom)
    NeuroDSL.set!(g, :W, randn(Float32, 8, 8); is_param=true)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:mid, [:x, :W], :matmul; attrs=Dict(:trans_b=>true)))
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:out, [:mid], :relu))   # :relu -- CTX_REBUILD vide (_ctx_empty)
    NeuroDSL.set!(g, :target, randn(Float32, 4, 8); atom_type=NeuroDSL.Datom)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:out, :target], :mse_loss))

    saved = Dict(op => NeuroDSL.CTX_REBUILD[op] for op in keys(NeuroDSL.CTX_REBUILD))
    empty!(NeuroDSL.CTX_REBUILD)   # force le repli (ré-exécution) pour TOUS les ops
    try
        NeuroDSL.invalidate_all!(g)
        NeuroDSL.demand!(g, :loss)
        NeuroDSL.backward_graph!(g, :loss; release_values=true)
    finally
        merge!(NeuroDSL.CTX_REBUILD, saved)
    end
    @test NeuroDSL.node(g, :W).gradient !== nothing
end

@testset "BACKWARD_RELEASE_VALUES -- garde-fou : second backward sans demand! intermédiaire" begin
    dev = NeuroDSL.Backend.CPUDevice()
    g = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g, :x, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom)
    out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, :x)
    NeuroDSL.set!(g, :target, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :target], :mse_loss))

    NeuroDSL.invalidate_all!(g)
    NeuroDSL.demand!(g, :loss)
    NeuroDSL.backward_graph!(g, :loss; release_values=true)
    # loss est gardée -- mais tout le reste est libéré : un second backward
    # SANS nouveau demand! doit lever le garde-fou explicite.
    @test_throws ErrorException NeuroDSL.backward_graph!(g, :loss; release_values=true)
end

@testset "BACKWARD_RELEASE_VALUES -- parité avec backward_graph_sparse!" begin
    dev = NeuroDSL.Backend.CPUDevice()
    Random.seed!(4040)

    function build(ns)
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom, namespace=ns)
        out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, :x; namespace=ns)
        NeuroDSL.set!(g, :target, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :target], :mse_loss; namespace=ns))
        return g
    end

    for prune in (false, true)
        g_off = build(Symbol(:brv_sparse_off_, prune))
        g_on  = build(Symbol(:brv_sparse_on_, prune))
        ns_off, ns_on = Symbol(:brv_sparse_off_, prune), Symbol(:brv_sparse_on_, prune)
        for (p_off, p_on) in zip(NeuroDSL.params(g_off; namespace=ns_off),
                                  NeuroDSL.params(g_on; namespace=ns_on))
            NeuroDSL.node(g_on, p_on.name; namespace=ns_on).value = copy(p_off.value)
        end
        x0 = randn(Float32, 6, 16); t0 = randn(Float32, 6, 16)
        NeuroDSL.node(g_off, :x; namespace=ns_off).value = copy(x0)
        NeuroDSL.node(g_on, :x; namespace=ns_on).value = copy(x0)
        NeuroDSL.node(g_off, :target; namespace=ns_off).value = copy(t0)
        NeuroDSL.node(g_on, :target; namespace=ns_on).value = copy(t0)

        NeuroDSL.invalidate_all!(g_off; namespace=ns_off)
        NeuroDSL.invalidate_all!(g_on; namespace=ns_on)
        NeuroDSL.demand!(g_off, :loss; namespace=ns_off)
        NeuroDSL.demand!(g_on, :loss; namespace=ns_on)
        NeuroDSL.backward_graph!(g_off, :loss; namespace=ns_off, sparse=true, prune_frozen=prune, release_values=false)
        NeuroDSL.backward_graph!(g_on, :loss; namespace=ns_on, sparse=true, prune_frozen=prune, release_values=true)

        for (p_off, p_on) in zip(NeuroDSL.params(g_off; namespace=ns_off),
                                  NeuroDSL.params(g_on; namespace=ns_on))
            ga = p_off.gradient; gb = p_on.gradient
            @test (ga === nothing) == (gb === nothing)
            ga !== nothing && @test Array(ga) == Array(gb)
        end
    end
end
