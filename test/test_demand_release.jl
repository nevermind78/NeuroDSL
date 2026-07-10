
# ─────────────────────────────────────────────────────────────────────────────
# demand_release!/release_intermediates! (src/demand_release.jl) -- libération
# proactive des activations forward pour un usage forward-SEUL (jamais suivi
# d'un backward_graph! sur ce même graphe/namespace). Cible val_window/
# gen_token (notebook/real_llm_vram_probe.jl), qui restent bornés au-dessus
# de PyTorch par la résidence permanente des .value du graphe mutable.
# ─────────────────────────────────────────────────────────────────────────────

@testset "demand_release! -- bit-exactitude vs demand! classique" begin
    dev = NeuroDSL.Backend.CPUDevice()
    Random.seed!(1010)

    g = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g, :ids, [1,3,2,4,1,2]; atom_type=NeuroDSL.Datom)
    tok = NeuroDSL.Embedding(5, 16)(g, :ids, :tok)
    out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, tok)
    logits = NeuroDSL.Linear(16, 5)(g, out, :head)

    y_ref = copy(Array(NeuroDSL.demand!(g, logits)))

    NeuroDSL.invalidate_all!(g)
    y_rel = copy(Array(NeuroDSL.demand_release!(g, logits)))

    @test y_ref == y_rel
end

@testset "demand_release! -- état post-libération correct" begin
    dev = NeuroDSL.Backend.CPUDevice()
    g = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g, :ids, [1,3,2,4]; atom_type=NeuroDSL.Datom)
    tok = NeuroDSL.Embedding(5, 16)(g, :ids, :tok)
    out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, tok)
    logits = NeuroDSL.Linear(16, 5)(g, out, :head)

    NeuroDSL.invalidate_all!(g)
    NeuroDSL.demand_release!(g, logits; keep=[tok])

    for (sym, nd) in g.nodes[g.active_ns]
        if sym == logits
            @test nd.value !== nothing   # la cible reste résidente
        elseif sym == tok
            @test nd.value !== nothing   # explicitement gardé via `keep`
        elseif nd.is_param
            @test nd.value !== nothing   # jamais libéré
        elseif !haskey(g.rules[g.active_ns], sym)
            @test nd.value !== nothing   # feuille (:ids) -- jamais libérée
        else
            @test nd.value === nothing && nd.valid == false
        end
    end
end

@testset "demand_release! -- garde-fou : backward_graph! après lève une erreur claire" begin
    dev = NeuroDSL.Backend.CPUDevice()
    g = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g, :x, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom)
    out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, :x)
    NeuroDSL.set!(g, :target, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :target], :mse_loss))

    NeuroDSL.invalidate_all!(g)
    NeuroDSL.demand_release!(g, :loss)
    @test_throws ErrorException NeuroDSL.backward_graph!(g, :loss)
    @test_throws ErrorException NeuroDSL.backward_graph!(g, :loss; sparse=true)
end

@testset "demand_release! -- cycle complet sans empoisonnement (demand_release! puis vrai entraînement)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    Random.seed!(2020)

    function build(ns)
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom, namespace=ns)
        out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, :x; namespace=ns)
        NeuroDSL.set!(g, :target, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :target], :mse_loss; namespace=ns))
        return g
    end

    g_a = build(:dr_cycle_a)   # jamais de demand_release!
    g_b = build(:dr_cycle_b)   # demand_release! une fois AVANT le vrai cycle
    for (p_a, p_b) in zip(NeuroDSL.params(g_a; namespace=:dr_cycle_a),
                           NeuroDSL.params(g_b; namespace=:dr_cycle_b))
        NeuroDSL.node(g_b, p_b.name; namespace=:dr_cycle_b).value = copy(p_a.value)
    end
    x0 = randn(Float32, 6, 16); t0 = randn(Float32, 6, 16)
    NeuroDSL.node(g_a, :x; namespace=:dr_cycle_a).value = copy(x0)
    NeuroDSL.node(g_b, :x; namespace=:dr_cycle_b).value = copy(x0)
    NeuroDSL.node(g_a, :target; namespace=:dr_cycle_a).value = copy(t0)
    NeuroDSL.node(g_b, :target; namespace=:dr_cycle_b).value = copy(t0)

    # g_b : une passe forward-seule "de validation" avec libération, PUIS un
    # cycle normal set!/invalidate_all!/demand!/backward_graph! -- doit
    # redevenir intégralement fonctionnel, sans trace de la libération.
    NeuroDSL.invalidate_all!(g_b; namespace=:dr_cycle_b)
    NeuroDSL.demand_release!(g_b, :loss; namespace=:dr_cycle_b)

    NeuroDSL.invalidate_all!(g_a; namespace=:dr_cycle_a)
    NeuroDSL.invalidate_all!(g_b; namespace=:dr_cycle_b)
    NeuroDSL.demand!(g_a, :loss; namespace=:dr_cycle_a)
    NeuroDSL.demand!(g_b, :loss; namespace=:dr_cycle_b)
    NeuroDSL.backward_graph!(g_a, :loss; namespace=:dr_cycle_a)
    NeuroDSL.backward_graph!(g_b, :loss; namespace=:dr_cycle_b)

    for (p_a, p_b) in zip(NeuroDSL.params(g_a; namespace=:dr_cycle_a),
                           NeuroDSL.params(g_b; namespace=:dr_cycle_b))
        @test Array(p_a.gradient) == Array(p_b.gradient)
    end
end

@testset "demand_release! -- cible plus profonde ensuite recalcule correctement" begin
    dev = NeuroDSL.Backend.CPUDevice()
    g = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g, :x, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom)
    mid = NeuroDSL.LlamaBlock(16, 4, 32)(g, :x, :b1)
    out = NeuroDSL.LlamaBlock(16, 4, 32)(g, mid, :b2)

    NeuroDSL.invalidate_all!(g)
    NeuroDSL.demand_release!(g, mid)   # libère les intermédiaires DANS b1, garde `mid`
    @test NeuroDSL.node(g, mid).value !== nothing

    y1 = copy(Array(NeuroDSL.demand!(g, out)))   # doit recalculer ce qui manque, pas planter

    g2 = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g2, :x, Array(NeuroDSL.node(g, :x).value); atom_type=NeuroDSL.Datom)
    for p in NeuroDSL.params(g)
        NeuroDSL.set!(g2, p.name, Array(p.value); is_param=true)
    end
    mid2 = NeuroDSL.LlamaBlock(16, 4, 32)(g2, :x, :b1)
    out2 = NeuroDSL.LlamaBlock(16, 4, 32)(g2, mid2, :b2)
    # (reconstruire avec les MÊMES poids échoue si les noms générés diffèrent
    # -- on vérifie donc plus simplement l'auto-cohérence : redemander `out`
    # sur g après une invalidation totale doit redonner la même chose que y1.)
    NeuroDSL.invalidate_all!(g)
    y2 = copy(Array(NeuroDSL.demand!(g, out)))
    @test y1 == y2
end

@testset "demand_release! -- changement de forme façon gen_token, pas de crash" begin
    dev = NeuroDSL.Backend.CPUDevice()
    g = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g, :ids, [1]; atom_type=NeuroDSL.Datom)
    tok = NeuroDSL.Embedding(5, 16)(g, :ids, :tok)
    out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, tok)
    logits = NeuroDSL.Linear(16, 5)(g, out, :head)

    last_val = nothing
    for t in 1:6
        NeuroDSL.set!(g, :ids, collect(1:t) .% 5 .+ 1; atom_type=NeuroDSL.Datom)
        NeuroDSL.invalidate_all!(g)
        last_val = copy(Array(NeuroDSL.demand_release!(g, logits)))
    end
    @test last_val !== nothing
    @test size(last_val) == (6, 5)
end

@testset "release_intermediates! -- balayage autonome" begin
    dev = NeuroDSL.Backend.CPUDevice()
    g = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g, :x, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom)
    out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, :x)
    NeuroDSL.demand!(g, out)   # demand! classique -- tout reste résident

    n_resident_before = count(nd.value !== nothing && haskey(g.rules[g.active_ns], sym) && !nd.is_param
                               for (sym, nd) in g.nodes[g.active_ns])
    @test n_resident_before > 0

    n_freed = NeuroDSL.release_intermediates!(g; keep=[out])
    @test n_freed > 0
    @test NeuroDSL.node(g, out).value !== nothing   # gardé explicitement

    n_resident_after = count(nd.value !== nothing && haskey(g.rules[g.active_ns], sym) && !nd.is_param && sym != out
                              for (sym, nd) in g.nodes[g.active_ns])
    @test n_resident_after == 0
end
