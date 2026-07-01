
# ─────────────────────────────────────────────────────────────────────────────
# test_compiler.jl — compile(), scan_graph(), RewriteRule, recompilation
# incrémentale, pooling mémoire.
#
# Avant ce fichier, aucun test n'exerçait compile()/scan_graph()/RewriteRule :
# _apply_fusion! délégait silencieusement à un FUSION_TABLE à 2 entrées et
# ignorait le RewriteRule.result qu'on lui passait — toute fusion déclarée
# au-delà de matmul+relu/relu+matmul était détectée par scan_graph mais jamais
# appliquée. Ces tests couvrent le fix (fusion honorée + garde training),
# la recompilation incrémentale, le pooling de buffers, et la preuve de
# concept de sélection par coût (sdpa_fusion vs flash_attention).
# ─────────────────────────────────────────────────────────────────────────────

@testset "Compiler — fusion application" begin
    dev = NeuroDSL.Backend.CPUDevice()

    @testset "matmul+relu fuse toujours (régression du chemin historique)" begin
        ns = :c_mr
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 4, 3); is_param=true)
        NeuroDSL.set!(g, :W, randn(Float32, 3, 5))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h, [:x, :W], :matmul; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:z, [:h], :relu; namespace=ns))

        rule = NeuroDSL.RewriteRule(:matmul_relu_fusion, (:matmul, :relu), :fused_matmul_relu; cost_delta=0.3f0)
        n_rules_before = length(g.rules[ns])
        plan = NeuroDSL.compile(g, NeuroDSL.CompilerConfig(rules=[rule]); namespace=ns)

        @test length(g.rules[ns]) < n_rules_before
        @test g.rules[ns][:z].op == :fused_matmul_relu
        @test length(plan.fused_ops) == 1
    end

    @testset "RewriteRule.result est honoré pour fused_matmul_add" begin
        ns = :c_ma
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 4, 3); is_param=true)
        NeuroDSL.set!(g, :W, randn(Float32, 3, 5))
        NeuroDSL.set!(g, :b, randn(Float32, 1, 5))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h, [:x, :W], :matmul; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:z, [:h, :b], :add; namespace=ns))

        rule = NeuroDSL.RewriteRule(:t_matmul_add, (:matmul, :add), :fused_matmul_add; cost_delta=0.2f0)
        plan = NeuroDSL.compile(g, NeuroDSL.CompilerConfig(rules=[rule]); namespace=ns)
        @test g.rules[ns][:z].op == :fused_matmul_add

        out_fused = Array(plan(g, :z))

        ns_ref = :c_ma_ref
        gref = NeuroDSL.NeuroGraph(namespace=ns_ref, device=dev)
        NeuroDSL.set!(gref, :x, Array(g.nodes[ns][:x].value))
        NeuroDSL.set!(gref, :W, Array(g.nodes[ns][:W].value))
        NeuroDSL.set!(gref, :b, Array(g.nodes[ns][:b].value))
        NeuroDSL.addrule!(gref, NeuroDSL.GraphRule(:h, [:x, :W], :matmul; namespace=ns_ref))
        NeuroDSL.addrule!(gref, NeuroDSL.GraphRule(:z, [:h, :b], :add; namespace=ns_ref))
        out_ref = Array(NeuroDSL.demand!(gref, :z; ctx_store=NeuroDSL.CtxStore(), namespace=ns_ref))

        @test isapprox(out_fused, out_ref; atol=1f-5)
    end

    @testset "RewriteRule.result est honoré pour fused_matmul_add_relu (chaîne à 3 nœuds)" begin
        ns = :c_mar
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 4, 3); is_param=true)
        NeuroDSL.set!(g, :W, randn(Float32, 3, 5))
        NeuroDSL.set!(g, :b, randn(Float32, 1, 5))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h, [:x, :W], :matmul; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:y, [:h, :b], :add; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:z, [:y], :relu; namespace=ns))

        rule = NeuroDSL.RewriteRule(:t_matmul_add_relu, (:matmul, :add, :relu), :fused_matmul_add_relu; cost_delta=0.5f0)
        plan = NeuroDSL.compile(g, NeuroDSL.CompilerConfig(rules=[rule]); namespace=ns)
        @test g.rules[ns][:z].op == :fused_matmul_add_relu

        out_fused = Array(plan(g, :z))

        ns_ref = :c_mar_ref
        gref = NeuroDSL.NeuroGraph(namespace=ns_ref, device=dev)
        NeuroDSL.set!(gref, :x, Array(g.nodes[ns][:x].value))
        NeuroDSL.set!(gref, :W, Array(g.nodes[ns][:W].value))
        NeuroDSL.set!(gref, :b, Array(g.nodes[ns][:b].value))
        NeuroDSL.addrule!(gref, NeuroDSL.GraphRule(:h, [:x, :W], :matmul; namespace=ns_ref))
        NeuroDSL.addrule!(gref, NeuroDSL.GraphRule(:y, [:h, :b], :add; namespace=ns_ref))
        NeuroDSL.addrule!(gref, NeuroDSL.GraphRule(:z, [:y], :relu; namespace=ns_ref))
        out_ref = Array(NeuroDSL.demand!(gref, :z; ctx_store=NeuroDSL.CtxStore(), namespace=ns_ref))

        @test isapprox(out_fused, out_ref; atol=1f-5)
    end

    @testset "gradient check : fused_matmul_add" begin
        ns = :g_ma
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :A, randn(Float32, 2, 3); is_param=true)
        NeuroDSL.set!(g, :B, randn(Float32, 3, 4))
        NeuroDSL.set!(g, :bias, randn(Float32, 1, 4))
        NeuroDSL.set!(g, :Z, zeros(Float32, 2, 4); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:C, [:A, :B, :bias], :fused_matmul_add; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [:C, :Z], :mse_loss; namespace=ns))
        # Mêmes eps/tol que les checks :matmul existants dans test_backward.jl —
        # le défaut (eps=1e-4) est calibré pour des ops lisses comme rmsnorm, pas
        # pour des matmuls de petite dimension où le bruit de différences finies
        # domine plus vite.
        ok, _ = grad_check(g, :A, :L; eps=Float32(1e-3), tol=Float32(5e-3))
        @test ok
    end

    @testset "gradient check : fused_matmul_add_relu" begin
        ns = :g_mar
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        # Décalage pour éviter que trop d'entrées post-ReLU tombent près de 0
        # (le point de non-différentiabilité du ReLU fausserait le gradient
        # numérique par différences finies indépendamment de la correctness).
        NeuroDSL.set!(g, :A, randn(Float32, 2, 3) .+ 0.5f0; is_param=true)
        NeuroDSL.set!(g, :B, randn(Float32, 3, 4))
        NeuroDSL.set!(g, :bias, randn(Float32, 1, 4) .+ 0.5f0)
        NeuroDSL.set!(g, :Z, zeros(Float32, 2, 4); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:C, [:A, :B, :bias], :fused_matmul_add_relu; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [:C, :Z], :mse_loss; namespace=ns))
        ok, _ = grad_check(g, :A, :L; eps=Float32(1e-3), tol=Float32(5e-3))
        @test ok
    end

    @testset "gradient check : fused_qkv_projection" begin
        ns = :g_qkv
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 2, 3); is_param=true)
        NeuroDSL.set!(g, :Wqkv, randn(Float32, 3, 6))
        NeuroDSL.set!(g, :Z, zeros(Float32, 2, 6); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:qkv, [:x, :Wqkv], :fused_qkv_projection; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [:qkv, :Z], :mse_loss; namespace=ns))
        ok, _ = grad_check(g, :x, :L)
        @test ok
    end

    @testset "gradient check : identity (double_transpose_elim / add_zero_elim)" begin
        ns = :g_id
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 2, 4); is_param=true)
        NeuroDSL.set!(g, :Z, zeros(Float32, 2, 4); atom_type=NeuroDSL.Datom)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h, [:x], :identity; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [:h, :Z], :mse_loss; namespace=ns))
        ok, _ = grad_check(g, :x, :L)
        @test ok
    end

    @testset "garde training : refuse par défaut, autorisé avec training=false" begin
        ns = :c_gate
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 2, 3); is_param=true)
        NeuroDSL.set!(g, :W, randn(Float32, 3, 4))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h, [:x, :W], :matmul; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:z, [:h], :relu; namespace=ns))
        made_up = NeuroDSL.RewriteRule(:made_up_fusion, (:matmul, :relu), :made_up_op_no_grad; cost_delta=0.9f0)

        NeuroDSL.compile(g, NeuroDSL.CompilerConfig(rules=[made_up]); namespace=ns)
        @test g.rules[ns][:z].op == :relu   # refusé : pas de GRAD_RULES pour :made_up_op_no_grad

        ns2 = :c_gate2
        g2 = NeuroDSL.NeuroGraph(namespace=ns2, device=dev)
        NeuroDSL.set!(g2, :x, randn(Float32, 2, 3); is_param=true)
        NeuroDSL.set!(g2, :W, randn(Float32, 3, 4))
        NeuroDSL.addrule!(g2, NeuroDSL.GraphRule(:h, [:x, :W], :matmul; namespace=ns2))
        NeuroDSL.addrule!(g2, NeuroDSL.GraphRule(:z, [:h], :relu; namespace=ns2))
        NeuroDSL.compile(g2, NeuroDSL.CompilerConfig(rules=[made_up], training=false); namespace=ns2)
        @test g2.rules[ns2][:z].op == :made_up_op_no_grad   # autorisé explicitement
    end
end

@testset "Compiler — recompilation incrémentale" begin
    dev = NeuroDSL.Backend.CPUDevice()

    function build_chain(ns)
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 4, 3); is_param=true)
        NeuroDSL.set!(g, :W, randn(Float32, 3, 5))
        NeuroDSL.set!(g, :b, randn(Float32, 1, 5))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h, [:x, :W], :matmul; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:y, [:h, :b], :add; namespace=ns))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:z, [:y], :relu; namespace=ns))
        return g
    end
    fusion_rule() = NeuroDSL.RewriteRule(:t_mar, (:matmul, :add, :relu), :fused_matmul_add_relu; cost_delta=0.5f0)

    @testset "is_dirty commence à false" begin
        ns = :r_init
        g = build_chain(ns)
        plan = NeuroDSL.compile(g, NeuroDSL.CompilerConfig(rules=[fusion_rule()]); namespace=ns)
        @test !NeuroDSL.is_dirty(plan)
    end

    @testset "set! marque le plan sale" begin
        ns = :r_dirty
        g = build_chain(ns)
        plan = NeuroDSL.compile(g, NeuroDSL.CompilerConfig(rules=[fusion_rule()]); namespace=ns)
        NeuroDSL.set!(g, :x, randn(Float32, 4, 3); is_param=true)
        @test NeuroDSL.is_dirty(plan)
    end

    @testset "recompile! est un no-op quand le plan n'est pas sale" begin
        ns = :r_noop
        g = build_chain(ns)
        plan = NeuroDSL.compile(g, NeuroDSL.CompilerConfig(rules=[fusion_rule()]); namespace=ns)
        NeuroDSL.recompile!(plan, g)
        @test plan.n_recompiles == 0
        NeuroDSL.recompile!(plan, g)
        @test plan.n_recompiles == 0
    end

    @testset "plan(g, out) déclenche recompile! et recalcule vraiment (pas de valeur mémoïsée)" begin
        ns = :r_auto
        g = build_chain(ns)
        plan = NeuroDSL.compile(g, NeuroDSL.CompilerConfig(rules=[fusion_rule()]); namespace=ns)
        plan(g, :z)
        @test plan.n_recompiles == 0

        x_new = randn(Float32, 4, 3)
        NeuroDSL.set!(g, :x, x_new; is_param=true)
        @test NeuroDSL.is_dirty(plan)

        out = Array(plan(g, :z))
        @test plan.n_recompiles == 1
        @test !NeuroDSL.is_dirty(plan)

        ns_ref = :r_auto_ref
        gref = NeuroDSL.NeuroGraph(namespace=ns_ref, device=dev)
        NeuroDSL.set!(gref, :x, x_new)
        NeuroDSL.set!(gref, :W, Array(g.nodes[ns][:W].value))
        NeuroDSL.set!(gref, :b, Array(g.nodes[ns][:b].value))
        NeuroDSL.addrule!(gref, NeuroDSL.GraphRule(:h, [:x, :W], :matmul; namespace=ns_ref))
        NeuroDSL.addrule!(gref, NeuroDSL.GraphRule(:y, [:h, :b], :add; namespace=ns_ref))
        NeuroDSL.addrule!(gref, NeuroDSL.GraphRule(:z, [:y], :relu; namespace=ns_ref))
        out_ref = Array(NeuroDSL.demand!(gref, :z; ctx_store=NeuroDSL.CtxStore(), namespace=ns_ref))

        @test isapprox(out, out_ref; atol=1f-5)
    end
end

@testset "Compiler — pooling mémoire" begin
    dev = NeuroDSL.Backend.CPUDevice()

    function build_mlp(ns, atom)
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x,  randn(Float32, 10, 6); is_param=true)
        NeuroDSL.set!(g, :W1, randn(Float32, 6, 6);  is_param=true)
        NeuroDSL.set!(g, :W2, randn(Float32, 6, 6);  is_param=true)
        NeuroDSL.set!(g, :W3, randn(Float32, 6, 1);  is_param=true)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h1, [:x, :W1],  :matmul; namespace=ns, atom_type=atom))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h2, [:h1],      :relu;   namespace=ns, atom_type=atom))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h3, [:h2, :W2], :matmul; namespace=ns, atom_type=atom))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h4, [:h3],      :relu;   namespace=ns, atom_type=atom))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:out, [:h4, :W3], :matmul; namespace=ns, atom_type=atom))
        return g
    end

    @testset "mode inference (training=false) recycle les intermédiaires Datom" begin
        ns = :p_inf
        g = build_mlp(ns, NeuroDSL.Datom)
        plan = NeuroDSL.compile(g, NeuroDSL.CompilerConfig(rules=NeuroDSL.RewriteRule[], training=false); namespace=ns)
        plan(g, :out)
        stats = NeuroDSL.pool_stats(plan.pool)
        @test stats.allocs + stats.hits > 0
        @test g.nodes[ns][:h1].value === nothing   # intermédiaire recyclé
    end

    @testset "mode training ne libère jamais un intermédiaire Quantom (backward reste valide)" begin
        ns = :p_tr
        g = build_mlp(ns, NeuroDSL.Quantom)
        plan = NeuroDSL.compile(g, NeuroDSL.CompilerConfig(rules=NeuroDSL.RewriteRule[], training=true); namespace=ns)
        plan(g, :out)
        @test g.nodes[ns][:h1].value !== nothing

        ctx = NeuroDSL.CtxStore()
        NeuroDSL.demand!(g, :out; ctx_store=ctx, namespace=ns)
        # Ne doit pas planter : backward_graph! a besoin des valeurs forward des
        # intermédiaires, qui ne doivent jamais avoir été recyclées ci-dessus.
        NeuroDSL.backward_graph!(g, :out; ctx_store=ctx, namespace=ns)
        @test g.nodes[ns][:W1].gradient !== nothing
    end

    @testset "params et sortie demandée jamais recyclés (les deux modes)" begin
        for (ns, atom, training) in [(:p_params_inf, NeuroDSL.Datom, false), (:p_params_tr, NeuroDSL.Quantom, true)]
            g = build_mlp(ns, atom)
            plan = NeuroDSL.compile(g, NeuroDSL.CompilerConfig(rules=NeuroDSL.RewriteRule[], training=training); namespace=ns)
            plan(g, :out)
            @test g.nodes[ns][:W1].value !== nothing
            @test g.nodes[ns][:W2].value !== nothing
            @test g.nodes[ns][:W3].value !== nothing
            @test g.nodes[ns][:out].value !== nothing
        end
    end
end

@testset "Compiler — sélection par coût (sdpa_fusion vs flash_attention, preuve de concept)" begin
    # _choose_cheapest_alternative n'est pas exporté (mécanisme interne, scopé à
    # une seule paire de règles) — testé directement avec des RuleMatch
    # synthétiques pour valider la logique de sélection indépendamment du fait
    # que sdpa_fusion et flash_attention produisent aujourd'hui des chaînes qui
    # ne se recoupent jamais littéralement (voir compiler_rules.jl).
    dev = NeuroDSL.Backend.CPUDevice()
    ns = :cost_sel
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.set!(g, :root, randn(Float32, 8, 8))

    cheap_rule = NeuroDSL.RewriteRule(:sdpa_fusion, (:matmul,), :fused_sdpa; cost_delta=0.1f0)
    expensive_rule = NeuroDSL.RewriteRule(:flash_attention, (:matmul,), :flash_attention_dummy; cost_delta=0.1f0)

    m_cheap     = NeuroDSL.RuleMatch(cheap_rule,     [:root], ns, 0.1f0)
    m_expensive = NeuroDSL.RuleMatch(expensive_rule, [:root], ns, 0.1f0)

    cost_fn(op, shapes; rule_cost_delta=0f0) = op == :fused_sdpa ? 1.0f0 : 100.0f0

    filtered = NeuroDSL._choose_cheapest_alternative(g, [m_cheap, m_expensive], cost_fn; namespace=ns)
    @test length(filtered) == 1
    @test filtered[1].rule.result == :fused_sdpa

    @testset "instances non recoupantes des deux règles sont toutes conservées" begin
        ns2 = :cost_sel2
        g2 = NeuroDSL.NeuroGraph(namespace=ns2, device=dev)
        NeuroDSL.set!(g2, :root_a, randn(Float32, 8, 8))
        NeuroDSL.set!(g2, :root_b, randn(Float32, 8, 8))
        ma = NeuroDSL.RuleMatch(cheap_rule,     [:root_a], ns2, 0.1f0)
        mb = NeuroDSL.RuleMatch(expensive_rule, [:root_b], ns2, 0.1f0)
        filtered2 = NeuroDSL._choose_cheapest_alternative(g2, [ma, mb], cost_fn; namespace=ns2)
        @test length(filtered2) == 2
    end
end
