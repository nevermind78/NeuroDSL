# Tests par invariants aléatoires sur le cœur réactif de NeuroDSL.
#
# Complète l'oracle de replay différentiel (test_oracle_replay.jl, UNE séquence fixe) en
# testant les mêmes classes d'invariants sur BEAUCOUP de topologies/séquences aléatoires
# différentes, pour attraper des bugs que la séquence fixe ne déclencherait pas par hasard.
#
# Aucune bibliothèque de property-based testing n'est utilisée (aucune n'est déclarée dans
# Project.toml et aucun ajout de dépendance n'est permis pour le moment) -- même idiome que
# le reste du dépôt : `Random.seed!`/`MersenneTwister` + boucles `for trial in 1:n`.
#
# Chaque invariant est vérifié contre un oracle de référence écrit INDÉPENDAMMENT (sans
# appeler les fonctions testées), pour ne pas se valider soi-même.

@testset "Invariants aléatoires du cœur réactif" begin
    dev = NeuroDSL.Backend.CPUDevice()

    # ── Oracle indépendant : table de consommateurs recalculée from scratch ────────────
    function _gt_consumers(g::NeuroDSL.NeuroGraph, ns::Symbol)
        idx = Dict{Symbol,Vector{Symbol}}()
        for (out_sym, rule) in g.rules[ns]
            for inp in rule.inputs
                push!(get!(idx, inp, Symbol[]), out_sym)
            end
        end
        return idx
    end

    function _consumers_maps_equal(a::Dict, b::Dict)
        Set(keys(a)) == Set(keys(b)) || return false
        return all(Set(a[k]) == Set(b[k]) for k in keys(a))
    end

    # ── Oracle indépendant : atteignabilité "aval" non conditionnée par `valid` ─────────
    # (reproduit la structure de _invalidate_downstream! -- consommateurs + watchers --
    # mais sans jamais lire ni muter `valid`, pour servir de vérité terrain neutre.)
    function _gt_reachable_downstream(g::NeuroDSL.NeuroGraph, ns::Symbol, target::Symbol)
        consumers = _gt_consumers(g, ns)
        visited = Set{Symbol}([target])
        queue = Symbol[target]
        while !isempty(queue)
            cur = pop!(queue)
            nd = get(g.nodes[ns], cur, nothing)
            if nd !== nothing
                for w in nd.watchers
                    w in visited || (push!(visited, w); push!(queue, w))
                end
            end
            for out_sym in get(consumers, cur, Symbol[])
                out_sym in visited || (push!(visited, out_sym); push!(queue, out_sym))
            end
        end
        return visited
    end

    # ── Oracle indépendant : composante faiblement connexe (JUMP+CLIMB de _invalidate_upstream!) ─
    function _gt_reachable_component(g::NeuroDSL.NeuroGraph, ns::Symbol, target::Symbol)
        consumers = _gt_consumers(g, ns)
        visited = Set{Symbol}([target])
        queue = Symbol[target]
        while !isempty(queue)
            cur = pop!(queue)
            for out_sym in get(consumers, cur, Symbol[])   # JUMP
                out_sym in visited || (push!(visited, out_sym); push!(queue, out_sym))
            end
            rule = get(g.rules[ns], cur, nothing)           # CLIMB
            if rule !== nothing
                for inp in rule.inputs
                    inp in visited || (push!(visited, inp); push!(queue, inp))
                end
            end
        end
        return visited
    end

    function _val_eq(a, b)
        (a === nothing) != (b === nothing) && return false
        a === nothing && return true
        return a == b
    end

    function _snapshot(g::NeuroDSL.NeuroGraph, ns::Symbol)
        Dict(sym => (nd.valid, nd.value === nothing ? nothing : copy(nd.value))
             for (sym, nd) in g.nodes[ns])
    end

    # ── Générateur 1 : DAG élémentaire minimal (structure pure, formes préservées) ──────
    function _random_elementwise_dag!(g::NeuroDSL.NeuroGraph, ns::Symbol, rng;
                                       n_leaves::Int=3, n_extra::Int=15, shape=(4, 4))
        syms = Symbol[]
        for i in 1:n_leaves
            s = Symbol(:leaf_, i)
            NeuroDSL.set!(g, s, randn(rng, Float32, shape...); namespace=ns)
            push!(syms, s)
        end
        for i in 1:n_extra
            op = rand(rng, (:add, :mul, :relu))
            out = Symbol(:n_, i)
            if op == :relu
                a = rand(rng, syms)
                NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out, [a], op; namespace=ns))
            else
                a, b = rand(rng, syms), rand(rng, syms)
                NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out, [a, b], op; namespace=ns))
            end
            push!(syms, out)
        end
        return syms
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Invariant : `_consumers_index!` reste synchronisé avec `g.rules[ns]` après
    # n'importe quelle séquence de `addrule!`/`set!`.
    # ═════════════════════════════════════════════════════════════════════════════
    @testset "_consumers_index! synchronisé avec g.rules[ns] (addrule!/set! aléatoires)" begin
        for trial in 1:15
            rng = MersenneTwister(1000 + trial)
            ns = Symbol(:consumers_trial_, trial)
            g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
            syms = Symbol[]
            for i in 1:rand(rng, 2:4)
                s = Symbol(:leaf_, i)
                NeuroDSL.set!(g, s, randn(rng, Float32, 4, 4); namespace=ns)
                push!(syms, s)
            end
            for step in 1:25
                if rand(rng) < 0.15 && !isempty(syms)
                    # set! sur une feuille existante -- ne doit pas désynchroniser le cache
                    NeuroDSL.set!(g, rand(rng, syms), randn(rng, Float32, 4, 4); namespace=ns)
                else
                    op = rand(rng, (:add, :mul, :relu))
                    out = Symbol(:n_, step)
                    if op == :relu
                        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out, [rand(rng, syms)], op; namespace=ns))
                    else
                        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out, [rand(rng, syms), rand(rng, syms)], op; namespace=ns))
                    end
                    push!(syms, out)
                end
                @test _consumers_maps_equal(_gt_consumers(g, ns), NeuroDSL._consumers_index!(g, ns))
            end
        end
    end

    @testset "_consumers_index! synchronisé après _fuse! (matmul+relu)" begin
        for trial in 1:5
            rng = MersenneTwister(2000 + trial)
            ns = Symbol(:fuse_trial_, trial)
            g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
            d = 4
            NeuroDSL.set!(g, :x, randn(rng, Float32, d, d); namespace=ns)
            NeuroDSL.set!(g, :W, randn(rng, Float32, d, d); namespace=ns)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:h, [:x, :W], :matmul; namespace=ns))
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:y, [:h], :relu; namespace=ns))
            @test _consumers_maps_equal(_gt_consumers(g, ns), NeuroDSL._consumers_index!(g, ns))
            ok = NeuroDSL._fuse!(g, [:h, :y]; ns=ns)
            @test ok
            @test _consumers_maps_equal(_gt_consumers(g, ns), NeuroDSL._consumers_index!(g, ns))
        end
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Invariant : topo_order! respecte inputs-avant-output, sur beaucoup de DAGs aléatoires
    # ═════════════════════════════════════════════════════════════════════════════
    @testset "topo_order! : inputs avant output, sur DAGs aléatoires" begin
        for trial in 1:20
            rng = MersenneTwister(3000 + trial)
            ns = Symbol(:topo_trial_, trial)
            g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
            _random_elementwise_dag!(g, ns, rng; n_leaves=rand(rng, 2:5), n_extra=rand(rng, 5:25))
            order = NeuroDSL.topo_order!(g; namespace=ns)
            pos = Dict(s => i for (i, s) in enumerate(order))
            @test length(order) == length(g.nodes[ns])
            for (out_sym, rule) in g.rules[ns]
                for inp in rule.inputs
                    @test pos[inp] < pos[out_sym]
                end
            end
        end
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Invariant : _invalidate_downstream! atteint exactement le cône aval, jamais plus
    # ═════════════════════════════════════════════════════════════════════════════
    @testset "_invalidate_downstream! : atteint le cône aval, jamais les nœuds hors-cône" begin
        for trial in 1:15
            rng = MersenneTwister(4000 + trial)
            ns = Symbol(:down_trial_, trial)
            g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
            syms = _random_elementwise_dag!(g, ns, rng; n_leaves=rand(rng, 2:4), n_extra=rand(rng, 8:20))
            # Tout valider d'abord (comme après un demand! complet)
            for (_, nd) in g.nodes[ns]
                nd.valid = true
            end
            target = rand(rng, syms)
            reachable = _gt_reachable_downstream(g, ns, target)
            before = Dict(s => nd.valid for (s, nd) in g.nodes[ns])

            NeuroDSL._invalidate_downstream!(g, target, ns)

            for (s, nd) in g.nodes[ns]
                if s in reachable
                    before[s] && @test !nd.valid   # devait être invalidé
                else
                    @test nd.valid == before[s]     # jamais touché hors du cône
                end
            end
        end
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Invariant : _invalidate_upstream! atteint exactement la composante connexe
    # (JUMP+CLIMB), jamais les sous-graphes disjoints
    # ═════════════════════════════════════════════════════════════════════════════
    @testset "_invalidate_upstream! : atteint la composante connexe, jamais les nœuds disjoints" begin
        for trial in 1:15
            rng = MersenneTwister(5000 + trial)
            ns = Symbol(:up_trial_, trial)
            g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
            # Deux composantes disjointes dans le même namespace, pour vérifier qu'on ne
            # déborde jamais de l'une vers l'autre.
            syms_a = _random_elementwise_dag!(g, ns, rng; n_leaves=2, n_extra=8, shape=(3, 3))
            g2_shape = (3, 3)
            syms_b = Symbol[]
            for i in 1:2
                s = Symbol(:leafB_, i)
                NeuroDSL.set!(g, s, randn(rng, Float32, g2_shape...); namespace=ns)
                push!(syms_b, s)
            end
            for i in 1:8
                op = rand(rng, (:add, :mul, :relu))
                out = Symbol(:nB_, i)
                if op == :relu
                    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out, [rand(rng, syms_b)], op; namespace=ns))
                else
                    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out, [rand(rng, syms_b), rand(rng, syms_b)], op; namespace=ns))
                end
                push!(syms_b, out)
            end

            target = rand(rng, syms_a)
            component = _gt_reachable_component(g, ns, target)
            @test isempty(intersect(component, Set(syms_b)))  # les deux DAGs sont bien disjoints

            for (_, nd) in g.nodes[ns]
                nd.gradient = randn(rng, Float32, 3, 3)   # forme arbitraire, non testée ici
                nd.backwarded = true
            end
            before_grad = Dict(s => copy(nd.gradient) for (s, nd) in g.nodes[ns])

            NeuroDSL._invalidate_upstream!(g, target, ns)

            for (s, nd) in g.nodes[ns]
                if s in component
                    @test nd.gradient === nothing
                    @test nd.backwarded == false
                else
                    @test nd.gradient == before_grad[s]  # jamais touché hors composante
                end
            end
        end
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Invariant : demand! ne recalcule QUE les nœuds invalides, jamais les autres
    # ═════════════════════════════════════════════════════════════════════════════
    @testset "demand! : pureté (ne touche que ce qui est invalide)" begin
        for trial in 1:10
            rng = MersenneTwister(6000 + trial)
            ns = Symbol(:demand_trial_, trial)
            g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
            syms = _random_elementwise_dag!(g, ns, rng; n_leaves=3, n_extra=rand(rng, 10:20))
            target = syms[end]
            NeuroDSL.demand!(g, target; namespace=ns)  # tout devient valide

            # (a) demand! sur cible déjà valide : rien ne doit changer nulle part.
            before = _snapshot(g, ns)
            NeuroDSL.demand!(g, target; namespace=ns)
            after = _snapshot(g, ns)
            @test before == after

            # (b) set! sur une feuille puis demand! : tout nœud dont la VALEUR change devait
            # être invalide juste avant le recalcul.
            leaf = syms[1]
            NeuroDSL.set!(g, leaf, randn(rng, Float32, 4, 4); namespace=ns)
            mid = _snapshot(g, ns)
            NeuroDSL.demand!(g, target; namespace=ns)
            aft = _snapshot(g, ns)
            for s in keys(aft)
                if !_val_eq(mid[s][2], aft[s][2])
                    @test mid[s][1] == false
                end
            end
        end
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Invariant 3 (patching.jl) : _adaptive_restore! choisit cache-ou-recalcul, mais le
    # résultat doit toujours être identique à une restauration par recalcul pur, y
    # compris quand les cônes de sites se recoupent.
    # ═════════════════════════════════════════════════════════════════════════════
    @testset "_adaptive_restore! == restauration par recalcul, cônes qui se recoupent" begin
        for trial in 1:8
            rng = MersenneTwister(7000 + trial)
            Random.seed!(7000 + trial)
            ns = Symbol(:adaptive_trial_, trial)
            n_heads = rand(rng, (2, 4))
            d_head = rand(rng, (4, 8))
            dim = n_heads * d_head
            n_layers = rand(rng, 2:3)
            hidden_dim = dim * 2
            seq_len = 8

            g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
            NeuroDSL.set!(g, :input, randn(Float32, seq_len, dim); namespace=ns)
            output_sym = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim)(g, :input; namespace=ns)
            NeuroDSL.demand!(g, output_sym; namespace=ns)
            clean_output = copy(NeuroDSL.demand!(g, output_sym; namespace=ns))
            clean_cache = NeuroDSL.capture_activations(g, ns)

            Xc = randn(Float32, seq_len, dim)
            Xc[1, :] .= randn(Float32, dim)
            NeuroDSL.set!(g, :input, Xc; namespace=ns)
            corrupted_output = copy(NeuroDSL.demand!(g, output_sym; namespace=ns))
            corrupted_cache = NeuroDSL.capture_activations(g, ns)

            # Deux têtes de la même couche 1 -- leurs cônes se recoupent forcément en aval
            # (hcat_heads puis toutes les couches suivantes sont partagées).
            site_a = Symbol(:layer_1_mha_ao_h, 1)
            site_b = Symbol(:layer_1_mha_ao_h, 2)
            cone_a = NeuroDSL._downstream_nodes(g, site_a, ns)
            cone_b = NeuroDSL._downstream_nodes(g, site_b, ns)
            @test !isempty(intersect(cone_a, cone_b))  # confirme le recoupement attendu

            # Site A déjà retenu (patché clean), puis on teste le candidat B par-dessus.
            NeuroDSL.patch_node!(g, site_a, clean_cache; namespace=ns)
            NeuroDSL.demand!(g, output_sym; namespace=ns)
            selected_cone_union = copy(cone_a)

            NeuroDSL.patch_node!(g, site_b, clean_cache; namespace=ns)
            NeuroDSL.demand!(g, output_sym; namespace=ns)

            NeuroDSL._adaptive_restore!(g, ns, site_b, cone_b, selected_cone_union, corrupted_cache, output_sym)
            state_adaptive = NeuroDSL.capture_activations(g, ns)

            # Référence indépendante : depuis le même point (B patché par-dessus A), annuler
            # B par pur recalcul (jamais par cache), quel que soit le recoupement.
            NeuroDSL.patch_node!(g, site_b, clean_cache; namespace=ns)
            NeuroDSL.demand!(g, output_sym; namespace=ns)
            NeuroDSL.patch_node!(g, site_b, corrupted_cache; namespace=ns)
            NeuroDSL.demand!(g, output_sym; namespace=ns)
            state_recompute = NeuroDSL.capture_activations(g, ns)

            @test Set(keys(state_adaptive)) == Set(keys(state_recompute))
            @test all(isapprox(Array(state_adaptive[k]), Array(state_recompute[k]); atol=Float32(1e-6))
                      for k in keys(state_adaptive))
        end
    end
end
