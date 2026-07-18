# Partitionnement d'un graphe en régions par découpe (src/graph_regions.jl).
#
# Le graphe de Bell/Stirling est le banc d'essai : Bₙ = Σₖ S(n,k) avec S(n,k) = k·S(n-1,k) +
# S(n-1,k-1). On DÉCLARE le graphe entier une fois, on résout la surface de coupe (la rangée n-1)
# depuis les nœuds nommés, on coupe, on colore, et on vérifie que chaque région restreinte calcule
# exactement sa tranche — la région aval ne reconstruit JAMAIS le triangle amont.

@testset "Partitionnement en régions (cut/assign/crossings/restrict)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    @defop scale_add out = attrs[:factor] * inputs[1] + inputs[2]
    @defop nsum (d, out, inputs, attrs, out_sym, nd, ctx) -> begin
        copyto!(out, inputs[1]); for i in 2:length(inputs); out .+= inputs[i]; end
    end

    BELL_N = 6                                  # B₆ = 203 ; rangée 5 = S(5,·) = [1,15,25,10,1]
    build_whole = function ()
        g = NeuroGraph(namespace = :main, device = dev)
        @neuro g ns=:main operators=[:scale_add] begin
            @node one  = 1.0
            @node zero = 0.0
            @rule stirling(n::Int, k::Int) = begin
                (n == 0 && k == 0) ? :one : (k == 0 || k > n) ? :zero :
                    scale_add(factor = k, stirling(n - 1, k), stirling(n - 1, k - 1))
            end
            @node bell = nsum([stirling(BELL_N, k) for k in 1:BELL_N]...)
        end
        g
    end

    @testset "surface résout la rangée nommée" begin
        g = build_whole()
        row5 = surface(g, :stirling, BELL_N - 1; ns = :main)
        @test length(row5) == BELL_N - 1                      # 5 entrées non nulles
        @test [g.nodes[:main][s].aux_data[:label] for s in row5] ==
              ["stirling(5, $k)" for k in 1:5]                # trié par l'argument k
        # les résultats constants (:zero / :one) ne sont jamais étiquetés → exclus
        @test all(s -> haskey(g.nodes[:main][s].aux_data, :call), row5)
    end

    @testset "cut partitionne en deux pièces" begin
        g = build_whole()
        row5 = surface(g, :stirling, BELL_N - 1; ns = :main)
        plan = RegionPlan(g; ns = :main)
        upstream, downstream = cut(plan, row5)
        # partition exacte du graphe entier
        @test isempty(intersect(upstream.nodes, downstream.nodes))
        @test union(upstream.nodes, downstream.nodes) == Set(keys(g.nodes[:main]))
        # la surface est dans l'amont ; bell est en aval ; le triangle amont ne l'est pas
        @test all(s -> s in upstream.nodes, row5)
        @test :bell in downstream.nodes
        @test !(:bell in upstream.nodes)
        # cut ne colore rien
        @test isempty(plan.region_of)
    end

    @testset "assign! colore, nothing laisse non coloré, pièces recoupables" begin
        g = build_whole()
        row5 = surface(g, :stirling, BELL_N - 1; ns = :main)
        plan = RegionPlan(g; ns = :main)
        lower, upper = cut!(plan, row5; above = :rega, below = :regb)
        @test all(n -> plan.region_of[n] === :rega, lower.nodes)
        @test all(n -> plan.region_of[n] === :regb, upper.nodes)

        # nothing des deux côtés : découpe pure, aucune couleur posée
        plan2 = RegionPlan(build_whole(); ns = :main)
        cut!(plan2, surface(build_whole(), :stirling, BELL_N - 1; ns = :main))
        @test isempty(plan2.region_of)

        # une pièce se recoupe : raffiner l'aval en deux
        top6 = surface(g, :stirling, BELL_N; ns = :main)      # rangée S(6,·)
        u2, d2 = cut!(upper, top6; above = :regb, below = :regc)
        @test plan.region_of[:bell] === :regc                 # bell est le seul aval de la rangée 6
        @test all(n -> plan.region_of[n] === :regb, top6)
    end

    @testset "crossings dérive les transferts (nœuds calculés seulement)" begin
        g = build_whole()
        row5 = surface(g, :stirling, BELL_N - 1; ns = :main)
        plan = RegionPlan(g; ns = :main)
        cut!(plan, row5; above = :rega, below = :regb)
        xs = crossings(plan)
        @test length(xs) == length(row5)                      # une traversée par entrée de rangée 5
        @test all(x -> x.src === :rega && x.dst === :regb, xs)
        @test Set(x.node for x in xs) == Set(row5)
        # zero est une feuille amont lue en aval : répliquée, JAMAIS expédiée
        @test !any(x -> x.node === :zero, xs)
    end

    @testset "restrict_to_region! : tranches disjointes, pas de recalcul amont" begin
        g = build_whole()
        row5 = surface(g, :stirling, BELL_N - 1; ns = :main)
        plan = RegionPlan(g; ns = :main)
        cut!(plan, row5; above = :rega, below = :regb)

        ga = restrict_to_region!(build_whole(), plan, :rega)
        gb = restrict_to_region!(build_whole(), plan, :regb)

        # région A : calcule le triangle et expose la rangée 5
        @test [demand!(ga, s; namespace = :main)[] for s in row5] == Float32[1, 15, 25, 10, 1]
        @test !haskey(ga.nodes[:main], :bell)

        # région B : n'a AUCUN nœud du triangle intérieur (rangées ≤ 4) — pas de reconstruction
        b_labels = [get(nd.aux_data, :label, "") for nd in values(gb.nodes[:main])]
        @test !any(l -> occursin("stirling(4,", l) || occursin("stirling(3,", l), b_labels)
        # la rangée 5 y est une FEUILLE injectable (sa règle a été retirée)
        @test all(s -> haskey(gb.nodes[:main], s) && !haskey(gb.rules[:main], s), row5)
        # zero répliquée localement (feuille avec valeur)
        @test haskey(gb.nodes[:main], :zero) && gb.nodes[:main][:zero].value !== nothing

        # inject la rangée 5 → B calcule Bₙ sans jamais toucher le triangle
        for s in row5
            set!(gb, s, demand!(ga, s; namespace = :main); namespace = :main)
        end
        @test demand!(gb, :bell; namespace = :main)[] == 203f0
    end

    @testset "piece : coloration d'ensembles explicites (front pour un solveur)" begin
        # Un solveur de placement colore des ENSEMBLES de nœuds directement (pas via une découpe) ;
        # `piece` fabrique la pièce depuis n'importe quel ensemble, et la coloration seule — pas la
        # forme du cône — dicte les traversées.
        g = build_whole()
        row5 = surface(g, :stirling, BELL_N - 1; ns = :main)
        plan = RegionPlan(g; ns = :main)
        up, down = cut(plan, row5)
        assign!(piece(plan, up.nodes), :rega)                 # colorer un ensemble de nœuds explicite
        assign!(piece(plan, down.nodes), :regb)
        @test all(s -> plan.region_of[s] === :rega, row5)
        @test plan.region_of[:bell] === :regb
        xs = crossings(plan)
        @test Set(x.node for x in xs) == Set(row5)            # coupe propre : 5 traversées
    end

    @testset "restrict signale un nœud calculé nourricier non coloré" begin
        g = build_whole()
        row5 = surface(g, :stirling, BELL_N - 1; ns = :main)
        plan = RegionPlan(g; ns = :main)
        cut!(plan, row5; below = :regb)                       # amont laissé non coloré (above=nothing)
        @test_throws ErrorException restrict_to_region!(build_whole(), plan, :regb)
    end
end
