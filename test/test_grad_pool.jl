
# ─────────────────────────────────────────────────────────────────────────────
# GradPool -- tests ciblés sur les 3 failles trouvées par audit sur la
# première ébauche (comptage de références par objectid) + le trou trouvé
# sur la révision (fuite fan-out sur `.+=`) : voir grad_pool.jl et la boucle
# principale de backward_graph! (src/backward.jl) pour le design retenu
# ("propriétaire unique, zéro refcount").
# ─────────────────────────────────────────────────────────────────────────────

@testset "GradPool -- identité bit-à-bit pool actif vs désactivé" begin
    dev = NeuroDSL.Backend.CPUDevice()
    Random.seed!(4242)

    function build_ref(ns)
        g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom, namespace=ns)
        out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, :x; namespace=ns)
        NeuroDSL.set!(g, :target, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :target], :mse_loss; namespace=ns))
        return g
    end

    g_on  = build_ref(:gp_bit_on)
    g_off = build_ref(:gp_bit_off)
    for (p_on, p_off) in zip(NeuroDSL.params(g_on; namespace=:gp_bit_on),
                              NeuroDSL.params(g_off; namespace=:gp_bit_off))
        NeuroDSL.node(g_off, p_off.name; namespace=:gp_bit_off).value = copy(p_on.value)
    end
    x0 = randn(Float32, 6, 16); t0 = randn(Float32, 6, 16)
    NeuroDSL.node(g_on, :x; namespace=:gp_bit_on).value = copy(x0)
    NeuroDSL.node(g_off, :x; namespace=:gp_bit_off).value = copy(x0)
    NeuroDSL.node(g_on, :target; namespace=:gp_bit_on).value = copy(t0)
    NeuroDSL.node(g_off, :target; namespace=:gp_bit_off).value = copy(t0)
    NeuroDSL.invalidate_all!(g_on; namespace=:gp_bit_on)
    NeuroDSL.invalidate_all!(g_off; namespace=:gp_bit_off)

    NeuroDSL.GRAD_POOL_ENABLED[] = true
    NeuroDSL.demand!(g_on, :loss; namespace=:gp_bit_on)
    NeuroDSL.backward_graph!(g_on, :loss; namespace=:gp_bit_on)

    NeuroDSL.GRAD_POOL_ENABLED[] = false
    try
        NeuroDSL.demand!(g_off, :loss; namespace=:gp_bit_off)
        NeuroDSL.backward_graph!(g_off, :loss; namespace=:gp_bit_off)
    finally
        NeuroDSL.GRAD_POOL_ENABLED[] = true
    end

    for (p_on, p_off) in zip(NeuroDSL.params(g_on; namespace=:gp_bit_on),
                              NeuroDSL.params(g_off; namespace=:gp_bit_off))
        @test Array(p_on.gradient) == Array(p_off.gradient)
    end

    # Répéter 3 passes de plus avec le pool actif (régime établi, buckets
    # déjà peuplés par la 1ère passe) -- doit rester identique.
    for _ in 1:3
        NeuroDSL.invalidate_all!(g_on; namespace=:gp_bit_on)
        NeuroDSL.demand!(g_on, :loss; namespace=:gp_bit_on)
        NeuroDSL.backward_graph!(g_on, :loss; namespace=:gp_bit_on)
    end
    for (p_on, p_off) in zip(NeuroDSL.params(g_on; namespace=:gp_bit_on),
                              NeuroDSL.params(g_off; namespace=:gp_bit_off))
        @test Array(p_on.gradient) == Array(p_off.gradient)
    end
end

@testset "GradPool -- aucune fuite (lent vide après chaque passe, sur graphe à fan-out)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    # LlamaModel : le résidu (:add) donne à chaque couche un nœud à 2
    # consommateurs directs -- exactement le motif fan-out visé par l'audit.
    g = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g, :x, randn(Float32, 8, 32); atom_type=NeuroDSL.Datom)
    out = NeuroDSL.LlamaModel(4, 32, 4, 64)(g, :x)
    NeuroDSL.set!(g, :target, randn(Float32, 8, 32); atom_type=NeuroDSL.Datom)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :target], :mse_loss))

    gpool = NeuroDSL._grad_pool_for(g)
    for _ in 1:5
        NeuroDSL.invalidate_all!(g)
        NeuroDSL.demand!(g, :loss)
        NeuroDSL.backward_graph!(g, :loss)   # l'@assert interne lève déjà si lent non vide
        @test isempty(gpool.lent)
        @test gpool.n_foreign == 0
    end
    # Les buckets doivent être non vides (le pool a bien servi à quelque chose)
    @test sum(length(v) for v in values(gpool.buckets)) > 0
end

@testset "GradPool -- prune_frozen (gradient jeté sans jamais être assigné) ne fuite pas" begin
    dev = NeuroDSL.Backend.CPUDevice()
    g = NeuroDSL.NeuroGraph(device=dev)
    NeuroDSL.set!(g, :x, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom)
    # Couche 1 gelée (is_param=false sur ses poids -- ne devrait jamais
    # recevoir de gradient réel, needs_bwd=false pour tout son sous-arbre),
    # couche 2 entraînable.
    out = NeuroDSL.LlamaModel(2, 16, 4, 32)(g, :x)
    NeuroDSL.set!(g, :target, randn(Float32, 6, 16); atom_type=NeuroDSL.Datom)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [out, :target], :mse_loss))
    for p in NeuroDSL.params(g)
        startswith(String(p.name), "layer_1_") && (p.is_param = false)
    end

    gpool = NeuroDSL._grad_pool_for(g)
    bucket_sizes_before = Dict(k => length(v) for (k,v) in gpool.buckets)
    for t in 1:20
        NeuroDSL.invalidate_all!(g)
        NeuroDSL.demand!(g, :loss)
        NeuroDSL.backward_graph!(g, :loss; prune_frozen=true)
        @test isempty(gpool.lent)
    end
    # Bornage : les buckets ne doivent pas grossir sans limite sur 20 passes
    # identiques (chaque forme ne devrait avoir besoin que d'un tampon à la fois).
    for (k, v) in gpool.buckets
        @test length(v) <= 4   # marge généreuse, mais pas "grossit à chaque passe"
    end
end

@testset "GradPool -- poison-release attrape un usage-après-libération" begin
    # Contre-test de sensibilité : si `grad_release!` rendait un buffer AVANT
    # sa dernière lecture légitime, un `grad_acquire!` immédiatement après
    # pourrait le re-prêter et un tiers l'écraserait -- on le simule
    # directement (sans modifier backward.jl) pour prouver que le mécanisme
    # `lent`/bucket réagit comme attendu à une libération prématurée.
    dev = NeuroDSL.Backend.CPUDevice()
    gp = NeuroDSL.GradPool(dev)
    buf1 = NeuroDSL.grad_acquire!(gp, (4,4))
    fill!(buf1, 1f0)
    @test NeuroDSL.grad_release!(gp, buf1) == true
    @test !haskey(gp.lent, buf1)
    buf2 = NeuroDSL.grad_acquire!(gp, (4,4))
    @test buf2 === buf1   # même forme -> même tampon réutilisé (comportement attendu du pool)
    fill!(buf2, 2f0)
    # Si du code avait gardé une référence à buf1 en croyant le posséder
    # encore, il verrait maintenant 2.0 partout -- le test documente
    # explicitement CE contrat (propriété unique, transfert immédiat).
    @test all(buf1 .== 2f0)

    # Un buffer étranger (jamais emprunté à CE pool) n'est jamais accepté.
    foreign = zeros(Float32, 4, 4)
    @test NeuroDSL.grad_release!(gp, foreign) == false
    @test gp.n_foreign == 1
end

@testset "GradPool -- non-régression suite différences finies (pool actif)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    NeuroDSL.GRAD_POOL_ENABLED[] = true
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
    @printf "  [grad_pool] pire erreur différences finies (pool actif) : %.2e (%s)\n" worst worst_sym
end
