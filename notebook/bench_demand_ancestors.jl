using NeuroDSL, Random, Printf

# Graphe principal : chaîne profonde (dim=64, depth=DEPTH), PLUS plusieurs
# branches "gelées" indépendantes accrochées au même namespace (jamais
# demandées, jamais invalidées après leur premier calcul -- simule des
# `graft_shadow_block!` non retenus qui restent dans le graphe) pour que le
# préfixe topologique soit bien plus grand que le cône ancêtre réel d'une
# cible peu profonde.
function build(depth::Int, n_frozen_branches::Int, dim::Int; ns::Symbol)
    dev = NeuroDSL.Backend.CPUDevice()
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    Random.seed!(1)
    NeuroDSL.set!(g, :x0, randn(Float32, dim, dim); namespace=ns)
    h = :x0
    for i in 1:depth
        Wsym = Symbol(:W, i)
        NeuroDSL.set!(g, Wsym, randn(Float32, dim, dim); is_param=true, namespace=ns)
        mm = Symbol(:h, i)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(mm, [h, Wsym], :matmul; namespace=ns))
        h = mm
    end
    shallow_target = Symbol(:h, 2)  # tout au début de la chaîne -- ancêtres = {x0, W1, h1, W2, h2}

    # branches gelées : chacune une petite chaîne indépendante, jamais
    # touchée par la cible peu profonde.
    for b in 1:n_frozen_branches
        NeuroDSL.set!(g, Symbol(:fx, b), randn(Float32, dim, dim); namespace=ns)
        fh = Symbol(:fx, b)
        for i in 1:5
            fW = Symbol(:fW, b, :_, i)
            NeuroDSL.set!(g, fW, randn(Float32, dim, dim); is_param=true, namespace=ns)
            fo = Symbol(:fh, b, :_, i)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(fo, [fh, fW], :matmul; namespace=ns))
            fh = fo
        end
        NeuroDSL.demand!(g, fh; namespace=ns)  # calculé une fois, jamais réinvalidé après
    end
    loss_target = h  # sortie finale -- ancêtres = presque tout le graphe
    return g, shallow_target, loss_target
end

function bench(depth, n_frozen, dim)
    g, shallow, losssym = build(depth, n_frozen, dim; ns=:bench)
    n_nodes = length(g.nodes[:bench])
    anc_shallow = NeuroDSL._ancestors_of!(g, :bench, shallow)
    anc_loss = NeuroDSL._ancestors_of!(g, :bench, losssym)
    @printf "  nœuds totaux dans le graphe        : %d\n" n_nodes
    @printf "  |ancêtres(cible peu profonde)|      : %d\n" length(anc_shallow)
    @printf "  |ancêtres(cible :loss/sortie)|      : %d\n" length(anc_loss)

    # invalide tout, puis chronomètre demand! sur les deux cibles séparément
    NeuroDSL.invalidate_all!(g; namespace=:bench)
    t_shallow = @elapsed NeuroDSL.demand!(g, shallow; namespace=:bench)
    for _ in 1:5  # warm-up JIT
        NeuroDSL.invalidate_all!(g; namespace=:bench)
        NeuroDSL.demand!(g, shallow; namespace=:bench)
    end
    ts = Float64[]
    for _ in 1:20
        NeuroDSL.invalidate_all!(g; namespace=:bench)
        push!(ts, @elapsed NeuroDSL.demand!(g, shallow; namespace=:bench))
    end
    @printf "  demand!(cible peu profonde) médiane : %.4f ms\n" (sort(ts)[10]*1000)

    NeuroDSL.invalidate_all!(g; namespace=:bench)
    for _ in 1:5
        NeuroDSL.invalidate_all!(g; namespace=:bench)
        NeuroDSL.demand!(g, losssym; namespace=:bench)
    end
    tl = Float64[]
    for _ in 1:20
        NeuroDSL.invalidate_all!(g; namespace=:bench)
        push!(tl, @elapsed NeuroDSL.demand!(g, losssym; namespace=:bench))
    end
    @printf "  demand!(cible :loss, cas dégénéré)  médiane : %.4f ms\n" (sort(tl)[10]*1000)
end

println("="^70)
println("depth=30, n_frozen_branches=40, dim=64  (préfixe >> cône réel pour la cible peu profonde)")
println("="^70)
bench(30, 40, 64)
