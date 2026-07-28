# ══════════════════════════════════════════════════════════════════════════════
# Prototype de benchmark : "Self-Healing Reactive Computational Graphs" -- idée
# proposée par l'utilisateur, évaluée et prototypée ici, PAS encore publiée.
#
# Reprend le noyau déjà validé de notebook/tri.ipynb (Bellman-Ford sur
# semi-anneau tropical, poids d'arêtes = paramètre entraînable, mutation
# structurelle à chaud) et l'étend à V=16..100 noeuds sur une grille 2D, avec
# un vrai protocole : (a) latence de ré-adaptation après un choc topologique,
# (b) surcoût mémoire (mesuré en allocations Julia -- PAS encore comparé à
# PyTorch, voir le rapport joint pour pourquoi), (c) écart à l'oracle Dijkstra.
#
# JUGEMENT (voir rapport complet ailleurs) : la comparaison "PyTorch/JAX
# subirait une latence de recompilation prohibitive" ne tient QUE pour un
# encodage sparse/à structure Python dynamique -- l'encodage naturel et
# honnête de CE problème (matrice d'adjacence dense V×V, arêtes coupées =
# valeur mise à Inf) ne force NI JAX ni NeuroDSL à rien "recompiler" : la
# forme du tenseur ne change jamais. Ce script mesure donc ce qui est
# RÉELLEMENT différent chez NeuroDSL sur ce cas précis : éviter de
# RECONSTRUIRE le graphe de règles à chaque perturbation (par opposition à
# une invalidation "en cône" qui économiserait du calcul -- elle ne le fait
# PAS ici, voir plus bas, tous les étages dépendent de :W).
# ══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra, NeuroDSL, Random, Statistics

# ── Opérateur tropical, IDENTIQUE à celui validé dans tri.ipynb ─────────────
register_op!(:tropical_matmul, (dev, out, inputs, attrs, out_sym, out_node, ctx) -> begin
    d = inputs[1]; W = inputs[2]
    V = size(W, 1)
    argmin_idx = zeros(Int, V)
    for j in 1:V
        min_val = Inf32; best_i = 1
        for i in 1:V
            val = d[i, 1] + W[i, j]
            if val < min_val
                min_val = val; best_i = i
            end
        end
        out[j, 1] = min_val
        argmin_idx[j] = best_i
    end
    if ctx !== nothing
        ctx[out_sym] = Dict{Symbol, Any}(:argmin_idx => argmin_idx, :V => V)
    end
end)
CUSTOM_SHAPE_RULES[:tropical_matmul] = (inputs, attrs) -> (size(inputs[2], 1), 1)
GRAD_RULES[:tropical_matmul] = (dev, dy, ctx, inputs) -> begin
    argmin_idx = ctx[:argmin_idx]; V = ctx[:V]
    dd = zeros(Float32, V, 1); dW = zeros(Float32, V, V)
    for j in 1:V
        best_i = argmin_idx[j]
        dd[best_i, 1] += dy[j, 1]
        dW[best_i, j] += dy[j, 1]
    end
    return (dd, dW)
end

# ── Génération de graphe : grille 2D (rows x cols), poids aléatoires ────────
function build_grid_W(rows::Int, cols::Int; seed=0, wmin=1f0, wmax=10f0)
    rng = MersenneTwister(seed)
    V = rows * cols
    W = fill(Inf32, V, V)
    for i in 1:V; W[i, i] = 0f0; end
    idx(r, c) = (r - 1) * cols + c
    edges = Tuple{Int,Int}[]
    for r in 1:rows, c in 1:cols
        if c < cols
            push!(edges, (idx(r, c), idx(r, c + 1)))
        end
        if r < rows
            push!(edges, (idx(r, c), idx(r + 1, c)))
        end
    end
    for (a, b) in edges
        w = wmin + (wmax - wmin) * rand(rng, Float32)
        W[a, b] = w
        W[b, a] = w   # grille non orientée
    end
    return W, edges
end

# ── Oracle : Dijkstra standard O(V^2), sans NeuroDSL, pour vérité-terrain ───
function dijkstra(W::Matrix{Float32}, source::Int)
    V = size(W, 1)
    dist = fill(Inf32, V)
    dist[source] = 0f0
    visited = falses(V)
    for _ in 1:V
        u = -1; best = Inf32
        for v in 1:V
            if !visited[v] && dist[v] < best
                best = dist[v]; u = v
            end
        end
        u == -1 && break
        visited[u] = true
        for v in 1:V
            if W[u, v] < Inf32 && dist[u] + W[u, v] < dist[v]
                dist[v] = dist[u] + W[u, v]
            end
        end
    end
    return dist
end

# ── Construction du graphe NeuroDSL (Bellman-Ford, V-1 étages chaînés) ──────
function build_bf_graph(dev, V::Int, W_init::Matrix{Float32}, source::Int; nsx=:bf)
    g = NeuroGraph(namespace=nsx, device=dev)
    d_init = fill(Inf32, V, 1); d_init[source, 1] = 0f0
    NeuroDSL.set!(g, :d_0, d_init; is_param=false, namespace=nsx)
    NeuroDSL.set!(g, :W, W_init; is_param=true, namespace=nsx)
    for step in 1:V-1
        in_sym = Symbol(:d_, step - 1); out_sym = Symbol(:d_, step)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out_sym, [in_sym, :W], :tropical_matmul; namespace=nsx))
    end
    return g, Symbol(:d_, V - 1)
end

# ── Choc topologique : coupe une fraction des arêtes EXISTANTES ────────────
function cut_edges(W::Matrix{Float32}, edges::Vector{Tuple{Int,Int}}, frac::Float64; seed=1)
    rng = MersenneTwister(seed)
    n_cut = round(Int, frac * length(edges))
    cut_idx = randperm(rng, length(edges))[1:n_cut]
    W2 = copy(W)
    for i in cut_idx
        a, b = edges[i]
        W2[a, b] = Inf32
        W2[b, a] = Inf32
    end
    return W2, n_cut
end

# ── Protocole de benchmark complet pour une taille V donnée ─────────────────
function run_benchmark(rows::Int, cols::Int; source::Int=1, shock_frac::Float64=0.2, seed=0)
    V = rows * cols
    println("="^70)
    println("BENCHMARK -- grille $(rows)x$(cols) (V=$V noeuds)")
    println("="^70)

    dev = NeuroDSL.Backend.CPUDevice()
    W0, edges = build_grid_W(rows, cols; seed=seed)
    println("Arêtes : $(length(edges))")

    # --- Construction initiale + premier calcul ---
    t_build0 = @elapsed begin
        g, final_node = build_bf_graph(dev, V, W0, source)
        d0 = NeuroDSL.demand!(g, final_node; namespace=:bf)
    end
    println("Construction + premier calcul (froid)  : $(round(t_build0*1000, digits=3)) ms")

    # --- Vérification de justesse vs oracle Dijkstra AVANT le choc ---
    oracle0 = dijkstra(W0, source)
    err0 = maximum(abs.(vec(d0) .- oracle0))
    println("Écart max vs oracle Dijkstra (avant choc) : $err0")

    # --- Choc topologique ---
    W1, n_cut = cut_edges(W0, edges, shock_frac; seed=seed + 1)
    println("Choc : $n_cut arêtes coupées sur $(length(edges)) ($(round(100*n_cut/length(edges),digits=1))%)")

    # --- Voie RÉACTIVE : set! + demand! sur le graphe EXISTANT ---
    mem_reactive = @allocated begin
        NeuroDSL.set!(g, :W, W1; is_param=true, namespace=:bf)
        global d1_reactive = NeuroDSL.demand!(g, final_node; namespace=:bf)
    end
    t_reactive = @elapsed begin
        NeuroDSL.set!(g, :W, W1; is_param=true, namespace=:bf)
        d1_reactive = NeuroDSL.demand!(g, final_node; namespace=:bf)
    end

    # --- Voie RECONSTRUCTION COMPLÈTE : nouveau graphe from scratch ---
    mem_rebuild = @allocated begin
        g2, final_node2 = build_bf_graph(dev, V, W1, source; nsx=:bf_rebuild)
        global d1_rebuild = NeuroDSL.demand!(g2, final_node2; namespace=:bf_rebuild)
    end
    t_rebuild = @elapsed begin
        g2, final_node2 = build_bf_graph(dev, V, W1, source; nsx=:bf_rebuild)
        d1_rebuild = NeuroDSL.demand!(g2, final_node2; namespace=:bf_rebuild)
    end

    # --- Oracle APRÈS choc, pour vérifier l'optimalité des deux voies ---
    oracle1 = dijkstra(W1, source)
    err_reactive = maximum(abs.(vec(d1_reactive) .- oracle1))
    err_rebuild  = maximum(abs.(vec(d1_rebuild) .- oracle1))

    println()
    println("── Après le choc ──")
    println("Latence RÉACTIVE (set! + demand!, graphe réutilisé)     : $(round(t_reactive*1000, digits=4)) ms")
    println("Latence RECONSTRUCTION (nouveau graphe from scratch)     : $(round(t_rebuild*1000, digits=4)) ms")
    println("Ratio rebuild/réactif                                    : $(round(t_rebuild/max(t_reactive,1e-9), digits=2))x")
    println("Mémoire allouée -- réactif    : $(mem_reactive) octets")
    println("Mémoire allouée -- rebuild    : $(mem_rebuild) octets")
    println("Ratio mémoire rebuild/réactif : $(round(mem_rebuild/max(mem_reactive,1), digits=2))x")
    println("Écart max vs oracle Dijkstra (voie réactive)  : $err_reactive")
    println("Écart max vs oracle Dijkstra (voie rebuild)   : $err_rebuild")
    println()

    return (; V, t_build0, t_reactive, t_rebuild, mem_reactive, mem_rebuild, err_reactive, err_rebuild, n_cut, n_edges=length(edges))
end

# ── Premier test À PETITE ÉCHELLE (obligatoire avant tout run coûteux) ─────
res16 = run_benchmark(4, 4; seed=0)

println("\n" * "#"^70)
println("# Second test, plus grand (V=100, grille 10x10) -- si le petit test")
println("# est concluant, coût O(V^3) attendu ~ (100/16)^3 ≈ 244x plus cher")
println("# que V=16, donc encore trivial en absolu (V=16 est de l'ordre du ms).")
println("#"^70)
res100 = run_benchmark(10, 10; seed=0)
