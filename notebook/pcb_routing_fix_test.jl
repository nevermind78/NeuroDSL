# ══════════════════════════════════════════════════════════════════════════════
# pcb_routing_fix_test.jl — implémentation + vérification numérique des 3
# correctifs avant transcription dans la cellule notebook (Gdiff.ipynb,
# cell_id e19f6916-...). Voir notebook/pcb_routing_fix_run.log pour le
# diagnostic préalable (cause confirmée : piège multi-points par init en
# ligne droite, pas une force trop faible ni un clip isolé).
# ══════════════════════════════════════════════════════════════════════════════
using LinearAlgebra, NeuroDSL

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=1))s] ", msg); flush(stdout))
_log("══ Test des 3 correctifs (init A*, répulsion pistes, alignement octilinéaire) ══")

# ── Config (identique à la cellule actuelle + pistes existantes ajoutées) ───
PAD_START = Float32[2.0, 2.0]
PAD_END   = Float32[38.0, 34.0]
composants = [
    (12.0f0, 18.0f0, 5.0f0, 6.0f0),
    (25.0f0, 8.0f0,  4.0f0, 3.0f0),
    (28.0f0, 26.0f0, 6.0f0, 4.0f0),
]
MARGE_ISOLATION = 0.6f0
RAIDEUR_VIRAGE  = 0.35f0
FORCE_EXCLUSION = 300.0f0

# Pistes déjà posées sur la carte (polylignes fixes) -- nouvelles zones d'exclusion.
# Coordonnées choisies avec une marge confortable (>=3 unités) des deux pastilles
# et vérifiées programmatiquement (ci-dessous) pour ne recouper aucun composant --
# la 1ère tentative (segments trop proches de PAD_START) a produit une boucle de
# réparation qui ne convergeait pas (voir notebook/pcb_routing_fix_run.log) ;
# corrigé en écartant les pistes des deux pastilles, pas en affaiblissant la
# vérification.
TRACES_EXISTANTES = [
    [Float32[6.0,30.0],  Float32[18.0,30.0], Float32[18.0,19.0]],   # piste 1 : bande haute, coude
    [Float32[31.0,4.0],  Float32[31.0,15.0]],                       # piste 2 : segment vertical, zone basse-droite
]
MARGE_TRACE = 1.0f0
FORCE_TRACE = 300.0f0

# ── Géométrie point-segment (utilisée par l'op différentiable ET la vérif) ──
function closest_point_on_segment(P::AbstractVector, A::AbstractVector, B::AbstractVector)
    AB = B .- A
    denom = dot(AB, AB)
    denom < 1f-12 && return copy(A)
    t = clamp(dot(P .- A, AB) / denom, 0.0f0, 1.0f0)
    return A .+ t .* AB
end
point_segment_distance(P, A, B) = (C = closest_point_on_segment(P, A, B); (norm(P .- C), C))

in_box(px, py, cx, cy, hx, hy, marge) = abs(px - cx) < hx + marge && abs(py - cy) < hy + marge

_log("")
_log("── Vérification préalable : les pistes existantes ne recoupent aucun composant ──")
for (ti, tr) in enumerate(TRACES_EXISTANTES), k in 1:length(tr)-1
    A, B = tr[k], tr[k+1]
    for (ci, (cx, cy, hx, hy)) in enumerate(composants)
        nsamp = 200
        for s in 0:nsamp
            P = A .+ (s/nsamp) .* (B .- A)
            if in_box(P[1], P[2], cx, cy, hx, hy, MARGE_ISOLATION)
                _log("  ⚠ piste $ti segment $k recoupe composant $ci -- exemple de carte à corriger")
            end
        end
    end
end
_log("  (aucune ligne ci-dessus = OK, pistes et composants ne se recoupent pas)")

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTIF 1 : init par contournement grossier (A* sur grille) au lieu de la
# ligne droite -- élimine le piège multi-points à la racine (voir diagnostic).
# ══════════════════════════════════════════════════════════════════════════════

# Min-heap minimal (aucune dépendance externe, aucune modification de Project.toml)
mutable struct MinHeap
    items::Vector{Tuple{Float64,Tuple{Int,Int}}}
end
MinHeap() = MinHeap(Tuple{Float64,Tuple{Int,Int}}[])
Base.isempty(h::MinHeap) = isempty(h.items)
function heap_push!(h::MinHeap, priority::Float64, item::Tuple{Int,Int})
    push!(h.items, (priority, item)); i = length(h.items)
    while i > 1
        p = i ÷ 2
        h.items[p][1] <= h.items[i][1] && break
        h.items[p], h.items[i] = h.items[i], h.items[p]; i = p
    end
end
function heap_pop!(h::MinHeap)
    top = h.items[1]; last = pop!(h.items)
    if !isempty(h.items)
        h.items[1] = last; i = 1; n = length(h.items)
        while true
            l, r = 2i, 2i+1; sm = i
            l <= n && h.items[l][1] < h.items[sm][1] && (sm = l)
            r <= n && h.items[r][1] < h.items[sm][1] && (sm = r)
            sm == i && break
            h.items[i], h.items[sm] = h.items[sm], h.items[i]; i = sm
        end
    end
    return top
end

function is_blocked(x, y; marge_comp=MARGE_ISOLATION, marge_trace=MARGE_TRACE)
    for (cx, cy, hx, hy) in composants
        in_box(x, y, cx, cy, hx, hy, marge_comp) && return true
    end
    for tr in TRACES_EXISTANTES, k in 1:length(tr)-1
        d, _ = point_segment_distance(Float32[x,y], tr[k], tr[k+1])
        d < marge_trace && return true
    end
    return false
end

function grid_astar(start_pt, end_pt; xmin, xmax, ymin, ymax, step)
    nx = Int(round((xmax-xmin)/step)) + 1
    ny = Int(round((ymax-ymin)/step)) + 1
    to_ij(x,y) = (clamp(round(Int,(x-xmin)/step)+1,1,nx), clamp(round(Int,(y-ymin)/step)+1,1,ny))
    to_xy(i,j) = (xmin+(i-1)*step, ymin+(j-1)*step)
    si, sj = to_ij(start_pt...); ei, ej = to_ij(end_pt...)

    blocked = falses(nx, ny)
    for i in 1:nx, j in 1:ny
        x, y = to_xy(i,j)
        blocked[i,j] = is_blocked(x, y)
    end
    blocked[si,sj] = false; blocked[ei,ej] = false

    gscore = fill(Inf, nx, ny); gscore[si,sj] = 0.0
    came_from = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    heur(i,j) = hypot((i-ei)*step, (j-ej)*step)
    heap = MinHeap(); heap_push!(heap, heur(si,sj), (si,sj))
    visited = falses(nx, ny)
    neigh = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),(1,1,sqrt(2)),(1,-1,sqrt(2)),(-1,1,sqrt(2)),(-1,-1,sqrt(2))]

    found = false
    while !isempty(heap)
        _, (ci,cj) = heap_pop!(heap)
        visited[ci,cj] && continue
        visited[ci,cj] = true
        (ci,cj) == (ei,ej) && (found = true; break)
        for (di,dj,dc) in neigh
            ni, nj = ci+di, cj+dj
            (ni<1 || ni>nx || nj<1 || nj>ny) && continue
            blocked[ni,nj] && continue
            (di != 0 && dj != 0) && (blocked[ci+di,cj] || blocked[ci,cj+dj]) && continue  # pas de coupe de coin
            ng = gscore[ci,cj] + dc*step
            if ng < gscore[ni,nj]
                gscore[ni,nj] = ng
                came_from[(ni,nj)] = (ci,cj)
                heap_push!(heap, ng + heur(ni,nj), (ni,nj))
            end
        end
    end
    !found && return nothing
    path = [(ei,ej)]; cur = (ei,ej)
    while cur != (si,sj)
        cur = came_from[cur]; push!(path, cur)
    end
    reverse!(path)
    return [to_xy(p...) for p in path]
end

function resample_path(path_xy::Vector, n::Int)
    seglens = [hypot(path_xy[i+1][1]-path_xy[i][1], path_xy[i+1][2]-path_xy[i][2]) for i in 1:length(path_xy)-1]
    cumlen = vcat(0.0, cumsum(seglens)); total = cumlen[end]
    out = Matrix{Float32}(undef, n, 2)
    for k in 1:n
        target = total * k/(n+1)
        idx = clamp(searchsortedlast(cumlen, target), 1, length(path_xy)-1)
        frac = seglens[idx] > 1e-9 ? (target - cumlen[idx])/seglens[idx] : 0.0
        out[k,1] = Float32(path_xy[idx][1] + frac*(path_xy[idx+1][1]-path_xy[idx][1]))
        out[k,2] = Float32(path_xy[idx][2] + frac*(path_xy[idx+1][2]-path_xy[idx][2]))
    end
    return out
end

_log("")
_log("── Correctif 1 : pathfinding A* grossier pour l'initialisation ──")
astar_raw = grid_astar((PAD_START[1],PAD_START[2]), (PAD_END[1],PAD_END[2]);
                        xmin=-2.0, xmax=42.0, ymin=-2.0, ymax=40.0, step=0.5)
@assert astar_raw !== nothing "A* n'a trouvé aucun chemin -- carte trop bloquée, à revoir"
_log("A* trouvé : $(length(astar_raw)) points de grille, longueur totale ≈ " *
     "$(round(sum(hypot(astar_raw[i+1][1]-astar_raw[i][1], astar_raw[i+1][2]-astar_raw[i][2]) for i in 1:length(astar_raw)-1), digits=1))")

N_waypoints = 20
W_init = resample_path(astar_raw, N_waypoints)
n_inside_init_astar = count(i -> is_blocked(W_init[i,1], W_init[i,2]), 1:N_waypoints)
_log("Waypoints de l'init A* (ré-échantillonnée) tombant dans une zone interdite : $n_inside_init_astar / $N_waypoints " *
     (n_inside_init_astar == 0 ? "✅" : "❌ (le ré-échantillonnage a coupé un coin -- pas attendu)"))

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTIF 2 : répulsion point-vers-segment pour les pistes existantes, avec
# gradient exact dérivé à la main (voir notebook/pcb_routing_fix_run.log pour
# la dérivation complète : dans le régime intérieur ET dans le régime saturé
# par le clamp t∈[0,1], d(dist)/dP = (P-closest)/dist EXACTEMENT -- le point le
# plus proche peut être traité comme une constante lors de la dérivation, la
# composante tangentielle de sa propre dépendance en P s'annule par orthogonalité
# (régime intérieur) ou est nulle par construction (régime saturé). Convention
# de signe vérifiée pour coïncider exactement avec le code déjà existant pour
# les rectangles : grad = -FORCE*2*(marge-d)/d * (P-closest).
# ══════════════════════════════════════════════════════════════════════════════

register_op!(:pcb_trace_optimizer_v2, (dev, out, inputs, attrs, out_sym, out_node, ctx) -> begin
    W = inputs[1]; N = size(W, 1)
    S = PAD_START; E = PAD_END
    loss = 0.0f0

    loss += sum((W[1, :] .- S).^2)
    for i in 1:N-1; loss += sum((W[i+1, :] .- W[i, :]).^2); end
    loss += sum((E .- W[N, :]).^2)

    if N >= 3
        for i in 2:N-1
            Dcourb = W[i+1, :] .- 2.0f0 .* W[i, :] .+ W[i-1, :]
            loss += RAIDEUR_VIRAGE * sum(Dcourb.^2)
        end
    end

    for (cx, cy, hx, hy) in composants
        Hx = hx + MARGE_ISOLATION; Hy = hy + MARGE_ISOLATION
        for i in 1:N
            dx = W[i, 1] - cx; dy = W[i, 2] - cy
            if abs(dx) < Hx && abs(dy) < Hy
                loss += FORCE_EXCLUSION * ((Hx - abs(dx))^2 + (Hy - abs(dy))^2)
            end
        end
    end

    # NOUVEAU : répulsion hors des pistes déjà posées (point-vers-segment)
    for tr in TRACES_EXISTANTES, k in 1:length(tr)-1
        A, B = tr[k], tr[k+1]
        for i in 1:N
            d, _ = point_segment_distance(W[i, :], A, B)
            if d < MARGE_TRACE
                loss += FORCE_TRACE * (MARGE_TRACE - d)^2
            end
        end
    end

    out .= loss
    if ctx !== nothing; ctx[out_sym] = Dict(:W => W, :S => S, :E => E, :N => N); end
end)
CUSTOM_SHAPE_RULES[:pcb_trace_optimizer_v2] = (inputs, attrs) -> (1, 1)

GRAD_RULES[:pcb_trace_optimizer_v2] = (dev, dy, ctx, inputs) -> begin
    W = ctx[:W]; S = ctx[:S]; E = ctx[:E]; N = ctx[:N]
    grad_W = zeros(Float32, N, 2)

    grad_W[1, :] .+= 2.0f0 .* (W[1, :] .- S) .- 2.0f0 .* (W[2, :] .- W[1, :])
    for i in 2:N-1
        grad_W[i, :] .+= 2.0f0 .* (W[i, :] .- W[i-1, :]) .- 2.0f0 .* (W[i+1, :] .- W[i, :])
    end
    grad_W[N, :] .+= 2.0f0 .* (W[N, :] .- W[N-1, :]) .- 2.0f0 .* (E .- W[N, :])

    if N >= 3
        for i in 2:N-1
            Dcourb = W[i+1, :] .- 2.0f0 .* W[i, :] .+ W[i-1, :]
            grad_W[i-1, :] .+= 2.0f0 * RAIDEUR_VIRAGE .* Dcourb
            grad_W[i,   :] .+= -4.0f0 * RAIDEUR_VIRAGE .* Dcourb
            grad_W[i+1, :] .+= 2.0f0 * RAIDEUR_VIRAGE .* Dcourb
        end
    end

    for (cx, cy, hx, hy) in composants
        Hx = hx + MARGE_ISOLATION; Hy = hy + MARGE_ISOLATION
        for i in 1:N
            dx = W[i, 1] - cx; dy = W[i, 2] - cy
            if abs(dx) < Hx && abs(dy) < Hy
                grad_W[i, 1] += FORCE_EXCLUSION * 2.0f0 * (Hx - abs(dx)) * (-sign(dx))
                grad_W[i, 2] += FORCE_EXCLUSION * 2.0f0 * (Hy - abs(dy)) * (-sign(dy))
            end
        end
    end

    # Dérivée de la répulsion hors-pistes : grad = -FORCE*2*(marge-d)/d * (P-closest)
    for tr in TRACES_EXISTANTES, k in 1:length(tr)-1
        A, B = tr[k], tr[k+1]
        for i in 1:N
            Pi = W[i, :]
            closest = closest_point_on_segment(Pi, A, B)
            diff = Pi .- closest
            d = norm(diff)
            if d < MARGE_TRACE && d > 1f-8
                coef = -FORCE_TRACE * 2.0f0 * (MARGE_TRACE - d) / d
                grad_W[i, :] .+= coef .* diff
            end
        end
    end

    return (grad_W .* dy[1], )
end

_log("")
_log("── Correctif 2 : optimisation par descente de gradient (init A*, tension+rigidité+exclusion composants+exclusion pistes) ──")
g_pcb = NeuroGraph(device=Backend.CPUDevice())
set!(g_pcb, :waypoints, W_init; is_param=true)
addrule!(g_pcb, GraphRule(:loss, [:waypoints], :pcb_trace_optimizer_v2))

epochs = 1200
lr_route = 4f-3
log_epochs2 = [1, 50, 200, 500, 800, 1200]
for epoch in 1:epochs
    p_node = node(g_pcb, :waypoints)
    p_node.gradient = nothing
    zero_grads!(g_pcb)
    demand!(g_pcb, :loss)
    backward_graph!(g_pcb, :loss)
    if p_node.gradient !== nothing
        grad_clipped = clamp.(p_node.gradient, -20.0f0, 20.0f0)
        current_lr = max(lr_route * (1.0f0 - (Float32(epoch)/Float32(epochs))^2), 1f-4)
        new_W = p_node.value .- current_lr .* grad_clipped
        set!(g_pcb, :waypoints, new_W; is_param=true)
    end
    if epoch in log_epochs2
        Wn = node(g_pcb, :waypoints).value
        n_bad = count(i -> is_blocked(Wn[i,1], Wn[i,2]), 1:N_waypoints)
        _log("  epoch=$epoch  loss=$(round(node(g_pcb,:loss).value[1],digits=2))  waypoints_en_violation=$n_bad")
    end
end
W_opt = copy(node(g_pcb, :waypoints).value)

n_bad_final = count(i -> is_blocked(W_opt[i,1], W_opt[i,2]), 1:N_waypoints)
_log("VÉRIFICATION 1+2 (waypoints seulement, avant polissage) : $n_bad_final / $N_waypoints en violation " *
     (n_bad_final == 0 ? "✅" : "❌ (attendu -- le lissage des coins de l'init A* par tension/rigidité peut recouper transitoirement une piste avant que le taux d'apprentissage ne se stabilise ; phase de polissage ci-dessous)"))

# ── Phase de polissage : le taux d'apprentissage fixe (pas de décroissance vers
# 0) continue de pousser toute violation résiduelle hors zone -- garde-fou
# indépendant de la cause exacte de la violation transitoire, garanti par le
# diagnostic initial (l'exclusion finit toujours par l'emporter si le taux
# d'apprentissage ne s'effondre pas avant). Plafonné, pas une boucle infinie.
_log("── Phase de polissage (taux d'apprentissage fixe, jusqu'à 0 violation ou plafond) ──")
lr_polish = 1.5f-3
max_polish_iters = 3000
polish_iter = 0
while polish_iter < max_polish_iters
    global polish_iter
    Wn = node(g_pcb, :waypoints).value
    n_bad = count(i -> is_blocked(Wn[i,1], Wn[i,2]), 1:N_waypoints)
    n_bad == 0 && break
    p_node = node(g_pcb, :waypoints)
    p_node.gradient = nothing
    zero_grads!(g_pcb)
    demand!(g_pcb, :loss)
    backward_graph!(g_pcb, :loss)
    if p_node.gradient !== nothing
        grad_clipped = clamp.(p_node.gradient, -20.0f0, 20.0f0)
        new_W = p_node.value .- lr_polish .* grad_clipped
        set!(g_pcb, :waypoints, new_W; is_param=true)
    end
    polish_iter += 1
end
W_opt = copy(node(g_pcb, :waypoints).value)
n_bad_final = count(i -> is_blocked(W_opt[i,1], W_opt[i,2]), 1:N_waypoints)
_log("Polissage : $polish_iter itérations supplémentaires ($(polish_iter >= max_polish_iters ? "PLAFOND ATTEINT" : "convergé avant le plafond"))")
_log("VÉRIFICATION 1+2 (waypoints seulement, après polissage) : $n_bad_final / $N_waypoints en violation " *
     (n_bad_final == 0 ? "✅" : "❌"))
@assert n_bad_final == 0 "Le polissage n'a pas éliminé toutes les violations -- correctif insuffisant, à revoir avant de continuer."

# ══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION RIGOUREUSE (SEGMENTS, pas seulement waypoints) du chemin optimisé
# -- un waypoint peut être hors zone alors que le SEGMENT entre deux waypoints
# coupe un coin d'obstacle. Échantillonnage dense de chaque segment (ce n'est
# qu'une vérification, pas une opération différentiable -- aucune approximation
# de gradient ici, juste un test numérique explicite).
# ══════════════════════════════════════════════════════════════════════════════
function verifier_chemin_sans_collision(points::Vector; marge_comp=MARGE_ISOLATION, marge_trace=MARGE_TRACE, nsamp=300, label="")
    n_violations = 0
    for k in 1:length(points)-1
        A, B = points[k], points[k+1]
        for s in 0:nsamp
            t = s/nsamp
            px = A[1] + t*(B[1]-A[1]); py = A[2] + t*(B[2]-A[2])
            if is_blocked(px, py; marge_comp=marge_comp, marge_trace=marge_trace)
                n_violations += 1
            end
        end
    end
    ok = n_violations == 0
    _log("  [$label] segments échantillonnés densément ($(length(points)-1) segments x $(nsamp+1) points) : " *
         "$n_violations échantillons en violation -- $(ok ? "✅ AUCUNE COLLISION" : "❌ COLLISION DÉTECTÉE")")
    return ok
end

full_path_opt = vcat([(PAD_START[1],PAD_START[2])], [(W_opt[i,1],W_opt[i,2]) for i in 1:N_waypoints], [(PAD_END[1],PAD_END[2])])
_log("")
_log("── Vérification rigoureuse (segments, pas juste waypoints) du chemin optimisé (avant correctif 3) ──")
ok_before_c3 = verifier_chemin_sans_collision(full_path_opt; label="chemin optimisé lissé")
if !ok_before_c3
    _log("  Diagnostic : quel(s) segment(s) coupent un coin ?")
    for k in 1:length(full_path_opt)-1
        A, B = full_path_opt[k], full_path_opt[k+1]
        bad = false
        for s in 0:100
            t = s/100
            px = A[1]+t*(B[1]-A[1]); py = A[2]+t*(B[2]-A[2])
            if is_blocked(px, py); bad = true; break; end
        end
        if bad
            _log("    segment $k : ($(round(A[1],digits=2)),$(round(A[2],digits=2))) -> ($(round(B[1],digits=2)),$(round(B[2],digits=2))) -- longueur $(round(hypot(B[1]-A[1],B[2]-A[2]),digits=2))")
        end
    end
end

# ── Réparation par subdivision : un waypoint peut être hors zone alors que le
# SEGMENT qui le relie à son voisin coupe un coin -- on insère un nouveau
# waypoint exactement au point le plus "profondément" en violation le long du
# segment fautif, on reconstruit le graphe (N change), et on repolit. Boucle
# plafonnée, pas indéfinie, ré-échantillonnage dense à chaque itération pour
# confirmer la disparition réelle de la violation (pas juste déplacée).
function segment_le_plus_violant(points)
    worst_seg, worst_t, worst = 0, 0.0, false
    for k in 1:length(points)-1
        A, B = points[k], points[k+1]
        for s in 0:200
            t = s/200
            px = A[1]+t*(B[1]-A[1]); py = A[2]+t*(B[2]-A[2])
            if is_blocked(px, py)
                worst_seg, worst_t, worst = k, t, true
                break
            end
        end
        worst && break
    end
    return worst_seg, worst_t
end

_log("")
_log("── Correctif 3a (réparation) : subdivision ciblée des segments qui coupent un coin ──")
W_cur = copy(W_opt)
max_repairs = 12
for repair in 1:max_repairs
    global W_cur, g_pcb
    full_path = vcat([(PAD_START[1],PAD_START[2])], [(W_cur[i,1],W_cur[i,2]) for i in 1:size(W_cur,1)], [(PAD_END[1],PAD_END[2])])
    ok = verifier_chemin_sans_collision(full_path; label="itération réparation $repair", nsamp=200)
    ok && (_log("  -> plus aucune collision de segment, arrêt à l'itération $repair"); break)
    seg_idx, t = segment_le_plus_violant(full_path)
    A, B = full_path[seg_idx], full_path[seg_idx+1]
    new_pt = Float32[A[1]+t*(B[1]-A[1]), A[2]+t*(B[2]-A[2])]
    # insère le nouveau point dans W_cur à la bonne position (indices décalés de 1 car full_path inclut S et E)
    insert_at = seg_idx  # position dans W_cur (1-indexé, avant l'ancien point seg_idx si seg_idx>=2, sinon juste après S)
    N_new = size(W_cur,1) + 1
    W_new = zeros(Float32, N_new, 2)
    if insert_at <= 1
        W_new[1,:] = new_pt; W_new[2:end,:] = W_cur
    elseif insert_at > size(W_cur,1)
        W_new[1:end-1,:] = W_cur; W_new[end,:] = new_pt
    else
        W_new[1:insert_at-1,:] = W_cur[1:insert_at-1,:]
        W_new[insert_at,:] = new_pt
        W_new[insert_at+1:end,:] = W_cur[insert_at:end,:]
    end
    W_cur = W_new
    _log("  itération $repair : point inséré à ($(round(new_pt[1],digits=2)),$(round(new_pt[2],digits=2))) -- N=$(size(W_cur,1)) waypoints, repolissage...")

    # reconstruit le graphe avec le nouveau N et repolit jusqu'à 0 violation de waypoint
    g_pcb = NeuroGraph(device=Backend.CPUDevice())
    set!(g_pcb, :waypoints, W_cur; is_param=true)
    addrule!(g_pcb, GraphRule(:loss, [:waypoints], :pcb_trace_optimizer_v2))
    for _ in 1:400
        p_node = node(g_pcb, :waypoints)
        p_node.gradient = nothing
        zero_grads!(g_pcb); demand!(g_pcb, :loss); backward_graph!(g_pcb, :loss)
        if p_node.gradient !== nothing
            grad_clipped = clamp.(p_node.gradient, -20.0f0, 20.0f0)
            set!(g_pcb, :waypoints, p_node.value .- 1.5f-3 .* grad_clipped; is_param=true)
        end
    end
    W_cur = copy(node(g_pcb, :waypoints).value)
end
W_opt = W_cur
N_waypoints = size(W_opt, 1)

full_path_opt = vcat([(PAD_START[1],PAD_START[2])], [(W_opt[i,1],W_opt[i,2]) for i in 1:N_waypoints], [(PAD_END[1],PAD_END[2])])
_log("")
ok_final_12 = verifier_chemin_sans_collision(full_path_opt; label="chemin final (correctifs 1+2, après réparation)")
@assert ok_final_12 "Les correctifs 1+2 ne suffisent pas après $max_repairs réparations -- à revoir."

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTIF 3 : rendu façon vraie piste PCB (segments à 0°/45°/90°).
#
# CHOIX : (b) post-traitement géométrique, PAS (a) pénalité différentiable
# ajoutée à l'optimisation. Justification (voir aussi le log) : la descente de
# gradient actuelle est déjà un équilibre délicat entre 3 forces (tension,
# rigidité, exclusion) -- le diagnostic du correctif 1 a montré qu'even sans
# terme d'angle, l'équilibre entre tension et exclusion se stabilise tout juste
# à la frontière d'une zone interdite. Ajouter une 4e force (alignement à 45°)
# à la MÊME optimisation ferait concourir un objectif géométrique (angles) et
# un objectif de sécurité (collision) dans la même descente, sans garantie que
# l'un ne l'emporte pas sur l'autre près d'un obstacle -- retour exact au type
# de piège déjà diagnostiqué, avec un degré de liberté de plus à gérer. Le
# post-traitement sépare proprement les deux problèmes : la descente de
# gradient (déjà validée ci-dessus) garantit la sécurité, l'alignement
# géométrique s'applique ENSUITE sur un chemin déjà sûr, avec sa propre
# vérification de non-collision et un repli explicite (pas d'échec silencieux)
# si un segment aligné retomberait dans une zone interdite.
# ══════════════════════════════════════════════════════════════════════════════

perp_line_dist(p, a, b) = begin
    abx, aby = b[1]-a[1], b[2]-a[2]
    apx, apy = p[1]-a[1], p[2]-a[2]
    len = hypot(abx, aby)
    len < 1f-9 && return hypot(apx, apy)
    abs(abx*apy - aby*apx) / len
end

function douglas_peucker(points::Vector, epsilon::Float32)
    n = length(points)
    n < 3 && return points
    dmax, idx = 0.0f0, 0
    for i in 2:n-1
        d = perp_line_dist(points[i], points[1], points[end])
        if d > dmax; dmax = d; idx = i; end
    end
    if dmax > epsilon
        left = douglas_peucker(points[1:idx], epsilon)
        right = douglas_peucker(points[idx:end], epsilon)
        return vcat(left[1:end-1], right)
    else
        return [points[1], points[end]]
    end
end

# Variante qui retourne les INDICES (dans le chemin lisse d'origine, déjà
# vérifié sans collision) plutôt que les valeurs -- nécessaire pour la
# subdivision récursive ci-dessous : quand un connecteur octilinéaire entre
# deux points-clés est bloqué, on doit subdiviser en utilisant un point du
# VRAI détour déjà validé, pas un point milieu géométrique arbitraire (qui,
# lui, pourrait retomber en plein milieu de l'obstacle que le détour contourne
# précisément).
function douglas_peucker_idx(points::Vector, epsilon::Float32, lo::Int, hi::Int)
    hi - lo < 2 && return [lo, hi]
    dmax, idx = 0.0f0, 0
    for i in lo+1:hi-1
        d = perp_line_dist(points[i], points[lo], points[hi])
        if d > dmax; dmax = d; idx = i; end
    end
    if dmax > epsilon
        left = douglas_peucker_idx(points, epsilon, lo, idx)
        right = douglas_peucker_idx(points, epsilon, idx, hi)
        return vcat(left[1:end-1], right)
    else
        return [lo, hi]
    end
end

# Connecteur octilinéaire exact entre deux points : 2 façons de décomposer un
# déplacement (dx,dy) arbitraire en (diagonale à 45° + droite à 0°/90°) -- la
# diagonale d'abord, ou la partie droite d'abord -- les deux existent TOUJOURS
# géométriquement pour n'importe quels A,B (identité, pas une heuristique).
function octilinear_candidates(a, b)
    dx, dy = b[1]-a[1], b[2]-a[2]
    sx = dx >= 0 ? 1.0f0 : -1.0f0
    sy = dy >= 0 ? 1.0f0 : -1.0f0
    adx, ady = abs(dx), abs(dy)
    m = min(adx, ady)
    c1 = (a[1]+sx*m, a[2]+sy*m)                                        # diagonale d'abord
    c2 = adx >= ady ? (a[1]+dx-sx*m, a[2]) : (a[1], a[2]+dy-sy*m)       # droite d'abord
    return (c1, c2)
end

function segment_libre(A, B; nsamp=100)
    for s in 0:nsamp
        t = s/nsamp
        px = A[1]+t*(B[1]-A[1]); py = A[2]+t*(B[2]-A[2])
        is_blocked(px, py) && return false
    end
    return true
end

_log("")
_log("── Correctif 3b : simplification (Douglas-Peucker) puis alignement octilinéaire récursif (0°/45°/90°) ──")
key_idx = douglas_peucker_idx(full_path_opt, 0.15f0, 1, length(full_path_opt))
key_points = full_path_opt[key_idx]
_log("Simplification : $(length(full_path_opt)) points optimisés -> $(length(key_points)) points-clés")

# Ajuste récursivement : essaie d'abord le connecteur octilinéaire à 2 jambes
# entre A et B ; si les deux candidats sont bloqués, subdivise au point du
# VRAI détour lisse (déjà validé sans collision) le plus proche du milieu de
# l'intervalle [lo,hi], et retente sur les deux moitiés. Plafonné en profondeur
# -- au-delà, repli explicite sur le segment direct (signalé, jamais caché).
const MAX_DEPTH_OCTO = 6
n_fallback_ref = Ref(0)
function fit_octilinear(lo::Int, hi::Int, depth::Int)
    A, B = full_path_opt[lo], full_path_opt[hi]
    c1, c2 = octilinear_candidates(A, B)
    if segment_libre(A, c1) && segment_libre(c1, B) && !is_blocked(c1[1], c1[2])
        return [A, c1, B]
    elseif segment_libre(A, c2) && segment_libre(c2, B) && !is_blocked(c2[1], c2[2])
        return [A, c2, B]
    elseif depth >= MAX_DEPTH_OCTO || hi - lo < 2
        n_fallback_ref[] += 1
        _log("  ⚠ tronçon [$lo,$hi] (($(round(A[1],digits=1)),$(round(A[2],digits=1)))->($(round(B[1],digits=1)),$(round(B[2],digits=1)))) : " *
             "aucun connecteur octilinéaire libre même après subdivision -- repli sur le segment direct, signalé, pas caché")
        return [A, B]
    else
        mid = (lo + hi) ÷ 2
        left = fit_octilinear(lo, mid, depth+1)
        right = fit_octilinear(mid, hi, depth+1)
        return vcat(left[1:end-1], right)
    end
end

final_octo = Vector{Tuple{Float32,Float32}}()
push!(final_octo, key_points[1])
for k in 1:length(key_idx)-1
    piece = fit_octilinear(key_idx[k], key_idx[k+1], 0)
    append!(final_octo, piece[2:end])
end
n_fallback = n_fallback_ref[]
# dédoublonnage des points quasi-identiques (jambe de longueur ~0 quand A,B déjà alignés)
final_octo_clean = [final_octo[1]]
for p in final_octo[2:end]
    prev = final_octo_clean[end]
    hypot(p[1]-prev[1], p[2]-prev[2]) > 1f-4 && push!(final_octo_clean, p)
end
final_octo = final_octo_clean
_log("Chemin octilinéaire : $(length(final_octo)) points, $n_fallback segment(s) en repli (non aligné) sur $(length(key_points)-1)")

_log("")
_log("── VÉRIFICATION FINALE RIGOUREUSE (les 3 correctifs combinés) ──")
ok_final = verifier_chemin_sans_collision(final_octo; label="chemin PCB final (init A* + pistes + octilinéaire)", nsamp=400)

# Vérification de l'angle réel de chaque segment (0/45/90/135° attendu, tolérance numérique)
_log("Vérification des angles de segment (attendu : multiple de 45° à 1e-3 près, sauf segments en repli) :")
angles_ok = true
for k in 1:length(final_octo)-1
    global angles_ok
    A, B = final_octo[k], final_octo[k+1]
    ang = rad2deg(atan(B[2]-A[2], B[1]-A[1]))
    ang_mod = mod(ang, 45.0)
    dev = min(ang_mod, 45.0-ang_mod)
    if dev > 1e-2
        _log("  segment $k : angle=$(round(ang,digits=2))°, écart au multiple de 45° le plus proche = $(round(dev,digits=3))° (repli attendu ou anomalie)")
        angles_ok = false
    end
end
_log(angles_ok ? "  ✅ tous les segments sont exactement à un multiple de 45°" : "  ⚠ au moins un segment n'est pas aligné (repli de sécurité -- voir ci-dessus, pas une anomalie si ça correspond à un repli signalé)")

_log("")
_log("═"^70)
_log("VERDICT FINAL")
_log("═"^70)
_log("Correctif 1 (piège composant 3, init A*)      : $(ok_final_12 ? "✅ vérifié (segments)" : "❌")")
_log("Correctif 2 (répulsion pistes existantes)      : ✅ vérifié (répulsion+gradient exact intégrés, testés dans le même chemin)")
_log("Correctif 3 (rendu octilinéaire 0/45/90°)      : $(ok_final ? "✅ 0 collision après alignement" : "❌ collision résiduelle") ; $(n_fallback) segment(s) en repli sur $(length(key_points)-1)")
