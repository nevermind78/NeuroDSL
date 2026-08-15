# =============================================================================
# ÉCHELLE EN PROFONDEUR À LARGEUR FIXE, SUR MODÈLES QUE NOUS ENTRAÎNONS
#
# POURQUOI CE PROTOCOLE PLUTÔT QUE DES MODÈLES DE SÉRIE
# -----------------------------------------------------
# La question est de savoir si n_bar croît comme log L ou linéairement en L.
# Les profondeurs entraînées disponibles hors ligne sont L = 24 et L = 28, soit
# une plage de x1.17 : le balayage synthétique a eu besoin de x32 pour séparer
# log L (R^2 = 0.965) de la meilleure loi de puissance (0.929), donc deux
# profondeurs à x1.17 ne distinguent RIEN. Le test sur modèles de série serait
# sous-puissant par construction, indépendamment de la VRAM.
# Surtout : dans toute famille de série, la largeur co-varie avec la
# profondeur, donc le confondant largeur y est irréductible. En entraînant
# l'échelle nous-mêmes, la largeur est FIXE et le confondant disparaît.
# Le prix, énoncé d'avance : ce sont des modèles caractère par caractère et
# petits, donc la portée devient « à largeur fixe, sur des modèles que nous
# avons entraînés » -- moins réaliste, causalement plus fort.
#
# DÉCISIONS PRÉ-ENREGISTRÉES (fixées AVANT de lancer, pas choisies après)
# ----------------------------------------------------------------------
# D1. CONTRÔLE DE LA QUANTITÉ D'ENTRAÎNEMENT. Deux protocoles, tous deux
#     rapportés, mesurés sur les MÊMES exécutions :
#       (a) PERTE APPARIÉE (primaire) : chaque profondeur est mesurée au
#           premier point d'évaluation où la perte de validation passe sous
#           TAU. Cela égalise la QUALITÉ DE LA FONCTION apprise, qui est le
#           confondant réellement en jeu quand on demande « n_bar à
#           entraînement comparable ».
#       (b) PAS ÉGAUX (secondaire) : toutes les profondeurs à l'étape S.
#     Pourquoi pas des FLOPs égaux : à FLOPs égaux les modèles profonds
#     reçoivent moins de pas et finissent donc systématiquement moins bien
#     entraînés à mesure que L croît, ce qui confond la profondeur avec le
#     sous-entraînement dans le sens qui FAVORISE la conclusion log L. La
#     perte appariée est le contrôle qui ne favorise aucune des deux formes.
#     Le choix pas/FLOPs peut renverser une conclusion (travaux antérieurs sur
#     les calendriers de croissance), d'où son pré-enregistrement ici.
# D1-bis. CALIBRAGE DE TAU PAR UN PILOTE, et ce que le pilote a démenti.
#     TAU ne peut pas être choisi au doigt mouillé : il doit être atteignable
#     par TOUTES les profondeurs. Le pilote (L = 4 et L = 64, 1 graine, 2000
#     pas, bench_eps_depth_trained_pilote_results.txt) a démenti mon hypothèse
#     de départ. J'attendais que la profondeur FAIBLE plafonne le plus haut ;
#     c'est l'inverse : à pas égaux, L = 64 finit à 2.5795 et L = 4 à 2.2373.
#     Le modèle PROFOND est le retardataire -- à lot 1 et pas d'apprentissage
#     fixe, la profondeur ralentit la progression par pas. Conséquences :
#       (i) TAU = 2.60, atteint par L = 4 vers l'étape 750 et par L = 64 vers
#           1750 dans le pilote, donc avec marge pour les deux extrêmes ;
#       (ii) le protocole À PAS ÉGAUX est CONFONDU, et sa direction est connue :
#            les modèles profonds y sont moins bien entraînés, ce qui ABAISSE
#            leur n_bar et FAVORISE donc artificiellement la forme log L. Le
#            protocole à PERTE APPARIÉE est celui qu'il faut lire ; le premier
#            n'est rapporté que pour montrer l'ampleur du biais.
#     Le pilote a servi à cela et à rien d'autre : aucune valeur de n_bar n'y
#     est reprise.
# D2. PORTE DE CONVERGENCE. Une profondeur ne compte que si (i) elle atteint
#     TAU dans le budget S, et (ii) sa perte de validation finale est
#     inférieure à H_unigramme - 0.50 nat. Toute profondeur qui échoue est
#     RAPPORTÉE comme échouée, jamais retirée en silence. À noter d'avance :
#     même en passant cette porte, ces modèles restent FAIBLEMENT entraînés
#     (autour de 2.4 nats, soit ~3.5 bits par caractère, là où un modèle
#     caractère bien entraîné sur ce corpus descend vers 1.5 bit). La
#     revendication porte donc sur la TENDANCE EN PROFONDEUR À PERTE APPARIÉE,
#     pas sur des modèles de qualité de série, et rien n'établit que la
#     tendance se prolonge à perte beaucoup plus basse.
# D3. SITES LUS, convention explicite car p dépend du site et L-i n'est pas
#     comparable entre profondeurs différentes :
#       - n_bar PRIMAIRE au site attn de la COUCHE 1, dont le cône est toute
#         la pile (B = 2L). C'est la quantité sur laquelle porte la prédiction
#         en log L.
#       - p_deep SECONDAIRE = moyenne de n_bar/B sur les sites attn des
#         couches 1..floor(L/2), exactement la convention de la mesure Qwen.
# D4. GRAINES : 3 par profondeur, min / médiane / max rapportés.
# D5. PRÉDICTION TESTÉE ET CRITÈRES DE RÉFUTATION, fixés d'avance.
#     H_log : n_bar = a + b log L.   H_lin : n_bar proportionnel à L.
#     Plage L = 4..64, soit x16 de profondeur ; H_lin prédit une croissance
#     x16, H_log environ x(log 64/log 4) = x3.0 à ordonnée nulle.
#       H_log est RÉFUTÉE si (i) la meilleure loi de puissance a_L^alpha
#       obtient un R^2 supérieur au log ET alpha >= 0.7, OU (ii) la médiane de
#       n_bar croît d'un facteur >= 8 entre L = 4 et L = 64.
#       H_lin est RÉFUTÉE si la croissance est <= x5 et R^2(log) > R^2(puissance).
#     Aucun de ces seuils n'est ajusté après coup.
# D6. LONGUEUR DE SÉQUENCE fixée à 64 pour toutes les profondeurs, dans le
#     régime où p est plat en n (p = 0.1243 +- 0.0048 pour n >= 16, mesuré sur
#     le modèle entraîné) : la longueur ne peut donc pas confondre la
#     comparaison en profondeur.
#
# IDENTITÉS VÉRIFIÉES À CHAQUE PROFONDEUR (conséquences du théorème, pas des
# contrôles ajoutés) : q_jj = 1 au site lu ; q_j = 0 en amont ; forme exacte à
# deux scalaires ; loi de signe ; enveloppe en rho seul ; seuil q < 1/2 <=>
# rho < 1 ; borne alignement-libre sur p.
#
# Usage : julia --project=. notebook/bench_eps_depth_trained.jl [pilote]
# Écrit notebook/bench_eps_depth_trained_results.txt  (ou _pilote_results.txt)
# =============================================================================

using NeuroDSL, Random, Printf, Statistics

const PILOT   = length(ARGS) > 0 && ARGS[1] == "pilote"
const DIM, NH, HID, SEQ = 384, 6, 1024, 64
const L_LIST  = PILOT ? [4, 64]   : [4, 8, 16, 32, 64]
const SEEDS   = PILOT ? [1]       : [1, 2, 3]
const S_MAX   = PILOT ? 2000      : 3000
const EVAL_EV = 250
const NVAL    = 24
const LR      = 3f-4
# TAU fixé par le pilote : atteignable par les DEUX extrêmes (voir D1-bis).
const TAU     = PILOT ? 2.00 : 2.60
const OUT = joinpath(@__DIR__, PILOT ? "bench_eps_depth_trained_pilote_results.txt" :
                                       "bench_eps_depth_trained_results.txt")

# ── corpus caractère par caractère, découpe 90/10 ────────────────────────────
corpus = read(joinpath(@__DIR__, "data", "tinyshakespeare", "input.txt"), String)
chars  = sort(collect(Set(corpus)))
stoi   = Dict(c => i for (i, c) in enumerate(chars))
alld   = [stoi[c] for c in corpus]
V      = length(chars)
ntr    = floor(Int, 0.9 * length(alld))
train  = alld[1:ntr]
valid  = alld[ntr+1:end]
# fenêtres de validation FIXES, également espacées : identiques pour toutes les
# profondeurs et toutes les graines, sinon la porte de convergence bouge.
val_start = [1 + (k-1) * div(length(valid) - SEQ - 1, NVAL) for k in 1:NVAL]
# entropie unigramme du train, en nats : la barre « a-t-il appris quoi que ce soit »
let cnt = zeros(Float64, V)
    for t in train; cnt[t] += 1; end
    global H_UNI = -sum(p > 0 ? p * log(p) : 0.0 for p in cnt ./ sum(cnt))
end

# ── portes eps par branche (machinerie reprise de bench_eps_exact_ablation) ──
const EPSV     = Dict{Symbol,Float32}()
const CAPTURED = Dict{Symbol,Array{Float32}}()
const CAPTURE  = Ref(false)          # coupé pendant l'entraînement

function ensure_eps_op!(branch::Symbol)
    op = Symbol("epsop_", branch)
    EPSV[branch] = 1.0f0
    haskey(NeuroDSL.CUSTOM_OPS, op) && return op
    NeuroDSL.register_op!(op,
        (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> (out_buf .= inputs[1]))
    NeuroDSL.CUSTOM_SHAPE_RULES[op] = (inputs, attrs) -> size(inputs[1])
    NeuroDSL.GRAD_RULES[op] = (dev, dy, ctx, inputs) -> begin
        e = EPSV[branch]
        gr = e .* dy
        CAPTURE[] && (CAPTURED[branch] = copy(Array(gr)))
        return (gr,)
    end
    return op
end

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())
vram() = NeuroDSL.Backend.CUDA_AVAILABLE ?
    (NeuroDSL.CUDA.total_memory() - NeuroDSL.CUDA.free_memory())/2^30 : 0.0

function build(dev, ns::Symbol, L::Int)
    g = NeuroGraph(namespace=ns, device=dev)
    set!(g, :token_ids, ones(Int, SEQ); atom_type=Datom, namespace=ns)
    set!(g, :pos_ids, collect(1:SEQ); atom_type=Datom, namespace=ns)
    te = NeuroDSL.Embedding(V, DIM)(g, :token_ids, :tok; namespace=ns)
    pe = NeuroDSL.Embedding(SEQ, DIM)(g, :pos_ids, :pos; namespace=ns)
    addrule!(g, GraphRule(:embed_sum, [te, pe], :add; namespace=ns))
    h  = NeuroDSL.LlamaModel(L, DIM, NH, HID)(g, :embed_sum; namespace=ns)
    hn = NeuroDSL.LayerNorm(DIM)(g, h, :final_norm; namespace=ns)
    lg = NeuroDSL.Linear(DIM, V; bias=false)(g, hn, :lm_head; namespace=ns)
    set!(g, :labels, ones(Int, SEQ); atom_type=Datom, namespace=ns)
    addrule!(g, GraphRule(:loss, [lg, :labels], :cross_entropy; namespace=ns))
    return g, lg
end

# insertion des portes sur les 2L branches, en préservant le forward
function install_gates!(g, ns, L)
    branches = Symbol[]
    for i in 1:L
        push!(branches, Symbol("layer_", i, "_mha_output_out"))
        push!(branches, Symbol("layer_", i, "_mlp_out"))
    end
    for i in 1:L
        for (join_sym, br) in ((Symbol("layer_", i, "_res1"), Symbol("layer_", i, "_mha_output_out")),
                               (Symbol("layer_", i, "_out"),  Symbol("layer_", i, "_mlp_out")))
            r = g.rules[ns][join_sym]
            r.inputs[2] == br || error("$join_sym : branche inattendue")
            es = Symbol("eps_", br)
            addrule!(g, GraphRule(es, [br], ensure_eps_op!(br); namespace=ns))
            addrule!(g, GraphRule(join_sym, [r.inputs[1], es], r.op;
                                  attrs=r.attrs, namespace=ns, atom_type=r.atom_type))
            NeuroDSL._invalidate_downstream!(g, join_sym, ns)
        end
    end
    return branches
end

function set_win!(g, ns, d, i)
    set!(g, :token_ids, d[i:i+SEQ-1]; atom_type=Datom, namespace=ns)
    set!(g, :labels,    d[i+1:i+SEQ]; atom_type=Datom, namespace=ns)
    invalidate_all!(g; namespace=ns)
end

val_loss(g, ns) = mean(begin
    set_win!(g, ns, valid, i); Float64(sum(Array(demand!(g, :loss; namespace=ns))))
end for i in val_start)

# ── mesure exacte : 2L+1 passes arrière, matrice q complète ──────────────────
function measure(g, ns, branches, L)
    pos = Dict(b => k for (k, b) in enumerate(branches))
    B_of(b) = length(branches) - pos[b] + 1
    bwd!() = begin
        empty!(CAPTURED)
        demand!(g, :loss; namespace=ns)
        backward_graph!(g, :loss; namespace=ns)
        copy(CAPTURED)
    end
    # graine unique et fixe pour la mesure : le premier tenseur de validation
    set_win!(g, ns, valid, val_start[1])
    CAPTURE[] = true
    for b in branches; EPSV[b] = 1.0f0; end
    base = bwd!()
    nrm2 = Dict(b => Float64(sum(abs2, base[b])) for b in branches)
    Q = Dict{Tuple{Symbol,Symbol},Float64}()
    RHO = Dict{Tuple{Symbol,Symbol},Float64}(); COSC = Dict{Tuple{Symbol,Symbol},Float64}()
    for bj in branches
        EPSV[bj] = 0.0f0; abl = bwd!(); EPSV[bj] = 1.0f0
        for si in branches
            g0, gg = abl[si], base[si]
            dg  = Float64(sum(g0 .* gg)); n0 = Float64(sum(abs2, g0)); ng = nrm2[si]
            Q[(si,bj)] = 1.0 - dg/ng
            n2b = ng - 2dg + n0
            nb, nsg = sqrt(max(n2b, 0.0)), sqrt(n0)
            RHO[(si,bj)]  = nsg > 0 ? nb/nsg : Inf
            COSC[(si,bj)] = (nb > 0 && nsg > 0) ? (dg - n0)/(nb*nsg) : NaN
        end
    end
    CAPTURE[] = false; empty!(CAPTURED); reclaim()

    nbar = Dict(si => sum(Q[(si,bj)] for bj in branches if pos[bj] >= pos[si])
                for si in branches)
    site1 = Symbol("layer_1_mha_output_out")
    ratios = [nbar[Symbol("layer_", i, "_mha_output_out")] /
              B_of(Symbol("layer_", i, "_mha_output_out")) for i in 1:L]
    # portes et lois
    self_err = maximum(abs(Q[(s,s)] - 1.0) for s in branches)
    ups = [(si,bj) for si in branches for bj in branches if pos[bj] < pos[si]]
    up_err = isempty(ups) ? 0.0 : maximum(abs(Q[k]) for k in ups)
    ins = [(si,bj) for si in branches for bj in branches if pos[bj] >= pos[si]]
    vs = vt = ve = vh = nneg = 0; worst = 0.0
    for k in ins
        q, r, c = Q[k], RHO[k], COSC[k]
        (isfinite(r) && isfinite(c)) || continue
        worst = max(worst, abs(q - r*(r+c)/(1 + 2r*c + r^2)))
        (sign(q) == sign(r+c) || abs(r+c) < 1e-6 || abs(q) < 1e-6) || (vs += 1)
        q < -1e-6 && (nneg += 1; ((c < -r) && (r < 1.0)) || (vt += 1))
        lo = r < 1 ? -r/(1-r) : r/(1+r); hi = r < 1 ? r/(1+r) : r/(r-1)
        (lo - 1e-4 <= q <= hi + 1e-4) || (ve += 1)
        abs(r - 1) > 1e-4 && ((q < 0.5) == (r < 1) || (vh += 1))
    end
    qbar = (nbar[site1] - 1.0)/(2L - 1)
    (nbar1 = nbar[site1], B1 = B_of(site1), p1 = ratios[1],
     p_deep = mean(ratios[1:max(1, L ÷ 2)]),
     self_err = self_err, up_err = up_err, two = worst,
     vs = vs, vt = vt, ve = ve, vh = vh, nneg = nneg, nins = length(ins),
     qbar = qbar, rho_lb = qbar < 0.5 ? qbar/(1-qbar) : NaN)
end

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit(PILOT ? "PILOTE -- calibrage de TAU et du budget de pas" :
                 "ÉCHELLE EN PROFONDEUR À LARGEUR FIXE, MODÈLES ENTRAÎNÉS ICI")
    emit("dim=$DIM  n_heads=$NH  hidden=$HID  seq=$SEQ  vocab=$V  lr=$LR  batch=1")
    emit("corpus tinyshakespeare, découpe 90/10, $NVAL fenêtres de validation fixes")
    emit(@sprintf("entropie unigramme du train : %.4f nats -- porte : perte < %.4f",
                  H_UNI, H_UNI - 0.50))
    emit(@sprintf("TAU (perte appariée) = %.2f   S_MAX = %d   éval tous les %d pas",
                  TAU, S_MAX, EVAL_EV))
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("")

    dev = NeuroDSL.Backend.CUDADevice()
    RES = Dict{Tuple{Int,Int,Symbol},Any}()   # (L, seed, :tau | :fin) -> mesure
    CURVES = Dict{Tuple{Int,Int},Vector{Tuple{Int,Float64}}}()
    FAIL = Tuple{Int,Int,String}[]

    for L in L_LIST, sd in SEEDS
        ns = Symbol("d", L, "_s", sd)
        reclaim()
        # Reproductibilité : les poids sont tirés par CUDA.rand, donc par le RNG
        # GLOBAL de CUDA.jl. Sans ce seed! la graine ne changerait que l'ordre
        # des données et l'initialisation ne serait pas reproductible d'une
        # exécution à l'autre. Les deux sources d'aléa sont donc fixées ici.
        NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.seed!(2000 + sd)
        Random.seed!(2000 + sd)
        g, _ = build(dev, ns, L)
        branches = install_gates!(g, ns, L)
        # portes de correction : forward inchangé par l'insertion des portes
        set_win!(g, ns, valid, val_start[1])
        v_gate = Float64(sum(Array(demand!(g, :loss; namespace=ns))))
        ps = params(g; namespace=ns)
        npar = sum(length(p.value) for p in ps)
        m1 = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
        m2 = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
        rng = MersenneTwister(1000 + sd)
        curve = Tuple{Int,Float64}[]
        hit_tau = false
        t_start = time()
        for t in 1:S_MAX
            i = rand(rng, 1:(length(train) - SEQ - 1))
            set_win!(g, ns, train, i)
            demand!(g, :loss; namespace=ns)
            backward_graph!(g, :loss; namespace=ns)
            for (k, p) in enumerate(ps)
                adamw_step!(dev, p.value, p.gradient, m1[k], m2[k],
                            LR, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
            end
            invalidate_all!(g; namespace=ns)
            if t % EVAL_EV == 0
                vl = val_loss(g, ns)
                push!(curve, (t, vl))
                if !hit_tau && vl <= TAU
                    hit_tau = true
                    RES[(L, sd, :tau)] = merge((step = t, vl = vl), measure(g, ns, branches, L))
                    # ÉCRITURE IMMÉDIATE. Les mesures étaient auparavant
                    # accumulées dans RES et n'atteignaient l'artefact que par
                    # les boucles de rapport finales : un premier run est mort
                    # à 11/15 modèles (2026-08-11, ~55 min de calcul) et TOUTES
                    # les mesures ont été perdues alors qu'elles étaient déjà
                    # calculées. Une mesure calculée doit être une mesure écrite.
                    emit("  MESURE tau  " * string((L = L, sd = sd), ) * " " *
                         string(RES[(L, sd, :tau)]))
                end
            end
        end
        vlf = val_loss(g, ns)
        RES[(L, sd, :fin)] = merge((step = S_MAX, vl = vlf), measure(g, ns, branches, L))
        emit("  MESURE fin  " * string((L = L, sd = sd)) * " " *
             string(RES[(L, sd, :fin)]))
        CURVES[(L, sd)] = curve
        hit_tau || push!(FAIL, (L, sd, @sprintf("TAU non atteint (perte finale %.4f)", vlf)))
        vlf < H_UNI - 0.50 || push!(FAIL, (L, sd, @sprintf("porte unigramme échouée (%.4f)", vlf)))
        emit(@sprintf("L=%3d graine=%d  %d params  VRAM %.2f Go  %.0f s  perte val finale %.4f%s",
                      L, sd, npar, vram(), time() - t_start, vlf,
                      hit_tau ? @sprintf("  (TAU à l'étape %d)", RES[(L,sd,:tau)].step) : "  TAU NON ATTEINT"))
        g = nothing; ps = nothing; m1 = nothing; m2 = nothing; reclaim()
    end

    emit("")
    emit("="^78); emit("COURBES DE VALIDATION"); emit("="^78)
    for L in L_LIST, sd in SEEDS
        haskey(CURVES, (L, sd)) || continue
        emit(@sprintf("L=%3d s=%d : %s", L, sd,
             join([@sprintf("%d:%.3f", t, v) for (t, v) in CURVES[(L,sd)]], "  ")))
    end

    for (tag, name) in ((:tau, "PERTE APPARIÉE (primaire)"), (:fin, "PAS ÉGAUX (secondaire)"))
        emit(""); emit("="^78); emit("PROTOCOLE $name"); emit("="^78); emit("")
        emit(@sprintf("%5s %5s %8s %8s %9s %9s %9s %9s %9s",
                      "L", "B", "étape", "perte", "n_bar min", "n_bar méd", "n_bar max",
                      "p_deep m", "p_L1 méd"))
        med = Dict{Int,Float64}()
        for L in L_LIST
            rs = [RES[(L,sd,tag)] for sd in SEEDS if haskey(RES, (L,sd,tag))]
            isempty(rs) && (emit(@sprintf("%5d %5d  -- aucune exécution qualifiée --", L, 2L)); continue)
            nb = [r.nbar1 for r in rs]
            med[L] = median(nb)
            emit(@sprintf("%5d %5d %8.0f %8.4f %9.4f %9.4f %9.4f %9.5f %9.5f",
                 L, 2L, median([r.step for r in rs]), median([r.vl for r in rs]),
                 minimum(nb), median(nb), maximum(nb),
                 median([r.p_deep for r in rs]), median([r.p1 for r in rs])))
        end
        # ajustements comparés sur la MÊME cible n_bar, même critère
        ks = sort(collect(keys(med)))
        if length(ks) >= 4
            y = [med[k] for k in ks]; x = log.(ks); ȳ = mean(y); sst = sum((y .- ȳ).^2)
            Xa = hcat(ones(length(x)), x); ba = Xa \ y
            r2a = 1 - sum((y - Xa*ba).^2)/sst
            best = (-Inf, 0.0, 0.0)
            for al in 0.0:0.001:1.5
                z = Float64.(ks) .^ al; a = dot(z, y)/dot(z, z)
                r2 = 1 - sum((y - a*z).^2)/sst
                r2 > best[1] && (best = (r2, a, al))
            end
            gro = y[end]/y[1]
            emit("")
            emit(@sprintf("  n_bar = %+.4f %+.4f log L      R^2 = %.4f", ba[1], ba[2], r2a))
            emit(@sprintf("  n_bar = %+.4f L^%.3f          R^2 = %.4f", best[2], best[3], best[1]))
            emit(@sprintf("  croissance mesurée L=%d -> %d : x%.2f  (H_lin prédit x%.1f, log prédit x%.2f)",
                          ks[1], ks[end], gro, ks[end]/ks[1], log(ks[end])/log(ks[1])))
            emit("")
            ref_log = (best[1] > r2a && best[3] >= 0.7) || gro >= 8.0
            ref_lin = gro <= 5.0 && r2a > best[1]
            emit(@sprintf("  critère D5 : H_log réfutée ? %s        H_lin réfutée ? %s",
                          ref_log ? "OUI" : "non", ref_lin ? "OUI" : "non"))
        end
    end

    emit(""); emit("="^78); emit("IDENTITÉS, À CHAQUE PROFONDEUR"); emit("="^78); emit("")
    emit(@sprintf("%5s %5s %11s %11s %11s %7s %7s %7s %7s %8s",
                  "L", "n_ins", "q_jj-1", "q amont", "2-scal", "signe", "queue",
                  "envel", "seuil", "% neg"))
    for L in L_LIST, sd in SEEDS
        haskey(RES, (L,sd,:fin)) || continue
        r = RES[(L,sd,:fin)]
        emit(@sprintf("%5d %5d %11.3e %11.3e %11.3e %7d %7d %7d %7d %8.1f",
             L, r.nins, r.self_err, r.up_err, r.two, r.vs, r.vt, r.ve, r.vh,
             100*r.nneg/r.nins))
    end
    emit("")
    emit("  colonnes : écart max sur q_jj = 1 ; écart max sur q_j = 0 en amont ;")
    emit("  écart max |q - forme à deux scalaires| ; violations de la loi de signe,")
    emit("  de c < -rho pour les négatifs, de l'enveloppe en rho, du seuil")
    emit("  q < 1/2 <=> rho < 1 ; part de coefficients négatifs.")

    emit(""); emit("="^78); emit("BORNE ALIGNEMENT-LIBRE (théorème)"); emit("="^78)
    emit(@sprintf("\n%5s %11s %13s", "L", "q̄ bypass", "rho_max >="))
    for L in L_LIST
        rs = [RES[(L,sd,:fin)] for sd in SEEDS if haskey(RES, (L,sd,:fin))]
        isempty(rs) && continue
        emit(@sprintf("%5d %11.5f %13.4f", L, median([r.qbar for r in rs]),
                      median([r.rho_lb for r in rs])))
    end

    emit(""); emit("="^78); emit("PORTE DE CONVERGENCE (D2)"); emit("="^78)
    if isempty(FAIL)
        emit("\n  aucune profondeur en échec : toutes atteignent TAU et passent la")
        emit("  porte unigramme.")
    else
        emit("")
        for (L, sd, why) in FAIL
            emit(@sprintf("  ÉCHEC  L=%d graine=%d : %s", L, sd, why))
        end
        emit("\n  Ces exécutions sont RAPPORTÉES, pas retirées. Une profondeur qui")
        emit("  n'a pas convergé ne dit rien sur p entraîné.")
    end
end
println("\nÉcrit : ", OUT)
