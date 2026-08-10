#=
PREUVE EMPIRIQUE DE LA LOCALITÉ D'INVALIDATION -- le claim phare de l'article 1.

POURQUOI CE BANC EXISTE : le théorème O(|V_θ|) est le titre de l'article et la
fondation réutilisée par les quatre articles compagnons, mais son seul support
empirique dans le papier est `tab:gpt2` -- DEUX chronométrages isolés
(166.8->0.6 ms, 158.7->29.7 ms), sans comptage de nœuds, sans contrôle négatif,
sans artefact rejouable. Autrement dit : la revendication la moins étayée du
papier est aussi la plus centrale. Ce banc la mesure.

QUANTITÉ PRIMAIRE, STRUCTURELLE (déterministe, sans chronomètre ni horloge) :
le nombre exact de nœuds traversés par une invalidation. Aucune modification du
moteur n'est nécessaire -- `_invalidate_downstream!(g, target, ns, visited)`
accepte déjà un `Set{Symbol}` injectable en 4e argument positionnel, donc
`length(visited)` après l'appel EST le compte de traversée.

LES DEUX AXES :
  (a) PROFONDEUR DU SITE : plus le site est profond, plus son cône aval est
      petit, donc moins on traverse. C'est la localité.
  (b) REMBOURRAGE DU GRAPHE (l'axe décisif) : on ajoute des branches gelées
      DISJOINTES qui gonflent |V| (1x, 2x, 4x) sans être atteignables depuis le
      modèle. Le compte d'un site profond doit rester **rigoureusement
      identique**. C'est exactement le sens de "indépendant de la taille totale
      du graphe" -- et c'est ce que deux chronométrages ne peuvent pas montrer.

CONTRÔLE NÉGATIF : éditer l'entrée du graphe (`:input`). Tout le modèle est
atteignable depuis elle, donc la localité ne peut RIEN économiser et le rapport
|visited|/|V_modele| doit être proche de 1. Un mécanisme qui paraîtrait local
même là serait la preuve que la mesure ne mesure rien.

ORACLE INDÉPENDANT : un BFS itératif écrit ici, sur l'adjacence inverse
reconstruite à la main depuis `rule.inputs`. Il n'appelle NI
`_consumers_index!` NI `_invalidate_downstream!` -- sinon on vérifierait
seulement qu'un code est d'accord avec lui-même. (Les oracles de
`test/test_property_invariants.jl` couvrent déjà l'égalité d'ENSEMBLES ; ce banc
mesure la TAILLE en fonction de |V|, ce qu'aucun test ne fait.)

CRITÈRES PRÉ-ENREGISTRÉS (fixés avant le premier lancement) :
  (a) comptes identiques sur 5 lancements indépendants ;
  (b) compte d'un site profond RIGOUREUSEMENT ÉGAL sous rembourrage 4x ;
  (c) accord avec l'oracle sur 100% des sites ;
  (d) contrôle `:input` à |visited|/|V_modele| >= 0.95.
FALSIFIÉ si le compte rembourré augmente, ne serait-ce que de 1, ou si le
contrôle apparaît lui aussi local.

UN SEUL (rembourrage, site) PAR EXÉCUTION : l'invalidation MUTE le graphe
(`valid=false`), donc mesurer deux sites de suite dans le même processus
fausserait le second. Processus neuf, graphe neuf, une seule mesure.

CONTRÔLE POSITIF (ajouté le 2026-08-08, 3e argument `padmode`) : les contrôles
ci-dessus sont tous NÉGATIFS -- ils montrent que la mesure renvoie une grande
valeur quand elle le doit. Aucun ne montrait qu'elle RÉAGIT au facteur confondant.
Un comptage qui serait constant *par erreur* passerait tous ces tests.
D'où `padmode=reachable` : le rembourrage consomme alors depuis `:input` au lieu
d'être disjoint, donc il EST atteignable. Deux prédictions exactes en découlent,
opposées et simultanées :
  - éditer `:input` doit visiter son cône PLUS tout le rembourrage, soit une
    croissance de EXACTEMENT (n_model-1) nœuds par unité de rembourrage ;
  - éditer un site profond (`layer6`, `layer12`) doit rester RIGOUREUSEMENT
    INCHANGÉ, le rembourrage n'étant pas en aval de ces sites.
Le même run teste donc que la mesure suit l'atteignabilité *et* que la localité
tient. Un comptage constant par artefact échoue à la première ; un comptage qui
compterait tout le graphe échoue à la seconde.

USAGE : julia --project=. notebook/bench_invalidation_locality.jl [pad] [site] [padmode]
        pad ∈ 1|2|4   site ∈ input|layer1|layer6|layer12   padmode ∈ disjoint|reachable
        (padmode par défaut = disjoint : comportement d'origine, résultats archivés valides)
=#
using NeuroDSL, Printf

const PAD  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
const SITE = length(ARGS) >= 2 ? ARGS[2] : "layer12"
const PADMODE = length(ARGS) >= 3 ? ARGS[3] : "disjoint"
PADMODE in ("disjoint","reachable") || error("padmode inconnu : $PADMODE")
PAD in (1,2,4) || error("pad doit valoir 1, 2 ou 4")
SITE in ("input","layer1","layer6","layer12") || error("site inconnu : $SITE")

const NLAYERS = 12
const DIM     = 32
const NHEADS  = 4
const HIDDEN  = 64
const SEQ     = 8

dev = NeuroDSL.Backend.CPUDevice()
ns  = Symbol(:invloc_, PAD, :_, SITE)
g   = NeuroDSL.NeuroGraph(namespace=ns, device=dev)

NeuroDSL.set!(g, :input, randn(Float32, SEQ, DIM); namespace=ns)
out_sym = NeuroDSL.LlamaModel(NLAYERS, DIM, NHEADS, HIDDEN)(g, :input; namespace=ns)
n_model = length(g.nodes[ns])          # taille du modèle seul, avant rembourrage

# ── rembourrage : branches DISJOINTES, non atteignables depuis le modèle ──────
# But : gonfler |V| sans rien ajouter au cône aval d'aucun nœud du modèle.
for k in 2:PAD
    prev = if PADMODE == "reachable"
        :input                      # le rembourrage devient EN AVAL de :input
    else
        root = Symbol(:pad_, k, :_root)
        NeuroDSL.set!(g, root, randn(Float32, SEQ, DIM); namespace=ns)
        root
    end
    for j in 1:(n_model - 1)           # même volume que le modèle par unité
        w = Symbol(:pad_, k, :_w, j)
        NeuroDSL.set!(g, w, randn(Float32, DIM, DIM).*0.02f0; is_param=false, namespace=ns)
        nxt = Symbol(:pad_, k, :_h, j)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(nxt, [prev, w], :matmul; namespace=ns))
        prev = nxt
    end
end
n_total = length(g.nodes[ns])

NeuroDSL.demand!(g, out_sym; namespace=ns)   # tout valide : un graphe vivant

# En mode `reachable`, il faut AUSSI calculer le rembourrage, sinon il reste
# `valid=false` et la traversée s'arrête legitimement dessus : `_invalidate_downstream!`
# ne propage qu'aux consommateurs VALIDES (`src/graph_api.jl` : la garde
# `out_nd.valid`), parce qu'invalider un nœud déjà invalide est un no-op et que son
# aval est déjà invalide par induction. Sans ce `demand!`, le contrôle positif
# mesurerait cette optimisation-là et non l'atteignabilité -- piège rencontré et
# diagnostiqué : le compte restait à 493 quand l'oracle en annonçait 1093.
if PADMODE == "reachable"
    for k in 2:PAD
        NeuroDSL.demand!(g, Symbol(:pad_, k, :_h, n_model - 1); namespace=ns)
    end
end

target = SITE == "input"  ? :input :
         SITE == "layer1"  ? :layer_1_out :
         SITE == "layer6"  ? :layer_6_out : Symbol(:layer_, NLAYERS, :_out)
haskey(g.nodes[ns], target) || error("site absent du graphe : $target")

# ── ORACLE : BFS itératif sur l'adjacence inverse, écrit indépendamment ───────
# N'utilise ni _consumers_index! ni _invalidate_downstream!.
function oracle_reachable(g, ns, start::Symbol)
    consumers = Dict{Symbol,Vector{Symbol}}()
    for (outs, rule) in g.rules[ns], inp in rule.inputs
        push!(get!(consumers, inp, Symbol[]), outs)
    end
    seen = Set{Symbol}([start]); queue = Symbol[start]
    while !isempty(queue)
        cur = pop!(queue)
        for c in get(consumers, cur, Symbol[])
            if !(c in seen); push!(seen, c); push!(queue, c); end
        end
    end
    return seen
end
oracle = oracle_reachable(g, ns, target)

# ── MESURE : le set visité injecté EST l'instrument ──────────────────────────
visited = Set{Symbol}()
NeuroDSL._invalidate_downstream!(g, target, ns, visited)

agree = (visited == oracle)

# Dénominateur honnête pour le contrôle négatif : les nœuds PARAMÈTRES sont des
# feuilles (aucune règle ne les produit) et ne peuvent donc JAMAIS se trouver en
# aval de quoi que ce soit. Le plafond atteignable depuis l'entrée n'est pas |V|
# mais le nombre de nœuds invalidables. Le critère pré-enregistré (d) avait été
# écrit contre |V| : mesuré à 0.8203, il échouait pour une raison structurelle
# et non parce que le mécanisme serait non local. On rapporte les DEUX rapports
# plutôt que de déplacer le seuil en silence.
n_param    = count(nd -> nd.is_param, values(g.nodes[ns]))
n_invalidable = count(nd -> !nd.is_param, values(g.nodes[ns]))

@printf("RESULT pad=%d padmode=%s site=%s n_model=%d n_total=%d n_param=%d n_invalidable=%d visited=%d oracle=%d agree=%s ratio_model=%.4f ratio_invalidable=%.4f\n",
        PAD, PADMODE, SITE, n_model, n_total, n_param, n_invalidable,
        length(visited), length(oracle), agree,
        length(visited)/n_model, length(visited)/n_invalidable)
