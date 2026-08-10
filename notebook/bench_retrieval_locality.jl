#=
PREUVE EMPIRIQUE DE LA LOCALITÉ DE RÉCUPÉRATION -- un résultat revendiqué NULLE PART.

CONTEXTE : le théorème publié de l'article 1 (et réutilisé par les quatre
articles compagnons) couvre la phase d'INVALIDATION : une édition ne touche que
le cône aval du nœud édité. Il ne dit rien de la phase de RÉCUPÉRATION.
Or `demand!` parcourait tout l'ordre topologique en cache DEPUIS LE DÉBUT à
chaque appel où la cible n'était pas déjà valide -- soit un coût en
O(position topologique de la cible), et non en O(cône ancêtre). L'asymétrie
était réelle : localité prouvée d'un côté, absente de l'autre.

Elle a été corrigée (`_ancestors_of!`, `src/graph_api.jl:66`, utilisé par
`demand!` à `src/dispatch.jl:795`). Ce banc mesure ce que la correction achète,
et ferme au passage un trou : `_ancestors_of!` n'a AUCUNE couverture de test
dans tout le dépôt -- l'oracle indépendant ci-dessous est sa première
vérification.

QUANTITÉ PRIMAIRE, STRUCTURELLE (déterministe, sans chronomètre) :
  ANCIEN coût = position de la cible dans `topo_order!` (nombre de nœuds que le
                parcours de préfixe visitait, cible incluse) ;
  NOUVEAU coût = `length(_ancestors_of!(g, ns, cible))` (le cône ancêtre réel).

SYMÉTRIE À NOTER, et c'est le point intéressant : pour l'INVALIDATION, ce sont
les sites PROFONDS qui sont bon marché (petit cône aval). Pour la RÉCUPÉRATION,
ce sont les cibles PEU PROFONDES (petit cône ancêtre). Les deux localités
jouent en sens inverse le long de la profondeur.

CONTRÔLE NÉGATIF : cible = sortie du modèle. Ses ancêtres sont presque tout le
modèle, donc ancêtres ≈ préfixe et la correction ne peut RIEN gagner. Un
mécanisme qui paraîtrait économe même là serait la preuve que la mesure ne
mesure rien.

ORACLE INDÉPENDANT : BFS d'ancêtres écrit ici, sur `rule.inputs` en marche
avant. N'appelle NI `_ancestors_of!` NI `topo_order!`. Doit inclure la cible
elle-même, comme `_ancestors_of!` (cf. le commentaire de `_ancestors_cache`
dans `src/types.jl`).

ORDRE DE CONSTRUCTION, dit explicitement : le rembourrage est construit AVANT le
modèle, afin que ses nœuds puissent occuper des positions topologiques
antérieures -- exactement la situation que l'ancien parcours de préfixe payait.
C'est un scénario légitime (un graphe qui contient plusieurs sous-graphes
indépendants, ce que NeuroDSL permet), et il est nommé pour que le lecteur sache
ce qui est mesuré.

CRITÈRES PRÉ-ENREGISTRÉS :
  (a) comptes identiques sur 5 lancements indépendants ;
  (b) NOUVEAU coût d'une cible peu profonde RIGOUREUSEMENT ÉGAL sous
      rembourrage x4, tandis que l'ANCIEN coût augmente ;
  (c) accord avec l'oracle sur 100% des points ;
  (d) contrôle (sortie du modèle) : ancien/nouveau <= 1.10, soit aucun gain.
FALSIFIÉ si le nouveau coût croît avec le rembourrage, ou si le contrôle
montre lui aussi un gain net.

UN SEUL (rembourrage, cible) PAR EXÉCUTION, processus neuf.

USAGE : julia --project=. notebook/bench_retrieval_locality.jl [pad] [cible]
        pad ∈ 1|2|4     cible ∈ input|layer1|layer6|output
=#
using NeuroDSL, Printf

const PAD    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
const TARGET = length(ARGS) >= 2 ? ARGS[2] : "layer1"
PAD in (1,2,4) || error("pad doit valoir 1, 2 ou 4")
TARGET in ("input","layer1","layer6","output") || error("cible inconnue : $TARGET")

const NLAYERS = 12
const DIM, NHEADS, HIDDEN, SEQ = 32, 4, 64, 8

dev = NeuroDSL.Backend.CPUDevice()
ns  = Symbol(:retloc_, PAD, :_, TARGET)
g   = NeuroDSL.NeuroGraph(namespace=ns, device=dev)

# ── rembourrage D'ABORD : branches disjointes, positions topologiques précoces ─
const PAD_UNIT = 600
for k in 2:PAD
    root = Symbol(:pad_, k, :_root)
    NeuroDSL.set!(g, root, randn(Float32, SEQ, DIM); namespace=ns)
    prev = root
    for j in 1:PAD_UNIT
        w = Symbol(:pad_, k, :_w, j)
        NeuroDSL.set!(g, w, randn(Float32, DIM, DIM).*0.02f0; is_param=false, namespace=ns)
        nxt = Symbol(:pad_, k, :_h, j)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(nxt, [prev, w], :matmul; namespace=ns))
        prev = nxt
    end
end

NeuroDSL.set!(g, :input, randn(Float32, SEQ, DIM); namespace=ns)
out_sym = NeuroDSL.LlamaModel(NLAYERS, DIM, NHEADS, HIDDEN)(g, :input; namespace=ns)
n_total = length(g.nodes[ns])

target = TARGET == "input"  ? :input :
         TARGET == "layer1" ? :layer_1_out :
         TARGET == "layer6" ? :layer_6_out : out_sym
haskey(g.nodes[ns], target) || error("cible absente : $target")

# ── ORACLE : BFS d'ancêtres, écrit indépendamment ────────────────────────────
function oracle_ancestors(g, ns, t::Symbol)
    seen = Set{Symbol}([t]); stack = Symbol[t]
    while !isempty(stack)
        cur = pop!(stack)
        r = get(g.rules[ns], cur, nothing)
        r === nothing && continue
        for inp in r.inputs
            if !(inp in seen); push!(seen, inp); push!(stack, inp); end
        end
    end
    return seen
end
oracle = oracle_ancestors(g, ns, target)

# ── NOUVEAU coût : le cône ancêtre réel ──────────────────────────────────────
anc = NeuroDSL._ancestors_of!(g, ns, target)
agree = (Set(anc) == oracle)

# ── ANCIEN coût : position de la cible dans l'ordre topologique ──────────────
order = NeuroDSL.topo_order!(g; namespace=ns)
idx = findfirst(==(target), order)
old_cost = idx === nothing ? -1 : idx

@printf("RESULT pad=%d target=%s n_total=%d new_cost=%d old_cost=%d oracle=%d agree=%s speedup=%.3f\n",
        PAD, TARGET, n_total, length(anc), old_cost, length(oracle), agree,
        old_cost > 0 ? old_cost/length(anc) : NaN)
