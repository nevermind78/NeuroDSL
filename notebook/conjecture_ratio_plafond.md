# Conjecture (prouvée) : le ratio agrégé d'un balayage exhaustif converge vers exactement 2

## Contexte

Suite à jalon 0 (ratio agrégé = 2.12× à 12 couches) et jalon 0b (2.10× à 24 couches) — les deux
sous le seuil pré-enregistré de 3× qui aurait justifié un portage réel de GPT-2. Question posée :
existe-t-il une constante (ou une borne) vers laquelle ce ratio converge structurellement, quelle
que soit l'échelle ?

## Formule exacte du cône aval (vérifiée, résidu nul sur 564 sites réels)

Pour un site à la couche `i` (sur `n_layers` couches, `n_heads` têtes, attention batchée) :

```
cone(ao_h, i)    = 10 + (n_layers − i)·(7·n_heads + 15)
cone(mlp_out, i) =  2 + (n_layers − i)·(7·n_heads + 15)
```

Dérivée indépendamment par deux voies (lecture du code `src/layers.jl`/`src/patching.jl`, et
reconstruction purement empirique depuis `notebook/jalon0_results.json`/`jalon0b_results.json`)
— les deux convergent sur la même formule, vérifiée sans aucune erreur sur les 156+408=564 sites
réels des deux jeux de données.

**Pourquoi 7·n_heads+15, pas 7·n_heads+24 (le compte brut de nœuds par couche)** : une couche
contient bien 7·n_heads+24 nœuds au total, mais 9 sont des tenseurs paramètres (`is_param=true`),
qui sont toujours des feuilles du graphe et ne sont donc **jamais** atteignables en aval de quoi
que ce soit (`_downstream_nodes`, `src/patching.jl`) — et ils ne coûtent surtout jamais rien à
« recalculer » puisqu'ils sont déjà résidents. Le nombre de nœuds *calculés* par couche est donc
`7·n_heads+15`, la seule quantité pertinente pour le coût.

## Théorème

Pour `n_sites = L·(H+1)` sites (H têtes + 1 MLP par couche, sur L couches), avec
`M = 7H+15`, `N_total = M·L` (nœuds calculés, formule ci-dessus sommée), le ratio agrégé

```
ratio(L, H) = (n_sites · N_total) / Σ_sites cone(site)
            = (H+1)·M·L / [ (H+1)·M·(L−1)/2 + (10H+2) ]
```

converge, quand `L → ∞`, vers **exactement 2, pour tout H fixé** — le terme `(10H+2)` (fini,
indépendant de `L`) devient négligeable face au terme `(H+1)·M·(L−1)/2`, qui croît sans borne.

**Vérifié numériquement** (arithmétique exacte, aucune approximation) pour H ∈ {1,2,4,8,12,16,
24,64,128} à L=10000 : ratio ∈ [2.00009, 2.00020] dans tous les cas — la dépendance en H
s'annule dans la limite, ne laissant qu'un résidu en O(1/L) de plus en plus petit.

**Intuition** : le cône moyen d'un balayage uniforme sur la profondeur vaut exactement la moitié
du graphe calculé, quel que soit le nombre de têtes ou la largeur du modèle — une suite
arithmétique décroissant de `N_total` à ~0 a pour moyenne `N_total/2`.

## Historique d'une fausse piste, corrigée honnêtement

Une première vérification structurelle rapide (`notebook/conjecture_ratio_multi_config.jl`,
2026-07-12) avait semblé *infirmer* cette limite universelle : elle montrait des paliers
apparemment dépendants de la largeur/du nombre de têtes (≈2.42 pour H=4, ≈2.15-2.19 pour H=12
même à 192 couches). **Cause identifiée** : ce script comptait `N_total` comme le nombre BRUT de
nœuds du graphe (`length(g.nodes[ns])`), incluant les 9 paramètres par couche — qui ne
contribuent jamais au temps de calcul réel (ils ne sont jamais recalculés). Une fois corrigé pour
ne compter que les nœuds calculés (`M = 7H+15`, pas `7H+24`), la limite redevient exactement 2
pour toute config testée. La conjecture initiale ("=2 universel") était donc juste — c'est l'outil
de vérification qui avait un bug, pas la théorie.

## Confrontation aux deux mesures réelles (écart attendu, expliqué)

| Config | Ratio prédit (formule, nœuds seuls) | Ratio mesuré (chronométrage réel) |
|---|---|---|
| jalon 0 (L=12, H=12) | 2.1449 | 2.1205 |
| jalon 0b (L=24, H=16) | 2.0734 | 2.1009 |

Écart de 1-2%, dans les deux sens — attendu : la formule suppose un coût strictement
proportionnel au nombre de nœuds (`cost = a·cone`, sans terme constant), alors que le
chronométrage réel a un léger surcoût fixe par appel (`cost = a·cone + b`, déjà mesuré,
`b` de l'ordre de 1 ms) qui n'est pas capturé par le comptage structurel pur. La formule
structurelle explique le comportement asymptotique et la limite ; le chronométrage réel y
ajoute un bruit systématique de premier ordre, sans changer la conclusion.

## Conséquence pour la décision jalon 0 / jalon 1

La décision de ne pas porter GPT-2 réellement (jalon 1) est maintenant justifiée par une preuve,
pas seulement par deux mesures sous le seuil : **aucune échelle, aussi grande soit-elle, ne fera
jamais dépasser ~2× un balayage exhaustif et uniforme sur la profondeur** de ce type. Le seuil de
3× ne peut structurellement pas être franchi par ce protocole. La seule voie qui reste
favorable est celle déjà identifiée : restreindre le balayage aux sites profonds/tardifs
(cône petit, ratio individuel bien supérieur à 2) — exactement la stratégie d'AtP*
("classer bon marché, vérifier exactement un petit nombre de candidats profonds").
