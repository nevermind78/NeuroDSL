# Pré-enregistrement — corriger un échec de raisonnement réel de Qwen2.5-1.5B-Instruct
# via une greffe Gradient Shadowing gelée

Date : 2026-08-31, écrit APRÈS les 3 checks de correction du mécanisme
(`graft_qwen_correctness_preregistration.md`, tous PASS) et APRÈS une
recherche exploratoire de l'échec (`graft_qwen_failure_search.jl`,
log `graft_qwen_failure_search.log`), mais AVANT toute greffe/entraînement
de CETTE expérience. Les nombres de succès ci-dessous ne sont pas encore
connus au moment de l'écriture.

## Rappel du précédent le plus proche (calibrage des attentes)

`artilce/patching_cost_and_circuits.tex`, §`sec:goulot` : le même mécanisme
greffe+gate+AdamW, sur un modèle plus petit, a échoué sur ses 3 critères
pré-enregistrés (avantage de magnitude du gate inversé sur une graine, la
greffe n'est jamais entrée dans le top-3 de la recherche de circuit, le
placement diagnostiqué ne bat pas un témoin naïf en perte). Cette expérience
n'a AUCUNE raison a priori de réussir mieux ; elle est conçue pour rapporter
honnêtement un résultat nul ou partiel si c'est ce qui se produit.

## 1. Recherche de l'échec — ce qui a été trouvé réellement (pas fabriqué)

`graft_qwen_failure_search.jl` a testé 4 familles de candidats sur le modèle
NON MODIFIÉ, poids réels, gabarit ChatML réel (`apply_chat_template`),
génération gloutonne :

| Famille | Résultat |
|---|---|
| Comparaison de décimaux (biais "version number", ex. 9.11 vs 9.9) | **PAS d'échec** — 4/4 corrects |
| Comptage de lettres (ex. "r" dans "strawberry") | Échec réel (2/3 faux : "strawberry"→2 au lieu de 3, "excellence"→3 au lieu de 4) mais probablement **une limite de tokenisation** (le modèle ne voit pas les caractères), pas un biais comportemental corrigible par une greffe additive dans le flux résiduel — écarté pour cette raison |
| Arithmétique à retenue (2-3 chiffres) | **PAS d'échec** — 3/3 corrects |
| **Suivi d'instruction sous distraction** (contrainte de format stricte + demande d'explication dans la même requête) | **Échec réel et net, 2/2** : `"Answer in exactly one word: ... Also, briefly explain..."` → réponse longue avec explication ; `"Reply with only 'yes' or 'no': ... Please justify..."` → réponse longue avec justification. Le contenu factuel est CORRECT dans les deux cas ("Paris", "17 is prime") — c'est uniquement la CONTRAINTE DE FORMAT qui est ignorée dès qu'une clause de distraction (demande d'explication) est ajoutée à la même requête. |

**Échec retenu** : suivi d'instruction sous distraction — le modèle respecte
une contrainte de format stricte ("réponds en un seul mot" / "réponds
seulement par oui/non") quand elle est seule, mais l'ignore systématiquement
dès qu'une clause "explique/justifie" est ajoutée à la même requête,
produisant une réponse longue au lieu du format demandé. Choisi plutôt que
le comptage de lettres parce que (a) c'est un échec comportemental
(pondération d'attention/décision de génération), pas une limite de
représentation d'entrée (le modèle "sait" la bonne réponse factuelle dans
les deux échecs testés — le contenu est correct, seul le format est violé),
donc plausiblement adressable par un signal additif dans le flux résiduel
tardif ; (b) il généralise naturellement par contenu ET par variante de
contrainte, ce qui permet un vrai test de généralisation disjoint (voir §2).

## 2. Jeu de données — entraînement, deux jeux "held-out" disjoints, témoin négatif

Tous les items ci-dessous sont des couples (prompt réel avec gabarit ChatML,
réponse cible attendue si le modèle SUIVAIT la contrainte de format). Le
prompt d'entraînement inclut TOUJOURS la clause de distraction ("explain" /
"justify") — c'est précisément la condition qui déclenche l'échec.

### Entraînement (6 exemples, contrainte "un seul mot")
Gabarit : `Answer in exactly one word: {question} Also, briefly explain your reasoning.`

1. capitale de la France → `Paris`
2. couleur du ciel par temps clair → `Blue`
3. plus grande planète du système solaire → `Jupiter`
4. auteur de Roméo et Juliette → `Shakespeare`
5. gaz nécessaire à la respiration humaine → `Oxygen`
6. saison après l'hiver → `Spring`

### Held-out A — DISJOINT en contenu, MÊME contrainte ("un seul mot", 4 exemples)
Teste : généralisation par contenu à contrainte fixe (pas juste mémorisation
des 6 exemples d'entraînement).
7. capitale du Japon → `Tokyo`
8. plus haute montagne de la Terre → `Everest`
9. symbole chimique de l'or → `Au`
10. nombre de continents → `Seven`

### Held-out B — DISJOINT en contenu ET en variante de contrainte ("oui/non", 3 exemples)
Gabarit : `Reply with only 'yes' or 'no': {question} Please justify your answer in detail.`
Teste : généralisation au-delà du gabarit de contrainte vu à l'entraînement
(pas seulement "j'ai appris à tronquer après le gabarit 'exactly one word'").
11. la Terre est-elle plate ? → `No`
12. l'eau est-elle composée d'hydrogène et d'oxygène ? → `Yes`
13. Paris est-elle la capitale de l'Allemagne ? → `No`

### Témoin négatif — mêmes questions, SANS contrainte de format (3 exemples)
`{question} Also, briefly explain why.` / `{question} Please justify your
answer in detail.` — PAS de "answer in exactly one word" / "reply with only
yes/no". Le comportement CORRECT ici est de rester verbeux (explication
normale). Détecte un effondrement dégénéré (la greffe apprend à tronquer
TOUT, pas seulement en présence d'une contrainte) plutôt qu'une vraie
correction conditionnelle.
14. capitale de l'Italie → verbeux attendu
15. 12 est-il un nombre pair ? → verbeux attendu
16. plus grand océan de la Terre → verbeux attendu

Aucun item du témoin négatif ni des deux jeux held-out ne partage son texte
de question avec l'entraînement.

## 3. Site et configuration de la greffe

Identique au mécanisme déjà validé par les 3 checks de correction :
- Site : `:layer_25_out` (sortie résiduelle après le 25e `LlamaBlock` sur 28).
- `graft_shadow_block!(g, ns, :layer_25_out, 1536, 4, 384; alpha0=0f0,
  zero_out_proj=false, prefix=:qwen_shadow_fix)` — Gradient Shadowing pur,
  theta aléatoire, gate à 0 (graine fixée : `Random.seed!(4242)` avant
  l'insertion, pour reproductibilité).
- Gel du backbone : même helper local `freeze_backbone!` (is_param=false sur
  tout nœud de poids Qwen d'origine, conservé pour le préfixe `qwen_shadow_fix`
  uniquement).
- `backward_graph!(...; prune_frozen=true)`.

## 4. Protocole d'entraînement (fixé AVANT tout run)

- Séquence par exemple : préfixe ChatML réel (`encode_chat`) + tokens de la
  réponse cible (encodage brut, `add_special_tokens=false`) + `EOS_ID`
  (`<|im_end|>`). Perte cross-entropie next-token standard sur la séquence
  ENTIÈRE (prompt inclus) — même schéma que les checks 1-3, pas de masquage
  de la perte sur le prompt (limite connue et assumée : voir §6).
- Optimiseur : `adamw_step_batched!`, LR=`3f-3` (repris du précédent
  goulot/témoin de cette session), β1=0.9, β2=0.999, ε=1e-8, weight_decay=0.
- 150 pas AdamW = 25 époques sur les 6 exemples d'entraînement, ordre
  cyclique FIXE (1,2,3,4,5,6,1,2,...), pas de mélange aléatoire — simplicité
  et reproductibilité plutôt que réalisme d'entraînement.
- Aucun arrêt anticipé conditionné aux résultats — 150 pas fixés a priori.
- Courbe de perte et trajectoire de `alpha` enregistrées à chaque pas.

## 5. Évaluation — AVANT et APRÈS entraînement, génération gloutonne réelle

Pour chaque item des jeux held-out A, B et témoin négatif : génération
gloutonne (recalcul complet, max 40 tokens, arrêt sur `EOS_ID`), AVANT tout
entraînement (alpha=0, doit être bit-exact au modèle non greffé — même
vérification que Check 2) et APRÈS les 150 pas.

**Vérificateur de conformité de format** (automatique, pas de jugement
humain) :
- Type "un mot" : texte réduit (`strip`), ponctuation finale retirée
  (`. ! ? , ; :`), conforme SSI il reste exactement 1 token séparé par un
  espace.
- Type "oui/non" : texte réduit, ponctuation finale retirée, mis en
  minuscules, conforme SSI égal exactement à `yes` ou `no`.
- Témoin négatif ("verbeux attendu") : conforme (= comportement correct
  préservé) SSI la réponse contient **au moins 5 mots** — teste que la
  greffe n'a pas appris à tout tronquer indistinctement.

Contenu factuel (secondaire, informatif seulement — pas l'axe principal du
verdict) : le mot/la réponse produite correspond-elle (comparaison
insensible à la casse) à la réponse cible ?

## 6. Critères de succès/partiel/échec — FIXÉS À L'AVANCE

Soit `held-out` = A ∪ B (7 items), `neg` = témoin négatif (3 items).

- **SUCCÈS COMPLET** si : conformité de format sur `held-out` passe d'une
  baseline basse (attendue proche de 0/7, à confirmer empiriquement) à
  **≥ 5/7 (≥ 70%)** APRÈS entraînement, ET conformité maintenue sur `neg`
  (**≥ 2/3** restent verbeux) — preuve que la correction est conditionnelle
  à la contrainte, pas un effondrement général.
- **SUCCÈS PARTIEL** si l'une des situations suivantes : (a) amélioration
  réelle mais modeste sur `held-out` (2-4/7, soit 29-57%) avec `neg`
  préservé (≥2/3) ; ou (b) généralisation forte sur held-out A (contenu
  disjoint, même contrainte, ≥3/4) mais faible/nulle sur held-out B
  (contrainte disjointe, ≤1/3) — généralisation étroite au gabarit de
  contrainte vu à l'entraînement, pas au-delà ; ou (c) conformité atteinte
  sur `held-out` mais au prix d'une dégradation notable de `neg` (≤1/3
  restent verbeux) — la greffe "corrige" en tronquant indistinctement,
  résultat gagné artificiellement, pas une vraie fonction conditionnelle.
- **ÉCHEC** si : amélioration sur `held-out` ≤ 1/7 au-dessus de la baseline
  (le gate a bougé — vérifié séparément, cf. mécanique déjà prouvée par
  Check 2 — mais n'a appris aucune fonction généralisable utile), OU la
  perte d'entraînement elle-même ne baisse pas de façon monotone sur les 150
  pas (signe que même la mémorisation des 6 exemples d'entraînement échoue).

Rapporté honnêtement quel que soit le résultat, avec les générations
réelles avant/après pour CHAQUE item des 3 jeux, comme le précédent
goulot/témoin.

## 7. Limites assumées, déclarées à l'avance

- Perte non masquée sur le prompt (§4) : dilue le signal de gradient vers
  les positions du prompt (déjà correctement prédites par construction du
  gabarit ChatML) plutôt que de le concentrer sur les tokens de réponse. Un
  échec pourrait donc refléter cette limite du protocole plutôt qu'une
  limite du mécanisme de greffe lui-même — signalé si observé.
- 6 exemples d'entraînement est un budget extrêmement petit pour une
  correction comportementale générale ; un succès partiel ou un échec
  seraient cohérents avec la littérature sur le few-shot fine-tuning ciblé,
  pas une surprise en soi.
- Un seul site de greffe (`:layer_25_out`), une seule graine d'insertion,
  aucun balayage d'hyperparamètre (LR, nombre de pas) — pas de recherche
  d'optimum, un seul point pré-enregistré et rapporté tel quel.

## 8. Addendum daté (2026-09-01, écrit APRÈS avoir tenté de reproduire l'expérience)

Non prévu au moment de l'écriture des sections ci-dessus. En construisant
le notebook `graft_qwen_experiment.ipynb`, un DEUXIÈME lancement de ce
protocole exact (même code, même graine `4242`) a **divergé numériquement**
(perte → NaN à partir du pas 93/150, génération dégénérée en `"!!!!..."`),
alors que le premier lancement avait convergé proprement (§6, résultat
rapporté ci-dessus).

Le script (`graft_qwen_experiment_run.jl`) a été rendu robuste à ce mode de
défaillance (détection de non-finitude, arrêt propre, verdict dédié
`DIVERGENCE_NUMERIQUE`) **sans changer le protocole pré-enregistré**
lui-même (LR, nombre de pas, jeu de données inchangés). Plusieurs lancements
frais supplémentaires ont ensuite été faits pour caractériser la fréquence
de cette divergence, suivant la convention de ce dépôt
(`docs/REPRODUCING.md`, "several independent launches").

### 8.1 Addendum 2 (même date, écrit APRÈS une investigation dédiée de la cause)

La phrase "cause plausible : les noyaux CUDA/cuBLAS ne garantissent pas le
déterminisme bit-à-bit entre lancements" écrite dans une version précédente
de cet addendum **n'avait pas été vérifiée**. Une investigation dédiée
(scripts `notebook/diag_graft_determinism_probe{,_v2}.jl`, logs
`graft_qwen_experiment_cleanA.log`/`cleanB.log`) a établi ce qui suit,
directement, pas par supposition :

- **Ce n'est pas un bug de graine** : `Random.seed!` + `CUDA.seed!` sont
  appelés correctement, à l'endroit correct. Vérifié en lisant le code.
- **Le calcul EST bit-déterministe entre lancements séparés, en isolation
  stricte**, sur un protocole tronqué (20 pas, sans phase d'évaluation
  avant-entraînement) : deux lancements séquentiels, `nvidia-smi` vérifié
  propre avant chacun, donnent des trajectoires identiques à 15 chiffres
  significatifs. Ceci réfute "cuBLAS jamais déterministe" comme explication
  générale.
- **Mais dès que la phase d'évaluation avant-entraînement réelle (10
  générations gloutonnes) est ajoutée**, la trajectoire diverge en VALEUR
  (pas en NaN) à partir du pas 4, de façon reproductible et déterministe
  (pas du bruit). Cette phase, à elle seule, pousse la VRAM à
  16041-16074/16384 Mio (aucun autre process actif) — quasi-saturation de
  la carte 16 Go. Diagnostic le plus plausible, non prouvé au niveau du
  noyau individuel : sélection heuristique d'algorithme cuBLAS/CUDA
  dépendante de l'état mémoire du device.
- **Sous isolation stricte et surveillée en continu, le protocole réel
  complet a quand même divergé, deux fois sur deux essais** (`cleanA` au
  pas 6/150, `cleanB` — isolation vérifiée du début à la fin — au pas
  4/150). L'isolation seule n'a donc PAS rendu ce protocole fiable dans cet
  échantillon.
- **Une contention GPU externe réelle a aussi été constatée directement**
  pendant cette investigation même (pas supposée) : un process `jupyter
  nbconvert --execute` de ce même notebook, démarré indépendamment,
  tournait en parallèle sur une fenêtre de temps qui recouvre celle de
  `cleanA` (retrouvé via `Get-CimInstance Win32_Process`, chaîne de
  parenté du process). Ce facteur de risque est confirmé réel sur cette
  machine, mais n'est ni nécessaire (`cleanB` a divergé sans lui) ni
  suffisant à lui seul (le lancement concurrent en question n'a lui-même
  pas divergé).

Sur **9 lancements indépendants connus au total** de ce protocole exact
(les 5 déjà comptés dans la version précédente de cet addendum, plus le
premier lancement notebook via `nbconvert`, plus les 3 lancements de cette
investigation dédiée — `cleanA`, `cleanB`, et le `nbconvert` concurrent
découvert en même temps) : **5 SUCCÈS PARTIEL** (held-out 3/7 une fois, 4/7
quatre fois ; témoin négatif 3/3 à chaque fois), **1 SUCCÈS COMPLET** (5/7),
et **3 DIVERGENCE_NUMERIQUE** (aux pas 93, 6 et 4/150 — point d'échec non
reproductible). Le taux de divergence mesuré ici (**3/9, ~33%**) est plus
élevé que ce que la première caractérisation partielle (5 lancements, 1
divergence, ~1/5) avait suggéré — corrigé ici plutôt que reconduit tel
quel. Voir `notebook/graft_qwen_experiment.ipynb` §3.4 et §4 pour la
discussion complète et tous les artefacts.

**Sur le critère de succès/partiel/échec §6** : il ne prévoyait aucune
exigence de stabilité d'un lancement à l'autre — il a été écrit pour UN
lancement, avant qu'aucune divergence ne soit connue. C'est une limite
réelle du pré-enregistrement original, signalée ici plutôt que dissimulée :
un protocole numériquement instable au point de diverger sur ~1/3 de ses
lancements peut passer un pré-enregistrement qui ne prévoit pas de critère
de reproductibilité.

## 9. Correctif de la divergence tardive + re-vérification (2026-09-01)

Écrit APRÈS implémentation du correctif et APRÈS les 7 lancements de
re-vérification décrits ci-dessous (6 lancements dédiés `fix1`-`fix6` plus
le lancement canonique de `graft_qwen_experiment.ipynb`, cellule 3.3). Fait
suite à §8.2 : les 9 lancements connus à l'époque avaient été regroupés sous
un taux de divergence unique (3/9, ~33%) sans séparer deux causes
distinctes identifiées séparément dans cette même section — cette section-ci
distingue les deux et rapporte l'effet mesuré d'un correctif ciblant
seulement l'une d'elles.

### 9.1 Les deux causes, redistinguées

1. **Divergence tardive** (le seul exemple connu : NaN au pas 93/150, le
   deuxième lancement historique du 2026-08-31). La perte atteint un
   plateau bas (~0.03-0.07) vers le pas ~26-35 sur tous les lancements
   archivés, puis y reste ~120 pas de plus — mais avec des pics transitoires
   récurrents de 4-6x l'amplitude du plateau qui se résorbent d'eux-mêmes
   (log complet `graft_qwen_experiment_results_run3.json` : ex. pas 127
   loss=0.212, pas 134 loss=0.277, plateau ~0.03-0.05). Mécanisme plausible,
   non prouvé au niveau du noyau CUDA individuel : une fois le gradient
   quasi nul en régime de plateau, l'estimateur du 2e moment d'Adam (m2, le
   dénominateur) devient minuscule, donc le pas normalisé m1/sqrt(m2+eps)
   peut devenir disproportionné même après le clip de gradient déjà actif.
   Vérifié directement dans le code (`src/kernels.jl`, noyau
   `_multi_adamw_kernel!`, appelé par `adamw_step_batched!` avec l'argument
   `clip=1f0` toujours passé depuis le premier run) : ce clip est PAR
   ÉLÉMENT et s'applique à `g_val` AVANT son accumulation dans `m1`/`m2` — il
   ne borne PAS le pas final normalisé qui en résulte. Une phrase d'une
   version antérieure de ce document affirmait « LR=3e-3 sans clip de
   gradient » ; c'était inexact (le clip existe et est actif depuis le
   début) — corrigé ici plutôt que reconduit tel quel.
2. **Divergence précoce** (`cleanA` pas 6, `cleanB` pas 4, §8.2 point 4) —
   une discontinuité de VALEUR (pas de bruit) liée à l'historique
   d'allocation mémoire GPU au moment de la phase d'évaluation
   avant-entraînement, qui influence probablement la sélection heuristique
   d'algorithme cuBLAS/CUDA (§8.2 point 3).

### 9.2 Correctif implémenté pour la cause 1 — décroissance de LR par plateau + arrêt anticipé

Dans `graft_qwen_experiment_run.jl` (commentaire complet au-dessus de la
constante `PLATEAU_WINDOW`) : une moyenne mobile courte de la perte (fenêtre
`PLATEAU_WINDOW=10` pas) est comparée à la meilleure moyenne mobile observée
jusqu'ici (`plateau_best_avg`, ratchet monotone — PAS une comparaison à une
fenêtre décalée fixe, rejetée après simulation hors-ligne sur les
`loss_history` déjà enregistrées de run1/run2/run3 : le bruit du plateau
produisait parfois une « amélioration » apparente de plus de 2% par pur
hasard, ce qui remettait le compteur de stagnation à zéro indéfiniment et ne
déclenchait AUCUNE décroissance sur `run1` avec ce schéma). Si la moyenne
courante n'améliore pas `plateau_best_avg` d'au moins
`PLATEAU_REL_IMPROVEMENT=15%` pendant `PLATEAU_PATIENCE=10` vérifications
consécutives, le LR est divisé par `PLATEAU_DECAY_FACTOR=4`. Un garde-fou
dur, `MIN_STEP_BEFORE_DECAY=40`, interdit toute décroissance avant ce pas —
marge confortable au-delà du pas ~26-35 où la convergence réelle est
observée sur les 4 lancements archivés analysés (run1, run2, run3, le
premier run du 2026-08-31), donc les 30-50 premiers pas (où l'apprentissage
réel a lieu) ne sont jamais affectés par construction. Après
`PLATEAU_MAX_DECAYS=3` décroissances (LR final ≈ LR initial / 64), le
protocole s'arrête — combine décroissance de LR et arrêt anticipé en un seul
mécanisme, plutôt que les traiter séparément. **Le protocole pré-enregistré
lui-même (LR initial `3e-3`, budget de 150 pas, jeu de données) n'a PAS été
modifié** — seul ce qui se passe APRÈS détection d'un plateau change.

Ces paramètres ont été choisis par simulation hors-ligne (rejeu du
détecteur sur les `loss_history` déjà enregistrées de run1/run2/run3, sans
aucun calcul GPU) avant tout nouveau lancement réel : le premier
déclenchement attendu tombait entre les pas 55 et 68 selon le run (jamais
avant le pas 40, jamais pendant la descente initiale), et les 3
décroissances se terminaient entre les pas 75 et 94 en simulation — avant le
pas 93 où l'unique divergence tardive documentée avait eu lieu. Ceci restait
une marge de sécurité visée, pas une garantie ; voir §9.4 pour la mesure
réelle.

### 9.3 Piste testée pour la cause 2 — préchauffage GPU dédié (résultat : partiel, pas résolu)

Script dédié : `diag_graft_determinism_probe_v3_warmup.jl`, calqué sur le
précédent JIT-warmup déjà appliqué à `qwen2.ipynb` (cellule 16, un appel
factice avant la vraie session pour payer le coût de compilation JIT une
seule fois) — hypothèse testée ici : un préchauffage GPU dédié (10
générations gloutonnes jetées sur le graphe NON MODIFIÉ, AVANT
`Random.seed!` et l'insertion de la greffe) pourrait porter la
VRAM/l'historique d'allocation cuBLAS dans un état stable AVANT que quoi que
ce soit de mesuré ne commence, éliminant la dépendance à l'ordre
d'exécution qui cause la divergence précoce (§8.2 point 3).

**Résultat mesuré (un lancement, `diag_graft_determinism_probe_v3_warmup.log`,
comparé aux logs déjà archivés `diag_graft_determinism_probe_{A,B,v2}.log`)** :
amélioration partielle, PAS une résolution complète.
- Sans préchauffage (`v2`) : bit-identique au probe sans éval-avant (`A`/`B`)
  jusqu'au pas 3 seulement ; diverge en VALEUR au pas 4 (`loss=4.4882` contre
  `4.3104`).
- Avec préchauffage (`v3warmup`) : bit-identique à `A`/`B` jusqu'au pas 5
  (2 pas de plus) ; **une NOUVELLE discontinuité apparaît au pas 6**, avec
  une TROISIÈME valeur de perte distincte (`loss=3.2164`), différente à la
  fois de `A`/`B` (`2.8239`) et de `v2` sans préchauffage (`2.9181`).

Le préchauffage retarde le point de divergence de 2 pas, il ne l'élimine
pas — un préchauffage de taille/composition fixe ne neutralise donc pas
entièrement la dépendance à l'historique d'allocation mémoire, cohérent avec
un mécanisme plus subtil que « seuil de VRAM total franchi ou non » (par
exemple, le calendrier exact des allocations intermédiaires du graphe
gréffé — buffers Adam `m1`/`m2`, backward élagué — pourrait lui-même
réintroduire une dépendance que le préchauffage, effectué avant la greffe,
ne couvre pas). **La cause 2 reste ouverte** ; aucune tentative
supplémentaire de réglage (taille de préchauffage, ordre) n'a été faite,
conformément à la consigne de ne pas forcer un correctif fragile.

### 9.4 Re-vérification — 9 lancements indépendants, GPU vérifié propre avant chacun

Protocole de vérification : avant CHAQUE lancement, `nvidia-smi
--query-gpu=memory.used,memory.total,memory.free` confirmé proche de la
ligne de base (~200-500 Mio, aucun autre process CUDA actif via
`nvidia-smi --query-compute-apps`) ET `Get-Process julia` confirmé sans
lancement précédent encore actif ; aucun lancement en parallèle ou en
arrière-plan superposé à un autre. 6 lancements dédiés (`fix1`-`fix6`,
`GRAFT_QWEN_RUN_ID=fix{1..6}`) strictement sérialisés, plus le lancement
canonique de `graft_qwen_experiment.ipynb` (cellule 3.3, `jupyter nbconvert
--execute --inplace`) exécuté à TROIS reprises pendant cette phase : une
première fois pour valider que le correctif s'exécute proprement de bout en
bout avant d'écrire la prose de cette section, une deuxième après une
première mise à jour de cette prose, une troisième (l'état final livré)
après la dernière mise à jour. Chacune de ces trois exécutions est un
lancement indépendant du même protocole corrigé (même code, même graine
`4242`) — 9 lancements post-correctif au total. Le fichier JSON canonique
(`graft_qwen_experiment_results.json`, sans suffixe de `RUN_ID`) est écrasé
à chaque exécution du notebook ; les deux premières exécutions canoniques
ont donc été archivées manuellement en résumé fidèle
(`graft_qwen_experiment_results_ipynb{3,4}.json`, mêmes champs que ceux
utilisés par le bilan, JSON complet original non conservé) avant d'être
écrasées, suivant la même convention que `_results_ipynb2.json` en §3.4 ;
seule la troisième (la plus récente) reste le JSON complet original sur
disque. La cellule de bilan du notebook (§3.5) charge maintenant les 6
relances dédiées ET les 3 exécutions canoniques (2 archivées + 1 courante)
de façon uniforme, sans plus rien recenser à la main.

| Lancement | Verdict | Held-out avant→après | Témoin nég. | Pas exécutés |
|---|---|---|---|---|
| fix1 | SUCCES_PARTIEL | 1/7 → 4/7 (A=1/4, B=3/3) | 3/3 | 87/150 (arrêt anticipé, 3 décroissances) |
| fix2 | SUCCES_COMPLET | → 6/7 (A=3/4, B=3/3) | 3/3 | 79/150 (arrêt anticipé, 3 décroissances) |
| fix3 | SUCCES_PARTIEL | → 4/7 (A=1/4, B=3/3) | 3/3 | 88/150 (arrêt anticipé, 3 décroissances) |
| fix4 | SUCCES_PARTIEL | → 4/7 (A=1/4, B=3/3) | 3/3 | 82/150 (arrêt anticipé, 3 décroissances) |
| fix5 | **DIVERGENCE_NUMERIQUE** | — | — | 6/150 (NaN au pas 6 — cause précoce, HORS périmètre de ce correctif) |
| fix6 | SUCCES_PARTIEL | → 4/7 (A=1/4, B=3/3) | 3/3 | 90/150 (arrêt anticipé, 3 décroissances) |
| canonique (1ère exéc., `ipynb3` archivé) | SUCCES_PARTIEL | 1/7 → 4/7 (A=1/4, B=3/3) | 3/3 | 86/150 (arrêt anticipé, 3 décroissances) |
| canonique (2e exéc., `ipynb4` archivé) | SUCCES_COMPLET | 1/7 → 5/7 (A=2/4, B=3/3) | 3/3 | 82/150 (arrêt anticipé, 3 décroissances) |
| canonique (3e exéc., finale, sur disque) | SUCCES_PARTIEL | 1/7 → 4/7 (A=1/4, B=3/3) | 3/3 | 86/150 (arrêt anticipé, 3 décroissances) |

*Note : ce tableau a été figé après la 4e exécution réelle du notebook (une
exécution supplémentaire, non prévue, s'est avérée nécessaire après le
passage du recensement manuel au chargement dynamique décrit ci-dessus —
voir §9.5) ; le libellé « 3e exéc. » de la dernière ligne désigne son rang
dans le tableau, pas un compte littéral d'exécutions du notebook.*

**Bilan chiffré, honnête** :
- **Taux de divergence global : 1/9 (~11.1%)**, contre 3/9 (~33%) avant
  correctif. Baisse mesurée dans le sens attendu, mais échantillon modeste —
  PAS présenté comme une preuve statistique définitive.
- **Séparé par cause** : la seule divergence post-correctif (`fix5`, pas
  6/150) est de la cause PRÉCOCE, celle que ce correctif ne cible pas
  (§9.3). **0 divergence tardive observée sur ces 9 lancements**, contre 1
  sur les 9 lancements précédents (pas 93/150) — cohérent avec le correctif
  visé, mais 0 observation sur 9 essais ne prouve pas un taux nul, seulement
  compatible avec une réduction réelle.
- **Conformité held-out des 8 lancements convergés : 4/7 (×6), 5/7 (×1),
  6/7 (×1)** — gamme comparable à, voire légèrement au-dessus de, celle
  d'avant correctif (3/7 une fois, 4/7 quatre fois, 5/7 une fois sur les 6
  lancements convergés historiques). **Témoin négatif préservé 3/3 sur les
  8 lancements convergés, sans exception** — le correctif ne dégrade pas le
  signal de généralisation, et l'arrêt anticipé (entre les pas 79 et
  90/150 selon le lancement, toujours nettement après la fenêtre de
  convergence réelle ~pas 26-35) n'a pas coupé l'entraînement avant que
  l'apprentissage réel n'ait eu lieu. Les trois exécutions canoniques
  recensées, lancées avec EXACTEMENT le même code et la même graine à
  quelques heures d'intervalle, donnent des verdicts qui varient (4/7, 5/7,
  4/7) — pas une contradiction mais une nouvelle illustration directe de la
  sensibilité d'un lancement à l'autre déjà caractérisée en §3.4/§8.2
  (cause 2, non résolue par ce correctif).
- Aucun `ECHEC` parmi les 8 lancements convergés (tous `SUCCES_PARTIEL` ou
  `SUCCES_COMPLET`), cohérent avec le bilan pré-correctif.

### 9.5 Discipline suivie, périmètre du correctif

- Le correctif vit entièrement dans `graft_qwen_experiment_run.jl` (script
  notebook) — **aucune modification de `src/`, `Project.toml` inchangé**,
  donc aucune ré-exécution de la suite de tests complète n'était requise par
  la consigne de cette phase (conditionnée à une modification de `src/`).
- Chaque lancement GPU de cette phase (9 lancements du protocole complet +
  1 sonde de préchauffage courte) a été vérifié isolé avant démarrage
  (`nvidia-smi`, `Get-Process julia`) et le process confirmé terminé avant
  le lancement suivant — aucun chevauchement, conformément à la contrainte
  de contention GPU documentée en tête de cette phase de travail. GPU et
  processus Julia confirmés propres/absents en fin de phase.
- `graft_qwen_experiment.ipynb` a été ré-exécuté intégralement de bout en
  bout (`jupyter nbconvert --execute --inplace`) à TROIS reprises : une
  première fois juste après implémentation du correctif (pour valider qu'il
  s'exécute proprement de bout en bout avant d'écrire la prose de cette
  section), une deuxième après une première mise à jour de la prose (§3.5,
  §4/§5), une troisième (l'état final livré) après la dernière mise à jour
  (rendue nécessaire par le passage d'un recensement manuel à un
  chargement dynamique des deux exécutions archivées `ipynb3`/`ipynb4`) —
  exécution propre confirmée les trois fois (`EXIT_CODE=0`), chacune comptée
  comme un lancement de re-vérification ci-dessus (les 7e, 8e et 9e du
  tableau §9.4).
- **Ce qui n'est PAS revendiqué** : que la divergence tardive est
  définitivement éliminée (échantillon de 9 lancements, 0 observation) ;
  que la divergence précoce est résolue (elle ne l'est pas — §9.3) ; qu'un
  réglage plus poussé des seuils du détecteur de plateau (fenêtre, seuil de
  progrès, patience) ne changerait pas ce bilan — aucun balayage
  d'hyperparamètre de ce détecteur n'a été fait au-delà de la simulation
  hors-ligne initiale (§9.2).
