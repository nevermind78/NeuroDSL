# Conjecture — surcoût de cône en attention batchée

**Contexte.** P1-bis (session du 2026-07-11) a mesuré, sur 9 sites de patch
(3 profondeurs × {`pr_h`/`sc_h`, `ao_h`}), l'écart de taille de cône (nombre de
nœuds à recalculer) entre l'attention **batchée** (`batched_attn=true`,
`gemm_strided_batched!`) et **non batchée** (`batched_attn=false`) de NeuroDSL.
La prédiction initiale de Fable (+4 constant pour les sites pré-softmax, 0
pour `ao_h`) ne collait pas aux mesures brutes (+12/+10/+6 et +8/+6/+2 selon la
profondeur). Une première conjecture (ci-dessous, §Énoncé initial) expliquait
ces 9 mesures exactement. Un second passage de Fable, chargé de la **démontrer
formellement à partir du code** (`src/patching.jl`, `src/layers.jl`) plutôt que
de la re-mesurer, a révélé qu'elle est correcte pour tout ce qui a été mesuré,
mais **incomplète** : deux types de site jamais testés par P1-bis (`q_h`,
`k_h`) suivent en réalité une autre loi. Le théorème corrigé est en §Théorème.

## Notation

- $s$ : le type de site patché.
- $n$ : nombre de têtes d'attention par couche.
- $D$ : nombre de couches d'attention **strictement en aval** de la couche
  contenant $s$.
- $\mathcal{A} = \{q_h, k_h, v_h, sc_h, sk_h, pr_h\}$ : sites calculés **avant**
  la fusion en tenseur partagé (par opposition à $ao_h$, qui en est extrait).

## Énoncé initial (vrai sur les 9 mesures, incomplet)

$$
\Delta(s, D) \;=\; n \cdot \mathbb{1}[s \in \mathcal{A}] \;+\; 2D
$$

## Démonstration

**Cadre.** Soient $G_u$ et $G_b$ les graphes émis par `LlamaModel` à $L$
couches, $n$ têtes, respectivement en mode non-batché et batché, avec pour
tout site $s$ le cône $C(s) = \{s\} \cup \mathrm{Desc}(s)$, où $\mathrm{Desc}(s)$
est l'ensemble des descendants stricts de $s$ dans le DAG des règles (arêtes :
entrée $\to$ sortie de chaque `GraphRule`). Cette identité tient car (H1)
aucun `watcher` n'est émis par les constructeurs de couches, et (H2) le cône
est mesuré sur un graphe entièrement valide (après un forward complet, avant
tout patch), donc la garde `out_nd.valid` de `_downstream_nodes`
(`src/patching.jl:117-142`) n'exclut rien : la fonction calcule exactement
l'atteignabilité structurelle. Ainsi $\Delta(s,D) = |\mathrm{Desc}_b(s)| -
|\mathrm{Desc}_u(s)|$, le nœud $s$ lui-même se compensant.

**Inventaire des différences.** $G_u$ et $G_b$ coïncident symbole pour symbole
et arête pour arête, sauf dans chaque bloc d'attention : $G_b$ ajoute deux
nœuds par couche ($sc3$, $ao3$) et recâble $sc3 \leftarrow \{q_1..q_n,
k_1..k_n\}$, $sc_h \leftarrow sc3$, $ao3 \leftarrow \{pr_1..pr_n, v_1..v_n\}$,
$ao_h \leftarrow ao3$, au lieu de $sc_h \leftarrow \{q_h,k_h\}$ et $ao_h
\leftarrow \{pr_h,v_h\}$ (`src/layers.jl:67-138`). Tout le reste (tranches
Q/K/V, $sk_h \leftarrow sc_h$, $pr_h \leftarrow sk_h$, `hcat_heads`, Linear de
sortie, résiduels, MLP, chaînage des couches) est commun aux deux modes.

**Terme local, couche par couche :**

- $s \in \{v_h, sc_h, sk_h, pr_h\}$ : ces sites précèdent la fusion $ao3$ mais
  la SUIVENT (ou n'y participent pas) pour ce qui concerne $sc3$. Exemple
  $v_h$ : $\mathrm{Desc}_u(v_h) = \{ao_h\}$ (1 nœud, via le `:matmul`
  $[pr_h,v_h]\to ao_h$) ; $\mathrm{Desc}_b(v_h) = \{ao3\} \cup
  \{ao_1,\dots,ao_n\}$ ($n+1$ nœuds, puisque $ao3 \leftarrow \{\dots,v_h,\dots\}$
  fait fan-out vers toutes les têtes). Excédent $= n$. Le même calcul
  (déplacé d'un cran) donne excédent $n$ pour $sc_h$, $sk_h$, $pr_h$.
- $s = ao_h$ : seul consommateur `concat` dans les deux modes ; $sc3$/$ao3$
  sont des **ancêtres** de $ao_h$ en mode batché, jamais des descendants.
  Excédent $= 0$.
- $s \in \{q_h, k_h\}$ : ces sites précèdent la fusion $sc3$, qui fait
  fan-out vers **toutes** les têtes dès la première étape (pas seulement la
  dernière). $\mathrm{Desc}_u(q_h) = \{sc_h,sk_h,pr_h,ao_h\}$ (4 nœuds, tête
  $h$ seule). $\mathrm{Desc}_b(q_h) = \{sc3,ao3\} \cup
  \{sc_j,sk_j,pr_j,ao_j\}_{j=1}^{n}$ ($4n+2$ nœuds, puisque $sc3$ irrigue les
  $n$ chaînes complètes, qui reconvergent par $ao3$). Excédent $= 4n-2$.

**Terme global ($2D$ exactement).** Depuis `concat_i`, le suffixe est commun
aux deux modes : la fin du bloc $i$ (contribution nulle), puis chaque bloc
$j$, $i<j\le L$, est intégralement descendant de `layer_{j-1}_out` dans les
deux modes, et ses symboles ne différent que par $\{sc3_j,ao3_j\}$ : $+2$ par
couche strictement en aval, soit $2D$. Ni $2(D{+}1)$ — $ao3_i$ (et $sc3_i$
pour $q/k$) est déjà compté dans le terme local, jamais dans le global — ni
$2D{+}1$ : aucun nœud asymétrique n'existe hors des blocs d'attention, et la
dernière couche n'a aucun statut particulier.

## Théorème (corrigé)

$$
\Delta(s, D) \;=\; c(s) + 2D, \qquad
c(s) = \begin{cases}
  n & s \in \{v_h, sc_h, sk_h, pr_h\} \\
  0 & s = ao_h \\
  4n-2 & s \in \{q_h, k_h\}
\end{cases}
$$

**Conditions de validité** : absence de `watchers` et de fusion (`_fuse!`)
sur les nœuds concernés ; graphe pleinement valide au moment de la mesure ;
`hcat_heads` et le `Linear` de sortie communs aux deux modes (ce qui annule
le suffixe) ; aucune op conditionnelle asymétrique entre les deux graphes
(un dropout présent dans un seul mode briserait l'annulation du suffixe).

## Vérification

**9 mesures P1-bis** (n=4, sites `pr_h`/`sc_h`/`ao_h` uniquement — jamais
$q_h$/$k_h$) :

| site | couche | $D$ | $\Delta$ prédit | $\Delta$ mesuré |
|---|---|---|---|---|
| `pr_h`/`sc_h` | layer_1 | 4 | $4+2(4)=12$ | **12** |
| `pr_h`/`sc_h` | greffe | 3 | $4+2(3)=10$ | **10** |
| `pr_h`/`sc_h` | layer_3 | 1 | $4+2(1)=6$ | **6** |
| `ao_h` | layer_1 | 4 | $0+2(4)=8$ | **8** |
| `ao_h` | greffe | 3 | $0+2(3)=6$ | **6** |
| `ao_h` | layer_3 | 1 | $0+2(1)=2$ | **2** |

**Cas non mesuré par P1-bis, prédit par le théorème corrigé, puis vérifié**
(`notebook/p1bis_batched_vs_nonbatched.jl`, sites_kind étendu à `q_h`/`k_h`,
2026-07-11) : pour $q_h$/$k_h$ ($n=4$), $\Delta = (4\cdot4-2)+2D$ prédisait
$+22$ (layer_1, $D=4$), $+20$ (greffe, $D=3$), $+16$ (layer_3, $D=1$) — PAS
$4+2D$ ($+12$/$+10$/$+6$) comme l'énoncé initial l'aurait prédit à tort. Les
trois valeurs mesurées sont **exactement +22/+20/+16** — la prédiction a été
faite avant la mesure, à partir du raisonnement sur le code, pas ajustée
après coup. **15/15 mesures désormais conformes au théorème** (9 originales
+ 6 nouvelles), aucune exception.

**Cas limite $D=0$ (dernière couche)** : $\Delta = c(s)$ seul, cohérent car
la dérivation locale n'utilise jamais $D>0$.

## Portée et limites

- Le terme global ($2D$) et le terme local pour $\{v_h,sc_h,sk_h,pr_h,ao_h\}$
  sont maintenant **démontrés** à partir du code (`_downstream_nodes`,
  `MultiHeadAttention`, `LlamaBlock`/`LlamaModel`), pas seulement mesurés.
- Le terme local pour $q_h/k_h$ ($4n-2$) est démontré ET vérifié
  empiriquement (+22/+20/+16, exact aux 3 profondeurs testées) — la
  prédiction précédait la mesure.
- La preuve suppose l'absence de `watchers`/fusion/ops asymétriques (voir
  conditions de validité ci-dessus) — vraie pour la config testée
  (`LlamaModel` standard, pas de dropout actif), pas garantie en général.
- Reste structurelle (comptage de nœuds), pas un modèle de coût en temps :
  le coût en millisecondes ne suit pas le même écart (cf. P4, où le surcoût
  de lancement de noyaux GPU domine à cône presque constant).



La forme générale correcte (celle déjà démontrée, rendue explicite)
$$
\Delta(s, i, L) = c(s) + 2(L - i), \qquad
c(s) = \begin{cases}
n & s \in {v_h, sc_h, sk_h, pr_h} \\
0 & s = ao_h \\
4n-2 & s \in \{q_h, k_h\}
\end{cases}
$$

où :

* $n$ = nombre de têtes par couche (n'importe quelle valeur, pas seulement 4),
* $L$ = nombre total de couches d'attention du modèle (couches originales + toute greffe insérée),
* $i$ = index de la couche contenant le site patché,
* $D = L - i$ = nombre de couches strictement en aval — c'est la forme explicite de ce qui était noté juste "$D$" dans le document.