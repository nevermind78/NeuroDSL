# Croissance de modèle pendant l'entraînement : modèle mathématique complet et validation sur 16 profondeurs

*NeuroDSL — synthèse théorique des jalons 0, 1, 1-corrigé et 1b, finalisée le 2026-07-14 avec
les runs depuis zéro à toutes les profondeurs L=1..16. Document autonome : toutes les valeurs
numériques sont recalculées à partir des fichiers JSON cités, aucune n'est reprise de mémoire.*

---

## 1. Contexte et motivation

NeuroDSL rend une greffe de couche (`insert_block!`) exacte au bit près au moment de l'insertion
et sans interruption de l'optimiseur (`extend_adamw_state!`), ce qui fait de la *densité* du
calendrier de croissance d'un transformer un axe expérimental gratuit — inaccessible en pratique
aux frameworks à bande (PyTorch/JAX), où chaque événement de croissance est une chirurgie
manuelle. Trois campagnes expérimentales (jalons 0, 1, 1b) ont établi *quand* la croissance bat
l'entraînement à taille fixe ; ce document clôt la question *pourquoi*, par un modèle de courbe
d'apprentissage ajusté sur des runs depuis zéro à chacune des 16 profondeurs, et trois théorèmes
vérifiés contre 18 calendriers mesurés.

## 2. Les trois résultats empiriques

Tâche : char-LM sur TinyShakespeare (vocabulaire 65 caractères, perte en nats/caractère),
dim 256, 4 têtes, hidden 512, séquence 256, AdamW lr 1e-3, batch 1. Le budget de calcul est
compté en « layer-steps » (Σ du nombre de couches à chaque pas), proxy FLOPs linéaire en
profondeur. Validation : moyenne de la cross-entropy sur 64 fenêtres régulièrement espacées du
split de validation (10 % final du corpus). Scripts : `growth_jalon0.jl` (module commun),
`growth_jalon0_full.jl`, `growth_jalon1_full.jl`, `growth_jalon1_fixed_full.jl`,
`growth_jalon1b_full.jl`.

**Résultat 1 (jalon 0) — la croissance bat le fixe à FLOPs égaux.** Budget 80 000 layer-steps,
3 seeds par bras (`growth_jalon0_results.json`). Bras A (4 couches fixes) : 1,6535 ± 0,0267 ;
bras B (2→3→4, parts de budget égales) : 1,6032 ± 0,0107 — chacune des trois valeurs de B bat
chacune des trois valeurs de A, et l'écart (0,0503) dépasse le critère pré-enregistré (≥ std de
A = 0,0267). Bras C (1→2→4) : 1,6366 ± 0,0025, critère non atteint.

**Résultat 2 (jalon 1 et jalon 1 corrigé) — le calendrier le plus grossier gagne, à FLOPs
égaux.** Budget 48 000 layer-steps, profondeur finale 16, 2 seeds, nombres de paliers
{1, 2, 4, 8, 16} (`growth_jalon1_results.json`, `growth_jalon1_fixed_results.json`). Sous deux
schémas d'allocation délibérément différents (parts égales, puis parts ∝ 1/L), le classement est
le même : le saut unique 1→16 gagne (1,7191 ± 0,0035 puis 1,6542 ± 0,0139), le fixe-16 perd
largement (2,0562 ± 0,0059 puis 2,0723 ± 0,0049), les densités intermédiaires sont entre les
deux. Le nombre total de pas de gradient achetés par le budget suit le même ordre — premier
indice que l'effet est une réallocation de calcul, pas une vertu de la gradualité.

**Résultat 3 (jalon 1b) — le classement s'inverse totalement à pas égaux.** Mêmes calendriers,
mais 4 000 pas de gradient pour tous, FLOPs libres (`growth_jalon1b_results.json`) : fixe-16
gagne (1,9570 ± 0,0086) et *tous* les calendriers de croissance perdent (de 2,0233 ± 0,0311 pour
16 paliers à 2,1195 ± 0,0278 pour 4 paliers). Conclusion mécanistique des trois résultats :
à cette échelle, le bénéfice de la croissance est intégralement un effet de réallocation de
calcul (les pas peu profonds coûtent moins cher, donc le même budget FLOPs achète plus de pas) ;
aucune trace d'un bénéfice intrinsèque de la gradualité.

Motif secondaire, constant mais jamais expliqué : 16 paliers bat 8 paliers dans les trois
campagnes (1,8053 vs 1,8151 ; 1,7593 vs 1,8082 ; 2,0233 vs 2,1169). Voir §7.

## 3. Le modèle mathématique

**Perte canonique.** Pour un entraînement depuis zéro à profondeur constante L, on modélise la
perte de validation après n pas de gradient par

```
perte(n, L) = a(L) + b(L) / (n + n0),      p = 1, n0 = 3000.
```

`a(L)` est le plancher asymptotique (capacité), `b(L)` le préfacteur de vitesse. Le choix p=1 et
n0=3000 est re-testé en §5.1 sur les données complètes et confirmé.

**Règle du pas effectif (continuité aux greffes).** `insert_block!` préserve exactement la
fonction calculée au moment de la greffe (identité au bit près, propriété F1 prouvée), donc la
perte est continue à la greffe. Le modèle postule qu'après une greffe vers la profondeur L′ à la
perte ℓ, le réseau évolue ensuite comme un réseau canonique de profondeur L′ ayant atteint la
même perte, c'est-à-dire repart du pas effectif

```
n_eff = b(L′) / (ℓ − a(L′)) − n0.
```

Ce n'est pas une hypothèse ad hoc : la greffe étant exactement préservante, l'état post-greffe
est un point légitime de l'espace des réseaux de profondeur L′, et la perte est la seule
coordonnée que le modèle suit. La règle est validée hors-échantillon en §6.4 (R² = 0,918 sur les
437 points de mesure strictement postérieurs à une première greffe, points jamais utilisés pour
l'ajustement).

**Vitesse de descente.** Sur la courbe canonique, dℓ/dn = −p·b^(−1/p)·(ℓ−a)^(1+1/p) ; on note
φ_L(ℓ) cette vitesse instantanée à perte ℓ et profondeur L (pour p=1 : φ_L(ℓ) = (ℓ−a_L)²/b_L).

## 4. Les trois théorèmes

Les preuves complètes (arguments de comparaison d'EDO) ont été établies le 2026-07-13 et sont
archivées dans la note mémoire du projet
(`project_neurodsl_growth_schedules_pitch_2026-07-13.md`) ; on en donne ici les énoncés et les
esquisses fidèles.

**Théorème A (régime à pas égaux).** L'entraînement fixe à profondeur L_f domine *tout*
calendrier de croissance, pour *tout* budget de pas N, si et seulement si
φ_{L_f}(ℓ) ≥ φ_L(ℓ) pour toute profondeur L disponible et toute perte ℓ entre la perte initiale
et a_{L_f}. *Esquisse* : la trajectoire de n'importe quel calendrier est solution de
ℓ′(n) = −φ_{L(n)}(ℓ) avec continuité aux greffes (règle du pas effectif) ; si φ_{L_f} majore
ponctuellement toutes les φ_L, un argument de comparaison de Grönwall borne toute trajectoire
par celle du fixe-L_f. Pour la nécessité, si le champ de vitesses se croise à une perte ℓ×, un
calendrier qui commute exactement à ℓ× bat le fixe. Une condition suffisante commode est b(L)
non croissante ; nos données donnent b croissante (§5), donc la dominance n'est pas automatique
et doit être vérifiée numériquement — c'est fait en §6.1.

**Théorème B (régime à FLOPs égaux, coût d'un pas ∝ L).** En temps-coût, la vitesse de descente
à profondeur L vaut φ_L(ℓ)/L = ((ℓ−a_L)/κ(L))² pour p=1, avec

```
κ(L) = (L^p · b(L))^(1/(p+1))   —   pour p=1 : κ(L) = √(L·b(L)).
```

Maximiser cette vitesse à perte ℓ revient à choisir le point (κ(L), a(L)) qui maximise la pente
(ℓ−a_L)/κ(L) du segment joignant (0, ℓ) à (κ(L), a(L)) : quand ℓ décroît, le maximiseur parcourt
de gauche à droite l'**enveloppe convexe inférieure** du nuage {(κ(L), a(L))}. Le calendrier
optimal visite exactement les profondeurs de l'enveloppe, en commutant aux pertes de tangence.
Corollaire : le saut unique L_s→L_f est optimal si et seulement si aucun point intermédiaire ne
passe sous la corde, c'est-à-dire si a est *concave* en κ — condition réelle et non triviale,
fausse par exemple pour a_L = a_∞ + c/L^α (qui rend a convexe en κ).

**Théorème C (seuil unifié).** Pour deux profondeurs L_s < L_f, une fonction de coût c(·), une
perte de départ commune ℓ0, on pose

```
G = (ℓ0 − a_{L_f}) / (ℓ0 − a_{L_s})  ≥ 1        (facteur d'écart de capacité)
ρ = κ_c(L_f) / κ_c(L_s),   κ_c(L) = (c(L)^p·b_L)^(1/(p+1))   (facteur d'échelle de coût)
```

Alors « rester à L_s puis sauter à L_f » bat le fixe-L_f à tout budget si et seulement si G < ρ ;
le fixe gagne si G ≥ ρ. À pas égaux (c ≡ 1), ρ = (b_f/b_s)^(1/2) pour p=1 : proche de 1 puisque
b varie peu. À FLOPs égaux (c(L) = L), ρ gagne un facteur (L_f/L_s)^(p/(p+1)) — **sous-linéaire**
dans le vrai rapport de coût (un rapport 16 ne contribue que √16 = 4 pour p=1), mais suffisant
pour franchir le seuil. C'est le mécanisme quantifié : le coût entre dans le seuil de façon
sous-linéaire, l'écart de capacité de façon linéaire.

## 5. L'ajustement a(L), b(L) sur les 16 profondeurs

### 5.1 Données et modèle de base

Sources, toutes depuis zéro (poids aléatoires, aucune greffe), configuration identique (le
script `growth_theory_missing_depths.jl` inclut `growth_jalon0.jl`, donc mêmes corpus, mêmes
hyperparamètres, même protocole de validation) :

| L | points | n_max (pas) | sources |
|---|--------|-------------|---------|
| 1 | 845 | 45 000 | segments pré-greffe des bras C (jalon 0, 3 seeds) et stages2/4/8/16 (jalon 1, 1-fixé, 1b, 2 seeds chacun) |
| 2 | 98 | 13 000 | `depth2` (nouveau, 4 000 pas) + segments pré-greffe B_2_3_4 (jalon 0, 3 seeds, 13 333 pas) |
| 3, 5..15 | 20 chacun | 4 000 | `growth_theory_missing_depths_results.json` (nouveau, seed unique) |
| 4 | 120 | 20 000 | bras A_fixed_4 (jalon 0, 3 seeds) |
| 16 | 88 | 8 000 | `depth16_extended` (nouveau, 8 000 pas) + stages1 de jalon 1/1-fixé (3 000 pas × 4) et jalon 1b (4 000 pas × 2) |

**Choix de n0 et p, re-testés.** À p=1, le n0 optimal global (balayage fin, SSE totale des 16
fits) est 3 270, contre 3 000 pour la valeur historique — gain de SSE de 1,05 % seulement
(3,2114 contre 3,2456) : **n0 = 3000 est conservé**. Sur la fenêtre complète, des exposants
p > 1 réduisent la SSE (2,50 à p=3), mais n0* dérive alors sans optimum net (3 250 → 10 750) :
cette « amélioration » absorbe le plateau de warmup (les 800–1 200 premiers pas, où toutes les
profondeurs stagnent vers 2,53–2,67, forme que l'hyperbole ne décrit pas). Restreinte à la
queue (n ≥ 1 500), la SSE préfère nettement **p = 1** (0,978 contre 1,012 à p=1,5 et 1,070 à
p=2). Le modèle p=1, n0=3000 est donc confirmé, avec un domaine de validité explicite :
n ≳ 200 pas.

### 5.2 Ajustements par profondeur et diagnostics d'honnêteté

Fits linéaires en (a, b) à n0=3000 (`perte = a + b·x`, x = 1/(n+3000)) :

| L | a ± SE | b ± SE | R² | points | n_max |
|---|--------|--------|-----|--------|-------|
| 1 | 1,5715 ± 0,0024 | 3 691 ± 21 | 0,974 | 845 | 45 000 |
| 2 | 1,4485 ± 0,0151 | 4 274 ± 97 | 0,953 | 98 | 13 000 |
| 3 | 1,9756 ± 0,0777 | 2 266 ± 365 | 0,682 | 20 | 4 000 |
| 4 | 1,4390 ± 0,0094 | 4 428 ± 82 | 0,961 | 120 | 20 000 |
| 5 | 1,5429 ± 0,0687 | 3 821 ± 322 | 0,887 | 20 | 4 000 |
| 6 | 1,5327 ± 0,0758 | 3 903 ± 356 | 0,870 | 20 | 4 000 |
| 7 | 1,4388 ± 0,0695 | 4 207 ± 326 | 0,902 | 20 | 4 000 |
| 8 | 1,4422 ± 0,0651 | 4 189 ± 305 | 0,913 | 20 | 4 000 |
| 9 | 1,4251 ± 0,0581 | 4 234 ± 273 | 0,931 | 20 | 4 000 |
| 10 | 1,4145 ± 0,0636 | 4 306 ± 298 | 0,920 | 20 | 4 000 |
| 11 | 1,3723 ± 0,0672 | 4 444 ± 315 | 0,917 | 20 | 4 000 |
| 12 | 1,4007 ± 0,0599 | 4 357 ± 281 | 0,930 | 20 | 4 000 |
| 13 | 1,3562 ± 0,0596 | 4 520 ± 280 | 0,936 | 20 | 4 000 |
| 14 | 1,3372 ± 0,0577 | 4 609 ± 271 | 0,942 | 20 | 4 000 |
| 15 | 1,3042 ± 0,0572 | 4 702 ± 269 | 0,945 | 20 | 4 000 |
| 16 | 1,2803 ± 0,0199 | 4 876 ± 99 | 0,966 | 88 | 8 000 |

Trois diagnostics, signalés avant toute sélection de forme :

**(i) Cohérence L=2, ancien contre nouveau.** L'ancienne source (segments pré-greffe B_2_3_4 de
jalon 0) est en fait la plus *longue* : 13 333 pas et 3 seeds, contre 4 000 pas et 1 seed pour le
nouveau run. Séparément : nouveau a = 1,549 ± 0,072, b = 3 735 ± 337 (R² = 0,872) ; ancien
a = 1,431 ± 0,016, b = 4 435 ± 113 (R² = 0,953). L'écart sur a est d'environ 1,6 σ du fit le
moins précis : compatible, pas de contradiction — mais c'est l'ancien, mieux contraint, qui est
le plus fiable. Les deux sources sont poolées (table ci-dessus).

**(ii) Biais de troncature à 4 000 pas — le point méthodologique central.** Sur les profondeurs
disposant de runs longs, refaire le fit en ne gardant que n ≤ 4 000 déplace a de
+0,055 (L=1), +0,092 (L=2), +0,114 (L=4) et +0,018 (L=16, comparaison 8 000 contre 4 000) : un
fit tronqué à 4 000 pas **surestime systématiquement le plancher a et sous-estime b**. Or les
profondeurs 3 et 5 à 15 n'ont *que* 4 000 pas. Les résidus le confirment : ces profondeurs sont
systématiquement au-dessus de la tendance ancrée par L = 1, 2, 4, 16. Le biais est donc modélisé
explicitement (§5.3) par un terme additif δ sur les profondeurs « 4 000 pas seulement », et son
estimation par le fit global (δa = +0,074) recoupe la mesure directe ci-dessus — cohérence
interne qui valide l'approche. (Exclure le warmup au lieu de modéliser le biais a été essayé et
rejeté : sans les premiers points, l'extrapolation devient non contrainte et le biais change de
signe en explosant, jusqu'à −0,29 sur a.)

**(iii) L'outlier L=3.** Le run depth3 stagne vers 2,43–2,51 jusqu'au pas 3 000 puis chute à
2,11 au pas 4 000 — un échappement de plateau tardif, à seed unique. Son fit (a = 1,976,
b = 2 266, R² = 0,682) est à ~5,7 σ de la tendance de ses voisins (L=2 : 2,28 au pas 2 000 ;
L=5 : 2,36 ; L=3 : 2,49). **L=3 est exclu des ajustements de forme fonctionnelle** et signalé
comme tel ; un re-run multi-seeds de cette profondeur est la première chose à faire pour
consolider ce point.

### 5.3 Sélection de forme fonctionnelle

Moindres carrés pondérés (poids 1/SE²) sur les 15 profondeurs (L=3 exclu), avec pour chaque
forme une variante incluant le dummy de troncature δ (valant 1 pour L ∈ {3, 5..15}, 0 sinon).
AIC = χ² + 2k, BIC = χ² + k·ln(15).

Pour a(L) :

| forme | k | χ² | AIC | BIC |
|-------|---|-----|-----|-----|
| a0 + a1·ln L | 2 | 27,7 | 31,7 | 33,2 |
| **a0 + a1·ln L + δa** | **3** | **16,5** | **22,5** | **24,6** |
| a∞ + c/L^q + δa | 4 | 16,5 | 24,5 | 27,3 |
| a∞ + c·e^(−L/τ) + δa | 4 | 23,4 | 31,4 | 34,2 |
| a∞ + c/L + δa | 3 | 40,9 | 46,9 | 49,0 |

La loi de puissance ne bat jamais le log : son exposant dégénère (q → 0,001 avec dummy, 0,12
sans), or L^q ≈ 1 + q·ln L pour q petit — elle *devient* le log en dépensant un paramètre de
plus. La saturation exponentielle et la forme en 1/L sont nettement rejetées : le plancher ne
sature pas sur la plage L ≤ 16, il décroît en ln L sans signe d'asymptote.

Pour b(L) :

| forme | k | χ² | AIC | BIC |
|-------|---|-----|-----|-----|
| b0 (constante) | 1 | 270,0 | 272,0 | 272,7 |
| b0 + b1·ln L | 2 | 26,7 | 30,7 | 32,1 |
| **b0 + b1·ln L + δb** | **3** | **10,7** | **16,7** | **18,8** |
| b0 + b1·L + δb | 3 | 57,4 | 63,4 | 65,5 |
| b∞ − c/L^q + δb | 4 | 5,5 | 13,5 | 16,3 |

La constante est exclue (ΔAIC > 250) : **b(L) est réellement croissante en L**, confirmant sur
16 profondeurs le constat contre-intuitif du premier ajustement (les réseaux profonds ont un
plancher plus bas ET un démarrage marginalement plus lent). La forme saturante bat le log de
ΔAIC = 3,2, mais le bootstrap la révèle non identifiée : q a pour IC90 [0,02 ; 1,35] et b∞ pour
IC68 [4 915 ; 7 922] — les données ne contraignent pas ces paramètres, et ses prédictions aval
sont indiscernables du log (R² de prédiction 0,773 contre 0,764, enveloppe convexe identique).
Elle est signalée comme statistiquement compétitive mais non retenue.

**Formule finale retenue** (bootstrap non paramétrique, 400 rééchantillonnages des points de
chaque profondeur puis re-fit complet ; IC68 entre crochets) :

```
perte(n, L) = a(L) + b(L)/(n + 3000)                      (nats/caractère, n ≥ ~200 pas)

a(L) = 1,5710 − 0,1042·ln L      a0 ∈ [1,5689 ; 1,5734],  a1 ∈ [−0,1101 ; −0,0989]
b(L) = 3701,6 + 471,0·ln L       b0 ∈ [3668 ; 3732],      b1 ∈ [429 ; 518]

δa = +0,0738 ∈ [0,038 ; 0,088]   (biais des runs 4 000 pas, cohérent avec la mesure directe)
δb = −449 ∈ [−524 ; −232]
```

χ²/dof : 1,37 pour a (léger excès de dispersion, cohérent avec le bruit de seed résiduel des
runs à seed unique), 0,89 pour b. Avec L=3 inclus, le log reste la meilleure forme mais le
χ²/dof est dégradé d'un facteur ~4 — la décision d'exclusion ne change pas la forme retenue,
seulement la qualité déclarée.

### 5.4 La forme log(L) est-elle dérivable de premiers principes, avec des constantes exactes ?

Question posée explicitement après la première version de ce document : `a0=1,571` et
`a1=-0,104` sont des coefficients de régression, pas des constantes dérivées. Deux agents Fable,
lancés indépendamment sur deux angles distincts (théorie de l'approximation/expressivité des
réseaux profonds ; théorie statistique de l'apprentissage et lois d'échelle), ont cherché une
dérivation rigoureuse. Verdict honnête, recoupé par les deux agents et revérifié
indépendamment ci-dessous : **non, aucune dérivation à constantes exactes n'existe** — mais la
forme fonctionnelle elle-même est solidement rattachée à un résultat publié et vérifiable, pas
choisie arbitrairement.

**Le pont architectural, vérifié dans le code.** Un `LlamaBlock` de ce projet (`src/layers.jl`,
dim=256, 4 têtes, hidden=512) compte exactement `4·dim² + 3·dim·hidden + 2·dim = 655 872`
paramètres, **indépendamment de L** — confirmé par calcul direct sur la config réelle
(`growth_jalon0.jl`). Le nombre total de paramètres (hors embeddings) est donc **exactement
affine en L** : `N(L) = 655 872·L + O(1)`. Toute loi en `ln N` est donc, à une constante additive
près, une loi en `ln L` — c'est le pont qui permet de relier `a(L)` aux lois d'échelle en
nombre de paramètres de la littérature, mesurées sur des modèles bien plus gros.

**Le test numérique.** Kaplan et al. 2020 (*Scaling Laws for Neural Language Models*,
arXiv:2001.08361) mesurent, sur des modèles de langage à grande échelle, une loi de puissance
`perte ∝ N^(-α_N)` avec **α_N ≈ 0,076** (vérifié par les deux agents, l'un via citation directe,
l'autre en confirmant la référence en ligne pendant la session). Un ajustement en loi de
puissance PURE (sans offset), refait indépendamment trois fois sur nos propres données
(profondeurs corrigées du biais de troncature) :

| Source | α ajusté |
|---|---|
| Agent 1 (approche indirecte, via exposant local) | ≈ 0,077 |
| Agent 2 (régression WLS directe, ln a vs ln L) | 0,0710 ± 0,0032 |
| Vérification indépendante (ci-dessous, régression simple) | 0,0792 |

**Les trois tombent dans la fourchette 0,071–0,079, encadrant l'exposant publié de 0,076.**
Ce n'est pas une coïncidence isolée : c'est une convergence à trois méthodes de calcul
indépendantes sur le même exposant, dans un intervalle étroit contenant la valeur publiée. La
forme log de nos données est donc l'apparence locale, sur une plage `L ∈ [1,16]` où
`α·ln(16) ≈ 0,2 ≪ 1`, d'une loi de puissance en nombre de paramètres numériquement cohérente
avec la littérature déjà publiée — pas une découverte ad hoc.

**Le seul mécanisme connu donnant du log EXACT (pas seulement approché).** Michaud, Liu, Girit
& Tegmark 2023 (*The Quantization Model of Neural Scaling*, NeurIPS 2023, arXiv:2303.13506)
modélisent l'apprentissage comme l'acquisition séquentielle de « quanta » de compétence, appris
par ordre de fréquence décroissante suivant une loi de Zipf. Sous l'hypothèse spécifique d'un
exposant de Zipf exactement égal à 1 (la valeur canonique pour les textes en langue naturelle,
Shakespeare inclus) — un calcul élémentaire de somme harmonique donne alors
`perte(N) − perte(∞) ∝ ln(K_max/K(N))`, c'est-à-dire **exactement** la forme log observée, sans
approximation. Ce n'est pas un théorème sur ce modèle précis (les hypothèses — quanta discrets à
coût de capacité égal, Zipf exactement 1, allocation gloutonne optimale par SGD — sont des
idéalisations fortes, non prouvées ici) : c'est un mécanisme publié, plausible, qui explique
*pourquoi* un log serait attendu plutôt qu'une coïncidence numérique.

**Ce qui n'est PAS acquis, dit explicitement :**
1. **Aucune valeur exacte de `a0` ou `a1` n'est dérivable.** Le produit `α·(a−a∞)` est contraint
   par les données, mais `α` (dépendant du spectre de la distribution des données, jamais dérivé
   de premiers principes pour le langage naturel) et `a∞` (le plancher irréductible, inconnu pour
   ce corpus) restent chacun non identifiés séparément — nos données sont aussi bien compatibles
   avec la paramétrisation de Kaplan (`a∞≈0`, α≈0,076) qu'avec celle de Hoffmann et al. 2022
   (Chinchilla : `a∞≈1,05` nat/caractère ≈ 1,5 bit/caractère, plausible comme entropie résiduelle
   d'un texte anglais, α≈0,34) — les deux passent par nos points à moins de 0,03 nat près, sous
   le bruit de nos mesures à seed unique.
2. **La monotonie (`a1<0`) a un fondement solide dans CE projet spécifiquement** : `insert_block!`
   (propriété F1, identité exacte au bit près déjà prouvée) garantit que la classe de fonctions
   représentables à profondeur L est strictement emboîtée dans celle à profondeur L+1 — le
   plancher représentationnel est donc nécessairement non croissant. C'est un argument propre à
   NeuroDSL (peu de frameworks peuvent le vérifier aussi directement), mais il ne borne
   qu'heuristiquement le plancher *atteint par SGD*, qui est ce que `a(L)` mesure réellement.
3. **Expériences discriminantes proposées par les deux agents**, pas encore menées : un balayage
   en largeur à profondeur fixe (pour vérifier que le plancher suit vraiment N et pas L en
   particulier) ; une extension à L=32/64 (pour chercher le coude de saturation que prédit
   Chinchilla mais pas Kaplan) ; une réduction du corpus d'entraînement (pour vérifier la
   dépendance en taille de données D prédite par les deux lois). Aucune n'a été exécutée dans
   cette passe — proposées comme suite possible, pas comme travail restant à faire
   immédiatement.

**Conclusion de cette sous-section** : la forme log(L) n'est pas un theorem, mais elle n'est pas
non plus un simple choix statistique arbitraire — elle est l'instance, numériquement vérifiée à
trois reprises indépendamment, d'une loi d'échelle en paramètres déjà publiée et largement
répliquée dans la littérature des grands modèles de langage, avec un mécanisme génératif
plausible (quantification zipfienne) qui prédirait exactement cette forme sous des hypothèses
explicites. C'est le niveau de rigueur honnêtement atteignable ici : la forme est justifiée,
les constantes exactes ne le sont pas — et c'est également le cas, sans exception connue, pour
toutes les lois d'échelle publiées à ce jour dans la littérature.

### 5.5 Dérivation rigoureuse par équation fonctionnelle : forme exacte, développement asymptotique, et une contrainte falsifiable

Le §5.4 établit que la forme log est *cohérente* avec la littérature, sans la *dériver*. Cette
sous-section pousse plus loin : on pose le modèle de quantification de Michaud et al. (déjà cité
en §5.4) comme une véritable **équation fonctionnelle** — une récurrence en L — et on la résout
exactement, puis on en tire un développement asymptotique rigoureux (pas une régression) avec une
prédiction croisée testable, vérifiée indépendamment.

**Position du modèle.** Un réseau de profondeur L a appris les `K(L) = ρ·L` premiers « quanta »
de compétence (classés par fréquence décroissante, capacité ∝ nombre de paramètres ∝ L — pont
déjà établi en §5.4), avec des poids de Zipf `p_k = c/k` (exposant exactement 1). La perte
résiduelle est `a(L) = a_∞ + Σ_{k=K(L)+1}^{K_max} p_k`. Le gain marginal d'une couche de plus
satisfait la récurrence linéaire du premier ordre

```
Δa(L) := a(L) − a(L+1) = c·[H_{ρ(L+1)} − H_{ρL}]        (H_n = Σ_{k=1}^n 1/k)
```

qui se résout par télescopage **exact**, sans aucune approximation, de `L` jusqu'à
`L_max = K_max/ρ` :

```
a(L) = a_∞ + c·[H_{K_max} − H_{ρL}]
```

**Forme fermée exacte, via la fonction digamma.** L'identité classique `H_n = ψ(n+1) + γ`
(ψ = fonction digamma, γ = constante d'Euler-Mascheroni) donne, pour L traité comme variable
continue (exact aux entiers) :

```
a(L) = a_∞ + c·[ψ(K_max+1) − ψ(ρL+1)]                    (EXACT)
```

C'est la forme close demandée : toute la richesse du modèle tient dans trois constantes
structurelles (`a_∞, c, ρ`) plus la borne `K_max`, pas dans un choix de forme fonctionnelle
arbitraire.

**Développement asymptotique rigoureux (Euler-Maclaurin), au-delà du terme dominant.** Le
développement classique et exact au sens asymptotique de Poincaré (reste borné par le premier
terme omis ; NIST DLMF §5.11.2, Abramowitz & Stegun 6.3.18) est

```
ψ(x) = ln x − 1/(2x) − 1/(12x²) + 1/(120x⁴) − …
```

Appliqué à `x = ρL+1`, ceci donne le développement complet de `a(L)` :

```
a(L) = A + a1·ln L + a2/L + a3/L² + O(L⁻⁴)
a1 = −c,    a2 = −c/(2ρ),    a3 = c/(12ρ²)
```

Le terme `a1·ln L` retrouve exactement la forme ajustée en §5.3 ; `a2/L` et `a3/L²` sont les
corrections d'ordre suivant, dérivées et non postulées.

**La contrainte croisée — la prédiction falsifiable centrale.** `a1, a2, a3` dépendent des deux
mêmes inconnues `(c, ρ)` ; les éliminer donne une relation exacte, sans paramètre libre :

```
a1·a3 / a2²  =  −1/3
```

**Vérifiée indépendamment** (implémentation manuelle de la digamma par récurrence + série
asymptotique de Stirling, sans dépendance ajoutée) : en générant `a(L)` à partir de la formule
digamma exacte pour plusieurs `ρ` et en ré-ajustant `{1, ln L, 1/L, 1/L²}`, le rapport
`a1·a3/a2²` converge vers **exactement −1/3** quand `ρL ≫ 1` (−0,2916 à ρ=1, −0,3322 à ρ=8,
−0,3333 à ρ=50 et 500) — confirmant la dérivation avant tout test contre les données réelles.

**Résolution de la tension log vs loi de puissance (§5.4) — pas numérique, structurelle.** En
généralisant à un exposant de Zipf `1+ε` (au lieu d'exactement 1), la somme tronquée devient une
fonction zêta de Hurwitz, `a(L) = a_∞ + c·[ζ(1+ε,ρL+1) − ζ(1+ε,K_max+1)]` — exacte, bien définie
sans coupure `K_max` dès que `ε>0` (contrairement au cas `ε=0`, où la coupure est obligatoire :
le cas « log pur » est le cas frontière singulier de cette famille). Le développement de Hurwitz
pour `q` grand (DLMF 25.11.43) donne au premier ordre `a(L) − A' ∝ (ρL)^{−ε}` — une loi de
puissance d'exposant `ε`. En développant `[(ρL)^{-ε}-1]/ε` en série de Taylor en `ε` à `L` fixé,
le terme d'ordre 0 est exactement `−ln L` : **le log et la loi de puissance ne sont pas deux
hypothèses concurrentes, mais la même fonction évaluée de deux façons différentes** (développer
en `ε` à `L` fixé, ou garder `ε` fixe sans développer). Elles coïncident tant que `ε·ln L ≪ 1` ;
avec `ε≈0,076` mesuré et `L≤16`, `ε·ln16≈0,21`, écart relatif attendu `O(ε²ln²L)≈4 %` — sous le
bruit de mesure actuel, mais pas rigoureusement nul. C'est l'explication *structurelle* (pas
seulement numérique) de la dégénérescence déjà observée en §5.3 (l'exposant de loi de puissance
`q→0` quand on ajoute le dummy de troncature).

**Test de la contrainte contre les données réelles — résultat honnête, non concluant.** Un
ajustement WLS étendu `a0+a1·ln L+a2/L+a3/L²` (+ dummy de troncature) sur les 15 profondeurs
(L=3 exclu) donne un ΔAIC apparemment favorable (−10,1) au modèle à 5 paramètres. **Quatre
vérifications de robustesse (χ²/dof, diagnostic résiduel, leave-one-out, bootstrap en grappes
rééchantillonnant aussi les profondeurs, pas seulement les points intra-profondeur) montrent que
ce gain est un artefact porté entièrement par un seul point à fort effet de levier (L=2, dont le
pooling de deux sources à ~1,6σ d'écart est déjà signalé en §5.2-i) : retirer ce seul point fait
perdre toute significativité (test emboîté p=0,26 sans L=2, contre p=0,0009 avec) ; le bootstrap
en grappes ne sélectionne le modèle étendu que dans 50 % des tirages, et son coefficient `a1`
devient instable au point de changer de signe.** Les coefficients nominaux
(`a1=−0,2585, a2=−1,2639, a3=+0,7686`) donnent `a1·a3/a2² ≈ −0,124` — loin de la prédiction
`−1/3` — mais ce chiffre n'est **pas interprétable** : il provient de coefficients déjà démontrés
non robustes, pas d'un désaccord théorie-données significatif. **Verdict honnête : la théorie
livre une prédiction exacte et testable ; les données actuelles (majoritairement à seed unique,
4000 pas, aux profondeurs intermédiaires) ne permettent pas encore de la tester proprement.** Un
test propre demanderait soit des runs multi-graines aux profondeurs 3–15, soit une extension à
L=32/64 pour entrer plus profondément dans le régime asymptotique `ρL≫1` où la contrainte est la
plus nette (et où le développement en `ε` de la sous-section précédente prédit aussi que log et
loi de puissance recommenceraient à diverger numériquement — un second test discriminant,
complémentaire du premier).

**Tentative sur `b(L)` — négatif, dit explicitement.** Un argument qualitatif (temps de
saturation des quanta croissant avec la profondeur, donc un ajustement hyperbolique sur une
fenêtre commune force artificiellement un `b(L)` croissant) est plausible mais non rigoureux : il
prédirait une dépendance en `n` logarithmique dans le régime pré-saturation, pas hyperbolique en
`1/n` comme le modèle de base (§3, confirmé en §5.1) — aucune dérivation ne réconcilie ces deux
formes sans hypothèse supplémentaire non testée. Aucune formule fermée pour `b(L)` n'est donc
disponible au même niveau de rigueur que pour `a(L)`.

**Conclusion de cette sous-section.** Le niveau de rigueur mathématique demandé est maintenant
atteint pour ce qui est atteignable : une forme fermée exacte (digamma), un développement
asymptotique classique avec reste contrôlé (Euler-Maclaurin), une prédiction croisée sans
paramètre libre (`a1·a3/a2²=−1/3`), et une résolution structurelle (pas numérique) de la tension
log/puissance via la fonction zêta de Hurwitz. Ce qui reste ouvert n'est plus une faiblesse de la
dérivation mais une limite des données actuelles, précisément caractérisée et avec un protocole
de test explicite pour la lever.

## 6. Re-vérification des théorèmes avec le modèle complet

### 6.1 Théorème A (pas égaux)

Avec les valeurs lisses, la perte de croisement des vitesses entre L=1 et L=16 est
ℓ× = (ρ·a1 − a16)/(ρ − 1) = 3,342 avec ρ = √(b16/b1) = 1,163. Or les courbes canoniques
*commencent* (valeur à n=0) à 2,805 (L=1) et 2,951 (L=16) : ℓ× est au-dessus de tout le domaine
de validité du modèle. **À pas égaux, φ_16 ≥ φ_L sur tout le trajet : le fixe-16 domine tout
calendrier dès le pas 0**, ce qui prédit exactement le résultat de jalon 1b (les cinq bras de
croissance perdent tous). L'ancien ajustement sur 4 points plaçait le croisement vers
1 600–2 000 pas dans la fenêtre de 4 000 ; les données complètes durcissent la conclusion.

### 6.2 Théorème B (FLOPs égaux) — le verdict qualitatif change

κ(L) = √(L·b(L)) va de 60,8 (L=1) à 89,8 (L=2) jusqu'à 283,1 (L=16). Enveloppe convexe
inférieure du nuage {(κ(L), a(L))} :

- valeurs empiriques brutes : enveloppe = {1, 2, 16} ;
- valeurs corrigées du biais de troncature : enveloppe = {1, 2, 15, 16} ;
- modèle lisse : a(κ) a une dérivée seconde discrète strictement positive sur tout [κ(1), κ(16)]
  — a est **convexe** en κ, donc **les 16 profondeurs sont toutes sur l'enveloppe**.

C'est un renversement qualitatif par rapport à l'ajustement sur 4 points, qui suggérait le saut
unique comme optimum (a concave en κ). Avec la forme log, a ≈ cst − 2·|a1|·ln κ est
mécaniquement convexe en κ : **le calendrier optimal théorique à FLOPs égaux est graduel, pas un
saut unique**. Quantitativement, au budget de jalon 1 (48 000 layer-steps), l'optimisation
numérique du calendrier sous le modèle donne :

| calendrier (optimisé sous le modèle) | perte finale prédite |
|---|---|
| optimum libre : L=1 (44,5 % du budget) → 2 (25,8 %) → 3 (29,7 %), arrêt à 3 | **1,626** |
| meilleur saut unique 1→16 (91,2 % / 8,8 %) | 1,643 |
| meilleur calendrier mesuré (jalon 1-fixé, 1→16) | 1,654 (mesuré) |
| fixe-16 | 2,117 (prédit) / 2,072 (mesuré) |

Deux lectures honnêtes. D'abord, l'essentiel du gain de gradualité vient des profondeurs 2 et 3
seulement (passer de la chaîne [1,16] à [1,2,16] gagne 0,016 ; ajouter 4..16 ne gagne plus
rien) : la « gradualité utile » est concentrée en bas de l'échelle, où κ varie vite. Ensuite, le
gain prédit du calendrier optimal sur le saut unique (~0,017 nat) est *inférieur* au RMSE de
prédiction du modèle (0,085, §6.4) : c'est une **prédiction testable**, pas une conclusion — et
elle est cohérente avec le fait observé que le saut unique a gagné parmi les calendriers
*effectivement testés*, aucun ne réalisant les points de commutation optimaux. La prédiction la
plus falsifiable est qu'à ce budget, un calendrier s'arrêtant à la profondeur 3 battrait tous
les calendriers finissant à 16 — aucun bras mesuré ne couvre ce cas.

### 6.3 Théorème C (seuil G contre ρ)

Pour L_s=1, L_f=16, valeurs lisses : ρ_pas = √(b16/b1) = **1,163** ;
ρ_FLOPs = √(16·b16/b1) = **4,652**. Le facteur G dépend de la perte de départ ℓ0 :

| convention ℓ0 | G | pas égaux (ρ=1,163) | FLOPs égaux (ρ=4,652) |
|---|---|---|---|
| ℓ0 = départ modèle, a1+b1/n0 = 2,805 | 1,234 | G ≥ ρ → **fixe gagne** ✓ | G < ρ → **croissance gagne** ✓ |
| ℓ0 = ln(65) = 4,174 (perte à l'init) | 1,111 | G < ρ → croissance ✗ | G < ρ → croissance ✓ |

À FLOPs égaux, le verdict « croissance » est robuste à la convention (marge ~4× sur le seuil) et
conforme aux jalons 0 et 1. À pas égaux, le cas est *au voisinage du seuil* et la convention
tranche : la convention cohérente avec le modèle (ℓ0 = valeur de la courbe canonique à n=0,
puisque le modèle ne décrit pas la phase d'init ln V → ~2,6 qui s'effectue en ~200 pas de façon
quasi indépendante de la profondeur) donne le verdict empiriquement correct. L'ancien ajustement
sur 4 points donnait G ≈ 1,086 < ρ ≈ 1,108, c'est-à-dire le *mauvais* côté du seuil à pas égaux ;
les données complètes déplacent le point (G = 1,234 contre ρ = 1,163) du bon côté. La proximité
du seuil n'est pas un artefact : simulé finement, le modèle chaîné littéral prédit même qu'une
très courte phase peu profonde suivie d'un saut gagnerait ~0,01–0,03 nat à pas égaux — avantage
fantôme issu de la zone warmup (où le modèle est mal spécifié) et effacé en pratique par le coût
transitoire des greffes (§6.4). Le régime à pas égaux est donc déclaré « au seuil, tranché
empiriquement », pas « prédit avec marge ».

### 6.4 Validation prédictive : les 18 calendriers

Chaque calendrier mesuré est prédit en chaînant les courbes canoniques lisses par la règle du
pas effectif, avec les bornes de palier exactes lues dans les `growth_events` des JSON — aucune
resimulation, aucun paramètre ajusté sur ces données. Mesuré = moyenne des seeds.

| bras | calendrier | mesuré (± std) | prédit | écart |
|---|---|---|---|---|
| j0 : A_fixed_4 | [4], 20 000 pas | 1,6535 ± 0,0267 | 1,6159 | −0,038 |
| j0 : B_2_3_4 | [2,3,4] | 1,6032 ± 0,0107 | 1,5852 | −0,018 |
| j0 : C_1_2_4 | [1,2,4] | 1,6366 ± 0,0025 | 1,5746 | −0,062 |
| j1 : stages1 | [16], 3 000 pas | 2,0562 ± 0,0059 | 2,1167 | +0,061 |
| j1 : stages2 | [1,16] | 1,7191 ± 0,0035 | 1,6599 | −0,059 |
| j1 : stages4 | [1,6,11,16] | 1,7838 ± 0,0082 | 1,6849 | −0,099 |
| j1 : stages8 | [1,3,5,7,10,12,14,16] | 1,8151 ± 0,0352 | 1,7125 | −0,103 |
| j1 : stages16 | [1..16] | 1,8053 ± 0,0543 | 1,7366 | −0,069 |
| j1f : stages1 | [16], 3 000 pas | 2,0723 ± 0,0049 | 2,1167 | +0,044 |
| j1f : stages2 | [1,16] | 1,6542 ± 0,0139 | 1,6432 | −0,011 |
| j1f : stages4 | [1,6,11,16] | 1,7520 ± 0,0003 | 1,6384 | −0,114 |
| j1f : stages8 | [1,3,5,7,10,12,14,16] | 1,8082 ± 0,0223 | 1,6356 | −0,173 |
| j1f : stages16 | [1..16] | 1,7593 ± 0,0213 | 1,6423 | −0,117 |
| j1b : stages1 | [16], 4 000 pas | 1,9570 ± 0,0086 | 1,9975 | +0,040 |
| j1b : stages2 | [1,16] | 2,0619 ± 0,0541 | 2,0115 | −0,050 |
| j1b : stages4 | [1,6,11,16] | 2,1195 ± 0,0278 | 1,9996 | −0,120 |
| j1b : stages8 | [1,3,5,7,10,12,14,16] | 2,1169 ± 0,0444 | 1,9971 | −0,120 |
| j1b : stages16 | [1..16] | 2,0233 ± 0,0311 | 1,9962 | −0,027 |

**R² = 0,764, RMSE = 0,0849 nat** (ancien modèle 4 points : R² = 0,75, RMSE = 0,087). Le modèle
reproduit correctement les deux renversements de régime (croissance gagne à FLOPs égaux dans
jalon 0/1, fixe gagne à pas égaux dans jalon 1b) et l'ordre grossier de chaque campagne.
L'amélioration globale est **marginale** — l'apport des 12 nouvelles profondeurs n'est pas une
meilleure précision de prédiction finale, mais la levée des ambiguïtés structurelles (forme de
a et b, convexité de l'enveloppe, côté du seuil G/ρ). Trois faits complémentaires :

- Le lissage est indispensable : prédire avec les 16 paires (a, b) brutes par profondeur donne
  R² = 0,42 (le bruit de fit domine) ; corrigées du biais, R² = 0,71 ; formes lisses, 0,76.
  La variante saturante pour b donne 0,77 — indiscernable.
- Validation trajectoire (points de perte individuels, pas seulement les finaux) : sur les 437
  points strictement postérieurs à une première greffe, R² = 0,918, RMSE = 0,081 ; sur les
  1 536 points de toutes les trajectoires, R² = 0,956, RMSE = 0,059. (La note précédente
  annonçait 0,981 sur les mêmes 437 points avec l'ancien modèle ; ce chiffre n'a pas pu être
  reproduit à protocole identique et le protocole exact d'origine n'est pas archivé — les
  valeurs ci-dessus font foi désormais.)
- **Les résidus sont structurés, et c'est informatif** : les 4 bras sans greffe ont un résidu
  moyen de −0,027 (le modèle est légèrement pessimiste, défaut de forme de l'hyperbole en milieu
  de courbe), tandis que les 14 bras greffés sous-performent la prédiction de +0,04 à +0,17,
  avec une corrélation de 0,43 entre résidu et nombre de greffes (+0,040 pour 1 greffe, +0,111
  pour 3, +0,132 pour 7 ; pente OLS ≈ +0,005 nat/greffe). La règle du pas effectif, exacte à
  l'instant de la greffe, est donc légèrement *optimiste* ensuite : il existe un coût transitoire
  de greffe réel non modélisé. C'est la principale hypothèse falsifiable dégagée par cette
  validation.

## 7. Limites honnêtes

1. **L'anomalie « 16 paliers bat 8 paliers » reste inexpliquée à FLOPs égaux.** Le modèle prédit
   le bon signe uniquement à pas égaux (jalon 1b : 1,9962 contre 1,9971, contre 2,0233/2,1169
   mesurés — signe correct, amplitude non). À FLOPs égaux, il prédit des quasi-égalités entre
   stages2/4/8/16 sous pondération 1/L (plage prédite 1,6356–1,6432) là où les mesures s'étalent
   sur 0,15 ; les écarts mesurés sont par ailleurs du même ordre que le bruit de seed (std
   jusqu'à 0,054 avec 2 seeds). Non résolu, non forcé.
2. **Le coût transitoire de greffe n'est pas modélisé** (résidus du §6.4). Un terme de pénalité
   par greffe améliorerait probablement la prédiction, mais l'ajuster sur les 18 mêmes points
   qu'il sert à prédire serait circulaire — il faudrait des runs dédiés (mesurer la trajectoire
   post-greffe contre la courbe canonique de même perte, à greffe unique et budgets longs).
3. **Le régime à pas égaux est au voisinage du seuil de Théorème C** : le verdict dépend de la
   convention ℓ0 et de détails de la zone warmup où le modèle est mal spécifié. La conclusion
   « fixe gagne à pas égaux » est établie *empiriquement* (jalon 1b) et *compatible* avec le
   modèle, mais pas prédite avec marge.
4. **a(L) et b(L) des profondeurs 3, 5–15 reposent sur un seul seed et 4 000 pas**, avec un
   biais de troncature corrigé par un paramètre global (δa, δb) mesuré sur 4 profondeurs. Le run
   L=3 est un outlier exclu (5,7 σ) — vraisemblablement un échappement de plateau tardif, mais
   un re-run multi-seeds est nécessaire pour l'affirmer.
5. **La forme log de a(L) ne peut pas être extrapolée loin au-delà de L=16** (elle finirait
   négative vers L ≈ 3,5·10⁶, et aucune saturation n'est encore visible sur 1..16) ; de même le
   verdict « toutes les profondeurs sur l'enveloppe » n'est démontré que sur la plage mesurée.
6. Tout ceci vaut à une seule échelle (char-LM ~5–20 M paramètres, batch 1, une tâche). La
   généralisation à d'autres échelles est exactement ce que le jalon 2 (85 M) devait tester.

## 8. Conclusion

Le mécanisme est clos : à cette échelle, le bénéfice de la croissance de modèle est un pur effet
de réallocation de calcul, quantifié par un seuil où le coût entre de façon sous-linéaire
(ρ ∝ √(L_f/L_s) pour p=1) et l'écart de capacité de façon linéaire — d'où croissance gagnante à
FLOPs égaux (ρ = 4,65 ≫ G ≈ 1,23) et perdante à pas égaux (ρ = 1,16 < G). L'apport des données
complètes L=1..16 n'est pas une meilleure précision brute (R² 0,75 → 0,76) mais trois
clarifications structurelles : a(L) et b(L) sont logarithmiques en L (avec b *croissante*,
confirmée sur 16 profondeurs), l'enveloppe convexe du Théorème B contient en réalité toutes les
profondeurs — l'optimum théorique à FLOPs égaux est un calendrier graduel concentré sur les
petites profondeurs (1→2→3 à notre budget), prédiction testable non couverte par les bras
existants —, et le seuil G/ρ à pas égaux bascule du bon côté une fois le plancher a(16) mesuré
sur 8 000 pas. La validation dégage en outre une hypothèse falsifiable neuve : un coût
transitoire par greffe (~0,005 nat/greffe) que la règle du pas effectif, exacte à l'instant de
la greffe, ne capture pas.

Sur la question des constantes exactes (§5.4) : `a(L)` et `b(L)` restent des ajustements
statistiques, pas des constantes dérivées de premiers principes — mais leur forme logarithmique
n'est pas arbitraire. Elle coïncide, à trois vérifications numériques indépendantes (α mesuré
entre 0,071 et 0,079), avec l'exposant publié des lois d'échelle en nombre de paramètres de
Kaplan et al. 2020 (α_N≈0,076), et le modèle de quantification de Michaud et al. 2023 fournit un
mécanisme publié qui prédirait exactement cette forme sous une hypothèse Zipf-1 plausible pour
le langage naturel. C'est le niveau de rigueur honnêtement disponible : la forme est justifiée
par la littérature, les constantes ne le sont pas — une limite partagée par toutes les lois
d'échelle publiées à ce jour, pas une faiblesse spécifique à ce travail.

## 9. Traçabilité

Données : `growth_theory_missing_depths_results.json` (L=2,3,5..15 à 4 000 pas, L=16 à 8 000,
seed 1), `growth_jalon0_results.json` (A/B/C, 3 seeds), `growth_jalon1_results.json`,
`growth_jalon1_fixed_results.json`, `growth_jalon1b_results.json` (5 calendriers × 2 seeds
chacun). Scripts générateurs : `growth_jalon0.jl` (module commun : corpus, graphe,
`train_growth_arm!`), `growth_theory_missing_depths.jl`, `growth_jalon1*_full.jl`,
`growth_jalon1b_equal_steps.jl`. Procédure d'analyse : fits linéaires (a, b) par profondeur à
n0 = 3000 ; balayages globaux de n0 et p ; WLS (poids 1/SE²) des formes fonctionnelles avec
dummy de troncature, sélection par AIC/BIC et bootstrap (400 rééchantillonnages) ; enveloppe
convexe par balayage monotone ; prédictions par chaînage aux bornes exactes des `growth_events`.
Preuves complètes des théorèmes : note mémoire
`project_neurodsl_growth_schedules_pitch_2026-07-13.md`.

Références externes citées en §5.4 (vérifiées en ligne pendant la session) : J. Kaplan et al.,
*Scaling Laws for Neural Language Models*, arXiv:2001.08361, 2020 ; J. Hoffmann et al.,
*Training Compute-Optimal Large Language Models* (Chinchilla), arXiv:2203.15556, 2022 ;
M. Hutter, *Learning Curve Theory*, arXiv:2102.04074, 2021 ; E. Michaud, Z. Liu, U. Girit,
M. Tegmark, *The Quantization Model of Neural Scaling*, NeurIPS 2023, arXiv:2303.13506.

Résultats mathématiques classiques utilisés en §5.5 (identités et développements asymptotiques
standards, pas des résultats du projet) : NIST *Digital Library of Mathematical Functions*
(DLMF), §5.11 (développement asymptotique de la fonction digamma) et §25.11 (développement de la
fonction zêta de Hurwitz) ; Abramowitz & Stegun, *Handbook of Mathematical Functions*, formules
6.3.18 et 6.1.40 (nombres harmoniques et digamma). La contrainte `a1·a3/a2²=−1/3` et le
développement `[(ρL)^{-ε}-1]/ε` sont des dérivations originales de cette session, vérifiées
numériquement (implémentation manuelle de la digamma par récurrence + série asymptotique, sans
dépendance ajoutée à `Project.toml`), pas des résultats cités.
