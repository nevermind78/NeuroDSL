# WIND-T13 — test du correctif « norme du MLP vivante » (pré-enregistré le 2026-10-02, AVANT tout calcul)

## Contexte
WIND-T12 (exploratoire, 10 prompts) suggère que la rétroaction qui fait s'effondrer la linéarisation à normes figées
(AN) passe entièrement par le gel de la norme d'entrée du MLP. Le gel de la norme d'attention n'y contribuerait pas.

Si c'est vrai :
- une linéarisation qui fige l'attention et sa norme mais garde la norme du MLP vivante (« AA », le correctif) ne
  s'effondre pas ;
- celle qui ne fige que la norme du MLP (« AM ») s'effondre comme AN.

Ce test est CONFIRMATOIRE. Il porte sur les 50 prompts de WIND-T8, avec des prédictions écrites ici avant tout calcul
des variantes AA et AM.

## Dérivation exacte (aucune approximation)
Bloc : h = x + a(n₁(x)), x' = h + m(n₂(h)).
- Jacobien vrai de la RMSNorm : r·diag(γ)·Q, avec Q = I − x xᵀ/(‖x‖² + nε) ; version figée : r·diag(γ).
- On pose E₁ = I − Q₁ = s₁ x̂x̂ᵀ et E₂ = s₂ ĥĥᵀ, avec s = ‖·‖² r²/n (= 1 − O(10⁻⁵) ici).
- A₁ = Jacobien de l'attention, probabilités figées et norme 1 figée ; M = Jacobien du MLP, norme 2 figée.

Variantes :
- J^A = (I + MQ₂)(I + A₁Q₁)
- J^AN = (I + M)(I + A₁)
- J^AA = (I + MQ₂)(I + A₁) = J^A + R¹, avec R¹ = s₁ c₁ x̂ᵀ et c₁ = A₁x̂ + MQ₂A₁x̂
- J^AM = (I + M)(I + A₁Q₁) = J^A + R², avec R² = s₂ (Mĥ)[ĥᵀ(I + A₁Q₁)]
- R = J^AN − J^A = R¹ + R² + s₁s₂(ĥᵀA₁x̂)(Mĥ)x̂ᵀ

Calcul, par bloc k = 1..27 :
- 3 JVP en différences finies centrées (outils WIND-3) donnent A₁x̂, MQ₂(A₁x̂) et Mĥ ;
- la ligne ĥᵀ(I + A₁) s'obtient exactement de R − R¹ = s₂ (Mĥ)·ĥᵀ(I + A₁), qui est de rang 1, avec R stocké (WIND-T8) ;
- on en tire J^AA et J^AM à partir du J^A stocké.

## Portes (sinon le prompt est exclu et signalé)
- **G1 (rang 1)** : ‖(R − R¹) − s₂(Mĥ)ρᵀ‖/‖R‖ < 5·10⁻², en médiane sur les blocs.
- **G2 (indépendante)** : JVP directe du bloc en variante AA (resp. AM) le long de 2 directions aléatoires, sur les
  blocs 2, 8, 14, 20 et 26, comparée à (J^A + R¹)v (resp. (J^A + R²)v). Erreur relative médiane < 10⁻².
- **G3 (gradients)** : ∂R/∂x_k au dernier token, issu du backward de la variante, égal à J_kᵀ ∂R/∂x_{k+1} avec les
  Jacobiennes reconstruites. Pire erreur relative < 10⁻² pour AA et AM.
- **G4 (cohérence)** : les attributions AN recalculées ici reproduisent celles de WIND-T9 (écart relatif max < 10⁻⁴).

## Prédictions
- **M1 (le gel de la norme MLP suffit à l'effondrement)** : AM s'effondre (erank₂(N·Π J^AM) ≤ 3, N vraie norme
  finale) sur au moins 10 des 12 prompts où AN s'effondre (ern_true(1) ≤ 3, WIND-T8), ET
  Spearman(θ_c^AM, θ_c^AN vrai) ≥ 0,8 sur les 50 prompts (θ_c censurés à 2,5).
- **M2 (le gel de la norme d'attention est inoffensif)** : AA ne s'effondre pas, erank₂(N·Π J^AA) > 3, sur au moins
  49/50 prompts, ET θ_c^AA > 2,5 (aucun franchissement sur la grille) sur au moins 90 % des prompts.
- **F1 (le correctif rend la fidélité)** : sur les 14 prompts effondrés prédits (θ_c^mod < 1, WIND-T8, figés), la
  médiane de ρ_AA − ρ_AN est ≥ +0,10, ET le test de Wilcoxon apparié unilatéral donne p < 0,05.
  - ρ = Spearman(attribution, ablation réelle) sur les 56 sous-couches, avec la vérité terrain de WIND-T9 réutilisée.
- **F2 (plus de dégradation spécifique)** : Δ_AA = médiane ρ_AA(effondrés) − médiane ρ_AA(non effondrés) ≥ −0,05.

Pour les attributions, chaque variante fige les probabilités d'attention et la norme finale :
- **AA** fige en plus les normes 1 (attention) et garde les normes 2 (MLP) vivantes ;
- **AM** fige en plus les normes 2 et garde les normes 1 vivantes.

## Descriptif (sans seuil)
- Fidélité globale T / AN / AA / AM sur les prompts non effondrés : le correctif coûte-t-il de la fidélité ailleurs ?
- Le prix conceptuel du correctif (perte de la conservation visée par la règle LN d'Ali et al.) est discuté, pas mesuré.

## Règles
Aucun seuil ne change après avoir vu des données. Tout est rapporté, y compris les échecs.
Le prompt 51 sert au test du pipeline, exclu des verdicts.

## Addendum EXPLORATOIRE WIND-T14 (écrit le 2026-10-02 après les verdicts T13, AVANT le calcul ci-dessous)
Question : l'écart résiduel du correctif (F2 FAUX, Δ_AA = −0,195) vient-il du gel de la NORME FINALE, ajouté à AA
et AM pour rester comparable à AN ? La variante A (WIND-T9), qui ne fige aucune norme, n'avait qu'un écart de −0,035.
Variantes nouvelles (probabilités d'attention figées dans les deux) :
- AAf = normes 1 (attention) figées, normes 2 (MLP) ET norme finale vivantes ;
- ANf = normes 1 et 2 figées, norme finale vivante.
Porte : AA recalculé ici (norme finale figée) = attributions AA de T13 (écart relatif max < 10⁻⁴).
Seuils (exploratoires, mêmes 50 prompts, mêmes groupes θ_c^mod < 1) :
- E1 : Δ_AAf = médiane ρ_AAf(eff) − médiane ρ_AAf(non) ≥ −0,05 → la norme finale explique l'écart résiduel.
- E2 : médiane appariée ρ_AAf − ρ_AN sur les 14 effondrés prédits ≥ +0,10, Wilcoxon unilatéral p < 0,05.
- Descriptif : ANf (le gel de la norme finale seul aggrave-t-il AN ?), et ρ_AAf contre ρ_A (WIND-T9).
