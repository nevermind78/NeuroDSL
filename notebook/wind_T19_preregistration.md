# WIND-T19 — l'effondrement se reproduit-il dans Gemma-2-2B ? (pré-enregistré le 2026-10-03, AVANT toute collecte)

## Question
Qwen2.5-1.5B (MLP SwiGLU) s'effondre quand on fige ses normes :
- rapport médian ern(N·P_AN) / ern(N·P_A) = 0,27 ;
- 38/50 prompts sous 0,5.

GPT-2 (MLP GELU sans porte) ne s'effondre pas : 1,04, 0/50. Mais GPT-2 était déjà concentré à 66 % sur une
direction (WIND-T18), ce qui en fait un mauvais témoin.

Gemma-2-2B est une seconde famille de LLM moderne à MLP À PORTE (GeGLU), avec 26 couches, RMSNorm et des normes
supplémentaires après chaque sous-couche. Question : l'effondrement est-il propre à Qwen, ou se reproduit-il ?

## Modèle et prompts
- Gemma-2-2B porté dans NeuroDSL, porte de parité contre transformers passée : états ≤ 1,2·10⁻⁶, logits ≤ 5,9·10⁻⁵,
  top-1 identique 5/5 ; pic de VRAM 10,8 Go.
- Prompts : les textes 1 à 25 des 50 de WIND-T8 (même ordre, choix fixé avant collecte, pour tenir en une nuit),
  tokenisés avec `<bos>` (`gemma-2-2b/qwen50_gemmatok.json`). Si la collecte finit tôt, les prompts 26 à 50 sont
  ajoutés, mais ils ne servent qu'au descriptif.

## Variantes et mesures
- **A** : probabilités d'attention épinglées, toutes les normes vivantes.
- **AN** : A + dénominateurs des 5 normes par bloc figés (norm1, post-attention, norm2, post-MLP) et norme finale.
- Jacobiennes de bloc au dernier token, collecteur WIND-T8 adapté (mêmes portes G1 à G4, G2 réécrite pour le MLP GeGLU
  avec ses normes avant et après).
- Produit P_V = J_25···J_1 (J_0, le plongement, est exclu) ; N = Jacobien de la vraie norme finale ; ern = erank₂(N·P_V).

## Critère principal (RELATIF, leçon de GPT-2)
**G1** : sur les prompts 1 à 25, la médiane de ern(N·P_AN) / ern(N·P_A) est ≤ 0,5, ET au moins 50 % des prompts ont un
rapport < 0,5.
- VRAI → l'effondrement se reproduit dans une seconde famille à MLP à porte : il n'est pas propre à Qwen.
- FAUX → il pourrait être propre à Qwen. On le rapporte, et l'article est recentré en conséquence.

## Descriptif (sans seuil)
- Métriques de WIND-T18, à comparer avec Qwen et GPT-2 :
  - g_raw, g_N (amplification du mode dominant par le gel, avant et après la norme finale) ;
  - c_dom (même direction dominante avec et sans gel) ;
  - p1_A (part de la direction dominante sans gel) ;
  - c_rad (alignement radial).
- Chemin θ : J^A + θ(J^AN − J^A), θ ∈ 0,25 : 0,125 : 2,5 ; ern(θ)/ern(0) ; premier θ où ce rapport passe sous 0,5.
- Gains d'état λ_k pour A et AN.
- Degré d'Euler effectif du MLP brut (norme d'entrée figée), mesure identique à WIND-T17.

## Règles
Le seuil de G1 ne change pas après avoir vu les données. Tout est rapporté. Un prompt dont une porte échoue est exclu
et signalé.

## Amendement de méthode (écrit le 2026-10-04 ~00:00Z, AVANT toute collecte Gemma ; seul un test de fumée à 96 colonnes a été lancé)
Constat : la collecte complète (2304 colonnes × 26 couches) coûte ~16 min par variante et par prompt (mesuré), soit
~14 h pour 25 prompts. Ça ne tient pas en une nuit.

Méthode retenue pour G1 : une ESQUISSE ALÉATOIRE du produit.
- k = 256 directions gaussiennes g_i (graine fixe), dérivées multi-couches centrées de x_1 vers x_26 au dernier token :
  P_V g_i (même routine que la porte G3 de WIND-T8).
- On applique N et on estime ern(N·P_V) par l'erank₂ des valeurs singulières de N·P_V·G/√k.
- La même G sert pour A et AN.
- Coût ≈ 9 fois moins que la collecte complète. Les 50 prompts deviennent possibles, donc **G1 porte sur les 50**
  (au lieu de 1 à 25).

Portes supplémentaires, fixées maintenant :
- **E1 (étalonnage de l'estimateur, sur Qwen, matrices exactes de WIND-T8, 20 premiers prompts)** : erreur absolue
  médiane du rapport esquissé ern_AN/ern_A contre l'exact ≤ 0,05, ET accord de la classe « rapport < 0,5 » ≥ 90 %.
  Si E1 échoue : retour à la collecte complète sur les prompts 1 à 15, avec le seuil de G1 inchangé.
- **E2 (vérification sur Gemma lui-même)** : les prompts 1 et 2 sont aussi collectés EN COMPLET (A et AN, 2304 colonnes).
  Le rapport esquissé doit être à ±0,05 de l'exact.
- **E3 (différences finies)** : sur 4 directions par variante et par prompt, ε contre 2ε d'écart relatif < 10⁻².

## Résultat de la porte E1 (2026-10-04, AVANT toute collecte Gemma utilisée pour G1)
E1 ÉCHOUE : erreur absolue médiane du rapport 0,027 (≤ 0,05, OK), mais accord de la classe « < 0,5 » = 85 % (< 90 %).
Désaccords : 3 prompts à la frontière (p2 0,473 → 0,501 ; p7 0,431 → 0,501 ; p9 0,458 → 0,518) ; l'esquisse surestime
légèrement le rapport. Résultats : `wind_T19_calib_results.txt`.
→ Application de la règle prévue : **G1 est évalué sur la collecte COMPLÈTE (A et AN, 2304 colonnes) des prompts 1 à 15**,
seuil inchangé (médiane ≤ 0,5 ET ≥ 50 % des prompts sous 0,5). L'esquisse n'est plus utilisée que pour du descriptif
(prompts 16 à 50, s'il reste du temps).

## Erratum (2026-10-04, pendant la collecte, sans effet sur le calcul)
« 5 normes par bloc » est une coquille : Gemma-2 a 4 RMSNorm par bloc (norm1, post-attention, norm2, post-MLP), plus la
norme finale. Toutes sont figées dans AN : 105 nœuds, vérifié dans le checkpoint et dans `norm_syms`.
