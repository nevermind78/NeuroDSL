# WIND-T15 — la règle du demi supprime-t-elle l'effondrement ? (pré-enregistré le 2026-10-02, AVANT tout calcul)

## Question
L'effondrement de la linéarisation à normes figées (AN) est porté par le gel de la norme du MLP, via le facteur 2
d'Euler du SwiGLU (WIND-T12, T13). Or les méthodes de pointe linéarisent le SwiGLU autrement :
- la « règle du demi » de la LRP (Transluce, Arora et al. arXiv:2601.22594, sur Llama 3.1 8B) ;
- la règle uniforme d'AttnLRP.

Ces règles suppriment exactement ce facteur 2.

Question : avec cette linéarisation (« AH »), l'effondrement disparaît-il, ou persiste-t-il ?

## Variante AH (linéarisation de Transluce, adaptée à Qwen2)
- Probabilités d'attention figées : l'attention est alors linéaire dans les valeurs, sans multiplication restante.
- Dénominateurs de TOUTES les RMSNorm figés, comme AN.
- SwiGLU f = silu(g)⊙u = g⊙σ(g)⊙u linéarisé comme Transluce : σ figé à sa valeur propre s₀ (f devient bilinéaire en
  g, u), puis règle du demi.
  - Différentielle : df = ½·s₀⊙(u₀⊙dg + g₀⊙du).
  - À comparer à AN (dérivée exacte) : df = u₀⊙silu'(g₀)⊙dg + silu(g₀)⊙du.
- Implémentation exacte : un op `:swiglu_half` (au niveau du script, aucune modification de src/) dont le forward
  est l'application affine f₀ + ½ s₀⊙(u₀⊙(g − g₀) + g₀⊙(u − u₀)), avec (g₀, u₀, f₀, s₀) capturés au point propre.
  - Sa valeur au point propre est exactement f₀ : le forward est inchangé.
  - Sa dérivée est exactement la linéarisation demi.
  - Les différences finies donnent donc la Jacobienne de bloc J^AH sans approximation.
  - Pour les gradients : règle de backward ½·dy⊙s₀⊙(u₀, g₀).

## Données
- Les 50 prompts de WIND-T8. Le prompt 51 sert au test du pipeline, exclu des verdicts.
- J^AH par bloc, au dernier token, collecté comme WIND-T8 (même script, mêmes portes, mêmes pas).
- J^A est repris de WIND-T8.
- N = vraie norme finale, comme pour toutes les mesures d'erank de WIND-T8.

## Portes (sinon le prompt est exclu et signalé)
- G1 : ε contre 2ε, comme WIND-T8.
- G3 : JVP multi-couches directe, normes figées, op demi, probabilités épinglées, comparée au produit des J^AH.
  Erreur médiane < 10⁻² (la composition est exacte, puisque le réseau AH est linéaire).
- G4a / G4b : forward avec ops substitués = forward propre (erreur relative < 10⁻⁵).
- GV : ∂R/∂x_k du backward AH, avec la règle demi analytique, comparé à J^AHᵀ ∂R/∂x_{k+1}.
  Pire erreur relative < 10⁻² sur k = 1..27. Deux implémentations indépendantes : différences finies affines
  contre backward analytique.

## Critère principal (décision)
- C_AN = les 12 prompts où AN s'effondre (ern_true(1) ≤ 3, WIND-T8).
- Effondrement AH : erank₂(N·Π_k J_k^AH) ≤ 3.
- **SUPPRIMÉ** : AH s'effondre sur ≤ 1 des 12 prompts de C_AN ET sur ≤ 2/50 au total.
- **PERSISTE** : AH s'effondre sur ≥ 6 des 12 prompts de C_AN.
- **PARTIEL** : sinon.

## Prédiction directionnelle (dérivée du mécanisme, avant mesure)
Le facteur d'Euler du SwiGLU le long de l'état passe de ≈ 2 (AN) à exactement 1 (AH, complétude). La rétroaction
d'échelle doit donc être affaiblie.
- **D1** : sur le chemin J(θ) = J^A + θ(J^AH − J^A), θ_c^AH > θ_c^AN (WIND-T8), en médiane appariée, test de Wilcoxon
  unilatéral p < 0,05. θ_c est censuré à 2,5 (« pas de franchissement » = 2,5).

## Fidélité (secondaire, même vérité terrain que WIND-T9, ablations au dernier token)
- **F1** : sur les 14 prompts effondrés prédits (θ_c^mod < 1, WIND-T8, figés), médiane appariée ρ_AH − ρ_AN ≥ +0,10,
  Wilcoxon unilatéral p < 0,05.
- **F2** : Δ_AH = médiane ρ_AH(effondrés) − médiane ρ_AH(non effondrés) ≥ −0,05.
- **Descriptif** : AHf (AH avec norme finale vivante) ; ρ médians T / A / AN / AH / AHf.

## Règles
Aucun seuil ne change après avoir vu des données WIND-T15. Tout est rapporté, y compris les échecs.
