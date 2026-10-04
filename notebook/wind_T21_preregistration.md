# WIND-T21 — tests des affirmations de `theo.tex` (pré-enregistré le 2026-10-04, AVANT tout calcul T21)

**Statut : EXPLORATOIRE.**
- Données existantes uniquement (Jacobiennes WIND-T8, T17, T19 ; facteurs T13), CPU, BLAS 4 fils.
- `theo.tex` n'est pas modifié.
- Script : `wind_T21_tests.jl`. Résultats : `wind_T21_tests_results.txt` et `.json`.

On teste les affirmations du texte telles qu'elles sont écrites, avec des critères opérationnels fixés ici. Pour
chacune, on indique aussi ma prédiction (ce que j'attends, avant calcul).

## Test A — lemme « Lyapunov Inversion »

**Affirmations du texte :**
- λ(J_exact) = ln‖J_exact e_x‖ ≈ 0 et ≤ 0 (« dissipatif ») ;
- λ(J_AN) > 0 (« strictement expansif ») ;
- ‖J_AN e_x‖ ≫ 1.

**Mesure.** Par bloc k, on calcule s_k^V = ln‖J_k^V x̂_k‖, avec x̂_k = x_k/‖x_k‖.
- Qwen (T, A, AN), prompts 1–10.
- GPT-2 (T, A, AN), prompts 1–10.
- Gemma (A, AN), prompts 1–5.

**Critères :**
- **A1 (texte).** s_k^T ≤ 0 pour TOUS les couples (prompt, bloc). Pour Gemma, on utilise A.
- **A2 (texte).** s_k^AN > 0 pour TOUS les couples.
- **A3 (texte).** Médiane de ‖J^AN x̂‖ ≥ 2 (lecture minimale de « ≫ 1 »).

**Mes prédictions :**
- A1 FAUX : s^T > 0 sur ≥ 20 % des couples Qwen, parce que le MLP voit h = x + a, et non x.
- A2 VRAI sur ≥ 80 % des couples mais pas sur 100 %.
- A3 FAUX : médiane < 1,5.

## Test B — théorème « Ho–Kalman Realization and Bifurcation Threshold »

**Affirmation du texte.** θ_c est la solution de det(I₂ − θ_c H₂) = 0, avec H₂ = C(I_D − J_exact)⁻¹B et ΔJ = BC.

**Mesure** (Qwen, 50 prompts de T8).
- Base J^A : c'est la base du θ-chemin qui DÉFINIT θ_c dans T8.
- Facteurs exacts de rang 2 (T13) : B_k = [s₁(A1+B1)_k, s₂MH_k], C_k = [x̂_k, ρ_k]ᵀ, de sorte que J_k^AN − J_k^A = B_kC_k.
  ρ_k est extrait comme dans T13 (porte de rang 1).
- Par bloc : H₂^(k) = C_k(I − J_k^A)⁻¹B_k, et θ^(k) = 1/(plus grande valeur propre réelle positive de H₂^(k)). On
  pose θ^(k) = +∞ s'il n'y en a aucune.
- Prédicteurs de θ_c :
  - θ_min = min_k θ^(k) ;
  - θ_med = médiane des θ^(k) finis.
- Conditionnement : on rapporte κ_k = σ_max/σ_min(I − J_k^A).

**Critères** (ceux de T8, appliqués à la formule du texte, avec θ_c^vrai de `wind_T8_truth.json`) :
- **B1.** Spearman(θ_pred, θ_c^vrai) ≥ 0,7, pour θ_min ou pour θ_med.
- **B2.** Fraction des prompts avec |θ_c^vrai − θ_pred| ≤ 0,15·θ_pred ≥ 0,7, pour le meilleur des deux prédicteurs.

**Mes prédictions :** B1 FAUX (Spearman < 0,5) et B2 FAUX (fraction < 0,3) pour les deux prédicteurs.
- Le système réel varie d'une couche à l'autre ; aucune quantité « à z = 1 » d'un bloc isolé ne fixe le seuil d'un
  produit de 27 blocs.
- (I − J_k^A) est mal conditionné, car J ≈ I + termes de bas rang.

## Test C — théorème « The Riemannian Funnel »

**Affirmation du texte.** Quand k → L, erank(g_k) = (tr g_k)²/tr(g_k²) s'effondre de D = 1536 à r ≈ 16, avec
g_k = T_{k→L}ᵀT_{k→L} et T_{k→L} = J_{L−1}···J_k. Le texte ajoute que ker(g_k) donne ds² = 0.

**Mesure** (Qwen, variante T, prompts 1–10, k = 1..27).
- erank_PR(g_k) : définition du texte, participation des σ_i².
- erank_H : entropie de WIND, exp H(σ_i²/Σσ²).
- On calcule les deux sans N et avec N (vraie norme finale).
- On rapporte aussi σ_min/σ_max de T_{k→L}.

**Critères :**
- **C1 (texte, direction).** Médiane sur les prompts de erank_PR(g_k) décroissante en k pour k ≥ 19, ET
  erank_PR(g_27) ≤ 30.
- **C2 (texte, noyau).** Il existe k avec σ_min(T_{k→L})/σ_max ≤ 10⁻⁸ (noyau numérique).

**Mes prédictions :**
- C1 FAUX : erank_PR(g_k) CROÎT quand k → L, puisque le produit a moins de facteurs ; erank_PR(g_27) > erank_PR(g_1)
  sur ≥ 80 % des prompts.
- C2 FAUX : pas de noyau numérique, σ_min/σ_max > 10⁻⁸ pour tout k.

## Contrôle D — chiffres cités par le texte

C'est une vérification de provenance, pas un test.
- Tableau T8 : comparer aux fichiers (`wind_T8_truth_results.txt`).
- Tableau « Depth Bias » (33/28/52/11 %) : recalculer à partir de T9 et T15, en précisant le groupe et les composantes.
- D de Gemma-2-2B : lire `gemma-2-2b/config.json`.

## Règles

Aucun seuil ne change après calcul. Tout est rapporté.
