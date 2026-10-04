# WIND-T9 — l'effondrement fausse-t-il de vraies attributions ? (pré-enregistré le 2026-10-02, AVANT tout calcul)

## Question
WIND-T8 a confirmé, sur 50 prompts neufs, que la linéarisation à normes figées (AN ≈ LRP(AH+LN), la linéarisation
des graphes d'attribution sans transcodeurs) s'effondre au gel complet quand le gain critique θ_c < 1. Cela concerne
24 % des prompts.

Question : sur ces prompts, les attributions linéaires calculées avec AN sont-elles nettement MOINS FIDÈLES aux
effets réels, alors que celles de la vraie dérivée (T) ne le sont pas ?

Issue possible et informative : NON. Si la direction qui explose est (quasi) orthogonale à la lecture, l'effondrement
est sans effet sur les attributions de R.

## Protocole
- **Prompts** : les 50 de WIND-T8 (`wind_T8_prompts.json`). Le prompt 51 sert au test du pipeline, exclu.
- **Lecture contrefactuelle** : R = logit(réponse) − logit(contrefactuel) = ⟨final_norm_out[n], W_U[réponse] − W_U[contrefactuel]⟩.
- **Composantes** : les 56 sous-couches au dernier token n, c'est-à-dire MHA_l (écrit dans h_l = layer_l_res1) et
  MLP_l (écrit dans x_l = layer_l_out), pour l = 1..28.
- **Référence d'ablation** : μ_c = moyenne de la sortie de c au dernier token sur les 50 prompts (méta WIND-T8).
- **Attribution linéaire de la variante V ∈ {T, A, AN}** : a^V_c = ⟨∂R/∂(point d'écriture de c), o_c − μ_c⟩.
  - Le gradient vient d'un backward exact (outils WIND-3, `wind3_common.jl`) avec stop-gradient script sur les
    probabilités d'attention (A, AN) et sur les dénominateurs RMSNorm, norme finale comprise (AN).
- **Vérité terrain** : ΔR_c = R(propre) − R(o_c remplacé par μ_c au dernier token), forward du VRAI modèle
  (normes vivantes, pas d'épinglage).
- **Porte GA** (gradient de T) : différence centrée locale (R(o + ε·d) − R(o − ε·d))/(2ε), avec d = o_c − μ_c et
  ε = 0,05, comparée à a^T_c sur les composantes où |a^T_c| ≥ 10⁻³·max|a^T|.
  - Seuils : médiane de l'erreur relative < 2·10⁻², sinon le prompt est exclu et signalé.
- **Porte GR** : top-1 = réponse attendue, et |R(propre) − (logit_answer − logit_cf)| ≤ 10⁻⁴·max(1, |logit_answer|),
  avec les logits de la méta WIND-T8 (écart relatif rapporté à la taille des logits).
- **Porte GV** (gradients de chaque variante) : au dernier token, ∂R/∂x_k doit valoir J_k^Vᵀ ∂R/∂x_{k+1}, avec les
  Jacobiennes WIND-T8 de la même variante. Pire erreur relative sur k = 1..27 < 10⁻² pour T, A et AN, sinon exclusion.
  (Précision ajoutée avant tout calcul.)

## Mesures (par prompt et par variante)
- ρ_V = Spearman(a^V, ΔR) sur les 56 composantes.
- top5_V = |top-5(|a^V|) ∩ top-5(|ΔR|)| / 5.
- err_V = ‖a^V − ΔR‖ / ‖ΔR‖.
- Versions « précoces » (l ≤ 10, 20 composantes) et « tardives » (l ≥ 11, 36 composantes).

## Groupes (fixés AVANT, prospectivement)
- Effondrés = θ_c^mod(ordre 2) < 1 dans `wind_T8_predictions.json`, figé le 2026-10-02 05:02Z (SHA-256 25cc3224…).
  Cela donne 14 prompts ; les 36 autres forment le groupe non effondré.
- Analyse de sensibilité : effondrement vrai, ern_true(1) ≤ 3 (12 prompts).

## Critères
- **B1 (cœur)** : médiane ρ_AN(effondrés) − médiane ρ_AN(non effondrés) ≤ −0,15, ET Mann–Whitney unilatéral p < 0,05.
- **B2 (spécificité)** : Δ_AN − Δ_T ≤ −0,10, où Δ_V = médiane ρ_V(effondrés) − médiane ρ_V(non). Autrement dit,
  c'est la linéarisation figée qui se dégrade, pas des prompts intrinsèquement difficiles.
- **B3 (pratique)** : moyenne top5_AN(effondrés) ≤ moyenne top5_AN(non) − 0,15.
- **B4 (dose-réponse)** : sur les 50 prompts, Spearman(θ_c^mod, ρ_AN) ≥ 0,3.
- **Lecture** :
  - B1 ∧ B2 → l'effondrement fausse les attributions AN, de façon spécifique et prévisible par θ_c ;
  - B1 faux → l'effondrement est sans effet notable sur ces attributions, rapporté tel quel.

## Descriptif (sans seuil)
- Même analyse avec les effondrements vrais.
- Précoces contre tardives (le mécanisme vit dans les couches 1–10).
- Variante A.
- **Contre-évidence d'Ali et al. 2022 / RelP** : sur les prompts NON effondrés, AN est-il plus fidèle que T
  (médianes de ρ et de err) ?

## Règles
Aucun seuil ne change après avoir vu des données. Tous les résultats sont rapportés, y compris les échecs.
