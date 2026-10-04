# WIND-T17 — loi d'architecture : MLP à porte ou sans porte (pré-enregistré le 2026-10-03, AVANT toute collecte GPT-2)

## Hypothèse
L'effondrement de la linéarisation à normes figées (AN) vient du facteur d'Euler ≈ 2 du MLP À PORTE (SwiGLU,
homogène de degré ≈ 2 quand sa norme d'entrée est figée), via une rétroaction d'échelle cumulée sur les couches
(WIND-T7 à T13, T15).

Un MLP SANS porte (GPT-2 : c_fc → GELU → c_proj, degré ≈ 1) ne devrait PAS produire cette rétroaction. Cela
expliquerait aussi pourquoi Ali et al. 2022 et RelP trouvent que figer les normes AIDE sur BERT / GPT-2.

## Modèles et prompts
- GPT-2 small (12 couches, d = 768, LayerNorm, MLP GELU sans porte). Le portage NeuroDSL a passé la porte de parité
  contre transformers : états ≤ 8·10⁻⁷, logits ≤ 2,2·10⁻⁴, top-1 identique.
- Qwen2.5-1.5B (données WIND-T8 existantes).
- Prompts : les MÊMES 50 textes que WIND-T8, tokenisés pour GPT-2 (`gpt2/qwen50_gpt2tok.json`), pour une comparaison
  appariée. GPT-2 en réussit 24/50 avec marge ≥ 2. Les critères géométriques (L1–L4) portent sur les 50 ; la fidélité
  (descriptif) porte sur les 24 réussis.

## Mesures (GPT-2, mêmes définitions que WIND-T8)
- Jacobiennes de bloc au dernier token pour T, A, AN : J_k, k = 1..11 (J_0, le plongement, est exclu comme pour Qwen).
- AN pour GPT-2 : dénominateurs (1/σ) des LayerNorm figés, le centrage reste.
- N = Jacobien de la LayerNorm finale : r·diag(γ)·(P − r² x̃x̃ᵀ/D), avec P = I − 11ᵀ/D.
- ern = erank₂(N·Π J_k) ; θ_c sur le chemin J^A + θ(J^AN − J^A), grille 0,25 : 0,125 : 2,5, seuil ern = 3 (règle WIND-T8).
- Gain d'état par couche λ_k = ⟨x̂_{k+1}, J_k x̂_k⟩, pour T et AN.
- Degré d'Euler effectif du MLP à norme figée, couche par couche, mesuré dans LES DEUX modèles par différence finie :
  d_l = ⟨∂_s m(s·h)|_{s=1}, m(h)⟩ / ‖m(h)‖², norme d'entrée du MLP figée, h = état après attention au dernier token.

## Prédictions (contrôle de profondeur intégré)
- **L1 (prémisse du mécanisme, indépendante de la profondeur)** : degré d'Euler médian (prompts × couches) ≤ 1,3 pour
  GPT-2 ET ≥ 1,7 pour Qwen.
- **L2 (excès de gain par couche, indépendant de la profondeur)** : médiane de (λ^AN − λ^T) pour GPT-2 < 0,5 × la même
  médiane pour Qwen (calculée sur les Jacobiennes WIND-T8).
- **L3 (conséquence, à profondeur égale)** : GPT-2-AN s'effondre (ern ≤ 3) sur ≤ 2/50 prompts, ET le contrôle de
  profondeur « Qwen-12 » s'effondre sur ≥ 6/50.
  - Qwen-12 : J^AN sur les couches 1 à 12, J^A sur les couches 13 à 27, vraie N.
  - Si GPT-2 ne s'effondre pas alors que Qwen avec seulement 12 couches figées s'effondre, la faible profondeur de
    GPT-2 n'explique pas la différence.
- **L4 (distance à la criticité)** : médiane de θ_c^AN de GPT-2 ≥ médiane de Qwen (1,14) + 0,5. θ_c est censuré à 2,5.

Décision : L1 ∧ L3 vrais → loi d'architecture soutenue (le facteur d'Euler du MLP décide de l'effondrement, la
profondeur ne l'explique pas). L1 faux → la prémisse est fausse et on le rapporte. L1 vrai et L3 faux → le mécanisme
n'est pas le seul déterminant.

## Descriptif
- Fidélité des attributions T / A / AN sur les 24 prompts réussis (protocole WIND-T9, vérité = ablation par la moyenne).
  Prédiction qualitative non chiffrée : AN ne doit pas être pire que A sur GPT-2, contrairement à Qwen.

## Portes
- Porte de parité : déjà passée.
- G1 (ε contre 2ε), G3 (JVP multi-couches contre produit), G4a et G4b, comme WIND-T8.
- G2 : Jacobienne analytique du MLP GELU à LayerNorm vraie, contre différences finies.

## Règles
Aucun seuil ne change après avoir vu des données WIND-T17. Tout est rapporté, y compris les échecs.
