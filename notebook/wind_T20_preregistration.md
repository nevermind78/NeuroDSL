# WIND-T20 — pré-enregistrement (écrit le 2026-10-04 ~07:35, AVANT tout test de prédiction)

**Statut : EXPLORATOIRE.** Ces données ont inspiré les conjectures (WIND-T6 à T19). Les seuils ci-dessous sont fixés
avant les calculs qui les testent ; ils ne changeront pas après.

## Ce qui a déjà été vu (échantillon de découverte, AVANT ce document)

Descriptif seulement : `wind_T20_explore.jl` et `wind_T20_lift.jl` sur Qwen p1–10, GPT-2 p1–10 et Gemma p1–2.
- Relèvement exact (lemme A de T7) : portes ≤ 2,5·10⁻⁴ ; rang de R_k = p (queue ≤ 10⁻³).
- Facteur de boucle (σ₁(Δ)/σ₁(Δ¹)), médiane : GPT-2 ≈ 1,1 ; Qwen ≈ 3,4 ; Gemma 6,7 et 12,1.
- Mémoire de boucle : ‖G‖/‖G_loc‖ ≈ 1 pour GPT-2, ≈ 5 pour Qwen, ≈ 20–70 pour Gemma.
- Gains radiaux relatifs ρ_k : la règle du demi (AH) est quasi neutre (≈ 0,93) ; AN est > 1 ; le réseau vivant
  vaut ≈ 0,75.
- Jamais calculés : les constantes du théorème 1 (partie A), la linéarisation REC (partie B), les prompts
  d'évaluation de la partie C.

---

## Partie A — Théorème 1 (règles neurone par neurone) : hypothèses et constantes

Le cadre et la preuve sont dans `wind_T20_theorem.md`. Ce qui est fixé ici, ce sont les points de mesure et ce qui
compte comme « hypothèse vérifiée ».

**Points de mesure.** Couches {3, 13, 23} pour Qwen et Gemma, {2, 6, 10} pour GPT-2, aux prompts 1 et 2, au
dernier token. Poids réels lus dans les safetensors ; état h depuis les méta WIND.

**Porte de parité P0.** f(v) recalculé à partir des poids = sortie MLP stockée dans la méta, en erreur relative :
< 10⁻² pour Qwen (poids bf16), < 10⁻³ pour GPT-2 et Gemma. Pour Gemma, la comparaison se fait après la post-norme.

**Hypothèses (critère 4) :**
- **H1 (excès d'Euler non nul).** ‖E‖/‖f(v) − f_base‖ ≥ 0,1 en chaque point.
- **H2 (injectivité / généricité).** La factorisation de Cholesky de la matrice de Gram 𝒢 réussit, et
  λ_min(𝒢)/λ_max(𝒢) ≥ 10⁻¹⁰ en chaque point (estimation par itération inverse).

**Question quantitative (descriptif, sans seuil de décision).** On note r* = τ*/‖JP‖_F, la distorsion transverse
minimale d'une règle neurone par neurone COMPLÈTE.
- Hypothèse annoncée : r* ≥ 0,05 (l'impossibilité est quantitativement significative).
- Contrepartie : r* < 0,05 voudrait dire qu'une règle neurone par neurone complète et presque transverse-exacte
  existe ; ce serait un résultat pratique à rapporter.
- Rapporté aussi : r_½, la distorsion de la règle du demi (Qwen, Gemma). Attendu : r_½ ≥ r* (optimalité, contrôle).

---

## Partie B — la linéarisation REC sur Qwen (50 prompts de WIND-T8)

REC est l'unique linéarisation complète et transverse-exacte du théorème 1(a), appliquée à chaque sous-couche :
- normes figées ;
- dérivée exacte sur ĥ⊥ ;
- gain radial du MLP ramené à J̃_m h = m.

Le script `wind_T20_rec.jl` est écrit ; il n'a pas été exécuté.

**Portes (sinon le prompt est exclu et signalé) :**
- G1 : rang 1 de R − R¹ < 5·10⁻², comme T13.
- G-T14 : ANf et AAf recalculés = WIND-T14 (MLP), écart relatif max < 10⁻⁶.
- G-T9 : A recalculé = WIND-T9 aA (MLP), < 10⁻³.

**Prédictions** (θ_c^AN vrai et d̄_p, Euler médian du prompt, viennent de T8 et T17) :
- **B1 (stabilité, mécanisme).**
  - REC s'effondre (ern(N·P^REC) ≤ 3) sur ≤ 6/50 prompts.
  - Si au moins 3 prompts s'effondrent : exactitude équilibrée ≥ 0,8 entre « REC s'effondre » et
    « θ_c^AN·d̄_p < 1 ».
  - Sur le chemin θ : médiane de θ_c^REC/θ_c^AN ∈ [1,4 ; 2,2], et Spearman(θ_c^REC, θ_c^AN) ≥ 0,8 (censure à 2,5 ;
    « pas de franchissement » = 2,5 dans les deux cas).
- **B2 (géométrie).**
  - Médiane de ern_REC/ern_A ∈ [0,5 ; 1,5].
  - Médiane de |ln(ern_REC/ern_T)| < médiane de |ln(ern_AN/ern_T)|, ET < médiane de |ln(ern_AH/ern_T)|.
- **B3 (fidélité, 28 composantes MLP, vérité WIND-T9).**
  - Médiane appariée ρ_REC − ρ_ANf ≥ +0,05, avec Wilcoxon unilatéral p < 0,05.
  - Médiane ρ_REC ≥ médiane ρ_AHf (WIND-T15, composantes MLP).
  - |médiane ρ_REC − médiane ρ_AAf| ≤ 0,05.
- **B4 (biais de profondeur, composantes MLP l ≤ 10).** Médiane de |early_REC − early_vérité| ≤ min(médiane
  |early_ANf − vérité|, médiane |early_AHf − vérité|).

---

## Partie C — mémoire de la boucle des normes, entre modèles (prompts d'évaluation)

**Prompts d'évaluation.**
- GPT-2 : prompts 11–30.
- Qwen : prompts 11–20.
- Gemma : prompts 3–8 (complets au moment de l'écriture).

Le script `wind_T20_lift.jl` est inchangé.

**Prédictions :**
- **C1 (facteur de boucle L_f = σ₁(Δ)/σ₁(Δ¹)).** Médiane GPT-2 ≤ 1,6 ; médiane Qwen ≥ 2,0 ; médiane Gemma ≥ 1,5 ×
  médiane Qwen.
- **C2 (mémoire M = ‖G‖/‖G_loc‖).** Médiane GPT-2 ≤ 1,5 ; médiane Qwen ≥ 3 ; médiane Gemma ≥ 3.
- **C3 (lien à l'effondrement, Qwen, découverte + évaluation, 20 prompts ; descriptif car en partie mécanique).**
  Spearman(ln L_f, θ_c^vrai) ≤ −0,6.

---

## Règles

Aucun seuil ne change. Tout est rapporté, échecs compris. Il n'y a aucun calcul GPU ; BLAS est limité à 4 fils.

---

## Déviations (écrites le 2026-10-04 ~08:05, après le test de fumée REC sur p1 seul, AVANT les 49 autres prompts)

**D1 — lecture.** La méta T8 stocke w_read = W_U[top1] − W_U[top2]. Or WIND-T9 utilise W_U[réponse] −
W_U[contrefactuel].
- Correction : `wind_T20_wread.json` (lignes de W_U lues dans le safetensors).
- Porte : R recalculé = logit(réponse) − logit(contrefactuel), écart relatif max 1,6·10⁻⁶ sur les 50 prompts.

**D2 — porte G-T14 mal spécifiée.** Le seuil 10⁻⁶ supposait le même chemin de code. Or WIND-T14 a calculé ses
attributions par le backward instrumenté (GPU, Float32), alors qu'ici on passe par le produit des Jacobiennes
stockées.
- Observé sur p1 : 5,1·10⁻⁵. C'est le niveau des portes GV de T9 (≤ 4,5·10⁻⁴ entre ces deux chemins).
- Le seuil devient 10⁻³, comme G-T9. Rien d'autre ne change.

**D3 — implémentation.** Pour l'efficacité, `wind_T20_rec.jl` évalue le θ-chemin de REC par le relèvement exact.
- Facteurs de rang 2 exacts.
- Porte : relèvement à θ = 1 contre produit direct.
- T, A, AN, AH sont repris des fichiers T8, T9, T14, T15 au lieu d'être recalculés.
- Les prédictions sont inchangées.
