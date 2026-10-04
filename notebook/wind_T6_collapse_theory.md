# WIND-T6 — Pourquoi les linéarisations figées (A, AN) s'effondrent : conjectures, preuves, tests

*2026-10-01. Qwen2.5-1.5B-Instruct, Jacobiennes WIND-1 (dernier token, 5 prompts). TOUT EST EXPLORATOIRE : ces
5 prompts ont révélé l'effet et ont servi à formuler les conjectures. Les prédictions P6.1–P6.4 ont été écrites dans
`wind_T3_preregistration.md` (section WIND-T6) avant `wind_T6_signflip.jl`, mais sur les mêmes prompts.*

---

## 0. Résumé

1. **L'effondrement est l'explosion d'une seule direction, pas une perte de « mélange ».** La masse résiduelle
   τ = (Σ_{i≥2} σ_i²)^{1/2} de N·P reste quasi identique entre T, A et AN (τ_V/τ_T ∈ [0,77 ; 1,01] sur les 10 couples).
   Pendant ce temps, σ₁ est multiplié par 22,2 (AN-p12), 4,5 (AN-p22) et 2,5 (A-p12). Par le lemme 1, cela suffit à
   faire tomber ern vers 1.
2. **L'excès vient exactement des termes de rétroaction des normaliseurs que la linéarisation figée supprime**
   (théorème 2, identité exacte). Pour les normes : P_AN − P_A = Σ_k (réponse aval) ⊗ ∇‖x_k‖, ∇‖h_k‖, soit un rang
   ≤ 2 par couche. Pour l'attention : P_T − P_A = Σ_k Σ_{h,j} (écart de valeur) ⊗ ∇(logit s^h_j), rang ≤ 12(n_tok−1)
   par couche (vérifié : rang numérique 82–149 ≤ 156). Les contributions se concentrent sur les couches 2–10.
3. **Normes figées : le mécanisme est une rétroaction positive d'échelle, cohérente d'une couche à l'autre.**
   (Proposition 3 exacte, conjecture C2 soutenue par P6.2–P6.4.) La linéarisation figée lit une fluctuation de
   ‖x_k‖ comme un changement de contenu. Elle ré-amplifie donc la sortie propre de chaque couche : ×2 à un défaut
   borné près pour le SwiGLU, avec un défaut ≤ 0,4392 par neurone. Ces termes s'additionnent avec un signe cohérent
   sur les couches 2–8. La direction de sortie de l'effondrement est alors la réponse à une mise à l'échelle commune
   des mises à jour précoces : |cos| = 0,998 (p12) et 0,995 (p22). Sous le nul « signes aléatoires », AN-p22 est
   au-delà des 40 tirages.
4. **Attention figée (A-p12) : l'hypothèse d'une compensation QK spécifique est RÉFUTÉE au seuil pré-enregistré**
   (P6.1 faux). Les chaînes « A + signes aléatoires des termes QK » s'effondrent presque toutes sur p12 (ern médiane
   3,97). T n'est qu'au 92,5ᵉ centile. La « protection » de T est une tendance faible (post hoc : T est dans la
   moitié haute sur les 5 prompts), et elle vient d'interactions entre couches, pas d'un effet de premier ordre.
5. **Restent ouverts :** pourquoi l'amplitude dépasse la masse résiduelle sur p12/p22 et pas sur p17 (qui présente
   pourtant la même cohérence), et l'origine de la sensibilité de p12 à l'attention figée.
6. **Non-trivialité :** aucun énoncé démontré ici n'est mathématiquement profond. Ce sont des conséquences directes
   de la règle de la chaîne, de Weyl, d'Eckart–Young, de Jensen et de l'orthogonalité de Walsh. La substance est la
   décomposition exacte plus les faits empiriques.

---

## 1. Cadre et notations

- Chaîne : P_V = J^V_27 ··· J^V_1 (J_0 exclu), V ∈ {T, A, AN}, n = 1536. Suffixes P^V_{k+1} = J^V_27···J^V_{k+1}
  (P^V_28 = I) ; préfixes Q^V_{k−1} = J^V_{k−1}···J^V_1 (Q_0 = I).
- Métrique : M_V = N·P_V avec la **vraie** norme finale N = r·diag(γ)(I − x xᵀ/(‖x‖²+nε)), x = x_28, pour toutes les
  variantes (convention « ernTN » de WIND-T4 ; elle isole l'effet des couches). ern = exp H(p), p_i = σ_i²/‖M‖_F².
- p₁ = σ₁²/‖M‖_F², masse résiduelle τ(M) = (Σ_{i≥2} σ_i²)^{1/2} = min_{rg X ≤ 1} ‖M − X‖_F (Eckart–Young),
  rapport d'explosion b = σ₁/τ.
- Termes de gel : A_k := J^A_k, **E_k := J^T_k − J^A_k** (terme QK, avec propagation MLP), **R_k := J^AN_k − J^A_k**
  (gel des normes).
- Bloc k (pré-norm) : h = x + a(n₁(x)), x' = h + m(n₂(h)). Exactement, pour chaque variante,
  **J_k = (I + J_m)(I + J_a)** (dérivée d'une composition). J_m est le même en T et A (même point propre, l'épinglage
  ne change pas le forward, porte G4a).

Scripts de ce travail (tous dans `notebook/`, CPU, données WIND-1 seulement) :
`wind_T6_explore.jl` (profils couche par couche), `wind_T6_hybrid.jl` (produits hybrides),
`wind_T6_normgrad.jl` (télescopage des normes), `wind_T6_cancel.jl` (télescopage le long de v₁),
`wind_T6_signflip.jl` (tests P6.1–P6.4 + contrôles des théorèmes), `wind_T6_qkrank.jl` (rang de E_k).

---

## 2. Observations exploratoires qui ont guidé les conjectures (post hoc, non prédites)

**O1. Explosion d'une direction, masse résiduelle conservée.** (`wind_T6_signflip_results.txt`)

| prompt | τ_T | τ_A/τ_T | τ_AN/τ_T | σ₁ A/T | σ₁ AN/T | b_T | b_A | b_AN | ern T / A / AN |
|---|---|---|---|---|---|---|---|---|---|
| p5  | 25,9 | 0,77 | 0,90 | 0,80 | 0,91 | 0,76 | 0,79 | 0,77 | 15,8 / 20,6 / 15,5 |
| p6  | 20,5 | 0,93 | 1,01 | 0,75 | 0,80 | 0,58 | 0,47 | 0,46 | 47,6 / 62,6 / 51,5 |
| p12 | 77,8 | 0,91 | 0,95 | **2,47** | **22,2** | 0,97 | **2,62** | **22,6** | 13,1 / **2,31** / **1,02** |
| p17 | 52,4 | 0,81 | 0,91 | 0,72 | 0,95 | 0,93 | 0,82 | 0,96 | 8,5 / 10,1 / 7,8 |
| p22 | 25,8 | 0,86 | 0,88 | 0,79 | **4,53** | 0,65 | 0,60 | **3,35** | 27,6 / 33,0 / **1,83** |

Les trois effondrements (ern ≤ 3) sont exactement les trois cas b ≥ 2,6. Dans tous les autres, b ≤ 0,97. La
masse résiduelle ne s'effondre jamais : elle est même légèrement plus faible dans les variantes figées.

**O2. Localisation causale (hybrides, `wind_T6_hybrid_results_*.txt`).**
- A-p12 : remettre la seule couche 6 en T dans A fait passer ern de 2,3 à 10,1 (couche 7 : 6,1). Figer
  l'attention sur la seule fenêtre 6–10 de T fait passer ern de 13,1 à 2,5. Figer 1–5 ou 11–27 ne change rien
  (13,9 et 20,3).
- AN-p22 : effet cumulatif, aucune couche seule n'est décisive. Geler les normes sur la fenêtre 1–10 de A fait
  passer ern de 33 à 3,85 ; sur 11–27, de 33 à 12,7.
- AN-p12 : figer 1–5 fait passer 2,3 à 1,09 ; figer 6–10 donne 1,47.
- p5, p6, p17 : aucun effet marqué, quelle que soit la fenêtre.

**O3. Télescopage le long de la direction dominante (`wind_T6_cancel_results.txt`).** On mesure la part de chaque
couche k dans la projection sur N·P_A·v, notée c_k.
- p12, v = v₁(N·P_A) : la correction QK retire 75 % (Σc = −0,75, cos = −0,96), surtout via les couches 6–9
  (−0,26, −0,24, −0,14, −0,30).
- p12 et p22, v = v₁(N·P_AN) : les termes de normes ajoutent +2,33 et +4,09. Contributions p22, k = 3..8 :
  +0,55, +0,84, +0,50, +0,44, +0,60, +0,43.
- p5, p6, p17 le long de v₁(N·P_A) : |Σc| ≤ 0,39, sans signe cohérent.
- Mise en garde : le long de v₁(N·P_T), la correction QK est toujours positive (+0,52 à +0,73). Le signe négatif le
  long de v₁(N·P_A) contient donc un **biais de sélection**, d'où le nul par retournement de signes (§6).

**O4. Trajectoire de la direction d'effondrement (`wind_T6_explore_results_*.txt`).**
La direction w_k = J_k···J_1 v₁(P_1) est alignée sur l'état x_{k+1} à 0,56–0,78 dans les couches 3–10 pour AN-p12,
AN-p22 et A-p12 (0,12–0,60 pour T-p12). Les gains par couche valent 1,0–2,5 (médiane 1,40), contre une médiane de
1,07 pour T-p12. Visibilité
α_k = ‖P_{k+1} u₁(J_k)‖/σ₁(P_{k+1}) :
- AN-p22 (k = 3..8) : 0,12–0,22 ;
- A-p12 (k = 4..6) : 0,14–0,16 ;
- T : 0,03–0,07 (hasard ≈ 1/√(n·p₁) ≈ 0,05).

Les couches 1–5 ont des pics de bas rang : srang(J_1 − I) = 3,3–3,4 (p12) et 7,5–7,7 (p17) ;
σ₁(J_1) ≈ 14,5 (p12) et 14,7 (p17). p17 a donc le même pic que p12 sans s'effondrer : le pic seul ne suffit pas.

**O5. Correction des normes quasi de rang 1 et grande sur p12/p22 (`wind_T6_normgrad_results.txt`).**
- σ₂/σ₁(N·Δ), Δ = P_AN − P_A : 0,057 (p12) et 0,124 (p22), contre 0,33–0,44 ailleurs.
- σ₁(N·Δ)/‖N·P_A‖_F : 8,2 et 2,76, contre 0,82 (p17), 0,58 (p5) et 0,45 (p6).

**O6. (négatif, non discriminant)** Le sous-espace S = span{Q^Aᵀ_{k−1}·(entrées de R_k)}, de dimension 54, contient
99,9 % de v₁(N·P_AN), mais aussi 77–99 % de v₁ pour T et A, sur tous les prompts. C'est attendu : tout vecteur tiré
en arrière par la chaîne amont se concentre sur ses directions d'entrée dominantes. **Ce test ne prouve rien** et
n'est pas utilisé.

---

## 3. Conjecture C1 — l'effondrement est une explosion de rang 1 à masse résiduelle conservée

**Énoncé.** (i) ern → 1 ⟺ p₁ → 1 ⟺ b → ∞, avec des bornes explicites. (ii) Pour le gel des normes, la
correction ne peut déplacer qu'au plus 54 valeurs singulières (entrelacement). Si elle est quasi de rang 1, la
masse résiduelle de AN est encadrée par celle de A. (iii) *Empirique :* τ_V/τ_T ∈ [0,7 ; 1,1] pour V ∈ {A, AN}.

### Lemme 1 (entropie ↔ masse dominante) — DÉMONTRÉ

Pour p ∈ Δ_n de plus grande masse p₁ : **1/p₁ ≤ ern ≤ e^{h(p₁)} (n−1)^{1−p₁}**, avec h(t) = −t ln t − (1−t) ln(1−t).

*Preuve.*
- Borne inférieure : H(p) = Σ p_i ln(1/p_i) ≥ Σ p_i ln(1/p₁) = ln(1/p₁).
- Borne supérieure (regroupement) : H(p) = h(p₁) + (1−p₁)·H(q), où q = (p₂,…,p_n)/(1−p₁) ∈ Δ_{n−1} ; puis
  H(q) ≤ ln(n−1). ∎

*Corollaire.* 1 − p₁ ≤ (ern − 1)/ern. Exemple : ern(AN-p12) = 1,021 donne 1 − p₁ ≤ 0,021. Réciproquement,
p₁ → 1 donne ern → 1 (n fixé).

*Vérification.* Les 15 cas satisfont l'encadrement, par exemple AN-p12 : 1,002 ≤ 1,021 ≤ 1,029.

### Proposition 1.1 — DÉMONTRÉ (identité)

p₁ = b²/(1+b²), car ‖M‖_F² = σ₁² + τ². Avec le lemme 1 : b ≥ 2,6 ⟹ p₁ ≥ 0,871 ⟹ ern ≤ 3,8. La réciproque
n'est pas garantie (ern ≤ 3 impose seulement p₁ ≥ 1/3). Empiriquement, sur les 15 cas, ern ≤ 3 ⟺ b ≥ 2,6.
C'est une reformulation. Le contenu empirique est O1 : **b varie par σ₁, pas par τ**.

### Théorème 1 (entrelacement pour le gel des normes) — DÉMONTRÉ sous F1 (vérifiée)

*Hypothèse F1* : rg R_k ≤ 2 pour tout k. Elle est exacte en arithmétique exacte (§4) et vérifiée sur les données :
σ₃/σ₁(R_k) ≈ 3·10⁻⁴, c'est-à-dire le bruit des différences finies.

*Énoncé.* rg(N(P_AN − P_A)) ≤ 54, et pour tout i :
**σ_{i+54}(N P_AN) ≤ σ_i(N P_A)** et **σ_{i+54}(N P_A) ≤ σ_i(N P_AN)**.

*Preuve.*
1. Identité télescopique, par récurrence sur le nombre de facteurs :
   Π_k (A_k + R_k) − Π_k A_k = Σ_k P^AN_{k+1} R_k Q^A_{k−1}.
2. Chaque terme est de rang ≤ rg R_k ≤ 2, donc la somme est de rang ≤ 54.
3. Inégalité de Weyl pour les valeurs singulières : σ_{i+j−1}(X+Y) ≤ σ_i(X) + σ_j(Y). Avec rg Y ≤ r et j = r+1,
   on obtient σ_{i+r}(X+Y) ≤ σ_i(X). On applique ceci dans les deux sens. ∎

*Vérification.* La plus grande valeur de max(σ_{i+54}(AN) − σ_i(A), σ_{i+54}(A) − σ_i(AN)) est négative sur les
5 prompts : aucune violation, même avec le bruit de mesure.

*Portée.* Le gel des normes **ne peut pas** contracter la masse spectrale au-delà du 55ᵉ indice. Un effondrement
AN ne peut venir que de l'explosion d'au plus 54 directions, en pratique une seule.

### Proposition 1.2 (encadrement de τ sous perturbation quasi de rang 1) — DÉMONTRÉ (sans hypothèse)

Si M = X + D₁ + D_r avec rg D₁ ≤ 1, alors **Σ_{i≥3} σ_i(X + D_r)² ≤ τ(M)² ≤ ‖X + D_r‖_F²**.

*Preuve.*
- Majoration : τ(M)² = min_{rg Y ≤ 1} ‖M − Y‖_F² ≤ ‖M − D₁‖_F².
- Minoration : X + D_r = M − D₁, et Weyl avec une perturbation de rang 1 donne σ_{i+1}(X + D_r) ≤ σ_i(M). On
  somme pour i ≥ 2. ∎

*Vérification* (X = N P_A, D₁ = meilleure approximation de rang 1 de N·Δ).

| prompt | minorant ≤ τ_AN ≤ majorant | ‖D_r‖_F/‖N P_A‖_F |
|---|---|---|
| p22 | 21,1 ≤ 22,6 ≤ 26,0 | 0,43 |
| p12 | 66,2 ≤ 74,2 ≤ 171,8 | 0,49 |
| p5 | 18,7 ≤ 23,3 ≤ 26,0 | 0,32 |

L'hypothèse « D_r petit » n'est **qu'à moitié** satisfaite. L'encadrement est serré pour p22 et lâche au-dessus
pour p12 : il explique partiellement, pas complètement, la conservation de τ.

### (iii) Conservation de τ entre A et T — EMPIRIQUE

Il n'y a pas de structure de rang (E_k est de rang ≈ 100–150 par couche). Observé sur les 10 couples :
τ_V/τ_T ∈ [0,77 ; 1,01] (post hoc). **Prédiction pour des prompts neufs :** τ_A/τ_T et τ_AN/τ_T ∈ [0,7 ; 1,1],
et ern ≤ 3 si et seulement si b ≥ 2,6.

### Ce que cela dit de l'hypothèse « Lyapunov / mélange » du brief

- **Réfutée**, la lecture selon laquelle « le gel retire un terme de mélange de haut rang » : le volume spectral hors
  tête (τ) est inchangé.
- **Confirmée**, la signature « écart entre les deux premiers exposants de Lyapunov en temps fini ». L'exposant de
  tête gagne (1/27)·ln(σ₁,AN/σ₁,T) = +0,115 par couche (p12) et +0,056 (p22), les autres restant en place.

**Statut C1 :** (i) démontré ; (ii) démontré (théorème 1 sous F1 vérifiée ; proposition 1.2 sans hypothèse, mais
son hypothèse d'application n'est satisfaite qu'à moitié) ; (iii) empirique, 10/10, post hoc.

---

## 4. Théorème 2 — identité de rétroaction des normaliseurs (exacte) — DÉMONTRÉ

**(a) Normes.** Notons r(y) = (‖y‖²/n + ε)^{−1/2} et β(y) = ‖y‖²/(‖y‖² + nε) (ici 1 − β < 10⁻⁷). Par F1,
J_n^{figé} − J_n^{vrai} = r·diag(γ)·y yᵀ/(‖y‖²+nε) = β·n(y)·ŷᵀ/‖y‖. Posons
- S_k = Σ_h p_ll^h W_O^h W_V^{g(h)} (voie « valeur du token lui-même ») ;
- G_k = ∂m/∂z au point z = n₂(h_k) ;
- a'_k = β₁·S_k n₁(x_k)/‖x_k‖ et m'_k = β₂·G_k n₂(h_k)/‖h_k‖.

Alors

> **R_k = c_k ∇_{x_k}‖x_k‖ᵀ + m'_k ∇_{x_k}‖h_k‖ᵀ**, avec c_k = (I + J_m^A) a'_k + (ĥ_kᵀ a'_k) m'_k,

où ∇_{x_k}‖x_k‖ = x̂_k et ∇_{x_k}‖h_k‖ = (I + J_a^A)ᵀ ĥ_k (gradients le long de l'application A : probabilités
épinglées, normes vivantes).

*Preuve.*
1. J_a^{AN} = J_a^A + a'_k x̂_kᵀ et J_m^{AN} = J_m^A + m'_k ĥ_kᵀ.
2. On développe J^{AN} = I + J_a^{AN} + J_m^{AN}(I + J_a^{AN}) et on soustrait J^A. Il reste
   R_k = a' x̂ᵀ + J_m^A a' x̂ᵀ + m' ĥᵀ(I + J_a^A) + m'(ĥᵀa') x̂ᵀ, qu'on regroupe. ∎

*Vérifications.* x̂_k ∈ span des 2 vecteurs singuliers droits de R_k : ‖V_kᵀ x̂_k‖ ≥ 0,99995 pour les 135 couples
(k, p). Le rang 2 est donné par F1.

**(b) Attention (QK).** Pour la tête h du dernier token l : logits s^h_j, probabilités p^h = softmax(s^h), valeurs
v_j, moyenne attendue v̄^h = Σ_j p^h_j v_j. Alors

> **E_k = (I + J_m) Σ_h W_O^h Σ_j p^h_j (v_j^{g(h)} − v̄^h) ∇_{x_k} s^h_jᵀ**, donc rg E_k ≤ 12 (n_tok − 1).

*Preuve.*
1. J_T = (I + J_m)(I + J_a^T) et J_A = (I + J_m)(I + J_a^A), avec le même J_m.
2. J_a^T − J_a^A = Σ_h W_O^h Σ_j v_j dp^h_j/dx, car la variante A ne garde que p_ll dv_l.
3. dp^h = (diag p − p pᵀ) ds^h, et V_hᵀ(diag p − p pᵀ) = Σ_j p_j (v_j − v̄) e_jᵀ.
4. rg(diag p − p pᵀ) ≤ n_tok − 1 (car 1 est dans le noyau). ∎

*Vérification.* Rang numérique de E_k (σ_i/σ₁ > 10⁻³) : 82–149 ≤ 156 pour p12, 87–125 ≤ 132 pour p22. Au-delà,
σ/σ₁ ≈ 4·10⁻⁵, niveau du bruit (`wind_T6_qkrank_results.txt`). La borne est respectée et non vide.

**(c) Télescopage.**
- P_AN − P_A = Σ_k P^AN_{k+1} [c_k ∇_{x_1}‖x_k‖ᵀ + m'_k ∇_{x_1}‖h_k‖ᵀ].
- P_T − P_A = Σ_{k,h,j} P^T_{k+1}(I + J_m) W_O^h p^h_j (v_j − v̄^h) ∇_{x_1} s^h_{j,k}ᵀ.

Les gradients sont pris par rapport à x_1 le long de l'application A. Les portes numériques (identité recalculée
terme à terme) sont à 10⁻¹⁵.

*Lecture.* **L'erreur d'une linéarisation figée = Σ (réponse aval à un changement du normaliseur) ⊗ (gradient du
normaliseur).** C'est la règle de la chaîne sur le graphe de calcul : exact mais élémentaire.

*Localisation (post hoc, O2/O3).* Les termes qui comptent sont aux couches 2–8 (normes) et 6–9 (QK, p12).

---

## 5. Conjecture C2 (normes figées) — rétroaction positive d'échelle, cohérente sur les couches précoces

### Proposition 3 (structure d'Euler exacte des colonnes de R_k) — DÉMONTRÉ (sans hypothèse)

Pour le SwiGLU m(z) = W_d(silu(W_g z) ⊙ W_u z), avec g = W_g z et u = W_u z :

> **G(z)·z = 2 m(z) + η(z)**, avec η(z) = W_d[(g² σ(g) σ(−g)) ⊙ u] et 0 ≤ g²σ(g)σ(−g) ≤ φ* ≈ 0,4392.

*Preuve.*
1. G z = W_d[silu'(g) ⊙ g ⊙ u + silu(g) ⊙ u].
2. g·silu'(g) = gσ(g) + g²σ(g)(1 − σ(g)) = silu(g) + g²σ(g)σ(−g).
3. Comme σ(g)σ(−g) = 1/(4 cosh²(g/2)), on a φ(g) = (g / (2 cosh(g/2)))².
4. Ce maximum est atteint pour g·tanh(g/2) = 2, soit g* ≈ 2,3994 et φ* ≈ 0,4392. ∎

Pour l'attention épinglée à norme figée, J_a^{fr} x = S_k n₁(x) =: ã_k, la contribution « valeur propre du
dernier token » (hors biais V). Donc m'_k = β₂ (2 m_k + η_k)/‖h_k‖ et a'_k = β₁ ã_k/‖x_k‖, d'où
**col(R_k) ⊆ span{(I + J_m) ã_k, 2 m_k + η_k}**. La colonne dominante est la mise à jour MLP de la couche elle-même,
doublée, à un défaut borné près. Avec un GLU-ReLU on aurait η ≡ 0 : homogénéité exacte de degré 2.

### Énoncé de la conjecture C2

Le long de toute direction d'entrée v, le théorème 2 donne exactement

  N(P_AN − P_A)v = Σ_k N P^AN_{k+1} [c_k δ_v‖x_k‖ + m'_k δ_v‖h_k‖],

où δ_v‖·‖ est la dérivée directionnelle de la norme du résiduel. Autrement dit, **la linéarisation figée transforme
une fluctuation de norme en une ré-injection de la propre sortie de la couche (≈ 2 m_k)**, là où la vraie norme
l'annulerait. C2 affirme :
- (i) pour la direction d'explosion, les contributions des couches 2–8 ont un signe cohérent (interférence
  constructive, et non une somme de signes aléatoires) ;
- (ii) la direction de sortie de l'explosion est la réponse de la sortie normalisée à une mise à l'échelle commune
  des mises à jour précoces, z = N Σ_{k≤10} P^AN_{k+1} Δ_k ;
- (iii) l'effondrement a lieu quand cette somme cohérente dépasse la masse résiduelle (b ≥ 2,6).

### Prédictions pré-enregistrées et résultats (`wind_T6_signflip_results.txt`)

- **P6.3 (colonnes de R_k ∋ mise à jour Δ̂_k), seuil : médiane ≥ 0,5 — VRAI.**
  - Médianes de ‖U_kᵀΔ̂_k‖² : p5 0,696 · p6 0,674 · p12 0,738 · p17 0,718 · p22 0,706 (hasard 0,0013).
  - Plus faible aux couches 1–5 : par exemple p17 k=5 : 0,00 ; p22 k=2 : 0,06 et k=5 : 0,05. C'est attendu, puisque
    la part « valeurs des autres tokens » de la mise à jour d'attention n'est pas dans col(R_k).
  - Couches 11–27 : 0,72–0,79.
- **P6.4 (|cos(u₁(N·P_AN), z)| ≥ 0,7 sur p12 et p22) — VRAI : 0,998 et 0,995** (hasard 0,026).
  - Descriptif : p5 0,684 · p6 0,669 · p17 0,917. L'alignement est donc élevé partout. Il caractérise la direction
    de la correction d'échelle, **pas** l'effondrement lui-même.
- **P6.2 (AN extrême sous le nul par retournement des signes de R_k) — VRAI.**
  - p12 : q(σ₁) = 0,975 et q(ern) = 0,025, **juste au seuil** (1 tirage sur 40 le dépasse).
  - p22 : q(σ₁) = 1,000 et q(ern) = 0,000, au-delà des 40 tirages. Le nul donne ern médiane 21,2 [5,8 ; 34,0]
    contre 1,83 observé, et σ₁ ≤ 35,6 contre 75,9 observé.
  - p5 : q(ern) = 0,225 ; p6 : 0,500 (typiques, comme prédit).
  - p17 (descriptif) : q(σ₁) = 0,975 et q(ern) = 0,05. **La cohérence de signe est présente aussi sur p17, qui ne
    s'effondre pas.**

### Ce que les résultats disent

- (i) Cohérence : soutenue. AN est extrême sous le nul sur p12, p22 et p17, typique sur p5 et p6. Mais la cohérence
  seule ne suffit pas : p17 l'a, avec b = 0,96.
- (ii) Direction : soutenue sur p12 et p22 (0,995–0,998), mais non spécifique (0,67–0,92 ailleurs).
- (iii) Seuil : c'est une tautologie (proposition 1.1), tant que l'amplitude n'est pas prédite indépendamment.
- Ordre de Walsh : le premier ordre P_A + Σ_k P^A_{k+1} R_k Q^A_{k−1} reproduit l'effondrement de p12 (ern 1,21 vs
  AN 1,02) mais pas entièrement celui de p22 (8,74 vs 1,83). Sur p22, les interactions entre couches gelées (la
  chaîne aval AN elle-même) comptent.

**Statut C2 :** la proposition 3 est démontrée (exacte). La lecture « colonnes ≈ mises à jour » est vérifiée
partiellement (≈ 70 %, plus faible aux couches 1–5). Le mécanisme de cohérence est **heuristique, soutenu** par
P6.2 et P6.4. Il n'explique **pas** l'amplitude, donc pas le choix p12/p22 contre p17.

---

## 6. Conjecture C3 (attention figée) — le terme QK compense spécifiquement la direction amplifiée par A

### Théorème 3 (ensemble à signes retournés) — DÉMONTRÉ

Soit P(ε) = Π_{k=27..1}(A_k + ε_k E_k), avec ε_k des variables de Rademacher i.i.d., et
W_S = Π_k (E_k si k ∈ S, A_k sinon), produit ordonné. Alors :
1. P(ε) = Σ_{S⊆[27]} ε^S W_S ;
2. **E P(ε) = P_A** ;
3. pour tous X, Y : E‖X P(ε) Y‖_F² = Σ_S ‖X W_S Y‖_F² ;
4. **E σ₁(X P(ε)) ≥ σ₁(X P_A)** ;
5. pour v fixé : E‖X P(ε) v‖² ≥ ‖X P_A v‖².

*Preuve.*
- (1) Développement multilinéaire.
- (2) et (3) : E ε^S ε^{S'} = δ_{SS'} (orthonormalité des caractères de Walsh).
- (4) Jensen, σ₁ étant convexe.
- (5) Cas particulier de (3). ∎

(Même énoncé avec R_k.) La chaîne à attention figée A est donc **exactement la moyenne** de l'ensemble où l'on tire
au hasard le signe de chaque terme QK. Ce nul est sans biais de sélection : la direction et la statistique sont les
mêmes pour tous les tirages.

### Prédiction P6.1 et résultat — FAUX

*Prédiction :* p12 : q_T(ern) ≥ 97,5 % et q_T(σ₁) ≤ 2,5 %. Autres prompts : q_T(ern) ∈ (2,5 % ; 97,5 %).

*Observé :*
- p12 : q_T(ern) = **0,925** et q_T(σ₁) = **0,075**, en dehors du seuil fixé. Le nul donne ern médiane 3,97
  [1,36 ; 16,29] contre 13,11 pour T.
- Autres prompts : q_T(ern) = 0,775 (p5), 0,725 (p6), 0,775 (p17), 0,95 (p22). Ces valeurs sont dans l'intervalle
  prédit.

Lecture honnête :
1. **Sur p12, presque toute chaîne voisine de A s'effondre** (ern médiane 3,97, minimum 1,36). La propension à
   l'effondrement est une propriété du voisinage de la chaîne moyenne A, pas une anomalie de A.
2. T est seulement dans la queue haute. **Post hoc** (non pré-enregistré), T est au-dessus de la médiane du nul pour
   ern sur les 5 prompts (q ∈ [0,725 ; 0,95]), et en dessous pour σ₁ (q ∈ [0,075 ; 0,375]). Sous un nul uniforme,
   P(5/5 > 0,7) ≈ 0,002. C'est une tendance anti-effondrement faible mais constante, qui devra être confirmée sur des
   prompts neufs.
3. Le premier ordre de Walsh ne reproduit pas T : P_A + Σ_k W_{k} donne ern 1,86 sur p12 contre 13,11 pour T. La
   protection, là où elle existe, est un effet d'**interaction entre couches**. Un terme QK seul (couche 6, O2)
   suffit à relever ern de 2,3 à 10,1, mais la somme des termes de premier ordre de toutes les couches ne le fait pas.
4. Les nuls QK contiennent souvent des chaînes effondrées sur des prompts où rien ne s'effondre : minimum 2,44 (p5)
   et 2,11 (p17). Le rang effectif de ces chaînes est très sensible aux signes des termes QK.

**Statut C3 :** le théorème 3 est démontré. **La conjecture forte (compensation spécifique) est réfutée au seuil
pré-enregistré.** La forme faible (tendance anti-effondrement de T) est post hoc et reste à confirmer.

---

## 7. Pourquoi p12/p22 ? Pourquoi les couches 1–10 ?

**p12/p22 (normes).** Les trois ingrédients mesurés sont :
- la correction quasi de rang 1 (O5) ;
- la cohérence de signe (P6.2) ;
- l'amplitude σ₁(N·Δ)/‖N·P_A‖_F = 8,2 / 2,76 contre ≤ 0,82.

Seule l'amplitude sépare p12/p22 de p17. **Son déterminant n'est pas identifié.** Pistes testées sans succès :
- la force du pic précoce (p17 a le même pic J_1 que p12) ;
- la taille relative des mises à jour ;
- la fraction de v₁ dans l'espace des gradients de norme (O6, non discriminante).

Descriptif, post hoc, 5 points, aucune valeur probante : l'excès du gain d'état AN,
log₁₀ Λ_AN − log₁₀(‖x_28‖/‖x_1‖), vaut +1,26 (p12), +0,15 (p22), −0,06 (p5), −0,06 (p6) et −0,11 (p17). Il
range les deux prompts effondrés en tête. **Ouvert.**

**p12 (attention).** Ce qui est acquis :
- le voisinage QK de la chaîne A est « effondrable » (§6) ;
- A contient deux pics précoces (J_1 : σ₁ 14,4 et srang 3,3 ; J_5 : 10,8 et 7,2) visibles de l'aval (α ≈ 0,14 contre
  0,05) ;
- les termes QK des couches 6–7 de T masquent cette visibilité (O2, O3).

**Pourquoi p12 : ouvert.**

**Couches 1–10.** Pour la couche k, la contribution le long de v est le produit d'un facteur amont (δ_v‖x_k‖) et
d'un facteur aval (N·P^AN_{k+1}·c_k).
- Pour k = 1, le facteur amont vaut ⟨x̂_1, v⟩, qui est petit : |cos(v₁, x̂_1)| ≈ 0,19–0,20.
- Pour les couches profondes, la chaîne aval est courte et les termes d'Euler relatifs sont plus petits.

Le produit culmine aux couches 3–8, ce qui est cohérent avec les profils du télescopage. **Statut : heuristique, non
testé quantitativement** (il faudrait factoriser les deux facteurs couche par couche).

---

## 8. Récapitulatif des statuts

| # | Énoncé | Statut | Chiffres clés |
|---|---|---|---|
| L1 | 1/p₁ ≤ ern ≤ e^{h(p₁)}(n−1)^{1−p₁} | démontré | 15/15 cas dans l'encadrement |
| P1.1 | p₁ = b²/(1+b²) ; b ≥ 2,6 ⟹ ern ≤ 3,8 | démontré (identité + L1) ; « ern ≤ 3 ⟺ b ≥ 2,6 » empirique (15/15) | effondrés : b = 22,6 / 3,35 / 2,62 ; autres ≤ 0,97 |
| T1 | entrelacement σ_{i+54} (AN vs A) | démontré sous F1 (vérifiée) | violation max ≤ 0 |
| P1.2 | encadrement de τ sous perturbation quasi de rang 1 | démontré ; hypothèse d'application à moitié satisfaite | p22 : 21,1 ≤ 22,6 ≤ 26,0 ; p12 lâche |
| C1(iii) | τ conservée entre variantes | empirique, post hoc, 10/10 | τ_V/τ_T ∈ [0,77 ; 1,01] |
| T2 | identité de rétroaction (normes rang ≤ 2, QK rang ≤ 12(n−1)) | démontré (règle de la chaîne) | ‖V_kᵀx̂_k‖ ≥ 0,99995 ; rang E_k 82–149 ≤ 156 |
| P3 | Euler SwiGLU : Gz = 2m + η, 0 ≤ défaut ≤ 0,4392 | démontré | — |
| C2 | rétroaction positive d'échelle cohérente (normes) | heuristique soutenue ; P6.2, P6.3, P6.4 VRAIS | p22 hors de 40/40 tirages ; cos(u₁, z) = 0,995–0,998 ; mais p17 cohérent sans effondrement |
| T3 | ensemble de Walsh : E P(ε) = P_A, Parseval, Jensen | démontré | — |
| C3 | compensation QK spécifique (A-p12) | **réfutée au seuil** (P6.1 FAUX) | q_T(ern) = 0,925, q_T(σ₁) = 0,075 ; nul médiane 3,97 |
| — | ce qui fixe l'amplitude (p12/p22 contre p17) | ouvert | σ₁(NΔ)/‖NP_A‖_F = 8,2 / 2,76 contre ≤ 0,82 |

Aucun théorème ne repose sur une hypothèse fausse pour Qwen. F1 est vérifiée. La proposition 1.2 est
inconditionnelle ; son hypothèse d'application (D_r petit) n'est satisfaite qu'à moitié, et c'est dit.

---

## 9. Verdict

**Ce qui explique l'effondrement.** Les linéarisations figées ne perdent pas de « mélange » : la masse spectrale hors
tête est conservée à ±25 %. Elles **libèrent une seule direction** que la vraie Jacobienne ne laisse pas croître.

Pour le gel des normes, le mécanisme est identifié et exact :
- l'erreur est une somme de termes de rang 2, (réponse aval) ⊗ (gradient de ‖x_k‖ ou de ‖h_k‖) ;
- les colonnes de ces termes sont, par Euler, la propre sortie de la couche doublée ;
- sur p12 et p22, ces ré-injections s'additionnent avec un signe cohérent sur les couches 2–8, en une direction de
  sortie unique ;
- le nul par signes aléatoires exclut le hasard (p22 : hors des 40 tirages).

C'est une **rétroaction positive d'échelle** : la linéarisation « graphes d'attribution / LRP(AH+LN) » traite une
variation de norme du résiduel comme une variation de contenu et la ré-amplifie à chaque couche. Conséquence
pratique : sur ces prompts, sa direction d'attribution dominante est un artefact d'échelle, pas une direction de
contenu. Les erreurs relatives de 20,3 et 2,6 de WIND-T4 viennent de là.

Pour l'attention figée (A-p12), l'effet QK est réel au niveau causal. Les hybrides montrent que la couche 6 seule fait
passer ern de 2,3 à 10,1. Mais **ce n'est pas une compensation spécifique** au sens pré-enregistré : presque toutes
les chaînes voisines de A s'effondrent, et T n'est qu'au 92,5ᵉ centile.

**Ce qui reste ouvert.**
1. Ce qui fixe l'amplitude de la direction libérée : pourquoi p12/p22 et pas p17, qui a la même cohérence.
2. Pourquoi le voisinage QK de p12 est effondrable.
3. La factorisation amont × aval qui localiserait quantitativement les couches 1–10.
4. Toute confirmation sur des prompts neufs : tout ici est exploratoire, sur les 5 prompts qui ont révélé l'effet.

**Non-trivialité mathématique : non.** Tous les énoncés démontrés sont courts et standards :
- identité de rétroaction : règle de la chaîne ;
- entrelacement : Weyl ;
- encadrement de τ : Eckart–Young et Weyl ;
- ensemble de Walsh : multilinéarité et Jensen ;
- borne d'Euler du SwiGLU : calcul d'une ligne, maximum 0,4392.

La combinaison (identité exacte + borne d'Euler + entrelacement de rang 54) donne un mécanisme **vérifiable et
falsifiable**, ce qui est utile. Ce n'est pas un théorème nouveau. Le contenu non trivial est empirique : conservation
de τ, cohérence de signe hors du nul, alignement de la direction d'explosion sur z à 0,995–0,998.

**Prédictions pour la confirmation sur prompts neufs** (à pré-enregistrer telles quelles) :
1. τ_A/τ_T et τ_AN/τ_T ∈ [0,7 ; 1,1] ;
2. ern ≤ 3 ⟺ b ≥ 2,6 ;
3. si AN s'effondre, alors q_AN(σ₁) ≥ 0,975 sous le nul des signes, et |cos(u₁, z)| ≥ 0,9 ;
4. T au-dessus de la médiane du nul QK en ern sur ≥ 80 % des prompts.

---

## 10. Fichiers

- Scripts : `wind_T6_explore.jl`, `wind_T6_hybrid.jl`, `wind_T6_normgrad.jl`, `wind_T6_cancel.jl`,
  `wind_T6_signflip.jl`, `wind_T6_qkrank.jl`.
- Résultats : `wind_T6_explore_results_p12.txt`, `wind_T6_explore_results_rest.txt`,
  `wind_T6_hybrid_results_p12.txt`, `wind_T6_hybrid_results_rest.txt`, `wind_T6_normgrad_results.txt`,
  `wind_T6_cancel_results.txt`, `wind_T6_signflip_results.txt`, `wind_T6_qkrank_results.txt` (+ journaux `.log`).
- Données dérivées : `wind_data/wind_T6_explore_p12.json`, `wind_data/wind_T6_explore_rest.json`,
  `wind_data/wind_T6_signflip.json`.
- Pré-enregistrement : section « WIND-T6 » ajoutée à la fin de `wind_T3_preregistration.md`.
