# WIND-T7 — La résonance de la boucle des normes : un théorème de dédoublement et le gain critique

*2026-10-01. Qwen2.5-1.5B-Instruct, Jacobiennes WIND-1 (dernier token). EXPLORATOIRE : mêmes 5 prompts que pour la
découverte. Les prédictions P7.1–P7.3 ont été écrites dans `wind_T3_preregistration.md` (section WIND-T7,
16:57:50Z) avant le script de test `wind_T7_sweep.jl` (16:58:36Z).*

---

## 0. En tête : ce qui est démontré, ce qui est mesuré

**Théorème 1 (non trivial) — dédoublement résonant non asymptotique.**
Cadre : une boucle de rétroaction linéaire variant dans le temps, de dimension d'état r = 2.
- Son opérateur entrée-sortie G est **exactement** la somme d'un terme de rang 1 et d'un reste.
- Le terme de rang 1 a pour norme A = ‖ĉ‖‖b̂‖ (produit cumulé des gains instables).
- Le reste a une norme β, explicitement bornée par des tests de Schur.
- Conséquence : **|σ₁(G) − A| ≤ β et σ₂(G) ≤ β**. Il y a un seul mode résonant dès que A ≫ β.

**Théorème 2 (non trivial, limite à constante exacte) — cas stationnaire.**
- **σ₁(G_L)·|λ_u|^{−L} → ‖Ce‖‖fᵀB‖/(λ_u² − 1)** et sup_L σ₂(G_L) < ∞.
- Le seuil est ρ(Φ + θBC) = 1 (rayon spectral de la boucle fermée).
- Contrôle synthétique : constante vérifiée à 7 chiffres.

**Lemmes (exacts, classiques)** : relèvement exact du produit gelé, déterminant 1, réponse en boucle fermée,
factorisation de Hankel.

**Ce que cela explique sur Qwen.** L'amplitude de l'effondrement AN est fixée par la **distance à la criticité
d'une boucle de rétroaction des normes effectivement de dimension 2**.

- *Réduction d'ordre 2 :* une réalisation d'ordre 2, ajustée sur W seul, reproduit σ₁ de la résolvante à 1–8 %
  (402,6 / 15,9 / 6,43 / 4,47 / 3,98 contre 398 / 14,6 / 6,49 / 4,46 / 3,98).
- *Gain critique (P7.1, pré-enregistré, VRAI) :* le modèle 2-D prédit le gain de rétroaction critique θ_c, mesuré
  ensuite sur les vrais produits 1536×1536.

  | | p22 | p17 | p5 | p6 |
  |---|---|---|---|---|
  | θ_c mesuré | **0,833** | **1,399** | 1,630 | 1,679 |
  | θ_c prédit | 0,851 | 1,546 | 1,621 | 1,646 |
  | erreur | 2 % | 9,5 % | 0,6 % | 2 % |

  **p22 s'effondre parce que sa boucle est sur-critique (θ_c < 1). p17 ne s'effondre pas parce qu'il faudrait
  40 % de rétroaction en plus.** Les deux ont la cohérence de signe (WIND-T6) ; ce qui les sépare, c'est la marge
  à la criticité.
- *Mode unique (théorème 1 sur le modèle) :* certifié pour p12 (A/β = 166) et p22 (A/β = 3,66), non certifié pour
  p17, p5 et p6 (A/β ≤ 1,94).

**Échecs pré-enregistrés.**
- P7.2 (fidélité des courbes, facteur 1,5) échoue pour p17 seul (écart 0,495 > 0,405).
- P7.3 (mode unique à θ = 2) échoue pour p17, p5 et p6 : un second mode apparaît. C'est cohérent avec la perte de
  l'hypothèse de dichotomie : le second exposant de Lyapunov remonte vers 0.

**Limites (honnêtes).**
- Le théorème 1 s'applique exactement au modèle d'ordre 2. Le passage modèle → vérité est **empirique**, non
  démontré. Exemple : σ₁(G) vrai de p12 = 402,55, juste hors de l'encadrement prouvé du modèle [395,98 ; 400,79].
- Le théorème 2 suppose la stationnarité, **fausse sur Qwen** : c'est l'idéalisation à profondeur infinie, vérifiée
  sur données synthétiques seulement.
- L'origine de l'ordre 2 n'est pas démontrée (h_k n'est pas stocké).
- Le cas A-p12 (QK) et l'entonnoir de T ne sont pas traités.

---

## 1. Cadre

On dispose, par couche k = 1..L (L = 27) :
- A_k = J_k^A (normes vivantes, attention épinglée) ;
- R_k = J_k^AN − J_k^A = U_kV_kᵀ, de rang 2 (WIND-T6 : σ₃/σ₁ ≈ 3·10⁻⁴).

V_k engendre exactement {∇‖x_k‖, ∇‖h_k‖} : ce sont les **capteurs de norme**. U_k engendre les injections
(théorème 2 de WIND-T6 ; ‖V_kᵀx̂_k‖ ≥ 0,99995). Pour le gain θ ∈ ℝ, on pose J_k(θ) = A_k + θR_k (θ = 0 : vrai,
θ = 1 : AN). N est la vraie norme finale.

Transports : Φ_θ(j,k) = J_{j−1}(θ)···J_k(θ), avec Φ(k,k) = I. Objets relevés (taille pL = 54) :

  𝒰_k = N Φ_0(L+1, k+1) U_k,   𝒱_k = V_kᵀ Φ_0(k, 1),   W_{jk} = V_jᵀ Φ_0(j, k+1) U_k (j > k), 0 sinon.

W est la **réponse impulsionnelle « injection de norme → capteur de norme » de la chaîne vivante**.

---

## 2. Lemmes exacts (classiques)

**Lemme A (relèvement exact).** Pour tout θ :

  N·Φ_θ(L+1, 1) = M_A + θ·𝒰 (I − θW)⁻¹ 𝒱,   avec M_A = N·Φ_0(L+1, 1).

I − θW est unipotente triangulaire inférieure par blocs, donc det = 1 et (I − θW)⁻¹ = Σ_{m=0}^{L−1} θ^m W^m (somme
finie). De plus, pour j > k :

  G(θ)_{jk} := [(I − θW)⁻¹]_{jk} = θ V_jᵀ Φ_θ(j, k+1) U_k.

G est donc la **réponse de la boucle fermée** (la chaîne figée lue par les mêmes capteurs).

*Preuve.*
1. On développe Π_k (A_k + θU_kV_kᵀ). Un terme est fixé par l'ensemble k₁ < … < k_m des facteurs θUVᵀ choisis ; il
   vaut θ^m 𝒰_{k_m} W_{k_m k_{m−1}} ··· W_{k₂k₁} 𝒱_{k₁}.
2. W étant strictement triangulaire, (W^{m−1})_{k_m k₁} est exactement la somme sur les chaînes croissantes. Donc
   la somme vaut θ^m 𝒰W^{m−1}𝒱, et l'on somme sur m.
3. Pour G_{jk}, on regroupe de même les chaînes entre k et j : Σ_{S⊆(k,j)} Π (A_i ou θU_iV_iᵀ) = Φ_θ(j, k+1). ∎

*Vérification.* Porte ‖relèvement − produit vrai‖/‖produit vrai‖ ≤ 3,6·10⁻⁴ pour θ ∈ [0,25 ; 2,5] et les
5 prompts (`wind_T7_sweep_results.txt`) : c'est le bruit des différences finies.

**Lemme B (Hankel / Kronecker).** Pour la coupure i, G_{(>i),(≤i)} = (I − W₂₂)⁻¹ H_i (I − W₁₁)⁻¹, avec
H_i = W_{(>i),(≤i)}. Donc rang G_{(>i),(≤i)} = rang H_i.

*Preuve.* Inverse par blocs d'une matrice triangulaire inférieure. ∎

*Observation (`wind_T7_resolvent_probe.jl`).* Pour p12, les blocs de Hankel de W sont de rang ≈ 1 à 6–12 % près,
mais ceux de G sont de rang 1 **à 10⁻² près** et de taille ≈ 400 à toutes les coupures. Les sous-résolvantes
amplifient une seule direction : c'est la « purification ».

Ces deux lemmes sont des faits classiques (relèvement d'un système linéaire variant dans le temps ; matrices
semi-séparables). Ils ne sont pas revendiqués comme nouveaux.

---

## 3. Théorème 1 — dédoublement résonant non asymptotique

**Hypothèse de forme (réalisation d'ordre r = 2).** G = I + K, avec K triangulaire inférieur strict par blocs
(p×p, ici p = 2) et

  K_{jk} = C_j M(j, k+1) B_k pour 1 ≤ k < j ≤ L,   M(j,k) = M_{j−1}···M_k,

où C_j ∈ ℝ^{p×2}, B_k ∈ ℝ^{2×p} et M_i ∈ GL₂(ℝ) (on pose C_1 = 0 et B_L = 0).

Soit un **scindement invariant quelconque** : vecteurs unitaires e_i, s_i, linéairement indépendants, avec
M_i e_i = μ_i e_{i+1} et M_i s_i = ν_i s_{i+1}. Il en existe toujours : il suffit de propager deux vecteurs
indépendants. On note (f_i, g_i) la base duale (f_iᵀe_i = 1, f_iᵀs_i = 0, etc.). On pose :
- m_j = Π_{i=2}^{j−1} μ_i ;
- ĉ_j = m_j C_j e_j ∈ ℝ^p et b̂_k = m_{k+1}⁻¹ f_{k+1}ᵀ B_k ∈ ℝ^{1×p} ;
- H = ĉ b̂, de **rang 1** ;
- U, la partie « anti-causale » de H : blocs (j,k) avec j ≤ k ;
- K^s_{jk} = C_j M(j,k+1) s_{k+1} g_{k+1}ᵀ B_k pour j > k.

**Énoncé.**
- (a) G = H + Y exactement, avec Y = I − U + K^s.
- (b) |σ₁(G) − ‖ĉ‖‖b̂‖| ≤ β et σ₂(G) ≤ β, où β = ‖Y‖₂.
- (c) β ≤ 1 + √(R_u C_u) + √(R_s C_s), où R, C sont les maxima des sommes de lignes et de colonnes des normes de
  blocs :
  - pour U : ‖C_je_j‖·‖f_{k+1}ᵀB_k‖·Π_{i=j}^{k}|μ_i|⁻¹ (j ≤ k) ;
  - pour K^s : ‖C_js_j‖·‖g_{k+1}ᵀB_k‖·Π_{i=k+1}^{j−1}|ν_i| (j > k).
- (d) Pour tous k < j : ‖ĉ‖‖b̂‖ ≥ ‖C_je_j‖·‖f_{k+1}ᵀB_k‖·Π_{i=k+1}^{j−1}|μ_i|.
- (e) **Uniformité en L (dichotomie).** Si |μ_i| ≥ μ > 1, |ν_i| ≤ ν < 1 et les quatre facteurs de couplage sont
  bornés par c_u b_u et c_s b_s, alors
  **β ≤ 1 + c_u b_u/(μ − 1) + c_s b_s/(1 − ν)** (indépendant de L),
  tandis que ‖ĉ‖‖b̂‖ ≥ ‖C_Le_L‖·‖f_3ᵀB_2‖·μ^{L−3}.
  Une seule valeur singulière croît alors exponentiellement, les autres restent bornées. Si au contraire la boucle
  fermée est contractante (‖M(j,k)‖ ≤ κν^{j−k}, ν < 1), alors ‖G‖ ≤ 1 + κ·max‖C‖·max‖B‖/(1 − ν) : pas de
  résonance.

**Preuve.**
- (a) On décompose B_k = e_{k+1}(f_{k+1}ᵀB_k) + s_{k+1}(g_{k+1}ᵀB_k). Par invariance,
  M(j,k+1)e_{k+1} = (Π_{i=k+1}^{j−1}μ_i)e_j = (m_j/m_{k+1})e_j. Donc K_{jk} = ĉ_j b̂_k + K^s_{jk} pour j > k,
  c'est-à-dire K = (H − U) + K^s, puis G = I + K = H + (I − U + K^s).
- (b) Inégalités de Weyl pour les valeurs singulières : σ_{i+j−1}(X+Y) ≤ σ_i(X) + σ_j(Y). Avec X = H (rang 1,
  σ₁(H) = ‖ĉ‖‖b̂‖) :
  - σ₂(G) ≤ σ₂(H) + σ₁(Y) = β ;
  - σ₁(G) ≤ σ₁(H) + β ;
  - σ₁(H) = σ₁(G − Y) ≤ σ₁(G) + β.
- (c) Test de Schur pour les matrices par blocs : ‖Z‖ ≤ (max_j Σ_k ‖Z_{jk}‖ · max_k Σ_j ‖Z_{jk}‖)^{1/2}. On
  l'applique avec ‖U_{jk}‖ = ‖ĉ_j‖‖b̂_k‖ = ‖C_je_j‖‖f_{k+1}ᵀB_k‖·|m_j/m_{k+1}|, où m_j/m_{k+1} = Π_{i=j}^{k}μ_i⁻¹
  pour j ≤ k. Pour K^s, on utilise M(j,k+1)s_{k+1} = (Π ν_i)s_j.
- (d) La norme d'un bloc d'une matrice de rang 1 est majorée par sa norme.
- (e) Sommes géométriques : Σ_{k≥j} μ^{−(k−j+1)} = 1/(μ−1) et Σ_{d≥0} ν^d = 1/(1−ν). Le cas contractant est un
  test de Schur direct sur K. ∎

**Ce qui le rend non trivial.** Le point clé est une **construction** : la normalisation par m_j transforme le
noyau causal à croissance exponentielle en **masque triangulaire d'une matrice de rang 1**, dont la partie
anti-causale U est contrôlée par l'inverse du cocycle. Ce n'est pas une inégalité appliquée une fois : l'énoncé (e)
est une dichotomie uniforme en L (un seul mode exponentiel contre aucun), avec constantes explicites.

**Corollaire 1 (amplitude de l'effondrement).** Par le lemme A, M_AN(θ) = Bulk + θ(𝒰ĉ)(b̂𝒱), avec
Bulk = M_A + θ𝒰Y𝒱. Comme τ(M) ≤ ‖M − rang 1‖_F (Eckart–Young),

  b(M_AN) ≥ (θ‖𝒰ĉ‖‖b̂𝒱‖ − ‖Bulk‖₂) / ‖Bulk‖_F.

Combiné au lemme 1 de WIND-T6 : si le terme résonant domine le Bulk, alors p₁ → 1 et ern → 1. (Conséquence directe,
statut : lemme.)

---

## 4. Théorème 2 — limite stationnaire à constante exacte et seuil de criticité

**Énoncé.** Supposons C_j ≡ C, B_k ≡ B et M_i ≡ M, avec des valeurs propres réelles λ_u, λ_s telles que
|λ_u| > 1 > |λ_s|. Notons e, s les vecteurs propres unitaires et f, g la base duale, et supposons Ce ≠ 0 et
fᵀB ≠ 0. Alors, quand L → ∞ :

  **σ₁(G_L)·|λ_u|^{−L} → ‖Ce‖·‖fᵀB‖ / (λ_u² − 1)**   et   **sup_L σ₂(G_L) ≤ 1 + ‖Ce‖‖fᵀB‖/(|λ_u|−1) + ‖Cs‖‖gᵀB‖/(1−|λ_s|)**.

Si |λ_u|, |λ_s| < 1, alors sup_L ‖G_L‖ < ∞. Pour la boucle à gain θ (M(θ) = Φ + θBC, Φ stable), la résonance
apparaît exactement quand ρ(Φ + θBC) franchit 1. C'est le seuil θ_∞ à profondeur infinie, et
(1/L) ln σ₁(G_L(θ)) → ln ρ(M(θ)).

**Preuve.**
1. On applique le théorème 1 avec le scindement propre (invariant), μ_i = λ_u, ν_i = λ_s, m_j = λ_u^{j−2}.
2. On a ‖ĉ‖² = ‖Ce‖² Σ_{j=2}^{L}|λ_u|^{2(j−2)} et ‖b̂‖² = ‖fᵀB‖² Σ_{k=1}^{L−1}|λ_u|^{−2(k−1)}. Le produit vaut
   ‖Ce‖²‖fᵀB‖²·λ_u²(|λ_u|^{L−1} − |λ_u|^{−(L−1)})²/(λ_u² − 1)². Donc
   ‖ĉ‖‖b̂‖ = ‖Ce‖‖fᵀB‖·|λ_u|^{L}(1 − |λ_u|^{−2(L−1)})/(λ_u² − 1).
3. Par (e), β est borné uniformément en L ; donc σ₁(G_L) = ‖ĉ‖‖b̂‖ + O(1), d'où la limite.
4. Le cas contractant est le dernier point de (e).
5. Seuil : par le lemme du déterminant matriciel, det(I − θW(z)) = det(I − z(Φ+θBC))/det(I − zΦ). Le nombre de
   modes résonants égale donc le nombre de valeurs propres de la boucle fermée hors du disque unité : c'est l'indice
   du symbole de Toeplitz. ∎

*Contrôle synthétique* (`wind_T7_synthetic_check_results.txt`, M de valeurs propres 1,25 et −0,55, C et B
aléatoires) :
- constante prédite 2,060886 ; mesurée 2,056814 (L = 20), puis 2,060885 (L = 40), puis 2,060886 (L = 60, 80) ;
- σ₂ = 6,25 → 6,89, toujours ≤ β (6,25 → 6,91) ≤ borne uniforme 13,20.

Au-delà de L ≈ 100, σ₁ > 10¹⁵ rend la SVD Float64 de σ₂ non fiable ; cette zone est exclue.

**Hypothèses sur Qwen : NON satisfaites.** La boucle est non stationnaire (μ_i varie de 0,08 à 1,85). Le théorème 2
est l'idéalisation à profondeur infinie qui donne le sens du « gain critique ». Il ne compte pas comme vérifié sur
Qwen (critère 4).

---

## 5. Vérification des hypothèses sur Qwen

**(H1) Réalisation d'ordre 2 de W.** Méthode de Ho–Kalman variant dans le temps : SVD des blocs de Hankel de W,
`wind_T7_realization_probe.jl` et `wind_T7_model.jl`.

| prompt | erreur ‖W₂ − W‖/‖W‖ | σ₁(G) vrai | σ₁(G₂) | σ₂ vrai / σ₂(G₂) |
|---|---|---|---|---|
| p12 | 0,10 | 402,55 | 398,38 | 3,40 / 2,39 |
| p22 | 0,31 | 15,89 | 14,64 | 4,46 / 4,02 |
| p17 | 0,37 | 6,43 | 6,49 | 3,68 / 2,89 |
| p5 | 0,32 | 4,47 | 4,46 | 3,08 / 2,77 |
| p6 | 0,28 | 3,98 | 3,98 | 2,47 / 2,46 |

Pour comparaison, l'ordre 1 donne σ₁ = 185 / 19,0 / 4,95 / 4,38 / 2,76, donc insuffisant ; l'ordre 4 donne
395 / 15,8 / 6,62 / 4,39 / 3,91.

**Verdict H1 : satisfaite au sens de la résolvante (σ₁ à 1–8 %), pas au sens de W (10–37 %).** C'est un fait
empirique. Aucun théorème de perturbation ne le garantit : ‖G‖ = 400 amplifie toute erreur de W. Il est donc testé
hors échantillon par le balayage en θ (§6).

**(H2) Structure du cocycle fermé 2×2** (`wind_T7_splitting_results.txt`, `wind_T7_ftle_results.txt`).

Théorème 1 avec le scindement adapté (celui qui minimise β ; choix post hoc, permis par le théorème pour tout
scindement invariant) :

| prompt | segment | A = ‖ĉ‖‖b̂‖ | β (Schur ≤) | A/β | encadrement prouvé du modèle |
|---|---|---|---|---|---|
| p12 | [2, 11] | 398,39 | 2,40 (7,16) | **166** | σ₁ ∈ [395,98 ; 400,79], σ₂ ≤ 2,40 |
| p22 | [3, 21] | 14,70 | 4,02 (11,18) | **3,66** | σ₁ ∈ [10,68 ; 18,72], σ₂ ≤ 4,02 |
| p17 | [12, 27] | 6,89 | 3,55 (10,50) | 1,94 | non informatif |
| p5 | [10, 12] | 6,34 | 4,87 (12,35) | 1,30 | non informatif |
| p6 | [5, 25] | 5,89 | 4,49 (11,54) | 1,31 | non informatif |

- Portes de l'identité (a) : ≤ 5·10⁻¹⁷.
- Les bornes sont respectées par le modèle (garanti). Sur la **vraie** G, elles tiennent pour p22 en σ₁ (15,89),
  mais pas tout à fait pour p12 : 402,55 contre la borne 400,79, et σ₂ = 3,40 > 2,40. L'écart est l'erreur du modèle,
  pas du théorème.
- Segments résonants (gains instables μ_i) :
  - p12 : μ = 1,32 ; 1,85 ; 1,76 aux couches 3–5, puis 1,51 ; 1,43 aux couches 8–9 ;
  - p22 : μ ≈ 1,12–1,26 aux pas 4–10 (et 12) du cocycle ;
  - p17 : aucun segment soutenu au-dessus de 1 (0,86–1,12).
- Exposant de Lyapunov maximal sur segments (≥ 4 couches), à θ = 1 : **λ₁* = +0,376 (p12), +0,196 (p22) contre
  +0,095 (p17), +0,091 (p5), +0,076 (p6)** ; λ₂* ≤ 0,009 partout, donc un seul mode instable. Post hoc,
  descriptif.

---

## 6. Prédictions pré-enregistrées (WIND-T7) et résultats

Intervention : J_k(θ) = J_k^A + θ(J_k^AN − J_k^A), produits vrais 1536×1536, grille θ = 0,25 : 0,125 : 2,5.
Aucun produit vrai à θ ≠ 1 n'avait été calculé avant le pré-enregistrement.

**P7.1 (gain critique prédit par le modèle 2-D, tolérance ±15 %, plus l'ordre p22 < 1 < autres) — VRAI.**

| prompt | θ_c pré-enregistré | θ_c vrai | écart |
|---|---|---|---|
| p22 | 0,851 | **0,833** | −2,1 % |
| p17 | 1,546 | **1,399** | −9,5 % |
| p5 | 1,621 | 1,630 | +0,6 % |
| p6 | 1,646 | 1,679 | +2,0 % |

L'ordre 0,833 < 1 < 1,399 / 1,630 / 1,679 est respecté. **P7.1b** (p12, ern_true(0,5) ∈ [1,10 ; 1,40]) : **VRAI**,
1,214 (modèle 1,23).

**P7.2 (fidélité des courbes ern, facteur 1,5) — FAUX**, à cause de p17 seul. Écarts max |Δ ln ern| : p12 0,017 ·
p22 0,090 · p5 0,102 · p6 0,178 · **p17 0,495**. Le modèle sous-estime l'effondrement de p17 entre θ = 1 et 1,4
(à θ = 1,375 : modèle 5,17 contre vrai 3,15). p17 est le prompt dont la réalisation d'ordre 2 est la moins fidèle
(erreur sur W 37 %).

**P7.3 (mode unique à θ = 2 : σ₂/σ₁ ≤ 0,10 et σ₂(G(2)) ≤ 3σ₂(G(1))) — FAUX.**

| | p12 | p22 | p17 | p5 | p6 |
|---|---|---|---|---|---|
| σ₂/σ₁ de G(2) | 0,000 | 0,031 | **0,402** | **0,141** | **0,125** |
| σ₂(G(2))/σ₂(G(1)) | 1,13 | 2,92 | **6,29** | 2,49 | 2,92 |

Lecture : le prédicat « un seul mode » était une extrapolation de ma part. Le théorème ne l'assure que sous
dichotomie (un exposant > 0, l'autre < 0). À θ = 2, λ₂* remonte à −0,014…+0,021 pour p17, p5, p6 et p22, et la
vraie G y a 2 valeurs singulières > 5 au lieu d'une. **Pour les prompts sous-critiques, l'hypothèse du théorème
tombe à fort gain, et la conclusion avec elle.** C'est cohérent, mais c'est un échec de prédiction et il est compté
comme tel.

**Réponse à la question d'amplitude.** Ce qui sépare p22 (effondré) de p17 (non effondré), malgré la même cohérence
de signe (WIND-T6, P6.2), c'est la **marge à la criticité de la boucle des normes** : θ_c = 0,83 contre 1,40,
prédit à ≤ 10 % par un modèle de dimension 2. De façon équivalente, c'est l'instabilité de la boucle fermée :
λ₁* = 0,196 contre 0,095. L'amplitude b n'est pas une propriété de chaque couche prise isolément : c'est la
croissance d'un mode de rétroaction sur un segment de 7–10 couches.

---

## 7. Classement des énoncés

| # | Énoncé | Classement | Vérification sur Qwen |
|---|---|---|---|
| A | relèvement exact, det G = 1, G = réponse en boucle fermée | lemme (classique) | portes ≤ 3,6·10⁻⁴ sur θ ∈ [0,25 ; 2,5] |
| B | factorisation de Hankel, rang conservé | lemme (classique) | purification observée (p12 : Hankel(G) de rang 1 à 10⁻²) |
| T1 | dédoublement non asymptotique (rang 1 + Y, bornes de Weyl et Schur, dichotomie uniforme en L) | **théorème (non trivial, modeste)** | exact sur le modèle d'ordre 2 ; informatif pour p12 (A/β = 166) et p22 (3,66) ; modèle → vérité à 0,4–8 % près |
| T2 | limite stationnaire, constante exacte ‖Ce‖‖fᵀB‖/(λ_u²−1), seuil ρ(Φ+θBC) = 1 | **théorème (non trivial)** | hypothèses NON satisfaites (non stationnaire) ; contrôle synthétique à 7 chiffres |
| C1 | corollaire d'amplitude b ≥ (ρ₁ − ‖Bulk‖)/‖Bulk‖_F | lemme | — |
| C7 | la boucle des normes est effectivement de dimension 2, et son gain critique fixe l'effondrement | **conjecture soutenue** | P7.1 VRAI (θ_c à ≤ 9,5 %), P7.1b VRAI ; P7.2 FAUX (p17) |
| P7.3 | mode unique à fort gain pour tous les prompts | **réfuté** | 3/5 ont un second mode à θ = 2 (dichotomie perdue) |

---

## 8. Nouveauté (évaluation franche)

**Lemme A.** C'est une série de Dyson / Duhamel discrète finie, sommée en résolvante d'un opérateur nilpotent : le
**relèvement** d'un système linéaire variant dans le temps (Dewilde–van der Veen, *Time-Varying Systems and
Computations*, 1998 ; Gohberg–Kaashoek). C'est aussi « Woodbury pour les produits ». **Pas nouveau.**

**Lemme B.** Semi-séparabilité et quasi-séparabilité des inverses (Eidelman–Gohberg 1999 ; Vandebril–Van Barel–
Mastronardi 2008). **Pas nouveau.**

**Théorème 2.** C'est, en langage d'état, le **phénomène de dédoublement des valeurs singulières de Toeplitz** :
le nombre de valeurs singulières exponentiellement petites de T_n(a) égale l'indice du symbole (Roch–Silbermann
1996 ; Böttcher–Grudsky, *Spectral Properties of Banded Toeplitz Matrices*, 2005), ici avec la constante dominante
explicite. Pour l'opérateur inverse, l'indice devient le nombre de valeurs propres de la boucle fermée hors du
disque (lemme du déterminant). **Très probablement connu sous une forme équivalente** ; la preuve donnée est directe.

**Théorème 1.** Il est voisin des résultats « dichotomie exponentielle ⟺ inversibilité des opérateurs
entrée-sortie variant dans le temps » (Ben-Artzi–Gohberg–Kaashoek, années 1990) et de « domination ⟺ écarts de
valeurs singulières » (Bochi–Gourmelon 2009, en théorie de Lyapunov). La version présentée est à section finie, avec
estimation bilatérale explicite, uniforme en L, et la construction de l'extension causale de rang 1. À ma
connaissance, elle n'est pas énoncée ainsi, mais elle est **à la portée d'un spécialiste** : nouveauté modeste.

**Application.** Je ne connais pas d'antécédent pour le contenu suivant :
- la linéarisation à normaliseurs figés (LRP(AH+LN), linéarisation des graphes d'attribution) transforme la
  rétroaction négative des RMSNorm en une **boucle de rétroaction positive quasi critique**, effectivement de
  dimension 2 ;
- son **gain critique** θ_c prédit quels prompts s'effondrent.

C'est la contribution réelle, et elle est empirique pour sa partie « ordre 2 ».

**Comparaison avec ρ → 2.** ρ → 2 était un théorème limite à constante exacte, vérifié à résidu nul sur des sites
réels. Ici, la constante exacte (théorème 2) n'est vérifiée que sur données synthétiques, car Qwen n'est pas
stationnaire. Ce qui est vérifié sur Qwen, ce sont le relèvement (exact), l'encadrement du théorème 1 sur le modèle
(exact par construction) et la prédiction hors échantillon des gains critiques (≤ 9,5 %). **C'est moins fort que
ρ → 2.**

---

## 9. Verdict

**Y a-t-il un vrai théorème au sens des 5 critères ?** Partiellement.
- Le théorème 1 satisfait les critères 1 (construction + dichotomie uniforme), 3 (preuve complète) et 5 (nouveauté
  évaluée : modeste).
- Il satisfait le critère 2 par sa conséquence : le gain critique θ_c, prédit et confirmé hors échantillon (P7.1).
  Cette prédiction passe toutefois par la réduction d'ordre 2, qui est empirique.
- Le critère 4 n'est satisfait que **pour le modèle d'ordre 2** de la boucle. Le passage à la vraie boucle n'est pas
  démontré : il est validé numériquement (σ₁ à 1–8 %, θ_c à ≤ 9,5 %) et échoue partiellement (P7.2 sur p17, P7.3).
- Le théorème 2 est un vrai théorème limite à constante exacte, mais ses hypothèses (stationnarité) sont fausses
  sur Qwen.

**Ce que l'on sait maintenant.** L'amplitude de l'effondrement AN est fixée par la distance à la criticité d'une
boucle de rétroaction des normes de dimension effective 2 : p22 sur-critique (θ_c = 0,83), p17 sous-critique
(θ_c = 1,40). La résonance est à mode unique quand la boucle a un seul exposant instable (p12, p22) ; elle peut en
avoir deux à fort gain.

**Ce qui manque pour aller plus loin.**
1. Une **preuve** que la boucle est d'ordre ≈ 2. Il faudrait identifier les deux canaux, a priori la fluctuation
   de ‖x‖ et celle de ‖h‖, et montrer que la chaîne vivante transmet les injections aux capteurs par un sous-espace
   de dimension 2. Cela exige **h_k et les sorties de sous-couches** a_k et m_k, **qui ne sont pas stockés**.
2. Un théorème de perturbation adapté aux résolvantes non normales, pour transférer les bornes du modèle à la
   vraie G. La borne naïve est inutilisable (facteur ‖G‖² ≈ 1,6·10⁵ pour p12).
3. Le cas A-p12 (QK). Le relèvement s'applique, mais le rang de E_k (82–149 par couche) donne un relèvement de
   dimension > n. Une réduction de bas ordre n'a pas été tentée.
4. L'entonnoir de T (cible (c)) n'a pas été traité.

---

## 10. Fichiers (tous dans `notebook/`)

- **Livrable** : `wind_T7_theorem.md`
- Pré-enregistrement : section « WIND-T7 » à la fin de `wind_T3_preregistration.md`
- Scripts :
  - `wind_T7_lift.jl` : relèvement exact, G, semi-séparabilité ;
  - `wind_T7_resolvent_probe.jl` : Hankel de W et G ;
  - `wind_T7_realization_probe.jl` : Ho–Kalman d'ordre r ;
  - `wind_T7_model.jl` : modèle d'ordre 2, théorème 1, courbes θ du modèle ;
  - `wind_T7_sweep.jl` : tests P7.1–P7.3 ;
  - `wind_T7_splitting.jl` : scindement adapté, bornes de Schur ;
  - `wind_T7_ftle.jl` : exposants de Lyapunov en temps fini ;
  - `wind_T7_synthetic_check.jl` : contrôle du théorème 2.
- Résultats : `wind_T7_lift_results.txt`, `wind_T7_resolvent_probe_results.txt`,
  `wind_T7_realization_probe_results.txt`, `wind_T7_model_results.txt`, `wind_T7_sweep_results.txt`,
  `wind_T7_splitting_results.txt`, `wind_T7_ftle_results.txt`, `wind_T7_synthetic_check_results.txt`
  (+ journaux `.log`).
- Données dérivées : `wind_data/wind_T7_lift.json`, `wind_data/wind_T7_lift_p{5,6,12,17,22}.bin` (𝒰, 𝒱, M_A, W),
  `wind_data/wind_T7_model_p{…}.json`.
