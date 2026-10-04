# WIND-T20 — Conservation contre exactitude transverse : le théorème derrière le dilemme effondrement / dispersion

*2026-10-04. Qwen2.5-1.5B, GPT-2 small, Gemma-2-2B. CPU seulement (BLAS/MKL ≤ 4 fils). Tout est EXPLORATOIRE : les
données ont inspiré les conjectures. Les prédictions sont écrites dans `wind_T20_preregistration.md` avant les
calculs qui les testent ; les déviations y sont consignées (D1–D3).*

---

## 0. En tête : classement des énoncés

| # | Énoncé | Classement | Vérifié sur les données réelles |
|---|---|---|---|
| **T1** | **Conservation contre exactitude transverse.** (a) Il existe une unique linéarisation d'un MLP (norme figée) à la fois complète et transverse-exacte : L* = J − E hᵀ/‖h‖². (b) Aucune règle neurone par neurone (gradient, règle du demi, règle uniforme, toute règle LRP de ce type) ne peut être à la fois complète et transverse-exacte. Toute règle complète de cette classe déforme la dérivée transverse d'au moins τ*, constante exacte et calculable, atteinte par une unique règle optimale. | **théorème non trivial, modeste** | hypothèses vraies aux 18 points (3 modèles, poids réels) : parité ≤ 1,1·10⁻⁶ ; ‖E‖/‖f‖ ≥ 0,87 ; Gram définie positive (λ_min/λ_max ≥ 3,7·10⁻⁵). r* = 0,15–0,23 (Qwen, Gemma), 0,23–0,61 (GPT-2) ; règle du demi r_½ = 0,51–0,64 |
| L2 | Geler un dénominateur, c'est exactement boucler la norme sur un gain multiplicatif, n_fr(z) = (ρ(z)/ρ₀)·n(z) (non linéaire) ; trichotomie radiale vivant / complet / gradient | lemme (élémentaire) | identité algébrique exacte ; sa forme linéarisée (relèvement de T7) est vérifiée, portes ≤ 2,7·10⁻⁴ sur les 3 modèles |
| L3 | J^AN − J^REC = (E_k/‖h_k‖)ρ_kᵀ, bloc par bloc : tout l'écart entre la linéarisation figée et la linéarisation complète et transverse-exacte est porté par les excès d'Euler propagés | lemme (élémentaire) | exact par construction (facteurs T13, porte G1 ≤ 4,8·10⁻³) |
| L4 | g_N ∈ [o·ℓ − 1, o·ℓ + 1], où o est la force en boucle ouverte et ℓ le facteur de boucle (Weyl) | lemme (classique) | — |
| B | Prédictions sur REC (= L* appliquée à chaque sous-couche, Qwen, 49 prompts) | **B2, B3, B4 VRAIS ; B1 FAUX** (un sous-critère : Spearman 0,58 < 0,8 sous forte censure ; 4 effondrements prédits, **0 observé**) | §5 |
| C | Question (a) : la marge de stabilité est fixée par la **mémoire longue** de la boucle des normes du réseau vivant, pas par le gain local ni le facteur d'Euler | **conjecture soutenue** (C1, C2 vrais sur les prompts d'évaluation ; C3 faux : −0,51 contre −0,6 requis) ; **aucun théorème** | §6 |
| — | Question (c) : relation quantitative auto-réparation ↔ boucle figée | **aucun théorème** ; réinterprétation exacte seulement (L2) | §7 |

**En une phrase.** Le dilemme « AN s'effondre, AH disperse » n'est pas un accident numérique. C'est un théorème
d'impossibilité pour toute la classe des règles neurone par neurone :
- le gradient (AN) est transverse-exact mais porte l'excès d'Euler E, qui alimente la boucle positive des normes
  figées ;
- les règles complètes de la classe (AH) déforment forcément la dérivée transverse, d'au moins 15 à 23 % sur
  Qwen et Gemma ; la règle du demi déforme de 51 à 64 % ;
- la seule échappatoire est un terme de rang 1 hors de la classe (REC). Testée : 0 effondrement, géométrie du vrai
  réseau, fidélité au niveau de la meilleure linéarisation non conservative, biais de profondeur corrigé.

---

## 1. Cadre et notations (une sous-couche MLP, au dernier token, norme d'entrée figée)

**Objets de base :**
- h ∈ ℝ^D \ {0} : l'entrée de la sous-couche (état résiduel après l'attention).
- F : la norme figée (dénominateur pris à sa valeur propre), linéaire.
  - RMSNorm : F = diag(γ)/ρ(h).
  - LayerNorm (GPT-2) : F = diag(γ)r·(I − 11ᵀ/D), plus un biais β.
- v = F h (+ β).

**MLP à porte sans biais** (Qwen : φ = SiLU ; Gemma : φ = GELU-tanh, post-norme figée absorbée dans W_d) :

  f(v) = W_d (φ(W_g v) ⊙ W_u v),   W_g, W_u ∈ ℝ^{n×D}, W_d ∈ ℝ^{D×n}

(n = 8960 pour Qwen, 9216 pour Gemma). On note g = W_g v, u = W_u v, A_g = W_g F, A_u = W_u F.

**Dérivée exacte par rapport à h, norme figée :**

  J = W_d [diag(u ⊙ φ'(g)) A_g + diag(φ(g)) A_u].

**Fait 0 (classique).** La dérivée avec la norme VIVANTE vaut J·(I − s ĥĥᵀ), avec s = ‖h‖²/(‖h‖² + Dε) ≈ 1.
Normes vivante et figée coïncident donc exactement sur l'hyperplan transverse h^⊥. On note P = I − ĥĥᵀ.

**Définitions.** Soit L une linéarisation de la sous-couche (application linéaire ℝ^D → ℝ^D).
- L est **transverse-exacte** si LP = JP : elle coïncide avec la VRAIE dérivée sur h^⊥.
- L est **complète** si L h = f(v) − f_base : c'est la conservation de la LRP le long de l'entrée propre. Sans biais,
  f_base = f(0) = 0 ; pour GPT-2, f_base = f(β) (h = 0).
- **Excès d'Euler** (défaut de complétude du gradient) : E = J h − (f(v) − f_base). Pour un MLP sans biais,
  E = W_d(u ⊙ φ'(g) ⊙ g).

**Classe 𝒩 des règles neurone par neurone.** On remplace les deux dérivées d'activation de chaque neurone par des
coefficients (β_i, γ_i) :

  L_{β,γ} = W_d [diag(β) A_g + diag(γ) A_u].

𝒩 contient :
- le **gradient** : β* = u ⊙ φ'(g), γ* = φ(g) ;
- la **règle du demi**, σ figé (Transluce 2601.22594 ; RelP 2508.21258 ; règle uniforme d'AttnLRP) : β = ½ u ⊙ s(g),
  γ = ½ φ(g), avec s = φ(g)/g ;
- toute règle « identité / ε / demi » qui agit neurone par neurone sur les activations.

Elle ne contient pas les règles qui modifient les poids (LRP-γ).

**Opérateurs associés.** On écrit δ = (β − β*, γ − γ*) ∈ ℝ^{2n}.
- L_{β,γ} − J = Λ(δ) := W_d[diag(δβ)A_g + diag(δγ)A_u].
- Partie radiale : Λ_h(δ) := Λ(δ)h = W_d(δβ ⊙ g + δγ ⊙ u), car A_g h = g et A_u h = u.
- Gram de Hadamard 𝒢 ∈ ℝ^{2n×2n}, définie de sorte que ‖Λ(δ)P‖_F² = δᵀ𝒢δ :

  𝒢 = [[(W_dᵀW_d) ⊙ (A_gPA_gᵀ), (W_dᵀW_d) ⊙ (A_gPA_uᵀ)], [(W_dᵀW_d) ⊙ (A_uPA_gᵀ), (W_dᵀW_d) ⊙ (A_uPA_uᵀ)]].

  Pour GPT-2 (sans porte), il n'y a qu'une famille : 𝒢 = (W_outᵀW_out) ⊙ (APAᵀ).

---

## 2. Théorème 1 — conservation contre exactitude transverse pour les règles neurone par neurone

**Énoncé.** Sous les définitions du §1 :

**(a) Unicité.** Il existe une et une seule application linéaire L à la fois transverse-exacte et complète :

  **L\* = J − E hᵀ/‖h‖²**   (le gradient corrigé d'un terme de rang 1, radial).

**(b) Impossibilité quantitative.** Supposons 𝒢 ≻ 0 (hypothèse H2) et Λ_h surjective. Alors toute règle neurone par
neurone COMPLÈTE L ∈ 𝒩 vérifie

  ‖(L − J)P‖_F ≥ τ*,   avec   **τ*² = Eᵀ (Λ_h 𝒢⁻¹ Λ_hᵀ)⁻¹ E**,

et cette borne est atteinte par une unique règle complète de 𝒩 :

  δ* = −𝒢⁻¹Λ_hᵀ(Λ_h𝒢⁻¹Λ_hᵀ)⁻¹E.

De plus, τ* ≥ √λ_min(𝒢)·‖E‖/σ_max(Λ_h) > 0 dès que E ≠ 0.

**(c) Conséquences.**
- (i) Le gradient est l'UNIQUE membre transverse-exact de 𝒩 ; son défaut de complétude vaut exactement E.
- (ii) Si E ≠ 0, aucune règle neurone par neurone n'est à la fois complète et transverse-exacte, et L* ∉ 𝒩.
- (iii) La règle du demi est complète ; sa distorsion transverse relative vérifie
  r_½ := ‖(L_½ − J)P‖_F/‖JP‖_F ≥ r* := τ*/‖JP‖_F.

**Preuve.**
- **(a)** ℝ^D = span(h) ⊕ h^⊥, et une application linéaire est déterminée par ses valeurs sur chaque facteur.
  - LP = JP fixe L sur h^⊥.
  - La complétude fixe L h.
  - Il y a donc au plus une solution. L* convient : L*P = JP − E(hᵀP)/‖h‖² = JP, car hᵀP = 0 ; et
    L*h = Jh − E = f − f_base. ∎
- **(b)** Soit L = L_{β*+δβ, γ*+δγ}. Les coefficients entrent linéairement, donc L − J = Λ(δ).
  - **Complétude** ⟺ Λ(δ)h = (f − f_base) − Jh = −E ⟺ Λ_hδ = −E.
  - **Distorsion transverse.** On a Λ(δ)P = Σ_i d_i ⊗ (δβ_i a_i + δγ_i b_i), avec d_i la i-ème colonne de W_d,
    a_i = P A_gᵀe_i et b_i = P A_uᵀe_i. Le produit scalaire de Frobenius vérifie
    ⟨d_i ⊗ a_i, d_l ⊗ b_l⟩_F = (d_iᵀd_l)(a_iᵀb_l). D'où ‖Λ(δ)P‖_F² = δᵀ𝒢δ.
  - **Minimisation.** On minimise une forme quadratique strictement convexe (𝒢 ≻ 0) sur le sous-espace affine non vide
    {Λ_hδ = −E}. Les multiplicateurs de Lagrange donnent l'unique minimiseur δ* et la valeur
    τ*² = Eᵀ(Λ_h𝒢⁻¹Λ_hᵀ)⁻¹E (moindres carrés généralisés).
  - **Borne inférieure.** δᵀ𝒢δ ≥ λ_min‖δ‖², et ‖E‖ = ‖Λ_hδ‖ ≤ σ_max(Λ_h)‖δ‖. ∎
- **(c)**
  - (i) 𝒢 ≻ 0 rend δ ↦ Λ(δ)P injective. Un membre transverse-exact de 𝒩 a donc δ = 0, c'est-à-dire L = J. Son
    défaut de complétude est Jh − (f − f_base) = E.
  - (ii) Une règle de 𝒩 complète et transverse-exacte aurait 0 ≥ τ* > 0 : contradiction. L* est complète et
    transverse-exacte (a), donc L* ∉ 𝒩.
  - (iii) δβ_½ ⊙ g + δγ_½ ⊙ u = ½u s(g) g − u φ'(g) g − ½ φ(g) u = −u φ'(g) g, car s(g)g = φ(g). Donc
    Λ_hδ_½ = −E : la règle du demi est complète, et (b) s'applique. ∎

**Remarques.**
- **Généricité.** 𝒩 a 2n degrés de liberté (17 920 pour Qwen) ; l'exactitude transverse impose D(D−1) ≈ 2,4·10⁶
  contraintes. H2 est donc génériquement vraie dès que 2n ≤ D(D−1).
  - Il en existe un point injectif explicite : on répartit les neurones par ligne de sortie, au plus (D−1)/2 par
    ligne, avec des vecteurs d'entrée indépendants dans h^⊥.
  - L'ensemble non injectif est alors une sous-variété algébrique propre.
  - On ne s'en sert pas : H2 est vérifiée directement sur les poids réels (§4).
- **La technique est élémentaire** : somme directe, puis moindres carrés sous contrainte. La non-trivialité est dans
  l'énoncé :
  - une impossibilité valable pour TOUTE la classe des règles neurone par neurone utilisées sur les MLP à porte ;
  - avec la constante optimale exacte, calculable sur les poids réels ;
  - et l'échappatoire explicite et unique (un terme de rang 1, hors de la classe).
- **Coût pratique de L\*.** E = Jh − f ne demande qu'UNE dérivée directionnelle du MLP le long de h. Le correctif est
  un nœud de rang 1 par MLP : même nature que les « nœuds dénominateurs » proposés pour les graphes d'attribution
  (interpretune #684), mais placé sur l'excès d'Euler du MLP plutôt que sur la norme.

---

## 3. Lemmes (exacts, élémentaires)

**Lemme 2 — geler un dénominateur, c'est boucler la norme sur un gain.** Pour une RMSNorm n(z) = γ⊙z/ρ(z), figée en
n_fr(z) = γ⊙z/ρ₀ avec ρ₀ = ρ(z₀), on a pour tout z :

  **n_fr(z) = (ρ(z)/ρ₀) · n(z)**.

*Preuve.* γ⊙z/ρ₀ = (ρ(z)/ρ₀)·γ⊙z/ρ(z). ∎

*Lecture.*
- Le réseau à normes figées est EXACTEMENT, de façon non linéaire, le réseau vivant dont chaque signal normalisé est
  multiplié par un gain c_i = ρ(z_i)/ρ_i⁰. Ce gain est piloté par la norme de sa propre entrée.
- Linéarisé : δ ln c_i = s_i δ ln‖z_i‖.
- Le θ-chemin de T7/T8 est le gain θ de cette boucle.
- Le noyau W du relèvement de T7 est la réponse des log-normes du réseau VIVANT à ces gains. Injections : vecteurs
  d'Euler des sous-couches ; capteurs : log-normes des entrées de normes.
- La « compensation par les normes » de T16 (TE_fn − TE) est l'effet de cette boucle à amplitude finie.
- Le forward figé (gain libre, réponse sur-linéaire du MLP au gain) peut exploser en temps fini : ce sont les
  23 divergences de T16.

**Corollaire (trichotomie radiale).** On prend une sous-couche à norme d'entrée figée et un pur changement d'échelle
δh = εh. Sa réponse selon la linéarisation vaut :

| Linéarisation | Réponse | Régime | Variante |
|---|---|---|---|
| Vivante | (1 − s)Jh ≈ 0 | contractant | AAf |
| Complète | f − f_base | neutre | AH, REC |
| Gradient | f − f_base + E | amplificateur dès que E a une composante radiale positive | AN |

**Lemme 3 — l'écart AN − REC est porté exactement par les excès d'Euler propagés.** Avec les facteurs exacts de
T13, bloc par bloc :

  **J_k^AN − J_k^REC = (E_k/‖h_k‖) ρ_kᵀ**,   avec ρ_k = (I + A₁)ᵀĥ_k et E_k = M h_k − m_k.

Sur toute la profondeur : P^AN − P^REC = Σ_k P^AN(L←k+1)(E_k/‖h_k‖)ρ_kᵀP^REC(k←1).

*Preuve.* J^AN = J^AA + s₂ Mĥ ρᵀ et J^REC = J^AA + cρᵀ, avec c = m/‖h‖ − (1 − s₂)Mĥ. Leur différence vaut
(Mĥ − m/‖h‖)ρᵀ = (Mh − m)ρᵀ/‖h‖. La seconde identité est l'identité télescopique des différences de produits. ∎

**Lemme 4 — décomposition boucle ouverte × boucle (Weyl).** On écrit N·P^AN = N·P^A + Δ, avec Δ = 𝒰(I − W)⁻¹𝒱
(lemme A de T7). On pose :
- o = σ₁(Δ¹)/σ₁(N P^A), la force en boucle ouverte, avec Δ¹ = 𝒰𝒱 (une seule injection) ;
- ℓ = σ₁(Δ)/σ₁(Δ¹), le facteur de boucle.

Alors l'amplification g_N = σ₁(N P^AN)/σ₁(N P^A) ∈ [o·ℓ − 1, o·ℓ + 1]. *Preuve :* inégalités de Weyl. ∎

---

## 4. Vérification des hypothèses sur les poids réels (critère 4)

`wind_T20_thm1.py`, résultats dans `wind_T20_thm1.json`.
- Poids lus dans les safetensors (lecteur maison) ; état h depuis les méta WIND.
- Points de mesure fixés avant calcul : couches 3, 13, 23 (Qwen, Gemma) et 2, 6, 10 (GPT-2), prompts 1 et 2, soit
  18 points.
- P0 = parité entre f(v) recalculé et la sortie MLP stockée.
- La borne inférieure de la dernière colonne est √λ_min‖E‖/σ_max(Λ_h), relative à ‖JP‖_F.

| modèle | couche | prompt | P0 parité | ‖E‖/‖f−f_base‖ | Euler eff. | Cholesky | λ_min/λ_max | **r*** | r_½ | borne inf. (rel.) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen | 3 | 1 | 5,6e-07 | 1,34 | 0,99 | oui | 1,2e-02 | **0,156** | 0,643 | 0,012 |
| qwen | 3 | 2 | 5,1e-07 | 1,30 | 1,51 | oui | 1,2e-02 | **0,158** | 0,627 | 0,013 |
| qwen | 13 | 1 | 4,1e-07 | 1,02 | 1,84 | oui | 2,2e-03 | **0,154** | 0,520 | 0,008 |
| qwen | 13 | 2 | 5,1e-07 | 0,94 | 1,74 | oui | 2,2e-03 | **0,145** | 0,515 | 0,006 |
| qwen | 23 | 1 | 4,5e-07 | 0,93 | 1,71 | oui | 1,2e-02 | **0,168** | 0,544 | 0,022 |
| qwen | 23 | 2 | 5,1e-07 | 0,98 | 1,84 | oui | 1,3e-02 | **0,185** | 0,542 | 0,022 |
| gemma | 3 | 1 | 5,4e-07 | 0,90 | 1,54 | oui | 7,4e-03 | **0,200** | 0,554 | 0,007 |
| gemma | 3 | 2 | 5,7e-07 | 0,89 | 1,36 | oui | 7,4e-03 | **0,192** | 0,566 | 0,006 |
| gemma | 13 | 1 | 6,8e-07 | 1,05 | 1,92 | oui | 5,6e-03 | **0,198** | 0,514 | 0,007 |
| gemma | 13 | 2 | 6,7e-07 | 1,08 | 1,99 | oui | 5,6e-03 | **0,213** | 0,512 | 0,008 |
| gemma | 23 | 1 | 5,5e-07 | 0,87 | 1,71 | oui | 7,2e-03 | **0,229** | 0,515 | 0,017 |
| gemma | 23 | 2 | 6,6e-07 | 0,95 | 1,83 | oui | 7,2e-03 | **0,227** | 0,519 | 0,017 |
| gpt2 | 2 | 1 | 1,1e-06 | 1,36 | −0,32 | oui | 3,7e-05 | **0,611** | — | 0,010 |
| gpt2 | 2 | 2 | 5,9e-07 | 1,44 | −0,38 | oui | 3,7e-05 | **0,566** | — | 0,012 |
| gpt2 | 6 | 1 | 4,4e-07 | 0,88 | 1,65 | oui | 1,9e-02 | **0,229** | — | 0,032 |
| gpt2 | 6 | 2 | 4,9e-07 | 0,87 | 1,64 | oui | 2,2e-02 | **0,248** | — | 0,039 |
| gpt2 | 10 | 1 | 2,6e-07 | 0,93 | 1,39 | oui | 1,7e-02 | **0,280** | — | 0,024 |
| gpt2 | 10 | 2 | 2,5e-07 | 1,03 | 1,36 | oui | 1,7e-02 | **0,299** | — | 0,027 |

**Verdicts pré-enregistrés (partie A) :**
- P0 VRAI (max 1,1·10⁻⁶).
- H1 VRAI (min 0,87).
- H2 VRAI (Cholesky réussie partout ; λ_min/λ_max ≥ 3,7·10⁻⁵).
- Question quantitative « r* ≥ 0,05 » VRAIE partout.
- Contrôle r_½ ≥ r* vrai partout.
- La règle optimale δ* est bien complète (écart ≤ 3·10⁻¹⁵). La règle du demi aussi (≤ 2,4·10⁻¹⁵), comme le prédit
  (c)(iii).

**Lecture.**
- **Le gradient (AN) est très incomplet.** Son défaut de complétude est aussi grand que la sortie elle-même :
  ‖E‖ ≈ 0,9–1,4·‖f‖.
- **E ne se réduit pas à « (d − 1)f ».** Pour Qwen, couche 3, le degré d'Euler effectif vaut 0,99 et pourtant
  ‖E‖/‖f‖ = 1,34. Le vecteur d'Euler a une forte composante non parallèle à f : 33 % en médiane sur les MLP de Qwen
  (`wind_T20_rec.json`, `euler_perp`).
- **Même la meilleure règle complète déforme beaucoup.** Elle distord la dérivée transverse de 15 à 23 % (Qwen,
  Gemma) et jusqu'à 61 % (GPT-2, couche 2). La règle du demi distord de 51 à 64 %, soit environ 3 fois l'optimum.

---

## 5. Ce que le théorème explique : AN, AH et REC sont les trois sommets du dilemme

| Variante (Qwen, 50 prompts) | Classe | Transverse-exacte ? | Complète ? | Effondrements | ern(N·P) médian (T : 17,0) | ρ médian, 28 MLP (T : 0,840) | part précoce l ≤ 10 (vérité 0,211) |
|---|---|---|---|---|---|---|---|
| AN / ANf (gradient, normes figées) | 𝒩 | oui | **non** (défaut E) | 12/50 (T8) | 5,45 | 0,593 (ANf) | 0,262 (sur-pondère) |
| AH / AHf (règle du demi) | 𝒩 | **non** (r_½ ≈ 0,5–0,64) | oui | 0/50 (T15) | 91,6 (disperse) | 0,559 (AHf) | 0,105 (sous-pondère) |
| **REC = L\*** | **hors 𝒩** | oui | oui | **0/49** | **17,3** | **0,691** | **0,202** |
| AAf (norme MLP vivante) | — | oui | non (contractant) | 0/50 (T13) | 19,1 | 0,729 | 0,197 |

Le théorème prédisait le dilemme, et les trois variantes en occupent les trois sommets :
- **AN** est l'unique membre transverse-exact de 𝒩, (c)(i). Son défaut de complétude E est exactement ce qui
  alimente la boucle positive des normes figées (lemme 3 ; trichotomie, lemme 2) : d'où l'effondrement.
- **AH** est complète, donc condamnée par (b) à une distorsion transverse ≥ r* (mesurée 0,51–0,64) : d'où la
  dispersion et la sous-pondération des couches précoces.
- **L\*** est la seule linéarisation complète et transverse-exacte, et elle est hors de 𝒩, (a) et (c)(ii).

**Test de L\* (REC = L\* sur chaque sous-couche ; prédictions B pré-enregistrées ; `wind_T20_rec.jl`).**
- 49 prompts retenus. p8 est exclu par la porte G-T14 (1,008·10⁻³ > 10⁻³) ; REC n'y est pas effondré (ern 20,5,
  ρ_REC 0,62 contre ρ_ANf 0,35).
- Portes : G1 ≤ 4,8·10⁻³ ; G-T14 ≤ 7,5·10⁻⁴ ; G-T9 ≤ 7,0·10⁻⁴ ; relèvement ≤ 3,2·10⁻¹⁵.

**B1 (stabilité) : FAUX au sens pré-enregistré.**
- Compte : **0/49 effondrements** (seuil ≤ 6) ✓.
- Rapport θ_c^REC/θ_c^AN médian 2,05 (dans [1,4 ; 2,2]) ✓.
- Spearman(θ_c^REC, θ_c^AN) = 0,58 < 0,8 ✗. 33/49 θ_c^REC sont censurés à 2,5, ce qui ôte presque toute
  information de rang.
- **Le mécanisme quantitatif était trop pessimiste.** Il prédisait 4 effondrements REC (p13, p25, p30, p43, ceux où
  θ_c^AN·d̄ < 1) ; il y en a 0. Sur ces quatre prompts, θ_c^REC/θ_c^AN = 4,1 ; 3,4 ; 4,4 ; 3,5, au lieu de ≈ d̄ ≈ 1,75.
- Explication (post hoc) : retirer E retire aussi sa composante non parallèle à m (33 %), qui nourrit la boucle
  autant que la partie « (d − 1)m ».

**B2 (géométrie) : VRAI.**
- ern_REC/ern_A médian 0,976.
- Distance médiane à la vraie géométrie, |ln(ern_V/ern_T)| : REC 0,29, contre AN 0,97 et AH 1,76.

**B3 (fidélité, 28 composantes MLP, vérité ablation WIND-T9) : VRAI.**
- ρ_REC − ρ_ANf médian +0,079 (Wilcoxon p = 2·10⁻⁶).
- REC (0,691) ≥ AHf (0,559).
- |REC − AAf| = 0,038 ≤ 0,05.
- *Descriptif, groupes effondrés / non effondrés (θ_c^mod < 1) :*

  | Variante | ρ effondrés | ρ non effondrés | écart |
  |---|---|---|---|
  | REC | 0,687 | 0,699 | **disparu** |
  | ANf | 0,515 | 0,635 | −0,12 |
  | AHf | 0,488 | 0,570 | −0,08 |

**B4 (biais de profondeur) : VRAI.** Écart médian |part précoce − vérité| : REC 0,038 (T 0,042, AAf 0,043), contre
ANf 0,073 et AHf 0,102.

**Conséquence pratique.** La conservation ne coûte pas de fidélité si l'on sort de la classe neurone par neurone.
- L* est complète, transverse-exacte et stable.
- Sa fidélité est au niveau de AAf (−0,04), sans effondrement ni dépendance au groupe.
- Elle coûte une dérivée directionnelle par MLP.

---

## 6. La question (a) : pourquoi Gemma ≫ Qwen ≫ GPT-2 ?

**Décomposition exacte (lemme A + lemme 4).** `wind_T20_lift.jl` ; portes de relèvement ≤ 2,7·10⁻⁴ ; rang de R_k
exact (queue ≤ 10⁻³). Médianes, découverte + évaluation :

| modèle (n) | o = σ₁(Δ¹)/σ₁(NP^A) | ℓ = facteur de boucle | ‖G_loc‖ (boucle d'un pas) | ‖G‖ | **mémoire M = ‖G‖/‖G_loc‖** | g_N | ern AN/A |
|---|---|---|---|---|---|---|---|
| GPT-2 (30) | 0,29 | 1,06 | 1,97 | 2,1 | **1,1** | 0,99 | 1,04 |
| Qwen (20) | 0,61 | 3,14 | 1,77 | 9,7 | **5,5** | 1,95 | 0,39 |
| Gemma (8) | 0,61 | 6,22 | 4,03 | 95 | **24** | 4,00 | 0,095 |

**Prédictions C (prompts d'évaluation : GPT-2 11–30, Qwen 11–20, Gemma 3–8).**
- **C1 VRAI** : L_f médian 0,96 / 2,54 / 5,42 (seuils ≤ 1,6 / ≥ 2,0 / ≥ 1,5 × Qwen).
- **C2 VRAI** : M médian 1,09 / 6,1 / 23 (seuils ≤ 1,5 / ≥ 3 / ≥ 3).
- **C3 FAUX** : Spearman(ln L_f, θ_c) = −0,51, pour −0,6 requis sur Qwen 1–20.

**Réponse à (a), telle qu'elle tient aujourd'hui.** La quantité du VRAI réseau qui fixe la marge de stabilité est
**la mémoire longue de sa boucle des normes**. C'est la réponse des log-normes du réseau vivant à des gains injectés
plusieurs couches plus tôt, injections données par les excès d'Euler (lemmes 2 et 3).
- Ce n'est pas le gain local : ‖G_loc‖ vaut 1,97 pour GPT-2 contre 1,77 pour Qwen.
- Ce n'est pas le facteur d'Euler : 1,47 / 1,77 / 1,62.
- GPT-2 n'a pas de mémoire (M ≈ 1) : son excès de gain local se dissipe en un pas.
- Qwen et Gemma l'ont (M = 5,5 et 24). Gemma ajoute une boucle locale deux fois plus forte, avec quatre normes par
  bloc, dont les post-normes.
- Le lemme 4 transforme o·ℓ en encadrement de l'amplification : GPT-2 o·ℓ ≈ 0,3 ; Qwen ≈ 1,9 ; Gemma ≈ 3,8.

**Ce qui manque pour un théorème sur (a).** Il faudrait prédire M à partir de l'architecture ou d'une grandeur du
forward seul. Tentative de la découverte (non retenue) : le « budget radial » Σ_k ln(1 + (d_k − 1)μ_k), avec μ_k la
fraction d'écriture radiale du MLP.
- Il classe bien GPT-2 (≈ 0,9) sous Qwen (≈ 2,2).
- Mais il met Gemma (≈ 2,1) au niveau de Qwen, alors que Gemma s'effondre 4 fois plus.

La mémoire est un effet de TRANSPORT par le réseau vivant (non normal, sur plusieurs couches) ; aucune quantité
locale ne la capture. Une borne utile demanderait une théorie de perturbation des cocycles non normaux, déjà
identifiée comme manquante en T7. **Pas de théorème : loi empirique confirmée hors échantillon.**

---

## 7. La question (c) : auto-réparation et boucle figée

**Ce qui est exact (lemme 2).** Les normes vivantes sont le réseau à gain fixé c ≡ 1. Les normes figées sont le même
réseau où le gain suit la norme, c_i = ρ(z_i)/ρ_i⁰. La compensation par les normes (T16) est donc exactement
l'effet de cette boucle. En réponse linéaire, c'est le terme 𝒰(I − W)⁻¹𝒱 du relèvement (d'où S3, 0,79, à
l'imperfection non linéaire près).

**Pourquoi il n'y a pas de théorème « l'une borne l'autre ».** Il faudrait relier la compensation moyenne d'un
prompt (Φ_p) à sa marge θ_c. Or les données le contredisent : T16-S4 donne −0,25 pour un seuil de −0,3. La raison
est structurelle :
- Φ_p mesure la boucle appliquée aux perturbations d'ABLATION (directions des sorties de sous-couches) ;
- θ_c mesure sa direction la plus instable.

Les deux ne sont liés que par la projection des ablations sur le mode résonant, qui varie d'un prompt à l'autre. Un
énoncé vrai serait une identité (lemme A appliqué à δ) ; une borne utile n'existe pas sans hypothèse d'alignement,
et cet alignement est faux sur les données. **Statut : aucun théorème.**

---

## 8. Nouveauté (évaluation franche)

**Antécédents les plus proches :**
- **Règles de conservation pour les produits** : la règle uniforme et la règle du demi d'AttnLRP (Achtibat et al.,
  ICML 2024), reprises par Transluce (Arora et al., arXiv 2601.22594) et RelP (2508.21258). Elles sont motivées
  par la conservation ; leur coût de fidélité n'est pas quantifié.
- **You et al. 2025** (arXiv 2510.18810) montrent que la règle bilinéaire d'AttnLRP viole l'invariance
  d'implémentation et s'écarte du leave-one-out. C'est le résultat publié le plus voisin : un défaut d'UNE règle.
  Ici, on a une borne inférieure pour TOUTE la classe, la constante optimale et l'échappatoire unique.
- **Axiomes de Sundararajan et al. 2017** : le gradient×entrée n'est pas complet pour les fonctions non homogènes.
  C'est connu ; ici, c'est (c)(i).
- **Bilodeau et al., PNAS 2024** : les attributions complètes et linéaires ne peuvent pas faire mieux que le hasard
  pour certaines tâches. L'objet est différent (tâches finales, classes de modèles), sans rapport technique.
- **Normes comme contrôle de gain** : Buehlmaier 2026 (transformers bouclés) ; « nœuds dénominateurs »
  (interpretune #684, Gemma-3 : 0,11 contre 0,81) ; auto-réparation par la norme finale (Rushing & Nanda,
  ICML 2024).
- **Algèbre linéaire** : Gram de Hadamard de matrices de rang 1 structurées, injectivité générique, moindres carrés
  sous contrainte. Rien de nouveau.

**Ce qui est, à ma connaissance, nouveau :**
1. Le critère « transverse-exacte » (accord avec la vraie dérivée hors de la direction radiale, exactement là où
   normes vivantes et figées coïncident), et l'impossibilité, pour la classe neurone par neurone, d'être à la fois
   complète et transverse-exacte, avec constante optimale calculée sur poids réels : r* = 0,15–0,23 sur Qwen et
   Gemma.
2. L'identification de l'unique échappatoire, L* = J − E hᵀ/‖h‖², et sa validation : stabilité, géométrie,
   fidélité et profondeur sur 49 prompts.
3. La lecture unifiée du dilemme. Le gradient porte E, donc l'effondrement par la boucle des normes ; les règles
   complètes de 𝒩 portent une distorsion transverse ≥ τ*, donc la dispersion.
4. Pour (a) : la mémoire de la boucle comme variable qui sépare les modèles. Loi empirique, pas un théorème.

**Ce qui n'est pas nouveau ou pas profond.** La technique de preuve (moindres carrés, somme directe) et les lemmes 2
à 4. Le théorème est **modeste mathématiquement**. Son poids vient de ce qu'il tranche une question de conception
réelle (quelle règle pour les MLP à porte) et de ce que sa conséquence non triviale (REC) a été prédite puis testée.

---

## 9. Verdict

**Y a-t-il un vrai théorème pertinent ? Oui, un, et modeste.**

Le **théorème 1** satisfait les cinq critères, avec une réserve franche sur le premier :
1. **Non trivial — limite.** Ce n'est ni une identité de rang ni une inégalité appliquée telle quelle : c'est une
   impossibilité pour toute une classe de règles, avec optimum exact et échappatoire unique. Mais la preuve tient en
   dix lignes de moindres carrés. Je le classe « théorème modeste », pas « théorème profond ».
2. **Pertinent.** Il explique le dilemme (b), effondrement contre dispersion contre biais de profondeur opposés, et
   il a fait une prédiction : l'échappatoire doit sortir de 𝒩. Cette prédiction est confirmée : REC, 0
   effondrement, géométrie 17,3 contre 17,0, fidélité 0,69, biais de profondeur 0,04. Le volet quantitatif (B1) est
   en revanche raté : REC est plus stable que prévu.
3. **Preuve complète.** Oui (§2).
4. **Hypothèses vérifiées sur les trois modèles.** Oui, aux 18 points pré-enregistrés, avec les poids réels.
5. **Nouveauté évaluée.** Oui (§8). Modeste mais réelle ; le plus proche est You et al. 2025.

**Ce qu'il n'explique pas :**
- **(a)** L'ordre Gemma ≫ Qwen ≫ GPT-2 est une loi empirique confirmée hors échantillon (mémoire de boucle,
  C1/C2), sans théorème.
- **(c)** Aucune relation quantitative démontrable ; les données contredisent la version simple.

**Comparaison avec ρ → 2.** ρ → 2 était un théorème limite à constante exacte, vérifié à résidu nul. Le théorème 1
est mathématiquement plus léger. Il est en revanche plus directement utile : il dit quelle linéarisation construire
pour les MLP à porte, et la linéarisation qu'il désigne a passé 3 tests pré-enregistrés sur 4.

**Limites :**
- Sous-couche MLP au dernier token, prompts courts.
- REC n'a été testée que sur Qwen ; Gemma et GPT-2 n'ont servi qu'aux hypothèses et à (a).
- La vérité terrain (ablation par la moyenne, dernier token) avantage les linéarisations proches du gradient.
- La classe 𝒩 exclut les règles qui modifient les poids (LRP-γ).

---

## 10. Fichiers (tous dans `notebook/`)

- **Livrables :** `wind_T20_theorem.md` (ce document), `wind_T20_preregistration.md` (prédictions et déviations
  D1–D3).
- **Scripts :**
  - `wind_T20_explore.jl` : gains radiaux, défauts d'Euler, découverte ;
  - `wind_T20_lift.jl` : relèvement exact, o, ℓ, M ;
  - `wind_T20_thm1.py` : constantes du théorème 1 sur poids réels ;
  - `wind_T20_rec.jl` : linéarisation REC = L*, tests B ;
  - `wind_T20_verdicts.py` : verdicts A, B, C.
- **Résultats :**
  - `wind_T20_thm1.json`, `wind_T20_thm1.log` ;
  - `wind_T20_rec.json`, `wind_T20_rec.log` ;
  - `wind_T20_wread.json` : lecture contrefactuelle W_U[réponse] − W_U[contrefactuel] ;
  - `wind_T20_lift_{qwen,gpt2,gemma}.json` ;
  - `wind_T20_explore_{qwen,gpt2,gemma}.json` ;
  - `wind_T20_verdicts_results.txt` ;
  - journaux `wind_T20_*.log`.
