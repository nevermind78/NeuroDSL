# WIND-T21 — Audit de rigueur de `theo.tex` (« Foundations of Geometric Interpretability »)

*2026-10-04. `theo.tex` n'est PAS modifié. Les tests sont exploratoires et pré-enregistrés dans
`wind_T21_preregistration.md` avant calcul ; script `wind_T21_tests.jl` ; résultats `wind_T21_tests_results.txt`
et `.json`. Le texte corrigé proposé est dans `wind_T21_theo_corrections.md`. Ce rapport est volontairement franc.*

---

## 0. Résumé exécutif

**Verdict global.** En l'état, le document n'est pas publiable. Il juxtapose trois choses de valeur très inégale.

1. **Des résultats empiriques solides, cités correctement** : le tableau WIND-T8, chiffres conformes aux fichiers.
2. **Des faits mathématiques vrais mais élémentaires** :
   - le Jacobien de la normalisation est un projecteur ;
   - la correction de gel est de rang 2 ;
   - l'identité de Sylvester.

   Leurs preuves sont souvent fausses dans le détail : mauvaise architecture, transposées incohérentes.
3. **Un habillage et des théorèmes qui sont faux ou ne sont pas démontrés** :
   - « invariance de jauge », « transport parallèle », « variété riemannienne courbe », « chaos » ;
   - le « théorème de Ho–Kalman », faux sur nos données (Spearman 0,28) ;
   - le « lemme d'inversion de Lyapunov », faux sur nos données ;
   - l'« entonnoir riemannien », faux dans sa direction ;
   - la « borne de Taylor–Hessienne », dimensionnellement incohérente ;
   - la description de NeuroDSL, qui ne correspond pas à ce qui a été fait ;
   - la mémoire VRAM « > 200 Go ».

**Les dix erreurs les plus graves :**

1. **Le théorème de bifurcation est faux sur les données, et ce n'est pas ce que le tableau T8 valide** (§4.3 ;
   test B). Appliquée à nos Jacobiennes, la formule det(I₂ − θH₂) = 0 avec H₂ = C(I − J)⁻¹B prédit θ_c avec un
   Spearman de 0,28 (θ_min) ou 0,22 (θ_méd), et ±15 % sur 2 % ou 30 % des prompts. Le tableau (0,971 ; 98 %) valide
   une autre méthode : une réalisation de Ho–Kalman d'ordre 2, VARIANT dans le temps, ajustée sur la boucle exacte
   relevée (T7/T8).
   - Le théorème suppose un système invariant dans le temps, qu'on n'a pas.
   - Il ne regarde que le passage d'une valeur propre en z = +1.
   - Il appelle G(1) une « réalisation de Ho–Kalman », ce qui est faux.
   - Pour un produit FINI de 27 blocs, il n'y a pas de bifurcation : (I − θW)⁻¹ est un polynôme en θ, car W est
     nilpotente.
   - (I − J_k^A) est très mal conditionnée : κ médian 1,6·10⁵, maximum 8,8·10⁷.
2. **Le lemme « Lyapunov Inversion » est faux** (test A). λ(J_exact) ≤ 0 n'est pas vrai partout : ‖J x̂‖ > 1 sur 21 %
   des couples (prompt, bloc) pour Qwen, 38 % pour GPT-2, 42 % pour Gemma. λ(J_AN) > 0 n'est pas strict : 4 % à 9 %
   des blocs ont ‖J^AN x̂‖ < 1. Et ‖J^AN x̂‖ n'est pas ≫ 1 : médiane 1,22 (Qwen, Gemma), 1,55 (GPT-2). La preuve
   suppose J x̂ ≈ x̂, faux pour un bloc pre-norm séquentiel, où le MLP voit h = x + a, pas x. « Chaotique » n'a pas de
   sens pour un produit fini de 27 matrices. Contre-exemple décisif : GPT-2 a le gain AN par bloc le plus fort
   (1,55) et ne s'effondre pas (ern AN/A 1,04).
3. **L'« entonnoir riemannien » est énoncé dans le mauvais sens** (test C). Quand k → L, erank(g_k) ne s'effondre
   pas : il CROÎT, de 4,4 (k = 1) à 452 (k = 27) en participation ; de 13,5 à 1 173 en entropie. La concentration
   vient des produits LONGS ; elle est construite progressivement par les couches ~10–27. La « preuve » est
   circulaire : elle suppose la décroissance spectrale. Le « r ≈ 16 » cité correspond à l'erank par ENTROPIE, pas à la
   définition donnée dans le texte (participation, qui vaut 4,4). L'entonnoir est déjà publié (Fernando &
   Guitchounts, arXiv 2605.14258) et n'est pas cité.
4. **La description de NeuroDSL ne correspond pas à ce qui a été fait.** Les Jacobiennes WIND ont été obtenues par
   différences finies centrées, bloc par bloc (seul le bloc k+1 est recalculé), avec portes ε contre 2ε. Les
   gradients viennent d'un backward instrumenté. Il n'y a pas eu de « Jacobiennes analytiques fermées propagées vers
   l'avant ». La formule J_{k+1} = J_k + ∂Attn/∂x J_k + ∂MLP/∂x J_k est fausse pour un bloc séquentiel. « Rank-1
   updates of SwiGLU » ne veut rien dire. Le stockage est en O(L·D²) (264 Mo par variante pour Qwen), pas O(D²).
5. **« > 200 Go de VRAM pour un seul token » est faux et non sourcé.** Une Jacobienne de bloc D×D au dernier token
   coûte D JVP en mode avant (jacfwd), ou des VJP par paquets. L'empreinte des activations d'un bloc pour 5–8 tokens
   est de l'ordre du mégaoctet. Le motif historique invoqué (« c'est pourquoi on gèle les normes ») est faux : on gèle
   pour la LINÉARITÉ et la conservation (Ali et al. 2022 ; graphes d'attribution), pas pour la mémoire.
6. **La « borne de Taylor–Hessienne » n'est pas une borne et elle est dimensionnellement incohérente.** On lit
   (J_exact − J_half)Δx ≈ ½ΔxᵀHΔx : un vecteur de ℝ^D égalé à un scalaire. Rien n'est démontré. La prémisse (« la
   règle du demi colle mieux aux ablations macroscopiques ») est contredite dans notre régime : le gradient exact est
   le plus fidèle (0,83), la règle du demi la moins fidèle (0,47). Ce n'est pas testé dans le régime de RelP. Une
   version EXACTE et correcte existe (§4.7) : sécante radiale le long du rayon vers 0, et théorème 1 de T20.
7. **L'architecture est fausse.** x_{k+1} = x_k + Attn(LN(x_k)) + MLP(LN(x_k)) est le bloc PARALLÈLE (GPT-J,
   PaLM). Qwen, Llama et GPT-2 sont séquentiels : h = x + Attn(LN(x)), x' = h + MLP(LN(h)). Gemma-2 a en plus deux
   post-normes par bloc. Conséquence : le « rang 2 exact » est faux pour Gemma-2, où le rang est ≤ 4 (vérifié : la
   queue au-delà du rang 4 vaut ≤ 3,4·10⁻⁴).
8. **Le rang 2 est faux si J_exact désigne la vraie dérivée T.** Si l'attention est vivante dans J_exact et figée
   dans J_AN, la différence contient les termes QK, de rang numérique 82–149 par couche (T6). Le rang ≤ 2 vaut pour
   J^AN − J^A, attention épinglée des deux côtés, ou pour « normes seules figées » contre T. Le texte ne précise pas
   lequel, et le θ-chemin réel de T8 part de J^A, pas de T.
9. **Survente du résumé :**
   - « full proofs » ;
   - « exact threshold » ;
   - « analytically predict » ;
   - « proving the intrinsic dimensional collapse » ;
   - « fully explaining the depth-bias » ;
   - « 98 % predictive accuracy on out-of-distribution prompts ». 98 % est la fraction à ±15 % ; les prompts sont
     NEUFS, des mêmes familles de tâches, pas hors distribution.
10. **Erreurs factuelles :**
    - D = 2048 pour « Gemma-2B » : Gemma-2-2B, le modèle étudié, a D = 2304.
    - « RelP (2026) » : RelP date de 2025 (arXiv 2508.21258).
    - La règle du demi n'est pas « proposée dans RelP » : elle vient de la LRP (règle uniforme d'AttnLRP 2024),
      reprise par RelP et Transluce.
    - **Aucune bibliographie** dans tout le document.

**Ce qui est récupérable**, avec énoncé et preuve corrects au §4 :
- le Jacobien exact de RMSNorm (avec ε et γ) ;
- un lemme de rang ≤ q (nombre de sites de normalisation par bloc, preuve générale) ;
- le relèvement exact (T7, lemme A) ;
- la version CORRECTE du critère de seuil (cas stationnaire : rayon spectral, réduction de Sylvester en +1, T7
  théorème 2) ;
- l'identité de la pente (T11) ;
- la sécante radiale exacte de la règle du demi ;
- surtout, le théorème 1 de T20, qui remplace avantageusement le « dilemme » et la « borne de Taylor–Hessienne ».

---

## 1. Inventaire des affirmations (statut)

Légende :
- **E** : exact et démontré dans le texte.
- **V-P** : vrai, mais la preuve est incomplète ou fausse.
- **V-H** : vrai seulement sous une hypothèse non énoncée.
- **F** : faux.
- **ND** : non démontrable, ou vocabulaire sans contenu.
- **S** : survente.

### Résumé (l. 98–104)

| # | Affirmation | Statut | Justification / donnée |
|---|---|---|---|
| R1 | « paradigme… flot géométrique discret sur une variété riemannienne primale » | ND | L'espace est ℝ^D euclidien plat. Aucune métrique non triviale ni courbure n'est utilisée dans le texte. |
| R2 | « attribution rigoureusement définie comme transport parallèle des 1-formes » | F (terminologie) / ND | C'est le tiré-en-arrière par la différentielle (ω_k = J_kᵀω_{k+1}), c'est-à-dire la rétropropagation. Un transport parallèle exige une connexion ; aucune n'est définie. |
| R3 | « preuves complètes que le gel viole l'invariance de jauge » | S + F (terme) | Le contenu vrai : chaque NORME est invariante d'échelle (Dn·x = (1−s)n ≈ 0), et le gel supprime cette invariance. Ce n'est pas une symétrie de jauge (locale) mais une symétrie d'échelle globale de chaque entrée de branche. Le réseau, lui, n'est pas invariant d'échelle. |
| R4 | « perturbation duale de rang 2 mathématiquement précise » | V-H | Rang ≤ 2 pour les blocs à 2 normes (Qwen, GPT-2), ≤ 4 pour Gemma-2, à condition que l'attention soit traitée de la même façon des deux côtés. |
| R5 | « du régime de Lyapunov strictement dissipatif (λ ≤ 0) au régime expansif chaotique (λ > 0) » | F + S | Test A : ‖J_exact x̂‖ > 1 sur 21–42 % des blocs ; ‖J_AN x̂‖ < 1 sur 4–9 %. Il n'y a pas de chaos dans un produit fini. |
| R6 | « prédire analytiquement le seuil EXACT θ_c par Ho–Kalman et Sylvester » | F + S | La formule du texte échoue (test B : Spearman 0,28). Le résultat validé (T8) vient d'une réalisation d'ordre 2 AJUSTÉE sur la boucle exacte relevée : précision ±15 % sur 98 %. Ce n'est ni exact ni purement analytique. |
| R7 | « NeuroDSL calcule le transport dual exact sans explosion mémoire » | F (description) | Différences finies par bloc, pas d'algèbre fermée (voir 5.4). |
| R8 | « formaliser l'entonnoir riemannien, en DÉMONTRANT l'effondrement dimensionnel » | F (direction) + ND (preuve) + S | Test C ; la preuve est circulaire ; résultat antérieur publié. |
| R9 | « proposer l'architecture Funnel-LLM » | ND / spéculatif | Non testé. L'inférence est invalide : le sous-espace dominant dépend du prompt (WIND-1 H2b : moyenner les Jacobiennes FAIT MONTER le rang, 96 contre 20). |
| R10 | « borne d'erreur de Taylor–Hessienne… expliquant entièrement le biais de profondeur » | F (incohérente) + S | Voir 7.4 ; ce n'est pas une borne ; non testé dans le régime RelP. |
| R11 | « 98 % de précision prédictive quantitative sur des prompts hors distribution » | S | 98 % = fraction à ±15 % (49/50) ; prompts neufs mais de mêmes familles ; « accuracy » est impropre. |

### Chapitre 1 (l. 109–154)

| # | Affirmation | Statut | Justification |
|---|---|---|---|
| 1.1 | Variété X ≅ ℝ^D ; D = 1536 (Qwen-1.5B), D = 2048 (« Gemma-2B ») | F (Gemma) | Gemma-2-2B : hidden_size 2304 (`gemma-2-2b/config.json`). 2048 est la dimension de Gemma-1 2B. |
| 1.2 | Définition de l'espace tangent | E (standard) | — |
| 1.3 | Définition de l'espace cotangent et des 1-formes | E (standard) | — |
| 1.4 | « l'attribution est une 1-forme » | E mais triviale | Un gradient est un covecteur. Dans ℝ^D euclidien, l'identification est canonique ; rien n'en dépend. |
| 1.5 | « flot géométrique discret : suite de difféomorphismes » | V-H / ND | L'inversibilité des blocs n'est ni montrée ni utilisée. σ_min/σ_max d'un bloc ≈ 10⁻⁴ : ce sont des blocs presque singuliers (test C). « Flot géométrique » désigne d'habitude une évolution de métriques : terme impropre. |
| 1.6 | FTLE λ(J,v) = ln‖Jv‖ ; dissipatif si λ < 0, chaotique si λ > 0 | F | Ce n'est pas un exposant (pas de normalisation par le nombre de pas, pas de limite, une seule direction). « Dissipatif » se rapporte au volume (déterminant). Le chaos exige un exposant ASYMPTOTIQUE positif dans un système borné ; un étirement en un pas n'est pas du chaos. |
| 1.7 | Théorème du déterminant de Sylvester | E (classique) | Correct. La phrase « clé de voûte de la réduction de dimension… réalisation minimale » est fausse : Ho–Kalman repose sur la factorisation de la matrice de Hankel, pas sur Sylvester. |

### Chapitre 2 (l. 157–207)

| # | Affirmation | Statut | Justification |
|---|---|---|---|
| 2.1 | x_{k+1} = x_k + Attn(LN(x_k)) + MLP(LN(x_k)) | F | Bloc parallèle ; Qwen, Llama et GPT-2 sont séquentiels ; Gemma-2 a des post-normes. Le bloc mélange aussi les positions : Φ_k n'est une application de ℝ^D dans ℝ^D qu'à états des autres positions fixés (bloc diagonal du dernier token). |
| 2.2 | « la sémantique voyage strictement vers l'avant » | ND | — |
| 2.3 | ω_k = J_kᵀω_{k+1} (« transport parallèle ») | E pour la formule ; F pour le nom | C'est la règle de la chaîne en mode inverse. Ce n'est pas un transport parallèle. |

### Chapitre 3 (l. 210–256)

| # | Affirmation | Statut | Justification |
|---|---|---|---|
| 3.1 | « la communauté gèle le dénominateur pour réduire le coût de calcul » | F | Le motif est la LINÉARITÉ et la conservation (LRP : Ali et al. 2022 ; graphes d'attribution : linéarité des arêtes), pas le coût. |
| 3.2 | « RMSNorm projette x sur une hypersphère e_x = x/‖x‖ » | V-H | Vrai à γ, √D et ε près : RMSNorm(x) = √D·γ⊙x/√(‖x‖² + Dε). |
| 3.3 | Lemme du projecteur ∂e_x/∂x = Π_x^⊥/‖x‖ | E pour e_x | Pour RMSNorm, le Jacobien vaut (Γ/ρ)(I − s x̂x̂ᵀ) avec s = ‖x‖²/(‖x‖²+Dε) < 1 ; ce n'est pas exactement un projecteur (1 − s ≈ 10⁻⁵). |
| 3.4 | « le réseau est invariant de jauge à l'échelle » | F | C'est la NORME (donc l'entrée de chaque branche) qui est invariante d'échelle, pas le réseau. Le bloc transmet la direction radiale par le saut : J x ≈ x + M Q₂ x ≠ 0. Le terme « jauge » est impropre. |
| 3.5 | Théorème « rang 2 exact par bloc » | V-P et V-H | **Énoncé :** « exactement rang 2 » doit se lire « rang ≤ 2 » (génériquement 2). Ce n'est vrai que si l'attention est traitée pareil des deux côtés, et seulement pour 2 normes par bloc (Gemma-2 : ≤ 4). **Preuve :** elle porte sur le mauvais bloc (parallèle). Les transposées sont incohérentes : (∂M/∂e·Π/‖x‖)ᵀ dans une Jacobienne avant. u_m est défini avec une transposée puis utilisé sans. **Version correcte :** §4.2. |

### Chapitre 4 (l. 259–331)

| # | Affirmation | Statut | Justification |
|---|---|---|---|
| 4.1 | Lemme « Lyapunov Inversion » | F | Test A (voir §3). La preuve suppose J_exact e_x ≈ e_x (faux pour le bloc séquentiel), puis écrit « = ln 1 = 0 » (passage illégitime de ≈ à =). |
| 4.2 | Système z_{k+1} = J z_k + Bv_k, v_k = θCz_k, avec J, B, C constants | F (hypothèse cachée) | Le système réel varie avec la couche (J_k, B_k, C_k). La figure, « True Plant (I − J)⁻¹ », est la résolvante en z = 1 d'un système invariant : elle ne s'applique pas. |
| 4.3 | Théorème « Ho–Kalman et seuil de bifurcation » | F (sur les données), V-H (dans un cadre idéalisé) | **Hypothèses cachées :** invariance dans le temps, passage en z = +1, I − J inversible. Sur Qwen : Spearman 0,28 et 0,22 ; ±15 % sur 2 % et 30 % (test B). Le pas « Sylvester » est algébriquement correct. Le critère correct pour un système invariant est ρ(Φ + θBC) = 1 (T7, théorème 2). Pour un produit fini, il n'y a pas de bifurcation (relèvement : polynôme en θ). H₂ = G(1) n'est pas une réalisation de Ho–Kalman. |
| 4.4 | « projeter l'erreur de dimension 1536 dans un sous-espace minimal 2×2 » | F (description) | La boucle exacte a la dimension 2L = 54, variant dans le temps. L'ordre 2 est une RÉDUCTION approchée, ajustée sur cette boucle (σ₁ à 1–8 %, T7). |

### Chapitre 5 (l. 334–351)

| # | Affirmation | Statut | Justification |
|---|---|---|---|
| 5.1 | « PyTorch/JAX sont conçus pour des VJP, pas pour extraire des matrices » | S | torch.func.jacfwd/jacrev et jax.jacfwd existent et extraient des Jacobiennes D×D par blocs. |
| 5.2 | Mémoire O(B·S·L·D²) ; « > 200 Go de VRAM pour un seul token » | F | Ordre de grandeur pour un bloc de Qwen à S = 6 tokens : 1536 VJP × activations d'un bloc (≈ S·D·10 flottants) ≈ 0,5 Go. Par paquets, c'est sans limite pratique. Aucune source ni mesure. |
| 5.3 | « raison historique du recours au gel des normes » | F | Voir 3.1. |
| 5.4 | « NeuroDSL abandonne la différentiation automatique, propage des Jacobiennes analytiques fermées » ; récurrence J_{k+1} = … ; « mises à jour de rang 1 du SwiGLU » ; mémoire O(D²) | F | Méthode réelle (WIND-1, T8) : différences finies centrées bloc par bloc, en ne recalculant que le bloc k+1, avec portes. La récurrence est fausse pour un bloc séquentiel. La dérivée du SwiGLU n'est pas de rang 1. Le stockage est O(L·D²). |

### Chapitre 6 (l. 354–376)

| # | Affirmation | Statut | Justification |
|---|---|---|---|
| 6.1 | g_k = TᵀT est une « métrique riemannienne » | F (terme) | C'est une forme bilinéaire semi-définie positive, dégénérée, le long d'UNE trajectoire. Une métrique riemannienne doit être définie positive et former un champ lisse. |
| 6.2 | « quand k → L, erank(g_k) passe de 1536 à r ≈ 16 » | F | Test C : erank_PR 4,4 (k = 1) → 452 (k = 27), croissance en k. |
| 6.3 | « preuve » par SVD et décroissance exponentielle du spectre | ND (circulaire) | Elle suppose ce qu'elle démontre. Elle confond « erank faible » (concentration du spectre) avec « noyau » (valeurs singulières nulles). |
| 6.4 | « tout v ∈ ker(g_k) donne ds² = 0 » | V-H | Tautologique. Des directions quasi dégénérées existent bien : σ_min/σ_max ≈ 10⁻¹⁹ pour les produits longs, ≈ 10⁻⁴ pour un bloc (test C). Mais un noyau exact n'est pas résoluble à la précision des différences finies (~10⁻⁴), et il n'a rien à voir avec les « 1536 − 16 » dimensions. |
| 6.5 | « goulot sémantique de 16 dimensions pour forcer la classification du vocabulaire » | ND | Interprétation sans test. |
| 6.6 | Funnel-LLM : « réduire D_k sans perdre de capacité géométrique » | ND / contredit | L'erank est faible pour UNE lecture et UN prompt. Le sous-espace dominant change d'un prompt à l'autre (WIND-1 H2b), et les autres positions et lectures utilisent les autres directions. |

### Chapitre 7 (l. 379–432)

| # | Affirmation | Statut | Justification |
|---|---|---|---|
| 7.1 | « règle du demi proposée dans RelP (2026) » | F | RelP : arXiv 2508.21258, 2025. La règle vient de la LRP (règle uniforme d'AttnLRP, 2024). |
| 7.2 | « la littérature soutient que J_AH colle mieux aux ablations macroscopiques » | V-H | RelP compare RelP et AtP à l'activation patching (IOI). Le régime n'est pas le nôtre. |
| 7.3 | Développement de Taylor ΔΦ = ⟨∇Φ, Δx⟩ + ½ΔxᵀHΔx + O(‖Δx‖³) | E (standard) | Le terme « Exact Tangent (J_exact) » est mal étiqueté : c'est ∇Φ. |
| 7.4 | Proposition « sécante » : (J_exact − J_half)Δx ≈ ½ΔxᵀHΔx | F (incohérente) + ND | Vecteur égalé à un scalaire, aucune dérivation. La version correcte, exacte et limitée à la direction RADIALE, est au §4.7. |
| 7.5 | « pour les ablations macroscopiques, la courbure domine » | V-H / contredit dans notre régime | Ablation par la moyenne au dernier token : le gradient exact est le plus fidèle (ρ 0,83 ; T9). |
| 7.6 | « la règle du demi écrase le routage local exact et crée un biais de profondeur » | V-P | Le biais est mesuré (exploratoire, puis pré-enregistré en T20-B4). Le mécanisme démontré est la distorsion transverse forcée par la complétude (T20, théorème 1 : r_½ = 0,51–0,64), pas la courbure. |
| 7.7 | Tableau T8 (C1 0,971 ; C2 98 % ; C3 0,974 ; C4 ; D1 24 %) | E (chiffres) / S (légende) | Les chiffres sont conformes à `wind_T8_truth_results.txt`. Mais « OOD » est faux, « exact moment » est une survente, C3 omet les 2 faux positifs, et SURTOUT le tableau valide la réalisation d'ordre 2 variant dans le temps (T7), PAS le théorème du chapitre 4. |
| 7.8 | Tableau « Depth Bias » 33/28/52/11 % | V-H (provenance cachée) | Ce sont les médianes sur les 14 prompts « effondrés prédits » (θ_c^mod < 1), 56 sous-couches, exploratoires (mon audit précédent). Sur les 36 autres : 30/26/37/9 %. « Horizontal routing » est faux : seuls les chemins verticaux au dernier token sont mesurés. Version pré-enregistrée (T20-B4, 49 prompts, 28 MLP) : vérité 21,1 %, T 23,7 %, ANf 26,2 %, AHf 10,5 %, REC 20,2 %. |

### Conclusion (l. 434–435)

| # | Affirmation | Statut | Justification |
|---|---|---|---|
| 8.1 | « dilemme mathématique impossible : une seule matrice du premier ordre ne peut pas… » | F tel quel ; V sous forme précise | Le dilemme démontré (T20, théorème 1) vaut à l'intérieur de la classe des règles neurone par neurone, et on en SORT : L* est complète ET transverse-exacte. Elle a été testée (0/49 effondrements, géométrie 17,3 contre 17,0, fidélité 0,69). |
| 8.2 | « prouvant que les LLM doivent être analysés comme flots non linéaires sur variétés riemanniennes courbes » | ND | Aucune courbure n'est définie ni calculée. |

**Transversal.** Il n'y a aucune bibliographie. Il manque au minimum :
- Ali et al. 2022 ; Achtibat et al. 2024 (AttnLRP) ; RelP 2025 ; Arora et al. 2026 (Transluce) ;
- Fernando & Guitchounts 2026 ; Ho & Kalman 1966 ; Dewilde & van der Veen 1998 ;
- Sundararajan et al. 2017 ; You et al. 2025 ; Rushing & Nanda 2024 ; Ameisen et al. 2025.

---

## 2. Confrontation avec nos résultats vérifiés (WIND-T8 à T20)

| Affirmation du texte | Nos données | Verdict |
|---|---|---|
| Correction de gel de rang 2 | T6/T7 : σ₃/σ₁(J^AN − J^A) ≈ 3·10⁻⁴ (Qwen) ; T20 lift : queue ≤ 10⁻³ (Qwen, GPT-2, rang 2), ≤ 3,4·10⁻⁴ (Gemma, **rang 4**) | **confirmé** pour Qwen et GPT-2 ; **contredit** pour Gemma-2 (4, pas 2) |
| Le vrai réseau contracte le radial ; le gel l'amplifie | gain d'état λ_T ≈ 0,846 (Qwen), 0,938 (GPT-2) ; λ_AN ≈ 1,130 et 1,411 (T17) ; ρ relatif : T 0,75, AN 1,04, AH 0,93 (T20) | **confirmé en médiane** ; **contredit** comme énoncé universel (test A) |
| λ > 0 ⇒ « chaos » et effondrement | GPT-2 : ‖J^AN x̂‖ médian 1,55 > Qwen 1,22, et pourtant pas d'effondrement (T17, ern AN/A 1,04) | **contredit** |
| θ_c « exact » par Ho–Kalman/Sylvester | T8 : réalisation d'ordre 2 ajustée, Spearman 0,971, ±15 % sur 98 % ; formule du texte : 0,28 / 2 % (test B) | le tableau est **confirmé** ; le théorème est **contredit** |
| 24 % d'effondrements (IC 14–37 %) | T8 D1 : 12/50, Wilson [0,143 ; 0,374] | **confirmé** |
| C3 « 12/12 effondrements détectés » | VP 12, FN 0, **FP 2**, VN 36 ; exactitude équilibrée 0,974 | **confirmé**, mais les 2 faux positifs sont omis |
| Entonnoir r ≈ 16 aux couches profondes | erank entropie de N·P_T (bout en bout) médian 17,0 (T8, 50 prompts) ; il croît quand k → L (test C) | le nombre est **confirmé** (mais avec la définition par entropie, et de bout en bout) ; la direction est **contredite** |
| Règle du demi meilleure pour les ablations macroscopiques | dans notre régime : la PIRE (ρ 0,47 contre T 0,83 ; T15) | non testé dans le régime RelP ; **contredit** dans le nôtre |
| Biais de profondeur 33/28/52/11 | groupe effondré, exploratoire ; pré-enregistré T20-B4 : 21,1 / 23,7 / 26,2 / 10,5 % | **confirmé qualitativement** ; provenance cachée |
| « Dilemme impossible » | T20 : impossibilité pour la classe 𝒩, échappatoire L* testée (B2–B4 vrais) | **réfuté** sous la forme générale ; **remplacé** par un énoncé exact |
| VRAM, NeuroDSL analytique | méthode réelle : différences finies par bloc ; aucune mesure de VRAM | **contredit** (description) ; non testé (VRAM) |
| Funnel-LLM | WIND-1 H2b : la moyenne des Jacobiennes a un erank de 96 contre 20 par prompt | **contredit** dans sa justification |
| Taylor–Hessienne explique RelP | non testé (régime IOI, grandes perturbations) | **non testé** |

**Chiffres du texte non conformes aux fichiers :**
- D(Gemma) = 2048 au lieu de 2304.
- « r ≈ 16 » présenté avec la définition par participation, qui donne en fait 4,4 de bout en bout.
- « 98 % accuracy » : c'est la fraction à ±15 %.
- Le tableau de profondeur est sans groupe ni composantes précisés.

Les autres chiffres du tableau T8 sont exacts.

---

## 3. Tests exploratoires T21 (pré-enregistrés, `wind_T21_tests_results.txt`)

### Test A — lemme d'inversion de Lyapunov

s_k = ln‖J_k x̂_k‖ ; Qwen et GPT-2 : 10 prompts × blocs ; Gemma : 5 prompts.

| Modèle | exact : fraction s > 0 | exact : ‖J x̂‖ médian | AN : fraction s > 0 | AN : ‖J x̂‖ médian [q10 ; q90] |
|---|---|---|---|---|
| Qwen (T) | 0,207 | 0,957 | 0,956 | 1,217 [1,05 ; 1,43] |
| GPT-2 (T) | 0,382 | 0,990 | 0,991 | 1,552 [1,21 ; 1,85] |
| Gemma (A) | 0,416 | 0,987 | 0,912 | 1,215 [1,01 ; 1,38] |

- **A1 FAUX** (prédit FAUX).
- **A2 FAUX** (prédit : vrai sur ≥ 80 % mais pas 100 % ; c'est le cas, 91–99 %).
- **A3 FAUX** (prédit FAUX).
- La partie juste du lemme : en MÉDIANE, le vrai bloc ne dilate pas la direction radiale (≈ 0,96–0,99) et le bloc
  figé la dilate modérément (×1,22 à ×1,55). Ce n'est ni universel ni « ≫ 1 », et ce n'est pas un prédicteur
  d'effondrement (GPT-2).

### Test B — théorème « Ho–Kalman/Sylvester »

Qwen, 50 prompts, facteurs exacts T13, porte de rang 2 ≤ 4,8·10⁻³.

| Prédicteur | Spearman(θ_pred, θ_c) | fraction ±15 % | médiane θ_pred (θ_c vrai : 1,136) |
|---|---|---|---|
| θ_min = min_k θ^(k) | 0,282 | 0,020 | 0,219 |
| θ_méd = médiane_k θ^(k) | 0,215 | 0,300 | 0,991 |

- κ(I − J_k^A) : médiane 1,6·10⁵, maximum 8,8·10⁷.
- **B1 FAUX, B2 FAUX** (prédits FAUX).
- La proximité des médianes (0,99 contre 1,14) est une coïncidence : il n'y a aucun accord de rang.

### Test C — entonnoir riemannien

Qwen, variante T, 10 prompts ; médianes par k.

| k | 1 | 5 | 10 | 15 | 19 | 23 | 26 | 27 |
|---|---|---|---|---|---|---|---|---|
| erank_PR(g_k) (déf. du texte) | 4,4 | 5,2 | 6,1 | 13,7 | 33,5 | 224 | 490 | 452 |
| erank entropie | 13,5 | 16,2 | 23,8 | 56,7 | 142 | 574 | 1 027 | 1 173 |
| erank_PR avec N | 4,7 | 5,7 | 7,5 | 14,8 | 40,0 | 183 | 307 | 322 |

- **C1 FAUX** : la direction est inversée ; erank_PR(g_27) > erank_PR(g_1) sur 10/10 prompts.
- **C2 VRAI**, contrairement à ma prédiction : σ_min/σ_max descend à 9·10⁻²³ pour les produits longs (≈ 10⁻¹⁹ dès
  k ≤ 15) et vaut ≈ 10⁻⁴ pour un seul bloc. Ces directions quasi dégénérées existent bien. Mais 10⁻¹⁹ est sous la
  précision de Float64, et ~10⁻⁴ est le plancher des différences finies : on ne peut pas affirmer un NOYAU exact. En
  tout état de cause, ces directions sont distinctes de la concentration du spectre, qui est le vrai phénomène
  d'entonnoir.

### Contrôle D — provenance

- Tableau T8 : conforme, avec les 2 faux positifs de C3 omis.
- Tableau de profondeur : groupe effondré, 56 sous-couches, exploratoire.
- Gemma-2-2B : D = 2304.

---

## 4. Ce qui se démontre ou se remplace (énoncés corrects et preuves)

### 4.1 Jacobien exact de RMSNorm (remplace le lemme 3.1 et l'« invariance de jauge »)

n(x) = γ⊙x/ρ(x), avec ρ(x) = (‖x‖²/D + ε)^{1/2}. Alors

  Dn(x) = (Γ/ρ)(I − s x̂x̂ᵀ),   s = ‖x‖²/(‖x‖² + Dε) ∈ (0, 1),   Dn(x)x = (1 − s)n(x).

*Preuve.* ∂ρ/∂x = x/(Dρ). Donc ∂(x_i/ρ)/∂x_j = δ_ij/ρ − x_ix_j/(Dρ³) = (1/ρ)(δ_ij − x_ix_j/(Dρ²)), et
Dρ² = ‖x‖² + Dε. ∎

Version figée : F = Γ/ρ, et F − Dn(x) = s·n(x)xᵀ/‖x‖² (rang 1). On a l'invariance d'échelle n(cx) = n(x) pour ε = 0.
C'est une symétrie de chaque entrée de branche, pas du réseau.

### 4.2 Lemme de rang ≤ q (remplace le théorème 3.2 ; vaut pour Qwen, GPT-2 et Gemma)

**Énoncé.** Soit un bloc dont la Jacobienne (pour une linéarisation donnée des autres opérations, par exemple
l'attention épinglée) est la composée de q sites de normalisation d'entrées z_1, …, z_q, rangés dans un ordre
topologique. Figer les q dénominateurs change la Jacobienne du bloc d'une matrice de rang ≤ q. Son espace des lignes
est engendré par les capteurs vivants c_i = (∂z_i/∂x)ᵀ_vivant ẑ_i.

**Preuve.**
- Au site i, Dn_i devient Dn_i + E_i avec E_i = s_i n_i(z_i)z_iᵀ/‖z_i‖² (§4.1).
- On développe la composée : la différence est la somme, sur les sous-ensembles non vides S de sites, des chaînes
  qui portent E_i aux sites i ∈ S et Dn_j ailleurs.
- Soit i le premier site de S. Tous les sites qui le précèdent sont vivants. La chaîne s'écrit donc
  (…)·n_i(z_i)·[z_iᵀ(∂z_i/∂x)_vivant]/‖z_i‖², de ligne c_iᵀ indépendante de S.
- En regroupant selon i, la différence vaut Σ_{i=1}^{q} w_i c_iᵀ : rang ≤ q. ∎

**Cas Qwen**, bloc séquentiel, attention épinglée (T13), forme fermée :

  J^AN − J^A = s₂(Mĥ)ĥᵀ(I + A₁) + s₁(I + MQ₂)A₁x̂ x̂ᵀ.

**Vérifié :**
- rang 2 pour Qwen et GPT-2 ; rang 4 pour Gemma-2 (2 pré-normes + 2 post-normes) ;
- queues ≤ 10⁻³ ; porte de reconstruction ≤ 4,8·10⁻³ (T13, T21-B).

Si l'attention est VIVANTE d'un côté seulement, la différence contient aussi les termes QK (rang 82–149) : le
lemme ne s'applique qu'à attention traitée de façon identique des deux côtés.

### 4.3 Relèvement exact et absence de bifurcation à profondeur finie (remplace le théorème 4.2)

**Énoncé (T7, lemme A).** On pose J_k(θ) = A_k + θU_kV_kᵀ. Alors

  N·Φ_θ(L+1, 1) = M_A + θ𝒰(I − θW)⁻¹𝒱,

avec W strictement triangulaire inférieure par blocs, donc nilpotente. (I − θW)⁻¹ = Σ_{m<L} θ^mW^m est un polynôme
en θ : aucun θ fini ne rend le produit singulier ou « bifurquant ». θ_c est un seuil sur l'erank, pas un pôle.

*Preuve :* T7 §2, par développement du produit et regroupement des chaînes croissantes.

*Vérifié :* portes ≤ 3,6·10⁻⁴ (T7), ≤ 2,7·10⁻⁴ (T20, 3 modèles).

### 4.4 Version CORRECTE du critère de Sylvester : cas stationnaire (T7, théorème 2)

**Énoncé.** Si A_k ≡ Φ, U_k ≡ B et V_kᵀ ≡ C (invariance dans le temps, hypothèse fausse sur les LLM réels), le mode
résonant apparaît exactement quand ρ(Φ + θBC) franchit 1. On a alors (1/L) ln σ₁(G_L) → ln ρ, et
σ₁(G_L)|λ_u|^{−L} → ‖Ce‖‖fᵀB‖/(λ_u² − 1).

**Lien avec le texte.** Dans le SOUS-cas où la valeur propre qui franchit est +1 et où I − Φ est inversible :
det(I − Φ − θBC) = 0 ⟺ det(I₂ − θC(I − Φ)⁻¹B) = 0, par Sylvester. C'est la formule du texte, juste sous ces deux
hypothèses.

**Test B.** Appliquée bloc par bloc aux données réelles (variant dans le temps), elle échoue : Spearman 0,28.

**Ce qui prédit θ_c (T8, pré-enregistré).** Une réalisation de Ho–Kalman d'ordre 2 variant dans le temps, ajustée sur
la boucle exacte W, sans aucun produit vrai à θ ≠ 0 : Spearman 0,971, ±15 % sur 98 %.

### 4.5 Pente du mode dominant (T11 ; énoncé et preuve)

**Énoncé.** Si σ₁(θ) de M(θ) = N·Π_k J_k(θ) est simple, de vecteurs singuliers (u, v), alors

  dσ₁/dθ = Σ_k uᵀ N P_θ(L←k+1) R_k P_θ(k←1) v.

*Preuve.* La perturbation au premier ordre d'une valeur singulière simple vaut dσ₁ = uᵀ(dM)v. La règle du produit
donne dM/dθ = Σ_k N P(L←k+1)R_k P(k←1). ∎

*Vérifié :* T11, porte ≤ 6·10⁻⁷ contre différences finies.

### 4.6 Remplace le « dilemme » et la « borne de Taylor–Hessienne » : théorème 1 de T20 (démontré, vérifié)

On prend un MLP à porte, une norme d'entrée figée, J la dérivée exacte et E = Jh − f(v) l'excès d'Euler.
- **(a)** La seule linéarisation à la fois complète (Lh = f) et transverse-exacte (LP = JP sur h^⊥) est
  L* = J − Ehᵀ/‖h‖².
- **(b)** Supposons que la Gram de Hadamard soit définie positive (vérifié sur poids réels, 18 points, 3 modèles).
  Alors toute règle neurone par neurone complète (règle du demi, règle uniforme…) a une distorsion transverse
  ≥ τ*, avec τ*² = Eᵀ(Λ_h𝒢⁻¹Λ_hᵀ)⁻¹E.
  - Mesures : r* = τ*/‖JP‖_F = 0,15–0,23 (Qwen, Gemma), 0,23–0,61 (GPT-2) ; règle du demi r_½ = 0,51–0,64.
- **Test de L\*** (REC, Qwen, 49 prompts, pré-enregistré) :
  - 0 effondrement ;
  - erank 17,3 contre 17,0 pour la vérité (AN 5,45 ; AH 91,6) ;
  - fidélité 0,691 (ANf 0,593, AHf 0,559, AAf 0,729) ;
  - part des couches précoces 0,202, contre 0,211 pour la vérité (ANf 0,262, AHf 0,105) ;
  - un sous-critère manqué (B1-Spearman).

*Preuve et données :* `wind_T20_theorem.md`.

### 4.7 Version exacte de l'intuition « sécante » (nouvelle, élémentaire)

**Énoncé.** On fige σ à σ₀ = σ(g₀) : f̃(v) = W_d((σ₀⊙W_g v)⊙W_u v) est bilinéaire, homogène de degré 2, et coïncide
avec f en v₀. Alors :
- (i) la règle du demi vaut exactement L_½ = ½Df̃(v₀) = ∫₀¹ Df̃(t v₀) dt. C'est l'opérateur sécant (gradient intégré
  depuis la base 0) le long du RAYON [0, v₀].
- (ii) pour toute lecture linéaire r, ⟨r, (Df̃(v₀) − L_½)v₀⟩ = ½ v₀ᵀ∇²(r·f̃)v₀. C'est exactement le terme de courbure
  de Taylor, le long du rayon seulement.

*Preuve.*
- f̃ est bilinéaire, donc Df̃(tv) = t·Df̃(v), et ∫₀¹ t dt = ½.
- Pour une forme quadratique q(v) = r·f̃(v), on a ∇q(v)·v = 2q(v) = vᵀ∇²q v. Donc (Df̃ − ½Df̃)v₀ projeté sur r vaut
  ½vᵀ∇²q v. ∎

**Portée.** C'est la seule forme exacte de la « sécante » : elle concerne l'ablation de l'entrée ENTIÈRE vers 0
(direction radiale), pas une ablation Δx quelconque. Pour Δx quelconque, aucune identité de ce type n'existe. Le
théorème de T20 montre que gagner l'exactitude radiale avec une règle neurone par neurone coûte au moins τ* sur
les directions transverses. C'est le mécanisme démontré du biais de profondeur, à la place d'un argument de courbure.

### 4.8 Entonnoir (énoncé empirique correct)

**Énoncé (empirique).** Pour Qwen2.5-1.5B au dernier token, le produit de bout en bout N·P_T a un erank (entropie)
médian de 17,0 sur 50 prompts (T8). L'erank de g_k = T_{k→L}ᵀT_{k→L} DÉCROÎT quand k diminue : de ~1 000 (k = 27) à
13,5 (k = 1). La concentration est construite cumulativement par les couches ~10–27 (WIND-T3 : ensemble de jonctions
11–26).

C'est empirique et cohérent avec Fernando & Guitchounts 2026, qui doit être cité. La version de T4 n'est PAS un
théorème.

---

## 5. Verdict et publiabilité

**Rigueur globale : faible.** Sur 50 affirmations inventoriées :
- 8 sont exactes (définitions standard, Sylvester, formule de la rétropropagation, Taylor, projecteur de e_x,
  chiffres T8) ;
- 8 sont vraies avec une preuve fausse ou une hypothèse cachée (rang 2, RMSNorm ↔ sphère, « kernel », biais de
  profondeur, régime RelP) ;
- 22 sont fausses (Lyapunov, Ho–Kalman sur données, entonnoir, NeuroDSL, VRAM, architecture, D de Gemma, RelP 2026,
  sécante, « jauge », FTLE, motif du gel, dilemme général) ;
- les 12 restantes sont du vocabulaire décoratif ou de la survente.

Les erreurs ne sont pas cosmétiques : le théorème central du chapitre 4 est faux sur les données, et le tableau qui
semble le valider porte sur une autre méthode. Un relecteur qui reproduirait la formule le découvrirait
immédiatement.

**Ce qui est publiable**, et sous quelle forme. Pas comme « monographie de géométrie riemannienne », mais comme un
article technique au cadre honnête, en réutilisant ce qui est vérifié :
1. les lemmes exacts (§4.1–4.3, 4.5) ;
2. la prédiction pré-enregistrée de θ_c par réalisation d'ordre 2 variant dans le temps (T8) ;
3. le théorème « conservation contre exactitude transverse » de T20, avec ses constantes sur poids réels et le test
   de L* ;
4. la mémoire de boucle entre modèles, comme loi empirique (T20-C).

Venue : TMLR, ou un atelier d'interprétabilité. Le texte corrigé proposé (`wind_T21_theo_corrections.md`) va dans ce
sens. Il garde le vocabulaire dual (covecteurs, tiré-en-arrière), qui est correct, et supprime « jauge », « transport
parallèle », « chaos », « courbure » et Funnel-LLM, faute de contenu démontré.

**Risque si le texte part en l'état.** Un relecteur compétent (contrôle, systèmes dynamiques, interprétabilité)
trouverait en quelques minutes :
- le bloc parallèle ;
- le théorème invariant dans le temps appliqué à un produit fini ;
- la direction de l'entonnoir ;
- la Taylor vecteur = scalaire ;
- l'absence de bibliographie ;
- D = 2048.

La crédibilité des résultats réellement solides (T8, T20) en pâtirait.

---

## 6. Fichiers (tous dans `notebook/`)

- **Livrables :** `wind_T21_theo_audit.md` (ce document), `wind_T21_theo_corrections.md` (texte de remplacement).
- **Pré-enregistrement :** `wind_T21_preregistration.md`.
- **Tests :**
  - script `wind_T21_tests.jl` ;
  - résultats `wind_T21_tests_results.txt`, `wind_T21_tests.json` ;
  - journal `wind_T21_tests.log`.
