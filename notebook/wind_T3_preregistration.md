# WIND-T3 / WIND-T4 — pré-enregistrement (2026-09-30, écrit AVANT les runs)

Contexte acquis (WIND-T2/T2b, 5 prompts, variante T) : après la RMSNorm finale, erank(N·J_{1→28}) = 15,8
(médiane) contre 45,8 pour les substituts à signes aléatoires (qui gardent spectres ET appariements P_k)
et 52,5 pour Haar. Théorème (impossibilité) : toute statistique fonction des spectres et des P_k est
identique sur le vrai réseau et sur ses substituts à signes → l'entonnoir vit dans une cohérence de phase
multi-couches. Métrique commune : ern = erank₂ de N·T, N = Jacobien de la norme finale.

Construction : randomiser la jonction k (entre J_k et J_{k+1}, k = 1..26) = insérer une réflexion
aléatoire R_k = U_k D_k U_kᵀ, i.e. J_k → (U_k D_k) Σ_k V_kᵀ. J_27 n'est jamais modifié.

## WIND-T3 — localisation (script `wind_localize_check.jl`)

On garde les vraies phases sur un ensemble W de jonctions, on randomise les autres (3 tirages).
Fraction de récupération par prompt : ρ(W) = [log ern_aucune − log ern_W] / [log ern_aucune − log ern_vrai].

- Porte GL : W = toutes → ern identique au vrai (écart relatif < 1e-8).
- HL1 (issue de WIND-1 : « l'entonnoir se construit dans les ~10 dernières couches ») :
  médiane ρ(suffixe ≥16) ≥ 0,7 ET médiane ρ(préfixe ≤15) ≤ 0,3 → VRAI ; inverse → FAUX ; sinon PARTIEL.
- HL2 : LOCALISÉ si une fenêtre de 5–6 jonctions atteint médiane ρ ≥ 0,5 ;
  sinon DISTRIBUÉ (aucune fenêtre ≥ 0,5).

## WIND-T4 — la linéarisation figée garde-t-elle l'entonnoir ? (script `wind_frozen_funnel_check.jl`)

Variantes : T (vrai), A (attention figée), AN (attention + normes figées ≈ LRP(AH+LN), linéarisation
des graphes d'attribution sans transcodeurs). Pour AN, la norme finale est figée aussi : N_AN = rms⁻¹·diag(γ).
Coefficient de cohérence c_V = ern_vrai(V) / moyenne ern_substituts-signes(V).

- HF1 (AN garde l'entonnoir) : ern(AN) ≤ 1,5 × ern(T) ET c_AN ≤ 0,6 (médianes) → VRAI.
- HF2 (même sous-espace) : moyenne des cos² principaux entre les sous-espaces de sortie top-10 de
  N·T_T et N_AN·T_AN ≥ 0,5 (hasard : 10/1536 ≈ 0,0065) → VRAI.
- Décomposition (descriptive) : A ≈ AN → effet dû au gel de l'attention ; A ≈ T → dû au gel des normes.

Décision : HF1 ET HF2 vrais → la linéarisation figée préserve la géométrie globale ; la critique des
graphes d'attribution reste locale (couches basses, WIND-1 H3). HF1 ou HF2 faux → la linéarisation
figée perd/déplace le canal dominant : critique indépendante des readouts (version forte de l'option A).
Limites connues : 5 prompts ; mesure géométrique, pas de fidélité (cf. Ali et al. 2022, RelP).

## WIND-T5 — mécanisme du sur-effondrement de la linéarisation figée (EXPLORATOIRE, écrit avant le run)

Observé en WIND-T4 (post hoc) : AN s'effondre vers le rang 1 sur p12 (ern 1,01, err ×20) et p22 (ern 2,06) ;
A aussi sur p12 (2,31). Ces 5 prompts ont révélé l'effet : tout résultat ici est exploratoire et devra être
confirmé sur de nouveaux prompts.

Faits exacts (dérivés avant mesure) :
- F1. Jacobien vrai de la RMSNorm = r₀·diag(γ)·(I − x xᵀ/(‖x‖² + n·ε)) ≈ version figée · (I − x̂x̂ᵀ). Figer une
  norme ajoute donc exactement un terme de rang 1 par sous-couche : J_figé − J_vrai = (J_figé x̂) x̂ᵀ.
- F2. À norme figée, J_mlp h = W_d[(silu'(g)⊙g + silu(g)) ⊙ u] → 2·m(h) là où les portes sont actives
  (SwiGLU ≈ homogène de degré 2) ; à attention figée, J_attn x = contribution de la valeur du token lui-même.
- F3. Donc la vraie couche n'a aucune auto-amplification de l'état courant, la version figée si
  (≈ x̂ + (a_self + 2m)/‖x‖) : prédiction d'un « mode d'état » amplifié dans AN.

Mesures (par prompt, variantes T/A/AN) : gain d'état par couche λ_k = ⟨x̂_{k+1}, J_k x̂_k⟩ et log10 Λ = Σ log10|λ_k| ;
pour les produits partiels P_k = J_27···J_k (k ∈ {1,6,11,16,21,26}) : s_k = ‖P_k x̂_k‖/σ₁(P_k) (part du mode
dominant portée par l'état), |cos(v₁, x̂_k)|, erank post-norme ; concentration de u₁ sur ses 5 plus grosses
coordonnées (masse5).

- P1 (auto-amplification) : médiane sur les prompts de log10 Λ_AN − log10 Λ_T ≥ 1.
- P2 (l'effondrement EST le mode d'état) : max_k s_k(AN) ≥ 0,5 sur p12 ET p22, alors que la médiane sur les
  prompts de max_k s_k(T) < 0,2.
- P3 (alternative : coordonnées massives) : masse5(u₁ de AN) ≥ 0,5 sur p12 ET p22.
Lecture : P1 ∧ P2 → mécanisme établi (théorème F1–F3 + mesure), à confirmer sur prompts neufs.
P2 faux et P3 vrai → artefact de coordonnées massives. Ni P2 ni P3 → mécanisme inexpliqué, on le dit.
A-p12 (normes vivantes) n'est PAS couvert par F1–F3 : décrit seulement.

### WIND-T5c (POST HOC, écrit avant le run, après P2/P3 faux)
Faille de P2 : le terme de rang 1 de la norme du MLP a pour entrée ĥ_k (état après attention, non stocké), pas x̂_k.
Le sous-espace d'entrée (2 vecteurs singuliers droits) R_k de ΔN_k = J_AN − J_A contient x̂_k et (I+J_a)ᵀĥ_k (F1).
P2' : pour AN sur p12 ET p22, max_{k=1..10} ‖R_kᵀ v₁(P_k)‖² ≥ 0,5 alors que la médiane (prompts) du même max pour
T < 0,1 (hasard 2/1536). Vrai → l'effondrement est alimenté par les termes de rang 1 des normes figées. Faux → inexpliqué.

## WIND-T6 — théorie de l'effondrement : prédictions (EXPLORATOIRE, écrit le 2026-09-30 ~23:35Z, AVANT le script `wind_T6_signflip.jl`)

Ces 5 prompts ont révélé l'effet et ont servi à formuler les conjectures (scripts `wind_T6_explore.jl`, `wind_T6_hybrid.jl`,
`wind_T6_normgrad.jl`, `wind_T6_cancel.jl`, déjà exécutés) : tout ce qui suit est EXPLORATOIRE et devra être confirmé sur des
prompts neufs. Déjà vu (donc NON prédit ici) : hybrides couche par couche, télescopages le long de v₁, masses résiduelles τ.

Notation : A_k = J_k^A, E_k = J_k^T − J_k^A (terme QK), R_k = J_k^AN − J_k^A (gel des normes, rang ≤ 2), N = vraie norme finale,
ern = erank₂(N·P), σ₁ = σ₁(N·P), P = produit k = 27..1. Ensembles « signes retournés » (ε ∈ {±1}^27 i.i.d. uniformes, K = 40
tirages, graine 20261001) : P_Q(ε) = Π(A_k + ε_k E_k) (ε ≡ +1 donne T) et P_R(ε) = Π(A_k + ε_k R_k) (ε ≡ +1 donne AN).
Fait exact : E_ε P(ε) = P_A (multilinéarité). q = rang centile de la vraie configuration (ε ≡ +1) parmi les 40 tirages.

- P6.1 (compensation QK non due au hasard des signes) : p12 : q_T(ern) ≥ 97,5 % (T au-dessus d'au moins 39 tirages) ET
  q_T(σ₁) ≤ 2,5 %. p5, p6, p17, p22 : 2,5 % < q_T(ern) < 97,5 %.
- P6.2 (cohérence de signe des termes de gel des normes) : p12 et p22 : q_AN(σ₁) ≥ 97,5 % ET q_AN(ern) ≤ 2,5 %.
  p5, p6 : 2,5 % < q_AN(ern) < 97,5 %. p17 : pas de prédiction (descriptif).
- P6.3 (structure d'Euler des colonnes de R_k) : soit U_k les 2 vecteurs singuliers gauches de R_k et Δ̂_k = (x_{k+1} − x_k)/‖·‖.
  Pour chaque prompt, médiane sur k = 1..27 de ‖U_kᵀ Δ̂_k‖² ≥ 0,5 (hasard 2/1536 = 0,0013). Entre 0,1 et 0,5 : partiel
  (la part « autres tokens » de la mise à jour d'attention n'est pas dans les colonnes) ; < 0,1 : faux.
- P6.4 (l'effondrement AN = réponse à une mise à l'échelle commune des mises à jour précoces) : z = N·Σ_{k=1..10} P^AN_{k+1} Δ_k
  (P^AN_{k+1} = J^AN_27···J^AN_{k+1}). Pour p12 et p22 : |cos(u₁(N·P_AN), z)| ≥ 0,7 (hasard ≈ 0,03). Descriptif pour les autres.
Décision : P6.1 vrai → le terme QK agit comme une rétroaction qui compense spécifiquement la direction amplifiée par la chaîne
à attention figée (pas un effet de sélection). P6.2 ∧ P6.3 ∧ P6.4 vrais → mécanisme « gel des dénominateurs = rétroaction
positive d'échelle, cohérente sur les couches précoces » soutenu. Tout échec est rapporté tel quel.

## WIND-T7 — boucle de rétroaction des normes : prédictions (EXPLORATOIRE, écrit le 2026-10-01 ~17:00Z, AVANT `wind_T7_sweep.jl`)

Mêmes 5 prompts (exploratoire). Déjà calculé et vu AVANT cette section : relèvement exact
N·Π(A_k + R_k) = M_A + 𝒰(I − W)⁻¹𝒱 (porte 2e-4), σ(G) à θ = 1, réalisation d'ordre r de W (Ho–Kalman variant dans
le temps) et, pour le modèle d'ordre 2 SEULEMENT, la courbe θ ↦ ern_mod(θ) = ern(M_A + θ𝒰(I − θW₂)⁻¹𝒱)
(`wind_T7_model_results.txt`). AUCUN produit vrai Π(A_k + θR_k) à θ ≠ 1 n'a été calculé.

Intervention testée : gain de rétroaction des normes θ, J_k(θ) = J_k^A + θ(J_k^AN − J_k^A) (θ = 0 : normes vivantes,
θ = 1 : normes figées). Vérité : ern_true(θ) = erank₂(N·Π_k J_k(θ)) avec la vraie N, grille θ = 0,25 : 0,125 : 2,5.
θ_c = premier franchissement de ern = 3 par valeurs décroissantes (interpolation linéaire), même règle pour modèle et vérité.

Prédictions du modèle d'ordre 2 (boucle fermée 2×2), à tester :
- P7.1 (gain critique) : |θ_c^true − θ_c^mod| ≤ 0,15·θ_c^mod pour p22 (θ_c^mod = 0,851), p17 (1,546), p5 (1,621),
  p6 (1,646). ET ordre : θ_c^true(p22) < 1 < min(θ_c^true(p17), θ_c^true(p5), θ_c^true(p6)).
  (p12 : A est déjà effondré à θ = 0 ; à la place, P7.1b : ern_true(0,5) ∈ [1,10 ; 1,40], modèle 1,23.)
- P7.2 (fidélité de la courbe) : pour chaque prompt, max sur la grille de |ln ern_mod(θ) − ln ern_true(θ)| ≤ ln 1,5.
- P7.3 (un seul mode résonant, théorème de dédoublement) : pour la vraie résolvante G(θ) = (I − θW)⁻¹ à θ = 2 :
  σ₂/σ₁ ≤ 0,10 pour les 5 prompts, et σ₂(G(2)) ≤ 3·σ₂(G(1)) (σ₂ reste O(1) pendant que σ₁ croît).
Décision : P7.1 ∧ P7.2 vrais → l'amplitude de l'effondrement AN est fixée par la distance à la criticité d'une boucle
de rétroaction de dimension 2 (p22 sur-critique, p17 sous-critique) ; P7.3 vrai → la résonance est à mode unique comme
le prédit le théorème. Tout échec est rapporté tel quel.
