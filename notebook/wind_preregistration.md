# WIND-1 — Pré-enregistrement (écrit AVANT toute mesure des quantités ci-dessous)

Date : 2026-09-28. Modèle : Qwen2.5-1.5B-Instruct natif NeuroDSL (28 couches, d=1536).
Seule mesure préalable : `wind_calib_probe.jl` (temps, normes, convergence en eps, porte
analytique MLP, porte de gel) — aucune des quantités H1–H3 n'y est calculée.

## Objet mesuré : le « vent vertical » du dernier token

x_k := ligne du dernier token (position n) de la sortie de couche k (k=0 : embedding).
J_k := ∂x_{k+1}/∂x_k (bloc diagonal dernier token → dernier token), 1536×1536, matérialisé
colonne par colonne par différence finie centrée (base canonique, eps = 5e-3·‖x_k‖ ; 1e-2 pour k=0),
en ne recalculant que la couche k+1 (cône réactif). Par causalité, perturber x_{k,n} ne modifie
que la position n en aval : le transport multi-couches est EXACTEMENT
J_{s→t} = J_{t-1}···J_s (pas d'approximation de composition) — c'est vérifié (G3), pas supposé.

Trois variantes par couche :
- **T** (vraie Jacobienne) ;
- **A** : probabilités d'attention (12 `pr_h` de la couche) épinglées à leur valeur propre
  (patch_node!, bit-exact) → supprime le canal QK ;
- **AN** : A + dénominateurs RMSNorm (norm1, norm2 ; + norme finale pour la lecture) gelés
  à leur valeur propre (op custom) → exactement la linéarisation « local replacement model »
  des attribution graphs (Ameisen et al. 2025 : stop-gradient sur motifs d'attention et
  dénominateurs de normalisation), hors transcodeurs.

Prompts (5, fixés maintenant) : indices 5 (Tour Eiffel), 6 (train 60 miles), 12 (Orwell),
17 (Q/A planète), 22 (induction « key under the mat ») de `qwen_sweep_prompts.json`.

## Portes (échec d'une porte = mesure invalide, rapportée comme telle)

- **G1** DF : 4 colonnes aléatoires/couche, écart relatif JVP(eps) vs JVP(2·eps) :
  médiane < 1e-3, max < 1e-2.
- **G2** Jacobienne MLP analytique (CPU Float64, poids) vs DF moteur, 3 couches × 2 directions :
  erreur relative < 1e-3.
- **G3** Composition : JVP multi-couches directe (perturber x_s, lire x_t) vs produit Π J_k v,
  s∈{1,7,14,21}, t∈{s+7, 28} : erreur relative médiane < 1e-2 (variante T ; + une paire en A et AN).
- **G4** Gels : épinglage `pr_h` sans perturbation bit-identique ; op RMSNorm gelée : forward
  propre reproduit à < 1e-5 relatif.
- **G5** (script séparé, 1 prompt) Adjoint : gradient rétro-propagé par le moteur (backward,
  capture aux branches) à chaque couche k vs g_28ᵀ Π J (DF avant) avec la même graine g_28 :
  erreur relative < 1e-2 pour tout k≥1.

## Hypothèses et seuils

**H1 — Chaînage (problème ouvert de crosslayer_interaction_qwen.tex, Remarque rem:crossterm-gap).**
G(s,t) := Π_{k=s}^{t-1} ‖J_k‖₂ / ‖J_{s→t}‖₂ (≥ 1). Avec des normes par couche EXACTES, G est
une borne INFÉRIEURE de la surestimation de toute borne chaînée couche par couche (publiée ou
améliorée). Seuils (médiane sur prompts, variante T) :
- H1 VRAIE (« chaînage structurellement inutilisable ») si G(1,28) > 100 ET G(s,s+7) > 10
  pour chaque s∈{1,7,14,21}.
- H1 FAUSSE (« chaînage quasi serré : seule la borne par couche est lâche ») si G(s,s+7) < 3
  pour tout s∈{1,7,14,21}.
- Sinon : MIXTE.
Exploratoire (pas de seuil) : G(k,k+2) couche par couche (signature possible de la correction
entre couches adjacentes, Patrawala et al.).

**H2 — Entonnoir de rang (Fernando & Guitchounts 2026, arXiv 2605.14258).**
erank₂(M) := exp(H(σ²/Σσ²)) (leur définition). Eux : produit de Jacobiennes MOYENNES
(sur 1000 échantillons), tronqué au rang 512 → erank ≈ 6.7 (Llama 3.1 8B).
- **H2a** (entonnoir par prompt, exact) : VRAIE si médiane erank₂(J_{1→28}) < 50 ; FAUSSE si > 200.
- **H2b** (artefact de moyennage) : produit des moyennes (sur nos 5 prompts) des J_k, plein et
  tronqué (rang 512 et rang 192 = 1536·512/4096, à chaque pas). VRAIE (« le moyennage/troncature
  abaisse substantiellement le rang ») si erank₂(produit des moyennes, tronqué 192) <
  (médiane erank₂ par prompt)/3 ; FAUSSE si le rapport est dans [1/2, 2]. Réserve déclarée :
  5 prompts moyennent beaucoup moins que 1000 — un écart ici est une borne basse de l'effet.

**H3 — Ce que la linéarisation gelée (attribution graphs) ne voit pas.**
Lecture : différence de logits top-1 − top-2 au dernier token (tokens pris dans les logits propres).
g_s := ∂(logit-diff)/∂x_s = g_28ᵀ J_{s→28} (T, norme finale vraie) ; g_s^AN idem en AN (norme
finale gelée). err_s := ‖g_s − g_s^AN‖/‖g_s‖.
- H3 VRAIE (« la linéarisation gelée manque une part substantielle du transport ») si la médiane
  (sur prompts) de err_s > 0.3 pour au moins la moitié des s∈{1..20}.
- H3 FAUSSE (« fidèle ») si médiane err_s < 0.1 pour tout s∈{1..27}.
- Sinon : PARTIELLE.
Décomposition rapportée (sans seuil) : part QK (T vs A) et part normalisation (A vs AN), par couche
(‖J_k^T − J_k^A‖_F/‖J_k^T − I‖_F, etc.) et sur la lecture.

## AMENDEMENT (2026-09-28, ajouté AVANT la collecte complète et avant toute analyse ;
## seul un test de fumée à 8 colonnes/couche avait tourné, inexploitable par construction)

Motif : découverte, pendant la revue de littérature, du « J-lens » d'Anthropic
(Verbalizable Representations Form a Global Workspace, arXiv 2607.15495, juillet 2026) :
J_ℓ = moyenne sur positions t, positions futures t' ≥ t et 1000 prompts de ∂h_{final,t'}/∂h_{ℓ,t}.
C'est une Jacobienne multi-couches MOYENNÉE ; la question naturelle qu'aucune source lue ne
quantifie est la dispersion par prompt autour de cette moyenne (« climat » vs « météo »).

**H4 — Spécificité par prompt du transport vers la sortie.**
P_s^(p) := J_{s→28} du prompt p ; P̄_s^(−p) := moyenne des P_s sur les 4 AUTRES prompts
(moyenne de produits, protocole J-lens restreint à t' = t = dernier token).
δ_s^(p) := ‖P_s^(p) − P̄_s^(−p)‖_F / ‖P_s^(p) − I‖_F.
- H4 VRAIE (« le transport est spécifique au prompt : la moyenne est un climat, pas la météo »)
  si la médiane (sur prompts) de δ_s > 0.5 pour au moins la moitié des s∈{1..20}.
- H4 FAUSSE (« la moyenne représente bien chaque prompt ») si médiane δ_s < 0.2 pour tout s∈{1..20}.
- Sinon : PARTIELLE.
Réserves déclarées : 4 prompts dans la moyenne (bruit d'échantillonnage qui gonfle δ) ; position
unique (t' = t) alors que le J-lens moyenne aussi sur t' > t.
Descriptif associé : différence de logits (top-1 − top-2) prédite par la lentille moyenne
P̄_s^(−p) x_s, la lentille exacte P_s^(p) x_s et la logit lens (x_s), chacune passée par la norme
finale, comparée à la vraie (le vocabulaire complet n'est pas chargé dans l'analyse CPU).

## Descriptif (sans seuil, annoncé d'avance)
‖J_k‖₂, σ_min(J_k) par couche ; 5 premières valeurs singulières de J_{s→28} ; cos(u₁,v₁)
(la direction la plus amplifiée est-elle tournée ?) ; alignement de v₁ avec x_s et avec g_s.

## Ce qui ne sera PAS affirmé
Rien sur les positions croisées (vent horizontal) — non mesuré ici. Rien de « certifié » :
tout est ponctuel (au point d'activation réel), comme la borne publiée.
