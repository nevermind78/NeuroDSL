# WIND-T8 — test CONFIRMATOIRE sur 50 prompts neufs (pré-enregistré le 2026-10-01, AVANT toute collecte)

## Objet
Confirmer sur des prompts qui n'ont servi à rien jusqu'ici :
- la trouvaille WIND-T7 : l'effondrement de la linéarisation à normes figées est fixé par le gain critique θ_c d'une
  boucle de rétroaction des normes effectivement de dimension 2 ;
- les prédictions WIND-T6 de Fable ;
- l'entonnoir de rang de T (WIND-T2b/T4).

## Prompts (figés avant collecte)
- 85 candidats neufs (aucun recouvrement avec les 22 anciens), tokenisés par `tokenizer.json` (porte : reproduit les
  22 anciennes tokenisations à l'identique).
- Criblage NeuroDSL (`wind_T8_screen.jl`) : top-1 = réponse attendue ET logit(réponse) − logit(contrefactuel) ≥ 2.
  Résultat : 67/85.
- Sélection déterministe :
  - le 1er candidat valide sert au test du pipeline (indice 51, « The capital of Italy is »), EXCLU des verdicts ;
  - les 50 autres sont tirés à tour de rôle par tâche (ordre alphabétique des tâches, ordre des candidats dans
    chaque tâche) : 17 factuels, 8 IOI, 6 induction, 6 antonymes, 6 expressions, 3 séquences, 2 arithmétique,
    1 grammaire, 1 code.
- Fichier : `wind_T8_prompts.json`.
- Un prompt dont une porte de collecte échoue (G1, G3, G4a, G4b) est exclu et signalé.

## Données
`wind_T8_collect.jl` (copie de WIND-1, mêmes portes) collecte, au dernier token, les Jacobiennes de couche A et AN
(1re passe, les 50 prompts), puis T (2e passe). Il stocke aussi h_k, les sorties MHA/MLP, les échelles des normes et la
dernière ligne d'attention.

## H1 (principale) — criticité de la boucle des normes
Notation identique à WIND-T7 :
- J_k(θ) = J_k^A + θ(J_k^AN − J_k^A) ;
- ern_true(θ) = erank₂(N·Π_k J_k(θ)), avec N la vraie norme finale ;
- θ_c = premier franchissement de ern = 3 par valeurs décroissantes (interpolation linéaire).

Vérité sur la grille θ = 0,25 : 0,125 : 2,5. Pas de franchissement → θ_c^true > 2,5, censuré à 2,5.

Prédiction (`wind_T8_predict.jl`) :
- θ_c^mod(r) vient de la réalisation d'ordre r de W (Ho–Kalman variant dans le temps, code identique à WIND-T7),
  sur la grille θ = 0,25 : 0,05 : 2,5, pour r = 1, 2, 3 ;
- ce script ne calcule aucun produit vrai à θ ≠ 0 ni aucune résolvante complète (I − θW)⁻¹ ;
- les prédictions sont écrites dans `wind_T8_predictions.json`, dont l'empreinte SHA-256 est consignée dans
  `wind_T8_hashes.txt` AVANT l'exécution de `wind_T8_truth.jl` ;
- le script de vérité refuse de tourner si ce fichier manque.

Critères (r = 2, sur les 50 prompts ; les deux θ_c sont censurés à 2,5) :
- **C1 (rangs)** : Spearman(θ_c^mod, θ_c^true) ≥ 0,7 → VRAI ; [0,4 ; 0,7[ → PARTIEL ; < 0,4 → FAUX.
- **C2 (précision)** : parmi les prompts avec θ_c^mod ≤ 2,3, la fraction vérifiant |θ_c^true − θ_c^mod| ≤ 0,15·θ_c^mod
  est ≥ 0,7 → VRAI ; [0,5 ; 0,7[ → PARTIEL ; sinon FAUX. Moins de 10 tels prompts → « puissance insuffisante ».
- **C3 (effondrement au gel complet)** :
  - effondrement prédit = θ_c^mod < 1 ; effondrement vrai = ern_true(1) ≤ 3 ;
  - s'il y a au moins 3 effondrements vrais et au moins 3 non-effondrements : exactitude équilibrée ≥ 0,8 → VRAI ;
  - sinon : descriptif.
- **C4 (l'ordre 2 est nécessaire)** : Spearman(r=1) ≤ Spearman(r=2) − 0,1, OU fraction C2(r=1) ≤ fraction C2(r=2) − 0,2
  → VRAI. L'ordre 3 est descriptif.
- **D1 (descriptif)** : fréquence des effondrements AN, ern_true(1) ≤ 3, avec IC de Wilson à 95 %.

## H2 (secondaire) — prédictions WIND-T6, telles que proposées par Fable
On note τ = sqrt(Σ_{i≥2} σ_i²) et b = σ₁/τ pour N·P, P étant le produit de la variante.
- **S1** : τ_A/τ_T et τ_AN/τ_T ∈ [0,7 ; 1,1] sur ≥ 80 % des prompts.
- **S2** : « ern ≤ 3 ⟺ b ≥ 2,6 » sur ≥ 95 % des couples (prompt, variante ∈ {T, A, AN}).
- **S3** : pour chaque prompt où AN s'effondre, q_AN(σ₁) ≥ 0,975 sous le nul des signes de WIND-T6 (K = 40) ET
  |cos(u₁(N·P_AN), z)| ≥ 0,9, avec z défini en WIND-T6 (P6.4). VRAI si c'est vérifié sur ≥ 80 % des prompts
  effondrés. Non applicable s'il y en a moins de 3.
- **S4** : ern de T au-dessus de la médiane du nul des signes QK de WIND-T6 sur ≥ 80 % des prompts (K = 20 tirages).

## H3 (réplication) — entonnoir de T
- **F1** : erank post-norme de T < moyenne − 2 écarts-types des substituts à signes de WIND-T2 (4 tirages) sur ≥ 80 %
  des prompts, ET médiane de c = ern_T / moyenne des substituts ≤ 0,6.
- **F2** : médiane du cos² moyen entre les sous-espaces de sortie top-10 de N·P_T et N_AN·P_AN ≥ 0,5
  (réplique HF2 de WIND-T4).

## Règles
- Aucun seuil ni critère ne change après avoir vu des données WIND-T8.
- Tous les résultats sont rapportés, y compris les échecs.
- Le prompt 51 ne sert qu'à tester le pipeline ; il est rapporté à part.
