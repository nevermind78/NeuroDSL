# WIND-3 — Pré-enregistrement : le « vent horizontal » (écrit AVANT toute mesure sur les 5 prompts)

Date : 2026-09-29. Modèle : Qwen2.5-1.5B-Instruct natif NeuroDSL (28 couches, d=1536, 12 têtes, GQA 2 KV,
pas de BOS : la position 1 est le premier mot du prompt).
Seule mesure préalable : `wind3_calib_probe.jl` sur le prompt 4 (« ramen », HORS échantillon) — mécanique,
temps, bruit des différences finies ; aucune quantité HH* n'y est calculée, aucun des 5 prompts n'y est touché.
Prompts mesurés (fixés, ceux de WIND-1) : 5 (Tour Eiffel), 6 (train), 12 (Orwell), 17 (planète), 22 (induction).
Lecture R = différence de logits top-1 − top-2 au dernier token (tokens de WIND-1, mêmes vecteurs w_read).

## 0. Positionnement (sources lues le 2026-09-29, pas citées de mémoire)

- **Causal tracing** (Meng et al. 2022, arXiv 2202.05262) : corruption par bruit des embeddings du sujet,
  restauration d'états cachés → MLP de milieu de réseau au dernier token du sujet.
- **Attention knockout** (Geva et al. 2023, arXiv 2304.14767, §5) : bloque l'arête dernière position ← c en
  posant M = −∞ dans le masque pré-softmax (donc les autres poids se RENORMALISENT), toutes têtes, sur une
  FENÊTRE de k couches (k=9 GPT-2 XL, k=5 GPT-J). Résultat : relation → dernier token d'abord (−35 à −45 %
  de probabilité), sujet → dernier token ensuite, couches milieu-haut (jusqu'à −60 %).
- **Information flow routes** (Ferrando & Voita 2024, arXiv 2403.00824) : importance d'arête par proximité
  ALTI, poids d'attention traités comme « constantes propres à la prédiction », LayerNorm linéarisée ;
  sans gradient, un forward ; ignore par construction l'effet des clés/requêtes (canal QK). Observent qu'un
  premier point « joue le rôle de BOS ».
- **Attention rollout/flow** (Abnar & Zuidema 2020, arXiv 2005.00928) : combinaison des matrices d'attention
  brutes ; corrèle mieux avec ablation et gradients d'entrée que l'attention brute. Aucun canal QK.
- **Puits d'attention** : Xiao et al. 2023 (StreamingLLM) ; Kobayashi et al. 2020 (EMNLP, arXiv 2004.10102) :
  les tokens spéciaux reçoivent de gros poids mais ‖α f(x)‖ petit ; Gu et al. 2024 (ICLR 2025,
  arXiv 2410.10781) : le puits « agit comme un biais de clé », valeurs peu informatives, disparaît sans la
  normalisation softmax ; Barbero et al. 2025 (arXiv 2504.02732) : puits = mécanisme anti-« over-mixing »,
  valeur du BOS de plus petite norme (tête « inactive par défaut »), sensibilité aux perturbations mesurée
  par normes de Jacobiennes (Gemma 7B), avec vs sans BOS.
- **Aubry et al. 2025** (ICLR, arXiv 2407.07810) : Jacobiennes de bloc inter-tokens J^l_{t1 t2} par autograd,
  couplage de leurs vecteurs singuliers ; pas de séparation des canaux QK/norme, pas d'influence vers une
  lecture.
- **Attribution graphs** (Ameisen et al. 2025, transformer-circuits.pub) : figent motifs d'attention et
  dénominateurs de normalisation ; écrivent explicitement que les graphes « ne contiennent pas l'influence
  via les motifs d'attention ». Suite QK (Kamath et al. 2025, « Tracing Attention Computation Through
  Feature Interactions ») : décomposition bilinéaire du score PRÉ-softmax par tête ; d'après des notes
  secondaires (T. Crosse 2026 — la page primaire, rendue en JS, n'a pas pu être lue), présentée comme
  explication par tête, pas intégrée comme arête vers la sortie.
- **AtP\*** (Kramár et al. 2024, arXiv 2403.00745) : l'approximation linéaire échoue sur les nœuds
  requête/clé (saturation de l'attention) → recalcul du softmax ; annulation effets directs/indirects.
- **J-lens** (arXiv 2607.15495, lu pour WIND-1) : moyenne de ∂h_{final,t'}/∂h_{ℓ,t} sur t' ≥ t — contient
  des blocs inter-positions, mais MOYENNÉS.

**Déjà connu** : où l'information traverse (Geva ; Ferrando & Voita) ; que le poids d'attention ≠ influence,
en particulier pour les puits (Kobayashi, Gu, Barbero) ; que les linéarisations à attention gelée omettent le
canal QK (Ameisen, dit qualitativement) et que le linéaire échoue aux nœuds Q/K (AtP*).
**Ce que la mesure ajoute (seulement ça)** : sur un vrai 1.5B, par prompt, par (position, couche), la PART
EXACTE (validée par différences finies et par l'adjoint indépendant de WIND-1) de l'influence inter-positions
sur la lecture qui passe par le canal QK et par les dénominateurs de norme — c.-à-d. le chiffre de ce que les
attribution graphs / flow routes / rollout omettent par construction — et sa comparaison au canal vertical ;
plus le recoupement avec les knockouts exacts (deux protocoles de renormalisation). Les parties HH1, HH3a,
HH3c sont des RÉPLICATIONS de résultats connus et seront étiquetées comme telles.

## 1. Objets mesurés

x_{i,s} := ligne i (position) de la sortie de couche s (s=0 : embedding). g^V_{i,s} := ∂R/∂x_{i,s}, pour
TOUTES les positions i et couches s, en UN backward du moteur par variante (instrumentation au niveau du
script, aucun changement de src/) :
- **T** : vrai modèle ;
- **A** : stop-gradient sur les 12×28 matrices de probabilités d'attention (canal QK supprimé partout) ;
- **AN** : A + dénominateurs RMSNorm (norm1, norm2 de chaque couche, norme finale) gelés à leur valeur propre
  = linéarisation « local replacement model » des attribution graphs, hors transcodeurs.
Aussi : c^V_ℓ := ∂R/∂res1_ℓ[n,:] (cotangent de la sortie d'attention de la couche ℓ au dernier token).

**Carte 1 (gradients)** : sensibilité relative S^V_{i,s} := ‖g^V_{i,s}‖·‖x_{i,s}‖ ; gradient×entrée
⟨g^V_{i,s}, x_{i,s}⟩ ; part horizontale h^V_s := Σ_{i<n} S^V_{i,s} / Σ_{i≤n} S^V_{i,s}.
Manque gelé horizontal, pondéré par l'activation :
E^{H}_s := sqrt(Σ_{i<n} ‖x_{i,s}‖² ‖g^T_{i,s} − g^{AN}_{i,s}‖²) / sqrt(Σ_{i<n} ‖x_{i,s}‖² ‖g^T_{i,s}‖²) ;
part QK E^{H,QK}_s (T vs A), part norme E^{H,N}_s (A vs AN) ; par cellule err_{i,s} = ‖g^T−g^AN‖/‖g^T‖.
Canal vertical (même backward) : err^{V,QK}_s := ‖g^T_{n,s} − g^A_{n,s}‖/‖g^T_{n,s}‖.

**Carte 2 (knockout exact)**, pour chaque couche ℓ=1..28 et source i=1..n−1, toutes têtes, UNE couche,
probabilités de la ligne n de la couche ℓ remplacées (patch_node!, bit-exact sinon) :
- **remove** (primaire) : p[n,i] := 0 SANS renormalisation → retire exactement le terme p·v_i ;
- **mask** (protocole Geva) : −∞ pré-softmax ⇔ p[n,j] := p[n,j]/(1−p[n,i]) ;
- **fenêtre** (descriptif, réplication Geva GPT-J) : mask sur les couches ℓ−2..ℓ+2.
Les couches en aval sont recalculées (vrai modèle). K^{rem}_{ℓ,i}, K^{mask}_{ℓ,i} := ΔR.
Prédictions linéaires de chaque arête : LIN^V_{ℓ,i} := ⟨c^V_ℓ, δ_{ℓ,i}⟩ où δ = variation exacte de la sortie
d'attention au dernier token causée par le knockout (T : courbure seule ; A, AN : + canaux gelés).

## 2. Portes (échec = mesure invalide pour ce prompt/variante, rapporté tel quel)

- **R0** : forward instrumenté ≡ WIND-1 bit à bit (x_{n,s} s=0..28 et logits top-1/top-2 stockés).
- **G4b** : forward à normes gelées vs propre, err rel logits < 1e-5.
- **GV** (adjoint indépendant) : lignes n de g^V_s vs produit (J^V)ᵀ des Jacobiennes DF de WIND-1 (même
  graine g^V_{n,28}), err rel < 1e-2 pour s = 1..27, pour T, A et AN.
- **GF** (différences finies horizontales) : cellules fixées a priori (i,s) ∈ {(1,3),(1,14),(2,1),(⌈n/2⌉,7),
  (n−1,20),(n−1,26),(n,10)}, directions ĝ + 3 aléatoires, JVP centrée à eps = 1e-3·‖x_{i,s}‖ (choisi par la
  sonde : plancher de bruit ≤ 7e-4·‖g‖ à ce pas, courbure visible au-delà), variante épinglée (A/AN : 12×(28−s)
  pr_h épinglés ; AN : normes gelées). e_v := |DF − ⟨g,v⟩|/‖g‖. Seuils : médiane < 1e-2 ET max < 5e-2.
  Une direction dont DF(eps) et DF(2eps) diffèrent de plus de 5e-2·‖g‖ (ou NaN) est déclarée « DF non
  résolue » (non-linéarité/explosion du modèle linéarisé sous perturbation finie, vue sur la sonde en AN à
  s=0) : exclue du max, comptée et rapportée.
- **K0** : knockout « sans blocage » (probabilités ré-imposées à leur valeur propre) bit-identique ; état
  restauré bit-identique après la carte 2.

## 3. Hypothèses et seuils (médianes sur les 5 prompts sauf mention)

**HH1 — Concentration et sources attendues (RÉPLICATION : Geva 2023 ; têtes d'induction).**
- HH1a : C₅ := part de Σ|K^{rem}| portée par les 5 % de cellules (ℓ, i<n) les plus fortes.
  VRAIE si médiane C₅ ≥ 0.5 ; FAUSSE si < 0.25 ; sinon PARTIELLE.
- HH1b : induction (p22, « …under the mat. The key is under the ») : source attendue i=6 (« mat »).
  Eiffel (p5) : sujet i ∈ {8,9,10,11} (« E iff el Tower »), traversée ℓ ∈ [8, 24].
  Critère par prompt (sources i ≥ 2, la position 1 est traitée en HH3) : (a) la cellule de |K^{rem}| max est
  dans la cible ; (b) la position de Σ_{s=1..27} S^T_{i,s} max est dans la cible (sujet pour p5, 6 pour p22).
  VRAIE si (a) et (b) sur les deux prompts ; FAUSSE si ni (a) ni (b) sur aucun des deux ; sinon PARTIELLE.

**HH2 — La linéarisation gelée manque une part substantielle de l'influence inter-positions.**
- HH2a : VRAIE si médiane E^H_s > 0.3 pour au moins la moitié des s ∈ {1..20} ; FAUSSE si < 0.1 pour tout
  s ∈ {1..26} ; sinon PARTIELLE (seuils de WIND-1 H3).
- HH2b (ouverte, à ma lecture) : l'influence horizontale dépend PLUS du canal QK que la verticale.
  VRAIE si médiane E^{H,QK}_s > médiane err^{V,QK}_s pour ≥ 2/3 des s ∈ {1..20} ; FAUSSE si l'inverse pour
  ≥ 2/3 ; sinon INDÉTERMINÉE.

**HH3 — Puits d'attention (position 1).**
- π₁ := Σ_{ℓ,h} p[n,1] / Σ_{ℓ,h} Σ_{i<n} p[n,i] (part d'attention du dernier token vers la position 1 parmi
  les sources i<n) ; κ₁ := Σ_ℓ |K^{rem}_{ℓ,1}| / Σ_{ℓ,i<n} |K^{rem}_{ℓ,i}| ; ρ₁ := κ₁/π₁.
- HH3a (RÉPLICATION Kobayashi/Gu/Barbero) : VRAIE si médiane π₁ ≥ 0.5 ET médiane ρ₁ ≤ 0.25 ; FAUSSE si
  ρ₁ ≥ 0.75 ou π₁ < 0.2 ; sinon PARTIELLE.
- HH3b (ouverte) : l'influence du puits passe surtout par le canal QK. q₁ := médiane sur s ∈ {3..20}
  (régime d'activation massive) de ‖g^T_{1,s} − g^A_{1,s}‖/‖g^T_{1,s}‖ ; q_rest := idem pondéré par
  l'activation sur i ∈ {2..n−1} (E^{H,QK}_s restreint). VRAIE si médiane q₁ > 0.5 ET médiane q₁ > médiane
  q_rest ; FAUSSE si q₁ < 0.2 ou q₁ < q_rest ; sinon PARTIELLE.
- HH3c (attendue d'après Gu 2024) : μ₁ := Σ_ℓ |K^{mask}_{ℓ,1}| / Σ_ℓ |K^{rem}_{ℓ,1}|. VRAIE si médiane μ₁ > 3
  (masquer le puits à la Geva agit par redistribution de masse, pas par son contenu) ; FAUSSE si < 1.5.

**HH4 — Recoupement carte 1 / carte 2** : Spearman(|LIN^T|, |K^{rem}|) sur les 28(n−1) cellules.
VRAIE (« la carte linéaire localise les traversées ») si médiane ≥ 0.7 ; FAUSSE si < 0.4 ; sinon PARTIELLE.
Rapporté aussi (sans seuil) : même chose pour LIN^A et LIN^AN, recouvrement des top-10, et, sur les 10
cellules de |K^{rem}| max par prompt, erreurs relatives |LIN^V − K^{rem}|/|K^{rem}| (T = courbure seule ;
AN − T = part attribuable aux canaux gelés si l'erreur T est < 0.3).

## 4. Descriptif annoncé (sans seuil)

h^V_s par couche ; carte err_{i,s} ; cartes K^{mask} et fenêtre ; profil par position ; attention vs influence
par position ; **HH5 (optionnel)** rang du canal horizontal : esquisse aléatoire (K=128 backwards de sonde
⟨x_{n,28}, u⟩, u gaussien) de J_{(i,s)→(n,28)} pour tout (i,s), T/A/AN, erank₂ ; porte GS : sur le bloc
vertical (i=n), erank₂ esquissé à un facteur 1.5 près de l'erank₂ exact de WIND-1 pour s ∈ {1,7,14} (sinon
seules les comparaisons relatives horizontal/vertical sont rapportées).

## 5. Ce qui ne sera PAS affirmé

Rien de général au-delà de 5 prompts ; tout est local (gradients au point réel, knockouts d'une couche) ;
les knockouts mono-couche sous-estiment les voies redondantes (d'où la fenêtre descriptive) ; aucune
prétention de nouveauté pour la localisation des traversées ni pour « attention ≠ influence » des puits.
