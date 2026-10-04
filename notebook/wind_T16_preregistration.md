# WIND-T16 — les normes comme rétroaction négative : une théorie de l'auto-réparation (pré-enregistré le 2026-10-03, AVANT tout calcul)

## Question
L'auto-réparation (Rushing & Nanda, ICML 2024 ; McGrath et al. 2023) désigne le fait que, lorsqu'on ablate une
composante, les couches suivantes compensent. Rushing & Nanda attribuent une partie de cette compensation à la NORME
FINALE, sans théorie quantitative.

Hypothèse : c'est la même boucle que l'effondrement des linéarisations figées, de signe opposé.
- Dans le vrai réseau, les RMSNorm forment une rétroaction NÉGATIVE : leur Jacobien annule la direction de l'état, ce qui
  donne une contraction mesurée de ×0,84 par couche.
- Figées, cette rétroaction devient positive (WIND-T7 à T15).

Si c'est la même boucle, la compensation par les normes doit être :
- un phénomène de réponse linéaire ;
- présente dans TOUTES les normes, pas seulement la finale ;
- plus forte là où la boucle figée est la plus instable (θ_c bas).

## Mesures (vrai modèle, forwards exacts, 50 prompts de WIND-T8 ; le prompt 51 sert au test, exclu)
Composantes c : les 56 sous-couches au dernier token n (MHA_l, MLP_l).
Ablation : o_c[n] → μ_c (moyenne sur les 50 prompts, référence identique à WIND-T9).
Lecture R = logit(réponse) − logit(contrefactuel). Pour chaque (p, c), six effets d'ablation TE = R(propre) − R(ablaté) :
- **TE** : vrai modèle (normes et attention vivantes) ;
- **TE_fn** : toutes les RMSNorm en aval figées (blocs + finale), attention vivante ;
- **TE_fb** : normes des blocs figées, norme finale vivante ;
- **TE_ff** : seule la norme finale figée ;
- **TE_fa** : probabilités d'attention des couches en aval épinglées, normes vivantes ;
- **TE_fan** : attention épinglée + toutes les normes figées.

« Figé » signifie que le dénominateur RMS garde sa valeur propre (op :rmsnorm_frozen, comme WIND-T8). Comme seules les
couches en aval de c changent, figer ou épingler s'applique de fait à l'aval.

Prédictions linéaires (gradients exacts déjà calculés dans WIND-T9) : a^A (attention figée, normes vivantes) et a^AN
(attention + normes figées).

Compensation par les normes : φ_c = 1 − TE_c / TE_fn,c. φ = 1 : compensation totale ; 0 : aucune ; < 0 : les normes
amplifient. Elle est calculée sur les couples (p, c) « actifs », c'est-à-dire |TE_fn,c| ≥ 0,05·max_c |TE_fn,c| (par prompt).

## Portes (sinon le prompt est exclu)
- **G0** : TE recalculé = vérité terrain de WIND-T9 (écart relatif max < 10⁻⁵).
- **G1** : sans ablation, chaque variante (normes figées, attention épinglée) redonne R propre (|ΔR| < 10⁻⁴·max(1, |R|)).

## Prédictions
- **S1 (les normes sont une rétroaction négative nette)** : sur les couples actifs, la fraction avec |TE| < |TE_fn|
  est ≥ 0,6 ET la médiane de φ est ≥ 0,10.
- **S2 (au-delà de la norme finale)** : Σ_{p,c} |TE_fb − TE| ≥ Σ_{p,c} |TE_ff − TE|. La compensation par les normes des
  blocs vaut au moins celle de la norme finale ; c'est nouveau par rapport à Rushing & Nanda.
- **S3 (réponse linéaire)** : la médiane sur les prompts de Spearman_c(TE_fan − TE_fa, a^AN − a^A) est ≥ 0,7. La
  compensation par les normes mesurée sur des ablations finies est alors prédite par la boucle linéarisée.
- **S4 (unification)** : Spearman sur les 50 prompts entre θ_c^AN (WIND-T8, censuré à 2,5) et Φ_p (médiane de φ_c du
  prompt) ≤ −0,3, ET p < 0,05 (permutation unilatérale, 10 000 tirages). La boucle figée la plus instable correspond
  alors à la rétroaction négative la plus forte.
- **S5 (mécanisme radial)** : Spearman, sur les couples actifs, entre φ_c et cos²(δ_c, ŝ_c) ≥ 0,2. δ_c = μ_c − o_c[n] ;
  ŝ_c = état normalisé au point d'écriture : h_l (layer_l_res1) pour MHA_l, x_l pour MLP_l.

## Descriptif (sans seuil)
- Auto-réparation par réacheminement de l'attention : Σ|TE_fa − TE|, à comparer à Σ|TE_fn − TE|.
- Profil de φ par couche et par type de composante.
- Dépendance au type de tâche.

## Règles
Aucun seuil ne change après avoir vu des données WIND-T16. Tout est rapporté, y compris les échecs.

## Déviation (écrite le 2026-10-03 APRÈS la collecte, AVANT tout calcul de verdict)
Constat : 23 effets sur 16 800 sont non finis (NaN/Inf, écrits `null` par JSON.jl) : 21 dans TE_fan, 1 dans TE_fn,
1 dans TE_fb. Ils touchent des ablations de couches basses (surtout MLP 6 et 8) dans 16 prompts, toutes portes OK.
Interprétation : divergence du forward NON LINÉAIRE à normes figées. Sans contrôle de gain, le SwiGLU (quadratique pour de
grandes entrées) fait croître la perturbation jusqu'au dépassement de capacité Float32.

Le protocole ne prévoyait pas ce cas. Règle retenue, CONSERVATRICE vis-à-vis des hypothèses :
- les couples (p, c) dont l'effet requis est non fini sont EXCLUS des statistiques S1, S2, S4, S5 (ce sont les cas de
  compensation la plus forte, donc les exclure biaise contre S1 et S4) ;
- pour S3, la corrélation de Spearman de chaque prompt est calculée sur les composantes finies ;
- l'ensemble actif est défini avec le max des |TE_fn| FINIS ;
- les divergences sont rapportées à part (nombre, couches, prompts), avec une analyse de sensibilité où φ = 1 pour
  les couples dont TE_fn diverge alors que TE est fini.
