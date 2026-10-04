# WIND-T18 (EXPLORATOIRE) — pourquoi GPT-2 ne s'effondre pas (écrit le 2026-10-03, AVANT tout calcul)

## Contexte
WIND-T17 (50 prompts) donne deux constats :
- figer les normes NE concentre PAS la linéarisation de GPT-2 (AN/A médian 1,04 ; 0/50 sous 0,5), alors que Qwen
  s'effondre (0,27 ; 38/50) ;
- pourtant le gel crée dans GPT-2 un excès de gain d'état par couche (+0,47) au moins aussi fort que dans Qwen (+0,30).

Ces hypothèses sont inspirées par T17 : le test est EXPLORATOIRE, mais ses seuils sont fixés ici avant tout calcul.

## Deux hypothèses
- **H_rad (effacement par la norme finale)** : le gel amplifie bien un mode dans GPT-2, mais ce mode est la direction
  de l'état final (x̃_L, centré pour la LayerNorm), que le Jacobien de la norme finale N annule exactement. Dans Qwen,
  le mode amplifié n'est pas radial et survit à N.
- **H_dom (déjà concentré)** : dans GPT-2, la direction dominante après N est déjà la même avec ou sans gel, et elle
  porte déjà l'essentiel de la sensibilité. Il n'y a donc « plus de place » pour un effondrement.

## Mesures (CPU, données existantes ; GPT-2 = wind_data_GPT2, Qwen = wind_data_T8 ; 50 prompts chacun)
P_V = J_{L−1}···J_1 (brut, sans N) pour V ∈ {A, AN}.
- g_raw = σ₁(P_AN) / σ₁(P_A) : amplification du mode dominant AVANT la norme finale.
- g_N = σ₁(N·P_AN) / σ₁(N·P_A) : la même APRÈS la norme finale.
- c_rad = |cos(u₁(P_AN), x̃_L / ‖x̃_L‖)| : alignement du mode amplifié (avant N) avec la direction annulée par N
  (x̃_L = x_L − moyenne pour GPT-2 ; x_L pour Qwen).
- c_dom = |cos(u₁(N·P_AN), u₁(N·P_A))| : même direction dominante avec et sans gel ?
- p1_A = σ₁² / Σσ² de N·P_A : part de la direction dominante sans gel.

## Seuils (médianes sur les 50 prompts)
- **H_rad VRAIE** si, pour GPT-2 : g_raw ≥ 2 ET g_N ≤ 1,3 ET c_rad ≥ 0,8, ET c_rad(GPT-2) ≥ c_rad(Qwen) + 0,3.
- **H_dom VRAIE** si c_dom(GPT-2) ≥ 0,9 ET p1_A(GPT-2) ≥ 0,4 ET c_dom(Qwen) ≤ 0,6.
- Les deux peuvent être vraies, ou aucune ; tout est rapporté.
