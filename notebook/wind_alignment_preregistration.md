# WIND-2 — Alignement spectral des couches adjacentes : pré-enregistrement

Date : 2026-09-29. Écrit AVANT tout calcul des quantités ci-dessous, sur les Jacobiennes WIND-1
déjà collectées (`notebook/wind_data/`, 5 prompts × 3 variantes T/A/AN). La seule quantité déjà
connue est G(k,k+2) = 1/γ_k (≈ 2.4–9.5, `wind_analysis_results.txt`) ; elle ne sert donc que de
porte de cohérence, pas d'hypothèse.

## Objet

J_k = U_k Σ_k V_kᵀ, J_{k+1} = U_{k+1} Σ_{k+1} V_{k+1}ᵀ. La composition vaut
J_{k+1}J_k = U_{k+1} (Σ_{k+1} O_k Σ_k) V_kᵀ avec O_k := V_{k+1}ᵀ U_k (orthogonale, n = 1536).
On compare la SORTIE de J_k (colonnes de U_k, ce que la couche k écrit) à l'ENTRÉE de J_{k+1}
(colonnes de V_{k+1}, ce que la couche k+1 amplifie) — deux sous-espaces du même espace x_{k+1}.
P_k := O_k ∘ O_k (doublement stochastique).

Sous-espaces « amplifiés » : r_a(k) = #{σ_i(J_k) > 1.5}, r_b(k+1) = #{σ_i(J_{k+1}) > 1.5} ;
« atténués » : r_d(k+1) = #{σ_i(J_{k+1}) < 0.5} (mêmes seuils que WIND-1).

## Métriques

- **Capture amplifiée** E_amp(k) = [‖O_k[1:r_b, 1:r_a]‖_F² / r_a] / (r_b / n) : enrichissement,
  par rapport au hasard, de la part de l'énergie du sous-espace amplifié écrit par k qui tombe dans
  le sous-espace amplifié lu par k+1. 1 = hasard.
- **Capture atténuée** E_damp(k) = [‖O_k[n−r_d+1:n, 1:r_a]‖_F² / r_a] / (r_d / n).
- **Gain de composition** γ_k = ‖J_{k+1}J_k‖₂ / (σ₁^(k+1) σ₁^(k)), comparé à un nul de Haar :
  γ_null = médiane sur 3 tirages de ‖Σ_{k+1} Q Σ_k‖₂ / (σ₁σ₁'), Q orthogonale de Haar
  (spectres réels conservés, appariement aléatoire).
- **Indice d'appariement exact** 𝒜_k = ‖J_{k+1}J_k‖_F² / Σ_i (σ_i^(k+1) σ_i^(k))² ∈ (0,1]
  (identité ‖J_{k+1}J_k‖_F² = Σ_ij σ_i'² P_ij σ_j² ; max 1 par Birkhoff + réarrangement) ;
  nul analytique 𝒜_rand = ‖J_{k+1}‖_F² ‖J_k‖_F² / (n Σ_i (σ_i' σ_i)²).
- **Angles principaux** (descriptif) : cos θ_i = valeurs singulières de O_k[1:r, 1:r],
  r ∈ {1, 2, 5, 10, 20, 50, 100, 200} ; α_r = (ρ_r − r/n)/(1 − r/n), ρ_r = ‖O_k[1:r,1:r]‖_F²/r.
- **Stabilité (Wedin)** : η_r = c·‖J‖_F / (σ_r − σ_{r+1}), c = 1.2e-3 (erreur relative max de la
  porte G1 de WIND-1, utilisée comme proxy de l'erreur de différence finie). Angle à r « certifié »
  si η_r < 0.1 pour J_k ET J_{k+1}. Rapporté, pas utilisé pour filtrer les verdicts.

## Portes (échec = résultats invalides, rapportés comme tels)

- **GA1** identité exacte : |‖J_{k+1}J_k‖_F² − Σ_ij σ_i'² P_ij σ_j²| / ‖J_{k+1}J_k‖_F² < 1e-10.
- **GA2** γ_k (T) = 1/G_adjacent de `wind_analysis_results.json` (calcul indépendant) : écart
  relatif < 1e-6.
- **GA3** P_k doublement stochastique : sommes de lignes et colonnes à 1 ± 1e-10.

## Hypothèses (variante T, médiane sur les 5 prompts × paires k = 1..26 ; la paire k = 0,
## qui implique la couche d'embedding atypique ‖J_0‖ ≈ 150, est rapportée à part)

- **HA1 — alignement des sous-espaces amplifiés.** ALIGNÉ si médiane E_amp > 2 ; HASARD si
  médiane E_amp ∈ [0.67, 1.5] ; ANTI-ALIGNÉ si < 0.67 ; sinon INTERMÉDIAIRE. Aucune prédiction
  directionnelle.
- **HA2 — signature de correction** (cf. Patrawala et al., couches adjacentes qui se corrigent) :
  PRÉSENTE si médiane E_damp > 1.5 ET médiane E_amp < 1 ; ABSENTE si médiane E_damp < 1.2 ;
  sinon AMBIGUË.
- **HA3 — appariement vs hasard** : médiane de γ_k/γ_null : > 1.2 « plus aligné qu'un appariement
  aléatoire », < 0.8 « moins aligné », sinon « indiscernable du hasard ».

Descriptif sans seuil : courbes α_r, 𝒜 normalisé, fraction certifiée par Wedin, et comparaison
T / A / AN de E_amp, E_damp et 𝒜 (le gel de l'attention change-t-il l'alignement ?).

## Ce qui ne sera PAS affirmé

Rien au-delà du dernier token et de ces 5 prompts ; rien de causal sur « ce que la couche
calcule » — seulement la géométrie locale du transport au point d'activation réel.
