# Pré-enregistrement — greffe Gradient Shadowing sur Qwen2.5-1.5B-Instruct réel

Date : 2026-08-31. Écrit AVANT tout run. Objectif : premier test de correction
minimal de la combinaison jamais exécutée dans ce dépôt — `graft_shadow_block!`
inséré dans les poids réels chargés de Qwen (pas un modèle NeuroDSL-natif),
backbone explicitement gelé (`is_param=false` sur tous les poids Qwen d'origine
après coup — jamais fait à cette échelle avant aujourd'hui), `backward_graph!`
avec `prune_frozen=true`, un pas AdamW sur les seuls paramètres de la greffe.
PAS l'expérience complète "corriger un échec de raisonnement" — seulement la
vérification de correction qui doit passer avant de la tenter.

## Setup commun aux 3 checks

- Modèle : Qwen2.5-1.5B-Instruct, poids réels HuggingFace, chargés depuis
  `notebook/qwen2.5-1.5b-instruct/qwen2_neurodsl.json/.bin` (déjà exporté par
  `load_qwen2.jl`, déjà validé bit-à-bit-proche de la référence HF ailleurs —
  `qwen2_parity_check.jl`, écart abs max 3.7e-5). Ce script ne re-parse PAS le
  safetensors, il recharge le format natif NeuroDSL déjà vérifié.
- Config : dim=1536, n_layers=28, n_heads=12, n_kv_heads=2, hidden_dim=8960,
  vocab=151936, rope_theta=1e6, rms_eps=1e-6, qkv_bias=true.
- Site de greffe : `:layer_25_out` (sortie résiduelle du 25e `LlamaBlock`,
  symbole produit par `LlamaBlock` — `Symbol(prefix,:_out)` dans
  `src/layers.jl`), sur les 28 couches — donc 24 couches en amont (1..24,
  coût nul attendu), 3 couches en aval (26,27,28) + norme finale + lm_head
  (coût aval attendu).
- Bloc greffé : `graft_shadow_block!(g, ns, :layer_25_out, 1536, 4, 384;
  alpha0=0f0, zero_out_proj=false, prefix=:qwen_shadow_l25)` — Gradient
  Shadowing pur (theta aléatoire, gate à 0), PAS Net2Net. dim=1536 (largeur
  pleine du residual stream — le site est le residual stream après le bloc,
  PAS un nœud par tête comme dans `real_llm_graft_experiment.jl`), n_heads=4
  choisi pour diviser 1536 exactement, hidden_dim=384 volontairement petit
  (greffe bon marché, pas un second FFN de la taille de Qwen).
- Prompt réel : `"If a train travels 60 miles in 2 hours, its speed is"`
  (`adhoc_prompts.json`, déjà tokenisé par le vrai tokenizer Qwen), 16 tokens.
  Perte : cross-entropie next-token standard (même schéma que
  `real_llm_graft_experiment.jl`) — entrée = tokens[1:15], labels =
  tokens[2:16].
- Gel du backbone : fonction locale `freeze_backbone!` copiée à l'identique de
  `real_llm_graft_experiment.jl` (déjà utilisée et correcte à petite échelle) —
  parcourt TOUS les nœuds du namespace, met `is_param=false` sur tout nœud
  `is_param=true` dont le symbole ne commence PAS par le préfixe de la greffe.
  AUCUNE modification de `src/` — le helper reste local au script, comme le
  précédent existant.
- Chaque mesure sensible (avant/après, coût comparatif) tourne dans un
  processus `julia` FRAIS séparé (contamination inter-run déjà observée cette
  session sur des mesures dans le même process).

## Check 1 — Correction du gel à l'échelle réelle de Qwen

**Ce qui est vérifié** : après `backward_graph!(g, :loss; namespace=ns,
prune_frozen=true)` sur le prompt réel, AUCUN des nœuds de poids Qwen
d'origine (~339 tenseurs de paramètres attendus : 28×12 par couche
[q/k/v/o_W, q/k/v_b, mlp_w1/w2/w3, norm1/norm2_gamma] + tok_E + lm_head_W +
final_norm_gamma) n'a de `.gradient` calculé/accumulé — doit être `nothing`
pour chacun. Inspection DIRECTE du champ `.gradient` sur LA TOTALITÉ des
nœuds de poids Qwen (pas juste un échantillon, le nombre le permet).
Vérification croisée : compter combien de nœuds ont `is_param=true` avant le
gel (devrait être ~339 + le nombre de tenseurs de la greffe) et après le gel
(devrait être exactement le nombre de tenseurs de la greffe).

**Verdict PASS** si et seulement si : (a) le gel amène `is_param=false` sur
100% des nœuds de poids Qwen d'origine (aucun raté par la boucle), ET (b)
`.gradient === nothing` pour 100% de ces mêmes nœuds après le backward.
**FAIL** si un seul nœud fait exception dans l'un ou l'autre cas.

Note pré-enregistrée sur ce qui est réellement testé : la relecture de
`src/backward.jl`/`src/backward_sparse.jl` montre qu'un balayage de nettoyage
final met TOUJOURS `nd.gradient = nothing` pour tout nœud `is_param=false` en
fin de passe, indépendamment de `prune_frozen`. Le test de correction
pertinent n'est donc pas seulement "le gradient est nul après coup" (garanti
par construction si le gel a réussi) mais surtout "le gel lui-même a-t-il
atteint 100% des nœuds Qwen sans exception" — un raté de nommage/couverture à
cette échelle (28 couches, ~339 tenseurs, jamais testé avant) serait le bug
réel à attraper ici, pas une anomalie du moteur backward lui-même.

## Check 2 — Le gate s'échappe de zéro

**Process A (frais)** : charger Qwen, PAS de greffe, calculer les logits sur
le prompt réel (forward seul). Sauvegarder.

**Process B (frais, séparé)** : charger Qwen, insérer la greffe
(`alpha0=0f0`), calculer les logits AVANT tout entraînement sur le même
prompt. Comparer aux logits du Process A : écart absolu maximal attendu
EXACTEMENT 0.0 (bit-exact — même standard que `Théorème
d'identité`/`Proposition prop:bitexact` de `math1.tex`, déjà vérifié ailleurs
pour ce mécanisme sur des modèles NeuroDSL-natifs, jamais sur Qwen chargé).

Puis, dans le MÊME Process B : geler le backbone (`freeze_backbone!`), un pas
AdamW (`adamw_step_batched!`) sur `params(g;ns)` (qui, après le gel, ne
contient plus que les tenseurs de la greffe — poids R(x;theta) + `alpha`),
avec la perte cross-entropie réelle. Relire `alpha` après ce pas.

**Verdict PASS** si et seulement si : (a) écart logits avant greffe vs après
greffe (alpha=0) = 0.0 exactement, ET (b) `alpha` vaut exactement `0.0f0`
avant le pas AdamW, ET (c) `alpha` est différent de `0.0f0` après UN SEUL pas
AdamW (mécanisme prédit par la Proposition `prop:escape` : `∂Loss/∂alpha =
⟨g, R(x;theta)⟩ ≠ 0` génériquement dès que `g ≠ 0`).

## Check 3 — Coût réel, gelé vs complet

**Process C (frais)** : charger Qwen, greffer, geler. Un appel de chauffe
(compilation CUDA) à `backward_graph!(...; prune_frozen=true)`, puis N=5
répétitions chronométrées (même token, `invalidate_all!` entre chaque). Après
le dernier appel, compter `count(nd.backwarded for nd in g.nodes[ns])` —
champ remis à `false` en tête de CHAQUE appel à `backward_graph!` puis mis à
`true` pour tout nœud effectivement visité pendant la passe (contrairement à
`.gradient`, PAS nettoyé en fin de passe pour les nœuds gelés — instrument
fiable de "quels nœuds ont réellement été touchés").

**Process D (frais, séparé)** : IDENTIQUE à C sauf `prune_frozen=false`
(passe complète, gaspillage volontaire pour établir la référence). Même
chauffe, même N=5, même comptage de `.backwarded`.

**Verdict attendu (pas juste "proche de zéro")** : temps médian(C) < temps
médian(D), avec un facteur cohérent avec le modèle de coût corrigé de
l'évaluation de faisabilité de ce jour — coût nul pour les couches 1-24,
coût aval réduit d'environ moitié pour 25-28+norme+lm_head (seul dA
nécessaire, dB sauté pour les poids gelés), plus le petit coût plein de la
greffe elle-même. Vérification quantitative : `count(backwarded)` en mode C
doit être de l'ordre du cône aval (couches 25-28 + norme + lm_head + greffe),
PAS de l'ordre du nombre total de nœuds du graphe (attendu ~2948) ; en mode D
il doit être beaucoup plus proche du total.

**Verdict FAIL/surprise à rapporter honnêtement** si : le facteur de gain
mesuré est absent, dans le mauvais sens, ou si `count(backwarded)` en mode C
n'est pas nettement inférieur à celui du mode D et proche du total — cela
indiquerait que `prune_frozen` ne taille pas correctement le cône sur ce
graphe réel (jamais testé à cette profondeur/largeur avant).

## Ce qui N'EST PAS testé ici

Pas de vérification que la greffe corrige un échec de raisonnement
spécifique de Qwen — c'est l'étape suivante, contingente à ces 3 checks.
Pas de comparaison de qualité de génération. Pas de mesure VRAM détaillée
(seulement le temps mur et le comptage de nœuds pour Check 3).
