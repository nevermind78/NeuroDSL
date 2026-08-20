"""
CORRELATION DE RANG : MAGNITUDE |q_j| vs INSTABILITE CROSS-SITE |q_j^(s)-q_j^(t)|
==================================================================================

CE QUE CA TESTE
----------------
Hypothese conversationnelle (non verifiee jusqu'ici, estimee a 0.80-0.91 a la
louche) : les branches de plus grande magnitude |q_j| seraient aussi celles
dont le rang change le plus quand on change de site de lecture -- i.e. la
magnitude au site de reference predirait le classement de l'instabilite
cross-site.

DONNEES
-------
Reutilise notebook/bench_eps_exact_ablation_qwen_multiprompt_matrix.csv
(deja calcule, CPU-only, aucun GPU requis). Colonnes utilisees :
  prompt_idx, site (nom), site_pos, branch (nom), branch_pos, q

Pour chaque prompt, la table contient q_j^(site) pour TOUS les 56 sites de
lecture (site_pos=1..56) et, a chaque site, pour toutes les branches j de son
cone (branch_pos >= site_pos). Site s=1 (layer_1_mha_output_out) a le cone
plein (56 branches) et sert de site de reference dans tout le travail de
session sur la dependance au site.

DEFINITION PRECISE
-------------------
Pour un prompt p et une paire de sites (s, t) avec s=1 (peu profond, cone
plein) et t plus profond (site_pos_t > 1) :
  - branches partagees = cone(t) = {j : branch_pos_j >= site_pos_t}
    (cone(s) est toujours un sur-ensemble puisque s=1)
  - magnitude(j)  := |q_j^(s)|            (magnitude au site de reference s)
  - deviation(j)  := |q_j^(s) - q_j^(t)|  (deviation absolue cross-site)
  - deviation_norm(j) := |q_j^(s)-q_j^(t)| / ((|q_j^(s)|+|q_j^(t)|)/2 + 1e-12)
    (deviation symetrique normalisee, bornee, robuste a l'echelle de |q|)

Pour chaque (prompt, paire de sites), on calcule le coefficient de Spearman
entre magnitude(j) et deviation(j) [et separement deviation_norm(j)], sur
toutes les branches partagees j. On rapporte la DISTRIBUTION de ces
coefficients sur (prompts x paires de sites), pas un seul point.

USAGE : python notebook/bench_eps_rank_correlation_magnitude_deviation.py
ECRIT  notebook/bench_eps_rank_correlation_magnitude_deviation_results.txt
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

HERE = Path(__file__).parent
CSV = HERE / "bench_eps_exact_ablation_qwen_multiprompt_matrix.csv"
OUT = HERE / "bench_eps_rank_correlation_magnitude_deviation_results.txt"

S_POS = 1  # site de reference : cone plein (layer_1_mha_output_out)
T_POS_LIST = [8, 13, 16, 21, 25, 33, 45, 55]  # paires span courte a tres profonde

df = pd.read_csv(CSV)
prompts = sorted(df.prompt_idx.unique())

lines = []
def emit(s=""):
    lines.append(s)
    print(s)

emit("CORRELATION DE RANG : |q_j| (magnitude au site s=1) vs |q_j^(s)-q_j^(t)| (deviation cross-site)")
emit(f"Site de reference s : site_pos={S_POS} (layer_1_mha_output_out, cone plein)")
emit(f"Sites profonds t testes : site_pos in {T_POS_LIST}")
emit(f"Prompts : {len(prompts)} (tous ceux du CSV)")
emit("=" * 92)

raw_rhos = []
norm_rhos = []
per_pair_raw = {t: [] for t in T_POS_LIST}
per_pair_norm = {t: [] for t in T_POS_LIST}

for p in prompts:
    dfp = df[df.prompt_idx == p]
    s_rows = dfp[dfp.site_pos == S_POS].set_index("branch_pos")["q"]
    for t_pos in T_POS_LIST:
        t_rows = dfp[dfp.site_pos == t_pos].set_index("branch_pos")["q"]
        if t_rows.empty:
            continue
        shared = t_rows.index  # cone(t) subset of cone(s)
        qs = s_rows.loc[shared].values
        qt = t_rows.loc[shared].values
        if len(qs) < 4:
            continue  # Spearman peu significatif sur trop peu de branches
        magnitude = np.abs(qs)
        deviation = np.abs(qs - qt)
        deviation_norm = np.abs(qs - qt) / ((np.abs(qs) + np.abs(qt)) / 2 + 1e-12)

        rho_raw, _ = spearmanr(magnitude, deviation)
        rho_norm, _ = spearmanr(magnitude, deviation_norm)
        raw_rhos.append(rho_raw)
        norm_rhos.append(rho_norm)
        per_pair_raw[t_pos].append(rho_raw)
        per_pair_norm[t_pos].append(rho_norm)

emit(f"\nTotal (prompt, paire de sites) exploitables : {len(raw_rhos)}")
emit("\n--- Spearman(magnitude, deviation ABSOLUE |q^s-q^t|) ---")
emit(f"  min={np.min(raw_rhos):.3f}  q25={np.percentile(raw_rhos,25):.3f}  "
     f"median={np.median(raw_rhos):.3f}  q75={np.percentile(raw_rhos,75):.3f}  "
     f"max={np.max(raw_rhos):.3f}  moyenne={np.mean(raw_rhos):.3f}  std={np.std(raw_rhos):.3f}")

emit("\n--- Spearman(magnitude, deviation SYMETRIQUE NORMALISEE) ---")
emit(f"  min={np.min(norm_rhos):.3f}  q25={np.percentile(norm_rhos,25):.3f}  "
     f"median={np.median(norm_rhos):.3f}  q75={np.percentile(norm_rhos,75):.3f}  "
     f"max={np.max(norm_rhos):.3f}  moyenne={np.mean(norm_rhos):.3f}  std={np.std(norm_rhos):.3f}")

emit("\nDetail par paire de sites (site_pos_t), moyenne +/- std sur les 22 prompts :")
emit(f"  {'t_pos':>6} {'n_branches_cone(t)':>20} {'rho_raw (moy+/-std)':>22} {'rho_norm (moy+/-std)':>22}")
for t_pos in T_POS_LIST:
    rr = per_pair_raw[t_pos]
    rn = per_pair_norm[t_pos]
    if not rr:
        continue
    nb = df[(df.prompt_idx == prompts[0]) & (df.site_pos == t_pos)].shape[0]
    emit(f"  {t_pos:>6} {nb:>20} {np.mean(rr):>10.3f} +/- {np.std(rr):<7.3f} "
         f"{np.mean(rn):>10.3f} +/- {np.std(rn):<7.3f}")

emit("\n" + "=" * 92)
emit("INTERPRETATION")
emit("=" * 92)
med_raw = np.median(raw_rhos)
med_norm = np.median(norm_rhos)
emit(f"Mediane observee : rho_raw={med_raw:.3f}, rho_norm={med_norm:.3f}.")
emit("Estimation conversationnelle prealable : 0.80-0.91.")
if med_raw >= 0.80 or med_norm >= 0.80:
    emit("-> La correlation forte tient au moins pour une des deux definitions.")
else:
    emit("-> La correlation mesuree est PLUS FAIBLE que l'estimation conversationnelle :")
    emit("   l'hypothese ne tient pas aussi fort que ce qui avait ete avance a la louche.")
emit("Rappel : rho_raw n'est pas normalise -- une branche de grande magnitude a mecaniquement")
emit("plus de marge pour deriver en valeur absolue (deviation bornee par la magnitude elle-meme")
emit("aux sites profonds ou q change de signe). rho_norm controle partiellement cet artefact")
emit("d'echelle ; l'ecart entre les deux mesure la part de la correlation due au seul effet")
emit("d'echelle plutot qu'a une vraie instabilite relative de rang plus grande pour les grandes")
emit("branches.")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nEcrit : {OUT}")
