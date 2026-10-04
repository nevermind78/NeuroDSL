"""WIND-T13 -- verdicts M1, M2, F1, F2 (pré-enregistrement : wind_T13_preregistration.md). Écrit AVANT les données.
Usage : python notebook/wind_T13_verdicts.py"""
import json, os, sys, statistics as st
import numpy as np
from scipy.stats import spearmanr, wilcoxon
sys.stdout.reconfigure(encoding="utf-8")
NB = os.path.dirname(os.path.abspath(__file__))
REC = json.load(open(os.path.join(NB, "wind_T13_reconstruct.json")))
FAC = json.load(open(os.path.join(NB, "wind_T13_factors.json")))
T8 = json.load(open(os.path.join(NB, "wind_T8_truth.json")))
PRED = json.load(open(os.path.join(NB, "wind_T8_predictions.json")))["predictions"]
T9 = json.load(open(os.path.join(NB, "wind_T9_attrib.json")))
ps = [p for p in map(str, range(1, 51)) if p in REC and p in FAC]
excl = [p for p in ps if not REC[p]["ok"]]; ps = [p for p in ps if REC[p]["ok"]]
# JSON.jl écrit Inf en null : θ_c = None <=> aucun franchissement (contrôlé sur la courbe)
for p in ps:
    for V in ("AA", "AM"):
        if REC[p][f"theta_c_{V}"] is None:
            assert min(REC[p][f"ern_{V}"]) > 3, (p, V)
            REC[p][f"theta_c_{V}"] = float("inf")
print(f"prompts retenus {len(ps)} ; exclus (portes) {excl}")
cap = lambda v: min(v, 2.5)
# M1
an_coll = [p for p in ps if T8[p]["ern_true_1"] <= 3]
am_hit = sum(REC[p]["ern1_AM"] <= 3 for p in an_coll)
sp_am = spearmanr([cap(REC[p]["theta_c_AM"]) for p in ps], [cap(T8[p]["theta_c_true"]) for p in ps]).correlation
print(f"M1 (AM s'effondre sur ≥ 10 des {len(an_coll)} effondrés AN, et Spearman(θ_c^AM, θ_c^AN) ≥ 0,8) : "
      f"{'VRAI' if am_hit >= 10 and sp_am >= 0.8 else 'FAUX'}  ({am_hit}/{len(an_coll)} ; Spearman {sp_am:.3f})")
# M2
aa_ok = sum(REC[p]["ern1_AA"] > 3 for p in ps); aa_nocross = sum(not np.isfinite(REC[p]["theta_c_AA"]) for p in ps)
print(f"M2 (AA non effondré sur ≥ 49/50, et pas de franchissement sur ≥ 90 %) : "
      f"{'VRAI' if aa_ok >= len(ps) - 1 and aa_nocross >= 0.9 * len(ps) else 'FAUX'}  ({aa_ok}/{len(ps)} ; {aa_nocross}/{len(ps)} sans franchissement)")
# F1, F2
dR = {p: np.array(T9[p]["dR"]) for p in ps}
rho = lambda p, V: spearmanr(np.array(FAC[p]["attr"][V]), dR[p]).correlation
rhoT = lambda p: spearmanr(np.array(T9[p]["aT"]), dR[p]).correlation
coll = [p for p in ps if PRED[p]["theta_c_mod2"] < 1]; non = [p for p in ps if p not in coll]
diff = [rho(p, "AA") - rho(p, "AN") for p in coll]
pw = wilcoxon(diff, alternative="greater").pvalue
print(f"F1 (médiane ρ_AA − ρ_AN ≥ +0,10 sur les {len(coll)} effondrés prédits, Wilcoxon p < 0,05) : "
      f"{'VRAI' if st.median(diff) >= 0.10 and pw < 0.05 else 'FAUX'}  (médiane {st.median(diff):+.3f}, p {pw:.2e})")
dAA = st.median(rho(p, "AA") for p in coll) - st.median(rho(p, "AA") for p in non)
print(f"F2 (Δ_AA = médiane ρ_AA(eff) − médiane ρ_AA(non) ≥ −0,05) : {'VRAI' if dAA >= -0.05 else 'FAUX'}  ({dAA:+.3f})")
print("\n--- DESCRIPTIF : médianes de ρ (effondrés / non effondrés) ---")
for nm, fn in (("T", rhoT), ("AN", lambda p: rho(p, "AN")), ("AA", lambda p: rho(p, "AA")), ("AM", lambda p: rho(p, "AM"))):
    print(f"  {nm:2s} {st.median(fn(p) for p in coll):.3f} / {st.median(fn(p) for p in non):.3f}")
print("Portes : G1 méd max {:.1e} ; G2 max {:.1e} ; G3 max {:.1e} ; G4 max {:.1e}".format(
    max(REC[p]["G1_med"] for p in ps), max(max(REC[p]["G2_AA"] + REC[p]["G2_AM"]) for p in ps),
    max(max(REC[p]["G3_AA"], REC[p]["G3_AM"]) for p in ps), max(REC[p]["G4"] for p in ps)))
