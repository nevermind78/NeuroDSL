"""WIND-T15 -- verdicts (pré-enregistrement : wind_T15_preregistration.md). Écrit AVANT les données.
θ_c = 99 signifie « pas de franchissement » (censuré à 2,5)."""
import json, os, sys, statistics as st, numpy as np
from scipy.stats import spearmanr, wilcoxon
sys.stdout.reconfigure(encoding="utf-8")
NB = os.path.dirname(os.path.abspath(__file__)); D8 = os.path.join(NB, "wind_data_T8")
J = lambda f: json.load(open(os.path.join(NB, f)))
AN_ = J("wind_T15_analysis.json"); AT = J("wind_T15_attrib.json"); T8 = J("wind_T8_truth.json")
PRED = J("wind_T8_predictions.json")["predictions"]; T9 = J("wind_T9_attrib.json")
def gates_ok(p):
    m = json.load(open(os.path.join(D8, f"wind_meta_T15_p{p}.json")))["gates"]
    ok = m["G1_AH_med"] < 1e-3 and m["G1_AH_max"] < 1e-2 and st.median(m["G3_AH"]) < 1e-2 and m["G4a_AH"] is True \
         and m["G4b_AH"] < 1e-5 and AT[p]["gate_GV"] < 1e-2 and AT[p]["gate_forward"] < 1e-5
    return ok, m
ps = [str(p) for p in range(1, 51) if str(p) in AN_ and str(p) in AT]
res = {p: gates_ok(p) for p in ps}; excl = [p for p in ps if not res[p][0]]; ps = [p for p in ps if res[p][0]]
print(f"prompts retenus {len(ps)} ; exclus (portes) {excl}")
cap = lambda v: min(v, 2.5)
CAN = [p for p in ps if T8[p]["ern_true_1"] <= 3]
hitC = sum(AN_[p]["ern1_AH"] <= 3 for p in CAN); hitAll = sum(AN_[p]["ern1_AH"] <= 3 for p in ps)
verdict = "SUPPRIMÉ" if hitC <= 1 and hitAll <= 2 else ("PERSISTE" if hitC >= 6 else "PARTIEL")
print(f"CRITÈRE PRINCIPAL : {verdict}  (AH s'effondre sur {hitC}/{len(CAN)} des effondrés AN ; {hitAll}/{len(ps)} au total)")
tAH = [cap(AN_[p]["theta_c_AH"]) for p in ps]; tAN = [cap(T8[p]["theta_c_true"]) for p in ps]
d = [a - b for a, b in zip(tAH, tAN)]
pw = wilcoxon(d, alternative="greater").pvalue if any(x != 0 for x in d) else 1.0
print(f"D1 (θ_c^AH > θ_c^AN, médiane appariée, p < 0,05) : {'VRAI' if st.median(d) > 0 and pw < 0.05 else 'FAUX'}"
      f"  (médianes {st.median(tAH):.3f} vs {st.median(tAN):.3f} ; diff {st.median(d):+.3f} ; p {pw:.1e} ; Spearman {spearmanr(tAH, tAN).correlation:.3f})")
dR = {p: np.array(T9[p]["dR"]) for p in ps}
rho = lambda a, p: spearmanr(np.array(a), dR[p]).correlation
coll = [p for p in ps if PRED[p]["theta_c_mod2"] < 1]; non = [p for p in ps if p not in coll]
diff = [rho(AT[p]["attr"]["AH"], p) - rho(T9[p]["aAN"], p) for p in coll]
pf = wilcoxon(diff, alternative="greater").pvalue
print(f"F1 (médiane ρ_AH − ρ_AN ≥ +0,10 sur {len(coll)} effondrés prédits, p < 0,05) : "
      f"{'VRAI' if st.median(diff) >= 0.10 and pf < 0.05 else 'FAUX'}  ({st.median(diff):+.3f}, p {pf:.1e})")
dAH = st.median(rho(AT[p]["attr"]["AH"], p) for p in coll) - st.median(rho(AT[p]["attr"]["AH"], p) for p in non)
print(f"F2 (Δ_AH ≥ −0,05) : {'VRAI' if dAH >= -0.05 else 'FAUX'}  ({dAH:+.3f})")
print("\n--- DESCRIPTIF : ρ médian (effondrés / non effondrés) ---")
for nm, f in (("T", lambda p: T9[p]["aT"]), ("A", lambda p: T9[p]["aA"]), ("AN", lambda p: T9[p]["aAN"]),
              ("AH", lambda p: AT[p]["attr"]["AH"]), ("AHf", lambda p: AT[p]["attr"]["AHf"])):
    print(f"  {nm:3s} {st.median(rho(f(p), p) for p in coll):.3f} / {st.median(rho(f(p), p) for p in non):.3f}")
print("ern_AH(1) médian {:.2f} (AN {:.2f}) ; portes GV max {:.1e}".format(
    st.median(AN_[p]["ern1_AH"] for p in ps), st.median(T8[p]["ern_true_1"] for p in ps), max(AT[p]["gate_GV"] for p in ps)))
