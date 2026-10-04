"""WIND-T9 -- verdicts B1–B4 + descriptifs (pré-enregistrement : wind_T9_preregistration.md).
Usage : python notebook/wind_T9_analyze.py"""
import json, os, sys, statistics as st
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
sys.stdout.reconfigure(encoding="utf-8")
NB = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(NB, "wind_T9_attrib.json")))
P = json.load(open(os.path.join(NB, "wind_T8_predictions.json")))["predictions"]
T = json.load(open(os.path.join(NB, "wind_T8_truth.json")))
ps = [p for p in map(str, range(1, 51)) if p in A]
excl = [p for p in ps if not A[p]["ok"]]; ps = [p for p in ps if A[p]["ok"]]
print(f"prompts : {len(ps)} retenus ; exclus (portes) : {excl}")
early = list(range(0, 10)) + list(range(28, 38)); late = [i for i in range(56) if i not in early]
def metrics(p, V, idx=None):
    a = np.array(A[p]["a" + V]); d = np.array(A[p]["dR"])
    if idx is not None: a, d = a[idx], d[idx]
    rho = spearmanr(a, d).correlation
    t5 = len(set(np.argsort(-np.abs(a))[:5]) & set(np.argsort(-np.abs(d))[:5])) / 5
    return rho, t5, np.linalg.norm(a - d) / np.linalg.norm(d)
M = {V: {p: metrics(p, V) for p in ps} for V in ("T", "A", "AN")}
def groups(rule):
    c = [p for p in ps if rule(p)]; nc = [p for p in ps if not rule(p)]; return c, nc
def report(name, rule, verdict):
    c, nc = groups(rule)
    print(f"\n=== Groupes : {name} -- effondrés {len(c)}, non effondrés {len(nc)}")
    for V in ("T", "A", "AN"):
        print(f"  {V:2s} ρ médian eff/non {st.median(M[V][p][0] for p in c):.3f} / {st.median(M[V][p][0] for p in nc):.3f} | "
              f"top5 moyen {np.mean([M[V][p][1] for p in c]):.3f} / {np.mean([M[V][p][1] for p in nc]):.3f} | "
              f"err médiane {st.median(M[V][p][2] for p in c):.3f} / {st.median(M[V][p][2] for p in nc):.3f}")
    dAN = st.median(M["AN"][p][0] for p in c) - st.median(M["AN"][p][0] for p in nc)
    dT = st.median(M["T"][p][0] for p in c) - st.median(M["T"][p][0] for p in nc)
    pval = mannwhitneyu([M["AN"][p][0] for p in c], [M["AN"][p][0] for p in nc], alternative="less").pvalue
    t5 = np.mean([M["AN"][p][1] for p in c]) - np.mean([M["AN"][p][1] for p in nc])
    if verdict:
        print(f"B1 (Δ_AN ≤ −0,15 et p < 0,05) : {'VRAI' if dAN <= -0.15 and pval < 0.05 else 'FAUX'}  (Δ_AN {dAN:.3f}, p {pval:.2e})")
        print(f"B2 (Δ_AN − Δ_T ≤ −0,10) : {'VRAI' if dAN - dT <= -0.10 else 'FAUX'}  (Δ_T {dT:.3f}, écart {dAN - dT:.3f})")
        print(f"B3 (top5_AN eff ≤ non − 0,15) : {'VRAI' if t5 <= -0.15 else 'FAUX'}  (écart {t5:.3f})")
    else:
        print(f"  Δ_AN {dAN:.3f} (p {pval:.2e}) ; Δ_T {dT:.3f} ; écart top5 AN {t5:.3f}")
    return c, nc
c, nc = report("θ_c^mod < 1 (prédits, figés)", lambda p: P[p]["theta_c_mod2"] < 1, True)
th = [min(P[p]["theta_c_mod2"], 2.5) for p in ps]; rAN = [M["AN"][p][0] for p in ps]
b4 = spearmanr(th, rAN).correlation
print(f"B4 (Spearman(θ_c^mod, ρ_AN) ≥ 0,3) : {'VRAI' if b4 >= 0.3 else 'FAUX'}  ({b4:.3f})")
print("\n--- DESCRIPTIF ---")
report("effondrement vrai ern_true(1) ≤ 3", lambda p: T[p]["ern_true_1"] <= 3, False)
for nm, idx in (("précoces l ≤ 10", early), ("tardives l ≥ 11", late)):
    for V in ("T", "AN"):
        ce = st.median(metrics(p, V, idx)[0] for p in c); ne = st.median(metrics(p, V, idx)[0] for p in nc)
        print(f"  {nm:16s} {V:2s} ρ médian eff/non {ce:.3f} / {ne:.3f}")
print("Ali / RelP (non effondrés) : ρ médian T {:.3f} vs AN {:.3f} ; err médiane T {:.3f} vs AN {:.3f}".format(
    st.median(M["T"][p][0] for p in nc), st.median(M["AN"][p][0] for p in nc),
    st.median(M["T"][p][2] for p in nc), st.median(M["AN"][p][2] for p in nc)))
print("Portes GA médiane {:.1e} ; GV max T {:.1e} A {:.1e} AN {:.1e}".format(
    st.median(A[p]["gate_GA"] for p in ps), *[max(A[p]["gate_GV"][V] for p in ps) for V in ("T", "A", "AN")]))
