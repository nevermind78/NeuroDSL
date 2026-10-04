"""WIND-T18 (exploratoire) -- verdicts H_rad, H_dom (pré-enregistrement : wind_T18_preregistration.md). Écrit AVANT les données."""
import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
NB = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(NB, "wind_T18_gpt2.json")))
Q = {}
for i in (1, 2, 3):
    Q.update(json.load(open(os.path.join(NB, f"wind_T18_qwen_s{i}.json"))))
ps = [str(p) for p in range(1, 51)]
m = lambda D, k: float(np.median([D[p][k] for p in ps if p in D]))
print(f"n : GPT-2 {len(G)} ; Qwen {len(Q)}")
for k in ("g_raw", "g_N", "c_rad", "c_rad_A", "c_dom", "p1_A", "p1_AN"):
    print(f"  {k:8s} médiane  GPT-2 {m(G, k):8.3f}   Qwen {m(Q, k):8.3f}")
hr = m(G, "g_raw") >= 2 and m(G, "g_N") <= 1.3 and m(G, "c_rad") >= 0.8 and m(G, "c_rad") >= m(Q, "c_rad") + 0.3
hd = m(G, "c_dom") >= 0.9 and m(G, "p1_A") >= 0.4 and m(Q, "c_dom") <= 0.6
print(f"H_rad (amplification effacée par la norme finale) : {'VRAIE' if hr else 'FAUSSE'}")
print(f"H_dom (GPT-2 déjà concentré, même direction dominante) : {'VRAIE' if hd else 'FAUSSE'}")
