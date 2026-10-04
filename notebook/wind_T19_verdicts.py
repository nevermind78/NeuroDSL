"""WIND-T19 -- verdict G1 (prompts 1 à 15, collecte complète) + descriptif. Écrit AVANT les données."""
import json, os, sys, statistics as st
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
NB = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(NB, "wind_T19_analysis.json")))


def gates_ok(r):
    g = r["gates"]
    return all(g[f"G1_{V}_med"] < 1e-3 and g[f"G1_{V}_max"] < 1e-2 and st.median(g[f"G3_{V}"]) < 1e-2 for V in ("A", "AN")) \
        and g["G2_max"] < 1e-3 and g["G4a_A"] is True and g["G4a_AN"] is True and g["G4b"] < 1e-5


ps = [str(p) for p in range(1, 16) if str(p) in A]
excl = [p for p in ps if not gates_ok(A[p])]; ps = [p for p in ps if p not in excl]
print(f"prompts retenus {len(ps)}/15 ; exclus (portes) {excl}")
r = np.array([A[p]["ratio"] for p in ps])
g1 = np.median(r) <= 0.5 and np.mean(r < 0.5) >= 0.5
print(f"G1 (médiane AN/A ≤ 0,5 ET ≥ 50 % sous 0,5) : {'VRAI' if g1 else 'FAUX'}  (médiane {np.median(r):.3f} ; {int(np.sum(r < 0.5))}/{len(ps)} sous 0,5)")
print("rapports par prompt :", ", ".join(f"p{p} {A[p]['ratio']:.2f}" for p in ps))
print("\n--- DESCRIPTIF (médianes) — repères : Qwen AN/A 0,27 ; GPT-2 1,04 ---")
for k in ("ern_A", "ern_AN", "g_raw", "g_N", "c_rad", "c_dom", "p1_A"):
    print(f"  {k:7s} {np.median([A[p][k] for p in ps]):8.3f}")
print("  excès de gain d'état (λ_AN − λ_A) médian {:+.3f}".format(np.median([a - b for p in ps for a, b in zip(A[p]['lam_AN'], A[p]['lam_A'])])))
print("  degré d'Euler médian du MLP {:.3f}".format(np.median([d for p in ps for d in A[p]["euler_d"]])))
th = [0.25 + 0.125 * i for i in range(19)]
cross = []
for p in ps:
    pr = A[p]["path_ratio"]; i = next((j for j, v in enumerate(pr) if v <= 0.5), None)
    cross.append(th[i] if i is not None else float("inf"))
print("  premier θ où ern(θ)/ern(A) ≤ 0,5 (médiane) :", np.median(cross))

# repère à pied d'égalité : Qwen sur les MÊMES prompts (exact, WIND-T8 secondaire), même métrique
S = json.load(open(os.path.join(NB, "wind_T8_secondary.json")))
rq = np.array([S[p]["AN"]["ern"] / S[p]["A"]["ern"] for p in ps])
print(f"\nQwen sur les mêmes {len(ps)} prompts : médiane AN/A {np.median(rq):.3f} ; {int(np.sum(rq < 0.5))}/{len(ps)} sous 0,5")
