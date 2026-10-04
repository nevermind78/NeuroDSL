"""WIND-T17 -- verdicts L1–L4 (pré-enregistrement : wind_T17_preregistration.md). Écrit AVANT les données."""
import json, os, sys, statistics as st
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
NB = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(NB, f)))
G = J("wind_T17_analysis_gpt2.json"); Q = J("wind_T17_analysis_qwen.json"); QE = J("wind_T17_qwen_euler.json"); T8 = J("wind_T8_truth.json")


def gates_ok(p):
    m = json.load(open(os.path.join(NB, "wind_data_GPT2", f"wind_meta_p{p}.json")))["gates"]
    return all(m[f"G1_{V}_med"] < 1e-3 and m[f"G1_{V}_max"] < 1e-2 and st.median(m[f"G3_{V}"]) < 1e-2 for V in ("T", "A", "AN")) \
        and m["G2_max"] < 1e-3 and m["G4a_A"] is True and m["G4a_AN"] is True and m["G4b"] < 1e-5


ps = [str(p) for p in range(1, 51) if str(p) in G and str(p) in Q and str(p) in QE]
excl = [p for p in ps if not gates_ok(p)]; ps = [p for p in ps if p not in excl]
print(f"prompts retenus {len(ps)} ; exclus (portes GPT-2) {excl}")
eg = np.median([d for p in ps for d in G[p]["euler_d"]]); eq = np.median([d for p in ps for d in QE[p]])
print(f"L1 (Euler GPT-2 ≤ 1,3 ET Qwen ≥ 1,7) : {'VRAI' if eg <= 1.3 and eq >= 1.7 else 'FAUX'}  (GPT-2 {eg:.3f} ; Qwen {eq:.3f})")
xg = np.median([a - b for p in ps for a, b in zip(G[p]["lam_AN"], G[p]["lam_T"])])
xq = np.median([a - b for p in ps for a, b in zip(Q[p]["lam_AN"], Q[p]["lam_T"])])
print(f"L2 (excès de gain GPT-2 < 0,5 × Qwen) : {'VRAI' if xg < 0.5 * xq else 'FAUX'}  (GPT-2 {xg:+.3f} ; Qwen {xq:+.3f})")
cg = sum(G[p]["ern_AN"] <= 3 for p in ps); cq = sum(Q[p]["ern_Q12"] <= 3 for p in ps)
print(f"L3 (GPT-2-AN effondré ≤ 2/50 ET Qwen-12 ≥ 6/50) : {'VRAI' if cg <= 2 and cq >= 6 else 'FAUX'}  (GPT-2 {cg}/{len(ps)} ; Qwen-12 {cq}/{len(ps)})")
tq = st.median(min(T8[p]["theta_c_true"], 2.5) for p in ps); tg = st.median(min(G[p]["theta_c_AN"], 2.5) for p in ps)
print(f"L4 (θ_c médian GPT-2 ≥ Qwen + 0,5) : {'VRAI' if tg >= tq + 0.5 else 'FAUX'}  (GPT-2 {tg:.3f} ; Qwen {tq:.3f})")
print("\n--- DESCRIPTIF ---")
print("ern médian GPT-2 : T {:.2f} ; A {:.2f} ; AN {:.2f}".format(*[st.median(G[p][k] for p in ps) for k in ("ern_T", "ern_A", "ern_AN")]))
print("Qwen-12 ern médian {:.2f} ; Qwen AN complet (T8) ern médian {:.2f}".format(
    st.median(Q[p]["ern_Q12"] for p in ps), st.median(T8[p]["ern_true_1"] for p in ps)))
print("gain d'état médian : GPT-2 T {:.3f} AN {:.3f} ; Qwen T {:.3f} AN {:.3f}".format(
    np.median([v for p in ps for v in G[p]["lam_T"]]), np.median([v for p in ps for v in G[p]["lam_AN"]]),
    np.median([v for p in ps for v in Q[p]["lam_T"]]), np.median([v for p in ps for v in Q[p]["lam_AN"]])))
print("Euler par couche (médiane) GPT-2 :", [round(float(np.median([G[p]["euler_d"][l] for p in ps])), 2) for l in range(12)])
print("Euler par couche (médiane) Qwen  :", [round(float(np.median([QE[p][l] for p in ps])), 2) for l in range(28)])
