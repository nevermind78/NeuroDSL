"""WIND-1 -- figure recapitulative (lit notebook/wind_analysis_results.json).

Panneaux : (a) norme vraie du transport J_{1->t} vs borne chainee prod ||J_k|| ;
(b) rang effectif erank2(J_{s->28}) ; (c) cos(gradient vrai, gradient gele AN) ;
(d) dispersion par prompt delta_s (H4). Lignes fines grises = prompts, epaisse = mediane.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "wind_analysis_results.json"), encoding="utf-8"))
PP = R["per_prompt"]
S1, S2, CTX = "#2a78d6", "#eb6834", "#c9c8c2"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))

# (a) chainage
a = ax[0, 0]
true_c, chain_c = [], []
for p, d in PP.items():
    c = d["G_curves"]["1"]
    nrm = np.array(c["norm"]); G = np.array(c["G"])
    t = np.arange(2, 2 + len(nrm))
    a.plot(t, np.log10(nrm), color=CTX, lw=0.8); a.plot(t, np.log10(nrm * G), color=CTX, lw=0.8)
    true_c.append(np.log10(nrm)); chain_c.append(np.log10(nrm * G))
t = np.arange(2, 2 + len(true_c[0]))
a.plot(t, np.median(chain_c, 0), color=S2, lw=2)
a.plot(t, np.median(true_c, 0), color=S1, lw=2)
a.text(t[-1], np.median(chain_c, 0)[-1], "  borne chaînée Π‖J_k‖₂", color=INK, va="center", fontsize=8)
a.text(t[-1], np.median(true_c, 0)[-1], "  vrai ‖J₁→t‖₂", color=INK, va="center", fontsize=8)
a.set_xlabel("couche cible t"); a.set_ylabel("log₁₀ amplification max")
a.set_title("(a) Chaîner les normes par couche : écart ~10¹⁷", loc="left", color=INK)
a.set_xlim(2, 36)

# (b) entonnoir
b = ax[0, 1]
E = []
for p, d in PP.items():
    e = np.array(d["erank2_s_to_28"]); b.plot(np.arange(len(e)), e, color=CTX, lw=0.8); E.append(e)
b.plot(np.arange(len(E[0])), np.median(E, 0), color=S1, lw=2)
b.set_yscale("log"); b.set_xlabel("couche source s"); b.set_ylabel("erank₂(J_s→28)  (sur 1536)")
b.set_title("(b) Rang effectif du transport vers la sortie", loc="left", color=INK)

# (c) linéarisation gelée
c = ax[1, 0]
C = []
for p, d in PP.items():
    v = np.array(d["cos_read_AN"]); c.plot(np.arange(len(v)), v, color=CTX, lw=0.8); C.append(v)
c.plot(np.arange(len(C[0])), np.median(C, 0), color=S1, lw=2)
c.set_xlabel("couche source s"); c.set_ylabel("cos(g_s vrai, g_s gelé)")
c.set_title("(c) Gradient logit-diff : vrai vs attention+normes gelées", loc="left", color=INK)

# (d) climat vs météo
dd = ax[1, 1]
if "H4" in R:
    delta = R["H4"]["delta"]
    s = np.arange(len(R["H4"]["median_delta"]))
    M = np.array([delta[str(k)] for k in s])      # (s, prompts)
    for j in range(M.shape[1]): dd.plot(s, M[:, j], color=CTX, lw=0.8)
    dd.plot(s, np.median(M, 1), color=S1, lw=2)
dd.set_xlabel("couche source s"); dd.set_ylabel("δ_s = ‖P⁽ᵖ⁾ − P̄⁽⁻ᵖ⁾‖_F / ‖P⁽ᵖ⁾ − I‖_F")
dd.set_title("(d) Transport d'un prompt vs moyenne des autres", loc="left", color=INK)

fig.suptitle("Qwen2.5-1.5B, dernier token : Jacobiennes exactes par prompt (5 prompts ; gris = prompts, trait épais = médiane)",
             color=INK, fontsize=10)
fig.tight_layout()
out = os.path.join(HERE, "wind_figure.png")
fig.savefig(out, dpi=150, facecolor="white")
print("Écrit :", out)
