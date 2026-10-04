"""WIND-3 -- figure recapitulative (lit notebook/wind3_analysis_results.json).

(a),(b) carte 2 : knockout exact « remove » (ΔR, dernier token <- source i, une couche) pour
        l'induction (p22) et la Tour Eiffel (p5) ; divergent bleu/rouge, milieu gris, échelle symlog.
(c) HH2 : part du gradient horizontal manquée par la linéarisation gelée (T vs AN), sa part QK
    (T vs A), et la part QK verticale (dernier token) ; gris = prompts, épais = médiane.
(d) HH3 : part d'attention vs part d'influence (knockout remove) par position source ;
    position 1 (puits) en orange.
(e) HH4 : prédiction linéaire d'arête |<c_l, δ>| vs knockout exact |ΔR| (vrai gradient T et gelé AN).
(f) part horizontale de la sensibilité h_s (T, A, AN), médianes.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "wind3_analysis_results.json"), encoding="utf-8"))
PP = R["per_prompt"]
S1, S2, S3, CTX = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2"
INK, INK2 = "#0b0b0b", "#52514e"
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9c2423"])

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))


def kmap(a, p, title):
    d = PP[p]
    K = np.array(d["Krem"])                      # L x (n-1)
    toks = d["tokens"][: d["n"] - 1]
    vmax = np.abs(K).max()
    im = a.imshow(K, aspect="auto", origin="lower", cmap=DIV, vmin=-vmax, vmax=vmax,
                  extent=(0.5, len(toks) + 0.5, 0.5, K.shape[0] + 0.5))
    a.set_xticks(range(1, len(toks) + 1))
    a.set_xticklabels([f"{i}:{t.strip() or repr(t)}" for i, t in enumerate(toks, 1)], rotation=60, ha="right", fontsize=7)
    a.set_ylabel("couche ℓ du knockout"); a.set_title(title, loc="left", color=INK)
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    cb.set_label("ΔR (échelle linéaire)", color=INK2)
    for sp in a.spines.values():
        sp.set_visible(False)


kmap(ax[0, 0], "22", f"(a) Induction : knockout exact (n ← i), ΔR ; R = {PP['22']['R']:+.2f}")
kmap(ax[0, 1], "5", f"(b) Tour Eiffel : knockout exact (n ← i), ΔR ; R = {PP['5']['R']:+.2f}")

# (c) HH2
c = ax[0, 2]
s = np.arange(28)
for p, d in PP.items():
    c.plot(s, d["E_H"], color=CTX, lw=0.8)
M = R["median"]
c.plot(s, M["E_H"], color=S1, lw=2, label="horizontal, T vs AN (QK + norme)")
c.plot(s, M["E_HQK"], color=S3, lw=2, label="horizontal, T vs A (QK seul)")
c.plot(s, M["errV_QK"], color=S2, lw=2, ls="--", label="vertical (dernier token), T vs A")
EX = os.path.join(HERE, "wind3_explore_results.json")
if os.path.exists(EX):
    ex = json.load(open(EX, encoding="utf-8"))
    c.plot(s, ex["E_HQK_nosink"], color=S3, lw=1.4, ls=":", label="horizontal QK, puits exclu (post hoc)")
c.axhline(0.3, color=INK2, lw=0.6, ls=":")
c.text(27.3, 0.3, "seuil 0.3", color=INK2, va="bottom", ha="right", fontsize=7)
c.set_ylim(0, 1.5)
c.text(5.5, 1.45, f"s=0,1,2 : médiane T vs AN = {M['E_H'][0]:.1f}, {M['E_H'][1]:.1f}, {M['E_H'][2]:.1f} (hors cadre)",
       color=INK2, fontsize=7, va="top")
c.set_xlabel("couche source s"); c.set_ylabel("‖g vrai − g gelé‖ / ‖g vrai‖ (pondéré ‖x‖)")
c.set_title("(c) Ce que la linéarisation gelée manque", loc="left", color=INK)
c.legend(frameon=False, fontsize=7.5, loc="lower left")

# (d) HH3
d_ = ax[1, 0]
xs, ys, xs1, ys1 = [], [], [], []
for p, d in PP.items():
    att = np.array(d["pos_attn"][: d["n"] - 1]); att = att / att.sum()
    kr = np.array(d["pos_krem"]); kr = kr / kr.sum()
    xs += list(att[1:]); ys += list(kr[1:]); xs1.append(att[0]); ys1.append(kr[0])
d_.scatter(xs, ys, s=18, color=S1, edgecolor="white", linewidth=0.6, label="positions 2..n−1", zorder=3)
d_.scatter(xs1, ys1, s=46, color=S2, edgecolor="white", linewidth=0.8, label="position 1 (puits)", zorder=4)
lim = [1e-3, 1.2]
d_.plot(lim, lim, color=INK2, lw=0.6, ls=":")
d_.text(0.5, 0.62, "influence = attention", color=INK2, rotation=38, fontsize=7)
d_.set_xscale("log"); d_.set_yscale("log"); d_.set_xlim(lim); d_.set_ylim(1e-3, 1.2)
d_.set_xlabel("part d'attention reçue du dernier token (moy. couches × têtes)")
d_.set_ylabel("part d'influence (Σ_ℓ |ΔR| knockout remove)")
d_.set_title("(d) Attention reçue ≠ influence (5 prompts)", loc="left", color=INK)
d_.legend(frameon=False, fontsize=7.5, loc="lower right")

# (e) HH4
e = ax[1, 1]
for V, col, lab in (("AN", S2, "gelé AN : |⟨c^AN, δ⟩|"), ("T", S1, "vrai T : |⟨c^T, δ⟩|")):
    X, Y = [], []
    for p, d in PP.items():
        X += list(np.abs(np.array(d["Krem"])).ravel()); Y += list(np.abs(np.array(d["LIN"][V])).ravel())
    e.scatter(X, Y, s=5, color=col, alpha=0.55, linewidth=0, label=lab)
lo, hi = 1e-6, 10
e.plot([lo, hi], [lo, hi], color=INK2, lw=0.6, ls=":")
e.set_xscale("log"); e.set_yscale("log"); e.set_xlim(lo, hi); e.set_ylim(lo, hi)
e.set_xlabel("|ΔR| knockout exact (remove)"); e.set_ylabel("|prédiction linéaire de l'arête|")
e.set_title("(e) Carte linéaire vs knockout exact (toutes arêtes)", loc="left", color=INK)
e.legend(frameon=False, fontsize=7.5, loc="upper left", markerscale=3)

# (f) part horizontale
f = ax[1, 2]
s29 = np.arange(29)
for V, col, lab in (("T", S1, "vrai (T)"), ("A", S3, "attention gelée (A)"), ("AN", S2, "attention + normes gelées (AN)")):
    H = np.array([d["hshare"][V] for d in PP.values()])
    f.plot(s29, np.median(H, 0), color=col, lw=2, label=lab)
if os.path.exists(EX):
    f.plot(s29, ex["hshare_nosink"], color=S1, lw=1.4, ls=":", label="vrai (T), puits exclu (post hoc)")
f.set_ylim(0, 1); f.set_xlabel("couche s"); f.set_ylabel("part de S = ‖g‖·‖x‖ portée par les positions i < n")
f.set_title("(f) Part horizontale de la sensibilité (médianes)", loc="left", color=INK)
f.legend(frameon=False, fontsize=7.5, loc="lower left")

fig.suptitle("WIND-3 — Qwen2.5-1.5B : le vent horizontal (influence entre positions vers la lecture top-1 − top-2), 5 prompts",
             color=INK, fontsize=10)
fig.tight_layout()
out = os.path.join(HERE, "wind3_figure.png")
fig.savefig(out, dpi=150, facecolor="white")
print("Écrit :", out)
