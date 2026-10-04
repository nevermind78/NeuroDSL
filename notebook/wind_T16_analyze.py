"""WIND-T16 -- verdicts S1–S5 + descriptif (pré-enregistrement : wind_T16_preregistration.md). Écrit AVANT les données.
Usage : python notebook/wind_T16_analyze.py"""
import json, os, sys, statistics as st
import numpy as np
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding="utf-8")
NB = os.path.dirname(os.path.abspath(__file__)); D8 = os.path.join(NB, "wind_data_T8")
J = lambda f: json.load(open(os.path.join(NB, f)))
SR = J("wind_T16_selfrepair.json"); T9 = J("wind_T9_attrib.json"); T8 = J("wind_T8_truth.json")
PR = J("wind_T8_prompts.json")
L = 28
ps = [str(p) for p in range(1, 51) if str(p) in SR]
excl = [p for p in ps if not SR[p]["ok"]]; ps = [p for p in ps if SR[p]["ok"]]
print(f"prompts retenus {len(ps)} ; exclus (portes) {excl}")
# null (NaN/Inf de JSON.jl) -> nan ; règle de la déviation pré-enregistrée (couples non finis exclus)
A = {p: {k: np.array([np.nan if v is None else v for v in SR[p][k]], dtype=float) for k in ("TE", "TE_fn", "TE_fb", "TE_ff", "TE_fa", "TE_fan")} for p in ps}
ndiv = {k: sum(int(np.sum(~np.isfinite(A[p][k]))) for p in ps) for k in ("TE", "TE_fn", "TE_fb", "TE_ff", "TE_fa", "TE_fan")}
print("effets non finis par condition :", ndiv)

def active(p):
    fn = A[p]["TE_fn"]; ok = np.isfinite(fn) & np.isfinite(A[p]["TE"])
    return ok & (np.abs(np.where(ok, fn, 0)) >= 0.05 * np.max(np.abs(fn[ok])))
phi = {p: 1 - A[p]["TE"] / A[p]["TE_fn"] for p in ps}

# S1
att = [(abs(A[p]["TE"][i]) < abs(A[p]["TE_fn"][i])) for p in ps for i in np.where(active(p))[0]]
ph = [phi[p][i] for p in ps for i in np.where(active(p))[0]]
s1 = np.mean(att) >= 0.6 and np.median(ph) >= 0.10
print(f"S1 (|TE| < |TE_fn| sur ≥ 60 % des couples actifs ET médiane φ ≥ 0,10) : {'VRAI' if s1 else 'FAUX'}"
      f"  ({np.mean(att):.3f} ; médiane φ {np.median(ph):+.3f} ; n = {len(ph)} couples)")
# S2
fin2 = lambda p: np.isfinite(A[p]["TE_fb"]) & np.isfinite(A[p]["TE_ff"]) & np.isfinite(A[p]["TE"])
sb = sum(np.sum(np.abs(A[p]["TE_fb"] - A[p]["TE"])[fin2(p)]) for p in ps); sf = sum(np.sum(np.abs(A[p]["TE_ff"] - A[p]["TE"])[fin2(p)]) for p in ps)
print(f"S2 (Σ|TE_fb − TE| ≥ Σ|TE_ff − TE|) : {'VRAI' if sb >= sf else 'FAUX'}  (blocs {sb:.2f} vs finale {sf:.2f} ; rapport {sb / sf:.2f})")
# S3
def _r3(p):
    m = A[p]["TE_fan"] - A[p]["TE_fa"]; f = np.isfinite(m)
    return spearmanr(m[f], (np.array(T9[p]["aAN"]) - np.array(T9[p]["aA"]))[f]).correlation
r3 = [_r3(p) for p in ps]
print(f"S3 (médiane Spearman(TE_fan − TE_fa, a^AN − a^A) ≥ 0,7) : {'VRAI' if np.median(r3) >= 0.7 else 'FAUX'}"
      f"  (médiane {np.median(r3):.3f} ; [Q1 {np.quantile(r3, .25):.3f}, Q3 {np.quantile(r3, .75):.3f}])")
# S4
Phi = np.array([np.median(phi[p][active(p)]) for p in ps]); th = np.array([min(T8[p]["theta_c_true"], 2.5) for p in ps])
r4 = spearmanr(th, Phi).correlation
rng = np.random.default_rng(20261003)
null = np.array([spearmanr(th, rng.permutation(Phi)).correlation for _ in range(10000)])
p4 = (np.sum(null <= r4) + 1) / (len(null) + 1)
print(f"S4 (Spearman(θ_c^AN, Φ_p) ≤ −0,3 et p < 0,05) : {'VRAI' if r4 <= -0.3 and p4 < 0.05 else 'FAUX'}  (ρ {r4:+.3f} ; p {p4:.4f})")
# S5
metas = {q: json.load(open(os.path.join(D8, f"wind_meta_p{q}.json"))) for q in range(1, 51)}
muM = [np.mean([np.array(metas[q]["MHA"][l]) for q in range(1, 51)], axis=0) for l in range(L)]
muP = [np.mean([np.array(metas[q]["MLP"][l]) for q in range(1, 51)], axis=0) for l in range(L)]
xs, ys = [], []
for p in ps:
    m = metas[int(p)]
    for i in np.where(active(p))[0]:
        l = (i % L) + 1
        if i < L:
            o = np.array(m["MHA"][l - 1]); d = muM[l - 1] - o; s = np.array(m["H"][l - 1])
        else:
            o = np.array(m["MLP"][l - 1]); d = muP[l - 1] - o; s = np.array(m["X"][l])
        xs.append(phi[p][i]); ys.append((d @ s) ** 2 / ((d @ d) * (s @ s)))
r5 = spearmanr(xs, ys).correlation
print(f"S5 (Spearman(φ_c, cos²(δ_c, ŝ_c)) ≥ 0,2 sur les couples actifs) : {'VRAI' if r5 >= 0.2 else 'FAUX'}  ({r5:+.3f})")

print("\n--- DESCRIPTIF ---")
fin3 = lambda p: np.isfinite(A[p]["TE_fa"]) & np.isfinite(A[p]["TE_fn"]) & np.isfinite(A[p]["TE"])
sa = sum(np.sum(np.abs(A[p]["TE_fa"] - A[p]["TE"])[fin3(p)]) for p in ps); sn = sum(np.sum(np.abs(A[p]["TE_fn"] - A[p]["TE"])[fin3(p)]) for p in ps)
print(f"compensation totale : normes Σ|TE_fn − TE| = {sn:.2f} ; réacheminement de l'attention Σ|TE_fa − TE| = {sa:.2f} ; Σ|TE| = {sum(np.sum(np.abs(A[p]['TE'])) for p in ps):.2f}")
for kind, rng_i in (("MHA", range(0, L)), ("MLP", range(L, 2 * L))):
    for band, ls in (("couches 1–10", range(0, 10)), ("11–20", range(10, 20)), ("21–28", range(20, 28))):
        v = [phi[p][list(rng_i)[j]] for p in ps for j in ls if active(p)[list(rng_i)[j]]]
        if v: print(f"  φ médian {kind} {band:13s} {np.median(v):+.3f}  (n = {len(v)})")
tasks = sorted({PR[int(p) - 1]["task"] for p in ps})
for t in tasks:
    v = [Phi[k] for k, p in enumerate(ps) if PR[int(p) - 1]["task"] == t]
    print(f"  Φ médian tâche {t:10s} {np.median(v):+.3f}  (n = {len(v)})")
print("Portes : G0 max {:.1e} ; G1 max {:.1e}".format(max(SR[p]["G0"] for p in ps), max(SR[p]["G1"] for p in ps)))

# sensibilité (déviation) : φ = 1 quand TE_fn diverge et TE est fini
ph2 = list(ph) + [1.0 for p in ps for i in range(2 * L) if (not np.isfinite(A[p]["TE_fn"][i])) and np.isfinite(A[p]["TE"][i])]
print(f"sensibilité S1 : médiane φ avec divergences comptées φ = 1 : {np.median(ph2):+.3f} (n = {len(ph2)})")
div = [(p, ("MHA" if i < L else "MLP") + str(i % L + 1)) for p in ps for i in range(2 * L) if not np.isfinite(A[p]["TE_fan"][i])]
print("divergences TE_fan (prompt, composante) :", div)
