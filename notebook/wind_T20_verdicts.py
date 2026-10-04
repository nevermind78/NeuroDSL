# WIND-T20 — verdicts des parties A, B, C (critères de wind_T20_preregistration.md, fixés avant calcul).
import json, os, statistics as st
import numpy as np
from scipy.stats import spearmanr, wilcoxon

NB = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(NB, f)))
med = lambda xs: float(np.median(xs))
V = lambda ok: "VRAI" if ok else "FAUX"
out = []
emit = lambda s: (out.append(s), print(s))

# ── Partie A ──────────────────────────────────────────────────────────────────────────────────────
A = J("wind_T20_thm1.json")
emit("=== Partie A — théorème 1 : hypothèses aux 18 points (Qwen, Gemma : couches 3/13/23 ; GPT-2 : 2/6/10 ; p1, p2)")
thr = {"qwen": 1e-2, "gemma": 1e-3, "gpt2": 1e-3}
p0 = all(r["P0"] < thr[r["model"]] for r in A.values())
h1 = all(r["E_rel"] >= 0.1 for r in A.values())
h2 = all(r["cholesky_ok"] and r["lam_ratio"] >= 1e-10 for r in A.values())
emit(f"P0 parité : {V(p0)} (max {max(r['P0'] for r in A.values()):.1e})")
emit(f"H1 excès d'Euler ‖E‖/‖f−f_base‖ ≥ 0,1 : {V(h1)} (min {min(r['E_rel'] for r in A.values()):.2f})")
emit(f"H2 injectivité (Cholesky, λ_min/λ_max ≥ 1e-10) : {V(h2)} (min ratio {min(r['lam_ratio'] for r in A.values()):.1e})")
for m in ("qwen", "gemma", "gpt2"):
    R = [r for r in A.values() if r["model"] == m]
    rh = [r["r_half"] for r in R if "r_half" in r]
    emit(f"  {m:5s} r* ∈ [{min(r['r_star'] for r in R):.3f} ; {max(r['r_star'] for r in R):.3f}]"
         + (f" | r_½ ∈ [{min(rh):.3f} ; {max(rh):.3f}] | r_½ ≥ r* partout : {all(r['r_half'] >= r['r_star'] for r in R)}" if rh else "")
         + f" | Euler effectif ∈ [{min(r['euler_deg'] for r in R):.2f} ; {max(r['euler_deg'] for r in R):.2f}]")
emit(f"Question quantitative r* ≥ 0,05 partout : {V(all(r['r_star'] >= 0.05 for r in A.values()))}")

# ── Partie B ──────────────────────────────────────────────────────────────────────────────────────
B = J("wind_T20_rec.json"); T8 = J("wind_T8_truth.json"); S8 = J("wind_T8_secondary.json")
E17 = J("wind_T17_qwen_euler.json"); A15 = J("wind_T15_analysis.json"); t15 = J("wind_T15_attrib.json"); t9 = J("wind_T9_attrib.json")
ps = [p for p in map(str, range(1, 51)) if p in B]
excl = [p for p in ps if not (B[p]["gate_G1_max"] < 5e-2 and B[p]["gate_T14"] < 1e-3 and B[p]["gate_T9A"] < 1e-3 and B[p]["gate_lift"] < 1e-6)]
ps = [p for p in ps if p not in excl]
emit(f"\n=== Partie B — REC sur Qwen : {len(ps)} prompts retenus, exclus {excl}")
emit(f"portes : G1 max {max(B[p]['gate_G1_max'] for p in ps):.1e} ; T14 max {max(B[p]['gate_T14'] for p in ps):.1e} ; "
     f"T9(A) max {max(B[p]['gate_T9A'] for p in ps):.1e} ; relèvement max {max(B[p]['gate_lift'] for p in ps):.1e}")
cap = lambda t: min(t, 2.5)
coll = [p for p in ps if B[p]["ern_REC"] <= 3]
dbar = {p: float(np.median(E17[p])) for p in ps}
thAN = {p: cap(T8[p]["theta_c_true"]) for p in ps}; thR = {p: cap(B[p]["theta_c_REC"]) for p in ps}
pred = [p for p in ps if thAN[p] * dbar[p] < 1]
emit(f"REC effondrés (ern ≤ 3) : {len(coll)}/50 {coll} ; prédits (θ_c^AN·d̄ < 1) : {pred}")
b1a = len(coll) <= 6
if len(coll) >= 3 and len(coll) < len(ps):
    tp = len(set(coll) & set(pred)); fn = len(set(coll) - set(pred)); fp = len(set(pred) - set(coll)); tn = len(ps) - tp - fn - fp
    bacc = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)); b1b = bacc >= 0.8
    emit(f"  exactitude équilibrée {bacc:.3f} (VP {tp} FN {fn} FP {fp} VN {tn})")
else:
    b1b = True; emit("  < 3 effondrements REC : critère d'exactitude non applicable")
ratio = [thR[p] / thAN[p] for p in ps]; sp = spearmanr([thR[p] for p in ps], [thAN[p] for p in ps]).correlation
b1c = 1.4 <= med(ratio) <= 2.2 and sp >= 0.8
emit(f"B1 (stabilité) : {V(b1a and b1b and b1c)}  [compte {V(b1a)} ; exactitude {V(b1b)} ; θ_c^REC/θ_c^AN médian {med(ratio):.3f}, "
     f"Spearman {sp:.3f} → {V(b1c)} ; θ_c^REC censuré à 2,5 sur {sum(B[p]['theta_c_REC'] >= 2.5 for p in ps)}/{len(ps)}]")
ernA = {p: S8[p]["A"]["ern"] for p in ps}; ernT = {p: S8[p]["T"]["ern"] for p in ps}
ernAN = {p: T8[p]["ern_true_1"] for p in ps}; ernAH = {p: A15[p]["ern1_AH"] for p in ps}; ernR = {p: B[p]["ern_REC"] for p in ps}
rA = med([ernR[p] / ernA[p] for p in ps])
dR_ = med([abs(np.log(ernR[p] / ernT[p])) for p in ps]); dN = med([abs(np.log(ernAN[p] / ernT[p])) for p in ps]); dH = med([abs(np.log(ernAH[p] / ernT[p])) for p in ps])
b2 = 0.5 <= rA <= 1.5 and dR_ < dN and dR_ < dH
emit(f"B2 (géométrie) : {V(b2)}  [ern_REC/ern_A médian {rA:.3f} ; |ln(ern_V/ern_T)| médian REC {dR_:.3f}, AN {dN:.3f}, AH {dH:.3f} ; "
     f"ern médian T {med(list(ernT.values())):.2f} A {med(list(ernA.values())):.2f} AN {med(list(ernAN.values())):.2f} AH {med(list(ernAH.values())):.2f} REC {med(list(ernR.values())):.2f}]")
spr = lambda a, b: spearmanr(a, b).correlation
rho = {V_: {p: B[p][f"rho_{V_}"] for p in ps} for V_ in ("A", "ANf", "AAf", "REC")}
rho["AHf"] = {p: spr(t15[p]["attr"]["AHf"][28:56], t9[p]["dR"][28:56]) for p in ps}
rho["T"] = {p: spr(t9[p]["aT"][28:56], t9[p]["dR"][28:56]) for p in ps}
diff = [rho["REC"][p] - rho["ANf"][p] for p in ps]; wp = wilcoxon(diff, alternative="greater").pvalue
b3a = med(diff) >= 0.05 and wp < 0.05; b3b = med(list(rho["REC"].values())) >= med(list(rho["AHf"].values()))
b3c = abs(med(list(rho["REC"].values())) - med(list(rho["AAf"].values()))) <= 0.05
emit(f"B3 (fidélité, 28 MLP) : {V(b3a and b3b and b3c)}  [REC − ANf médian {med(diff):+.3f} p {wp:.1e} → {V(b3a)} ; "
     f"REC ≥ AHf {V(b3b)} ; |REC − AAf| ≤ 0,05 {V(b3c)}]")
emit("  ρ médians (28 MLP) : " + " ; ".join(f"{k} {med(list(v.values())):.3f}" for k, v in rho.items()))
early = lambda a: float(np.sum(np.abs(a[:10])) / np.sum(np.abs(a)))
eT = {p: B[p]["early_truth"] for p in ps}
dev = {"REC": med([abs(B[p]["early_REC"] - eT[p]) for p in ps]), "ANf": med([abs(B[p]["early_ANf"] - eT[p]) for p in ps]),
       "AHf": med([abs(early(np.array(t15[p]["attr"]["AHf"][28:56])) - eT[p]) for p in ps]),
       "T": med([abs(early(np.array(t9[p]["aT"][28:56])) - eT[p]) for p in ps]), "AAf": med([abs(B[p]["early_AAf"] - eT[p]) for p in ps])}
b4 = dev["REC"] <= min(dev["ANf"], dev["AHf"])
emit(f"B4 (biais de profondeur) : {V(b4)}  [écart médian |early_V − vérité| : " + " ; ".join(f"{k} {v:.3f}" for k, v in dev.items())
     + f" ; part précoce médiane vérité {med(list(eT.values())):.3f}, REC {med([B[p]['early_REC'] for p in ps]):.3f}, "
     f"ANf {med([B[p]['early_ANf'] for p in ps]):.3f}, AHf {med([early(np.array(t15[p]['attr']['AHf'][28:56])) for p in ps]):.3f}]")
# descriptif : groupes θ_c^mod < 1 (T9)
P8 = J("wind_T8_predictions.json")["predictions"]
eff = [p for p in ps if P8[p]["theta_c_mod2"] < 1]; non = [p for p in ps if p not in eff]
emit("  descriptif ρ médian effondrés/non (θ_c^mod<1, 14/36) : " + " ; ".join(
    f"{k} {med([v[p] for p in eff]):.3f}/{med([v[p] for p in non]):.3f}" for k, v in rho.items()))

# ── Partie C ──────────────────────────────────────────────────────────────────────────────────────
emit("\n=== Partie C — mémoire de boucle (prompts d'évaluation pré-enregistrés)")
Lq, Lg, Lm = J("wind_T20_lift_qwen.json"), J("wind_T20_lift_gpt2.json"), J("wind_T20_lift_gemma.json")
sel = {"gpt2": (Lg, [str(p) for p in range(11, 31)]), "qwen": (Lq, [str(p) for p in range(11, 21)]), "gemma": (Lm, [str(p) for p in range(3, 9)])}
Lf, M = {}, {}
for m, (d, pl) in sel.items():
    pl = [p for p in pl if p in d]
    Lf[m] = med([d[p]["loop_factor"] for p in pl]); M[m] = med([d[p]["normG"] / d[p]["normGloc"] for p in pl])
    emit(f"  {m:5s} n {len(pl):2d} | L_f médian {Lf[m]:.3f} | M médian {M[m]:.3f} | portes relèvement max {max(d[p]['gate_lift'] for p in pl):.1e}")
c1 = Lf["gpt2"] <= 1.6 and Lf["qwen"] >= 2.0 and Lf["gemma"] >= 1.5 * Lf["qwen"]
c2 = M["gpt2"] <= 1.5 and M["qwen"] >= 3 and M["gemma"] >= 3
emit(f"C1 (facteur de boucle) : {V(c1)}")
emit(f"C2 (mémoire) : {V(c2)}")
q20 = [str(p) for p in range(1, 21) if str(p) in Lq]
c3 = spearmanr([np.log(Lq[p]["loop_factor"]) for p in q20], [T8[p]["theta_c_true"] for p in q20]).correlation
emit(f"C3 (descriptif, Qwen 1–20) Spearman(ln L_f, θ_c^vrai) = {c3:.3f} → {V(c3 <= -0.6)}")
open(os.path.join(NB, "wind_T20_verdicts_results.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
