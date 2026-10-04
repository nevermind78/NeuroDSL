"""WIND-T8 : liste des prompts 1..50 complets et dont les portes de collecte passent (pré-enregistrement).
Usage : python wind_T8_gatecheck.py A,AN [--missing]   -> imprime la liste (virgules) ; exclus sur stderr."""
import json, os, sys, statistics
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wind_data_T8")
V = sys.argv[1].split(","); missing_mode = "--missing" in sys.argv
ok, miss, bad = [], [], []
for p in range(1, 51):
    m = os.path.join(D, f"wind_meta_p{p}.json")
    if not os.path.isfile(m) or not all(os.path.isfile(os.path.join(D, f"wind_J_p{p}_{v}.bin")) for v in V):
        miss.append(p); continue
    g = json.load(open(m))["gates"]; why = []
    for v in V:
        if not (g.get(f"G1_{v}_med", 1) < 1e-3 and g.get(f"G1_{v}_max", 1) < 1e-2): why.append(f"G1_{v}")
        if not (statistics.median(g.get(f"G3_{v}", [1])) < 1e-2): why.append(f"G3_{v}")
        if v in ("A", "AN") and g.get(f"G4a_{v}") is not True: why.append(f"G4a_{v}")
    if "AN" in V and not (g.get("G4b", 1) < 1e-5): why.append("G4b")
    (bad.append((p, why)) if why else ok.append(p))
print(",".join(map(str, miss if missing_mode else ok)))
print(f"manquants {miss} ; exclus (portes) {bad}", file=sys.stderr)
