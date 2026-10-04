"""Suivi en direct du run WIND-T8 (lecture seule).
Usage (depuis le dossier NeuroDSL) :  python notebook/wind_T8_progress.py      (Ctrl+C pour quitter)
                                      python notebook/wind_T8_progress.py --once
"""
import glob, os, re, sys, time
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
NB = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(NB, "wind_data_T8")
P = range(1, 51)


def read(name):
    try:
        return open(os.path.join(NB, name), encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def bar(done, total, width=40):
    f = 0 if total == 0 else min(1.0, done / total)
    return "█" * int(f * width) + "░" * (width - int(f * width)) + f" {100 * f:5.1f} %"


def hms(s):
    s = int(max(0, s)); return f"{s // 3600}h{(s % 3600) // 60:02d}"


def current_layers():
    """Avancement de la variante en cours (dernière ligne « couches 0..X faites »)."""
    txt = read("wind_T8_collect_results.txt")
    tail = txt[txt.rfind("PROMPT"):] if "PROMPT" in txt else ""
    m = re.findall(r"\[(\w+)\] couches 0\.\.(\d+) faites", tail)
    done_v = re.findall(r"\[(\w+)\] collecte :", tail)
    return (int(m[-1][1]) + 1) / 28 if m and (not done_v or done_v[-1] != m[-1][0] or len(m) > 4 * len(done_v)) else 0.0


def show():
    log = read("wind_T8_run_all.log")
    start = re.search(r"\[(\S+Z)\] passe 1", log)
    t0 = datetime.fromisoformat(start.group(1).replace("Z", "+00:00")) if start else None
    now = datetime.now(timezone.utc)
    nA = sum(os.path.isfile(os.path.join(DATA, f"wind_J_p{p}_A.bin")) for p in P)
    nAN = sum(os.path.isfile(os.path.join(DATA, f"wind_J_p{p}_AN.bin")) for p in P)
    nT = sum(os.path.isfile(os.path.join(DATA, f"wind_J_p{p}_T.bin")) for p in P)
    npred = len(re.findall(r"^\s+p\d+\s+ern_A", read("wind_T8_predict_results.txt"), re.M))
    ntru = len(re.findall(r"θ_c\^true", read("wind_T8_truth_results.txt")))
    # secondaire : tranches de la reprise (s1..s3) + passe finale ; un prompt compte une fois
    pat = re.compile(r"^\s+p(\d+)\s+ern T.*?\|\s*(\d+) s\s*$", re.M)
    shards, ids = [], set()
    for i, size in ((1, 17), (2, 16), (3, 16)):
        got = pat.findall(read(f"wind_T8_secondary_s{i}.txt"))
        ids.update(int(p) for p, _ in got)
        shards.append((i, size, len(got), [int(s) for _, s in got]))
    ids.update(int(p) for p, _ in pat.findall(read("wind_T8_secondary_results.txt")))
    nsec = len(ids)
    os.system("cls" if os.name == "nt" else "clear")
    print(f"WIND-T8 -- suivi  ({datetime.now().strftime('%H:%M:%S')})\n")
    done1 = nA + nAN + (current_layers() if nAN < 50 else 0)
    eta1 = ""
    if t0 and 0 < done1 < 100:
        el = (now - t0).total_seconds()
        eta1 = f"  écoulé {hms(el)}  reste ~{hms(el / done1 * (100 - done1))}"
    print(f"1. Collecte A + AN  {bar(done1, 100)}  ({nA + nAN}/100 matrices){eta1}")
    print(f"2. Prédictions θ_c  {bar(npred, 50)}  ({npred}/50)")
    print(f"3. Vérité + verdict {bar(ntru, 50)}  ({ntru}/50)")
    doneT = nT + (current_layers() if 0 < nT < 50 and nAN == 50 else 0)
    print(f"4. Collecte T       {bar(doneT, 50)}  ({nT}/50)")
    print(f"5. Secondaire       {bar(nsec, 50)}  ({nsec}/50)")
    for i, size, n, secs in shards:
        if n or os.path.isfile(os.path.join(NB, f"wind_T8_secondary_s{i}.txt")):
            eta = f"  reste ~{hms((size - n) * sum(secs) / len(secs))}" if secs and n < size else ""
            print(f"   tranche {i}       {bar(n, size, 30)}  ({n}/{size}){eta}")
    t9 = re.findall(r"^\s+p(\d+)\s.*\|\s*(\d+) s\s*$", read("wind_T9_attrib_results.txt"), re.M)
    if t9 or os.path.isfile(os.path.join(NB, "wind_T9_attrib.log")):
        secs = [int(s) for _, s in t9]
        eta = f"  reste ~{hms((50 - len(t9)) * sum(secs) / len(secs))}" if secs and len(t9) < 50 else ""
        print(f"6. Attributions T9 {bar(len(t9), 50)}  ({len(t9)}/50){eta}")
    g13 = re.findall(r"^\s+p(\d+)\s+facteurs.*\|\s*(\d+) s\s*$", read("wind_T13_factors_results.txt"), re.M)
    if g13:
        secs = [int(s) for _, s in g13]
        eta = f"  reste ~{hms((50 - len(g13)) * sum(secs) / len(secs))}" if len(g13) < 50 else ""
        print(f"7. Correctif (GPU)  {bar(len(g13), 50)}  ({len(g13)}/50){eta}")
    c13 = set()
    for f in glob.glob(os.path.join(NB, "wind_T13_reconstruct*_results.txt")):
        if "smoke" not in f:
            c13 |= set(re.findall(r"^\s+p(\d+)\s+ern\(θ=1\)", read(os.path.basename(f)), re.M))
    if c13 or g13:
        print(f"8. Correctif (CPU)  {bar(len(c13), 50)}  ({len(c13)}/50)")
    log15 = read("wind_T15_run_all.log")
    if log15:
        nAH = sum(os.path.isfile(os.path.join(DATA, f"wind_J_p{p}_AH.bin")) for p in P)
        t15 = re.search(r"\[(\S+Z)\] collecte AH", log15)
        eta = ""
        if t15 and 0 < nAH < 50:
            el = (now - datetime.fromisoformat(t15.group(1).replace("Z", "+00:00"))).total_seconds()
            eta = f"  reste ~{hms(el / nAH * (50 - nAH))}"
        print(f"\n9. Règle du demi — collecte AH  {bar(nAH, 50)}  ({nAH}/50){eta}")
        nat = len(re.findall(r"^\s+p\d+\s+porte forward", read("wind_T15_attrib_results.txt"), re.M))
        print(f"10. Règle du demi — attributions {bar(nat, 50)}  ({nat}/50)")
        nan_ = sum(len(re.findall(r"^\s+p\d+\s+ern_AH", read(f"wind_T15_analysis_s{i}_results.txt"), re.M)) for i in (1, 2, 3))
        print(f"11. Règle du demi — seuils      {bar(nan_, 50)}  ({nan_}/50)")
        l15 = [l for l in log15.strip().splitlines() if l.strip()]
        print("    dernière étape T15 : " + l15[-1])
        if "FIN" in log15:
            print("\n*** RÈGLE DU DEMI : TERMINÉ -- verdicts dans notebook/wind_T15_verdicts_results.txt ***")
    last = [l for l in log.strip().splitlines() if l.strip()]
    print("\nDernière étape : " + (last[-1] if last else "(pas encore démarré)"))
    if "FIN" in log:
        print("\n*** RUN TERMINÉ ***")


while True:
    show()
    if "--once" in sys.argv:
        break
    time.sleep(10)
