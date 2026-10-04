"""Suivi en direct du test Gemma-2-2B (WIND-T19), lecture seule.
Usage (depuis le dossier NeuroDSL) :  python notebook/wind_T19_progress.py      (Ctrl+C pour quitter)
"""
import json, os, re, sys, time
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
NB = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(NB, "wind_data_GEMMA")


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


def show():
    res = read("wind_T19_collect_results.txt")
    done = sum(all(os.path.isfile(os.path.join(DATA, f"wind_J_p{p}_{V}.bin")) for V in ("A", "AN")) for p in range(1, 16))
    tail = res[res.rfind("PROMPT"):] if "PROMPT" in res else ""
    nv = len(re.findall(r"^\[(A|AN)\] collecte :", tail, re.M))
    lay = re.findall(r"\[(A|AN)\] couches 0\.\.(\d+) faites", tail)
    frac = 0.0
    if done < 15 and tail:
        cur = (int(lay[-1][1]) + 1) / 26 if lay and len(lay) > 3 * nv else 0.0
        frac = (nv + cur) / 2
    durs = [float(x) for x in re.findall(r"^\[(?:A|AN)\] collecte : (\d+) s", res, re.M)]
    eta = f"  reste ~{hms((15 - done - frac) * 2 * (sum(durs) / len(durs) + 30))}" if durs and done < 15 else ""
    cur_prompt = re.findall(r"^PROMPT (\d+) : (.*)$", res, re.M)
    fails = len(re.findall(r"ÉCHEC", res))
    an = json.load(open(os.path.join(NB, "wind_T19_analysis.json"))) if os.path.isfile(os.path.join(NB, "wind_T19_analysis.json")) else {}
    sk = sum(os.path.isfile(os.path.join(DATA, f"wind_sketch_p{p}.bin")) for p in range(16, 51))
    os.system("cls" if os.name == "nt" else "clear")
    print(f"WIND-T19 -- Gemma-2-2B : l'effondrement se reproduit-il ?  ({datetime.now().strftime('%H:%M:%S')})\n")
    print(f"1. Collecte complète A + AN (prompts 1-15)  {bar(done + frac, 15)}  ({done}/15){eta}")
    if cur_prompt and done < 15:
        print(f"   prompt en cours : {cur_prompt[-1][0]}  {cur_prompt[-1][1][:60]}")
    print(f"   portes en échec : {fails}" + ("" if fails == 0 else "   <-- À VÉRIFIER"))
    print(f"2. Analyse (CPU)                            {bar(len(an), 15)}  ({len(an)}/15)")
    if os.path.isfile(os.path.join(NB, "wind_T19_verdicts_results.txt")):
        g1 = [l for l in read("wind_T19_verdicts_results.txt").splitlines() if l.startswith("G1")]
        print("\n*** VERDICT : " + (g1[0] if g1 else "voir notebook/wind_T19_verdicts_results.txt") + " ***")
    print(f"3. Esquisses descriptives (prompts 16-50)   {bar(sk, 35)}  ({sk}/35)")


while True:
    try:
        show()
    except Exception as e:
        print("lecture en cours…", e)
    if "--once" in sys.argv:
        break
    time.sleep(10)
