"""Suivi en direct du test GPT-2 (WIND-T17), lecture seule.
Usage (depuis le dossier NeuroDSL) :  python notebook/wind_T17_progress.py      (Ctrl+C pour quitter)
                                      python notebook/wind_T17_progress.py --once
"""
import json, os, re, sys, time
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
NB = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(NB, "wind_data_GPT2")
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


def show():
    res = read("wind_T17_collect_results.txt")
    done = sum(all(os.path.isfile(os.path.join(DATA, f"wind_J_p{p}_{V}.bin")) for V in ("T", "A", "AN")) for p in P)
    # avancement fin du prompt en cours : variantes terminées + couches de la variante en cours
    tail = res[res.rfind("PROMPT"):] if "PROMPT" in res else ""
    nv = len(re.findall(r"^\[(T|A|AN)\] collecte :", tail, re.M))
    lay = re.findall(r"\[(T|A|AN)\] couches 0\.\.(\d+) faites", tail)
    frac = 0.0
    if done < 50 and tail:
        cur = (int(lay[-1][1]) + 1) / 12 if lay and len(lay) > 2 * nv else 0.0
        frac = (nv + cur) / 3
    durs = [float(x) for x in re.findall(r"^\[(?:T|A|AN)\] collecte : (\d+) s", res, re.M)]
    eta = f"  reste ~{hms((50 - done - frac) * 3 * (sum(durs) / len(durs) + 8))}" if durs and done < 50 else ""
    fails = len(re.findall(r"ÉCHEC", res))
    eul = [float(x) for x in re.findall(r"Degré d'Euler effectif du MLP \(norme figée\) : médiane ([\d.]+)", res)]
    cur_prompt = re.findall(r"^PROMPT (\d+) : (.*)$", res, re.M)
    os.system("cls" if os.name == "nt" else "clear")
    print(f"WIND-T17 -- GPT-2, loi d'architecture  ({datetime.now().strftime('%H:%M:%S')})\n")
    print(f"1. Collecte GPT-2 (T, A, AN)  {bar(done + frac, 50)}  ({done}/50 prompts){eta}")
    if cur_prompt and done < 50:
        print(f"   prompt en cours : {cur_prompt[-1][0]}  {cur_prompt[-1][1][:60]}")
    print(f"   portes en échec : {fails}" + ("" if fails == 0 else "   <-- À VÉRIFIER"))
    if eul:
        s = sorted(eul); med = s[len(s) // 2]
        print(f"   degré d'Euler du MLP GPT-2 (médiane par prompt, provisoire) : {med:.2f} sur {len(eul)} prompts  [prédit ≤ 1,3]")
    qe = os.path.isfile(os.path.join(NB, "wind_T17_qwen_euler.json"))
    ag = len(json.load(open(os.path.join(NB, "wind_T17_analysis_gpt2.json")))) if os.path.isfile(os.path.join(NB, "wind_T17_analysis_gpt2.json")) else 0
    aq = len(json.load(open(os.path.join(NB, "wind_T17_analysis_qwen.json")))) if os.path.isfile(os.path.join(NB, "wind_T17_analysis_qwen.json")) else 0
    print(f"\n2. Degré d'Euler Qwen (GPU)    {'terminé' if qe else 'en attente (après la collecte)'}")
    print(f"3. Analyse GPT-2 (CPU)        {bar(ag, 50)}  ({ag}/50)")
    print(f"4. Analyse Qwen + Qwen-12     {bar(aq, 50)}  ({aq}/50)")
    if os.path.isfile(os.path.join(NB, "wind_T17_verdicts_results.txt")):
        print("\n*** VERDICTS ÉCRITS : notebook/wind_T17_verdicts_results.txt ***")


while True:
    try:
        show()
    except Exception as e:                       # fichier en cours d'écriture : on réessaie au tour suivant
        print("lecture en cours…", e)
    if "--once" in sys.argv:
        break
    time.sleep(10)
