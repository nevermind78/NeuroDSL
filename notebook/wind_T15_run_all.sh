#!/bin/bash
# WIND-T15 -- nuit : collecte AH (50 prompts) -> attributions -> seuils (3 tranches) -> verdicts.
cd /c/Users/Nevermind/Desktop/NeuroDSL
LOG=notebook/wind_T15_run_all.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
missing() { for p in $(seq 1 50); do [ -f notebook/wind_data_T8/wind_J_p${p}_AH.bin ] || printf "%s," $p; done | sed 's/,$//'; }
echo "[$(ts)] collecte AH démarre" >> $LOG
WIND_PROMPTS=$(seq -s, 1 50) julia --project=. notebook/wind_T15_collect.jl >> notebook/wind_T15_collect.log 2>&1
M=$(missing); [ -n "$M" ] && { echo "[$(ts)] relance collecte pour : $M" >> $LOG; WIND_PROMPTS=$M julia --project=. notebook/wind_T15_collect.jl >> notebook/wind_T15_collect.log 2>&1; }
echo "[$(ts)] collecte AH terminée ; manquants : '$(missing)'" >> $LOG
julia --project=. notebook/wind_T15_attrib.jl >> notebook/wind_T15_attrib.log 2>&1
echo "[$(ts)] attributions terminées" >> $LOG
i=0
for S in "$(seq -s, 1 17)" "$(seq -s, 18 34)" "$(seq -s, 35 50)"; do
  i=$((i+1))
  WIND_PROMPTS=$S WIND_BLAS=10 WIND_T15_AN_OUT=notebook/wind_T15_analysis_s$i.json WIND_T15_AN_RES=notebook/wind_T15_analysis_s${i}_results.txt \
    julia --project=. notebook/wind_T15_analyze.jl >> notebook/wind_T15_analyze.log 2>&1 &
done
wait
python - <<'PY'
import json
m = {}
for i in (1, 2, 3):
    m.update(json.load(open(f"notebook/wind_T15_analysis_s{i}.json")))
json.dump(m, open("notebook/wind_T15_analysis.json", "w"))
print("fusion :", len(m))
PY
echo "[$(ts)] seuils terminés" >> $LOG
PYTHONIOENCODING=utf-8 python notebook/wind_T15_verdicts.py > notebook/wind_T15_verdicts_results.txt 2>&1
echo "[$(ts)] verdicts écrits -- FIN" >> $LOG
