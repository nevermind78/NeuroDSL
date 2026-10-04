#!/bin/bash
# WIND-T8 -- reprise après le redémarrage Windows (mise à jour) du 2026-10-02 08:12 : T de p50 + analyse secondaire.
cd /c/Users/Nevermind/Desktop/NeuroDSL
LOG=notebook/wind_T8_run_all.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "[$(ts)] reprise après redémarrage Windows : collecte T de p50 + secondaire en 3 tranches" >> $LOG
( WIND_PROMPTS=50 WIND_VARIANTS=T julia --project=. notebook/wind_T8_collect.jl >> notebook/wind_T8_collect_pass2.log 2>&1
  echo "[$(ts)] collecte T de p50 terminée" >> $LOG ) &
i=0
for S in "$(seq -s, 1 17)" "$(seq -s, 18 33)" "$(seq -s, 34 49)"; do
  i=$((i+1))
  WIND_PROMPTS=$S WIND_BLAS=10 WIND_SEC=notebook/wind_T8_secondary_s$i.json WIND_SEC_RES=notebook/wind_T8_secondary_s$i.txt \
    julia --project=. notebook/wind_T8_secondary.jl >> notebook/wind_T8_analysis.log 2>&1 &
done
wait
python - <<'PY'
import json
m = {}
for i in (1, 2, 3):
    m.update(json.load(open(f"notebook/wind_T8_secondary_s{i}.json")))
json.dump(m, open("notebook/wind_T8_secondary.json", "w"))
print("fusion :", len(m), "prompts")
PY
OK2=$(python notebook/wind_T8_gatecheck.py T,A,AN 2>>$LOG)
echo "[$(ts)] prompts retenus pour le secondaire : $OK2" >> $LOG
WIND_PROMPTS=$OK2 julia --project=. notebook/wind_T8_secondary.jl >> notebook/wind_T8_analysis.log 2>&1
echo "[$(ts)] secondaire (H2, H3) terminé -- FIN" >> $LOG
