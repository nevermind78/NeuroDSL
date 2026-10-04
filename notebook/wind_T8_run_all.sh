#!/bin/bash
# WIND-T8 -- orchestration de la nuit (pré-enregistrement : notebook/wind_T8_preregistration.md)
cd /c/Users/Nevermind/Desktop/NeuroDSL
LOG=notebook/wind_T8_run_all.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
collect() { WIND_PROMPTS=$1 WIND_VARIANTS=$2 julia --project=. notebook/wind_T8_collect.jl >> notebook/wind_T8_collect_$3.log 2>&1; }
echo "[$(ts)] passe 1 (A, AN) démarre" >> $LOG
collect "$(seq -s, 1 50)" A,AN pass1
MISS=$(python notebook/wind_T8_gatecheck.py A,AN --missing 2>>$LOG)
[ -n "$MISS" ] && { echo "[$(ts)] relance passe 1 pour : $MISS" >> $LOG; collect "$MISS" A,AN pass1; }
OK=$(python notebook/wind_T8_gatecheck.py A,AN 2>>$LOG)
echo "[$(ts)] passe 1 terminée ; prompts retenus : $OK" >> $LOG
( collect "$OK" T pass2
  MISS2=$(python notebook/wind_T8_gatecheck.py T --missing 2>>$LOG)
  [ -n "$MISS2" ] && collect "$MISS2" T pass2
  echo "[$(ts)] passe 2 (T) terminée" >> $LOG ) &
PID2=$!
WIND_PROMPTS=$OK julia --project=. notebook/wind_T8_predict.jl >> notebook/wind_T8_analysis.log 2>&1
python -c "import hashlib,datetime;f='notebook/wind_T8_predictions.json';print(datetime.datetime.now(datetime.timezone.utc).isoformat(), f, hashlib.sha256(open(f,'rb').read()).hexdigest())" >> notebook/wind_T8_hashes.txt
echo "[$(ts)] prédictions figées (empreinte dans wind_T8_hashes.txt)" >> $LOG
WIND_PROMPTS=$OK julia --project=. notebook/wind_T8_truth.jl >> notebook/wind_T8_analysis.log 2>&1
echo "[$(ts)] vérité + verdicts H1 terminés" >> $LOG
wait $PID2
OK2=$(python notebook/wind_T8_gatecheck.py T,A,AN 2>>$LOG)
WIND_PROMPTS=$OK2 julia --project=. notebook/wind_T8_secondary.jl >> notebook/wind_T8_analysis.log 2>&1
echo "[$(ts)] secondaire (H2, H3) terminé -- FIN" >> $LOG
