#!/bin/bash
# WIND-T19 -- après la collecte complète (PID en argument) : analyse -> verdict G1 -> esquisses descriptives 16..50.
cd /c/Users/Nevermind/Desktop/NeuroDSL
LOG=notebook/wind_T19_run.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "[$(ts)] attente de la collecte complète (PID $1)" >> $LOG
while [ "$(powershell -NoProfile -Command "(Get-Process -Id $1 -ErrorAction SilentlyContinue | Measure-Object).Count")" != "0" ]; do sleep 30; done
echo "[$(ts)] collecte terminée : $(ls notebook/wind_data_GEMMA/wind_J_p*_AN.bin 2>/dev/null | wc -l)/15 prompts" >> $LOG
julia --project=. notebook/wind_T19_analyze.jl >> notebook/wind_T19_analyze.log 2>&1
PYTHONIOENCODING=utf-8 python notebook/wind_T19_verdicts.py > notebook/wind_T19_verdicts_results.txt 2>&1
echo "[$(ts)] verdict G1 écrit -- $(grep '^G1' notebook/wind_T19_verdicts_results.txt)" >> $LOG
WIND_PROMPTS=$(seq -s, 16 50) julia --project=. notebook/wind_T19_sketch.jl >> notebook/wind_T19_sketch.log 2>&1
echo "[$(ts)] esquisses descriptives 16..50 terminées -- FIN" >> $LOG
