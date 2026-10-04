#!/bin/bash
# WIND-T17 -- suite automatique après la collecte GPT-2 (PID donné en argument) :
# degré d'Euler Qwen (GPU) -> analyses GPT-2 et Qwen (CPU, en parallèle) -> verdicts.
cd /c/Users/Nevermind/Desktop/NeuroDSL
LOG=notebook/wind_T17_run.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "[$(ts)] attente de la fin de la collecte (PID $1)" >> $LOG
while [ "$(powershell -NoProfile -Command "(Get-Process -Id $1 -ErrorAction SilentlyContinue | Measure-Object).Count")" != "0" ]; do sleep 30; done
echo "[$(ts)] collecte terminée ; prompts complets : $(ls notebook/wind_data_GPT2/wind_J_p*_AN.bin 2>/dev/null | wc -l)" >> $LOG
julia --project=. notebook/wind_T17_qwen_euler.jl >> notebook/wind_T17_qwen_euler.log 2>&1
echo "[$(ts)] degré d'Euler Qwen terminé" >> $LOG
WIND_T17_MODEL=gpt2 WIND_BLAS=6 julia --project=. notebook/wind_T17_analyze.jl >> notebook/wind_T17_analyze_gpt2.log 2>&1 &
WIND_T17_MODEL=qwen WIND_BLAS=10 julia --project=. notebook/wind_T17_analyze.jl >> notebook/wind_T17_analyze_qwen.log 2>&1 &
wait
echo "[$(ts)] analyses terminées" >> $LOG
PYTHONIOENCODING=utf-8 python notebook/wind_T17_verdicts.py > notebook/wind_T17_verdicts_results.txt 2>&1
echo "[$(ts)] verdicts écrits -- FIN" >> $LOG
