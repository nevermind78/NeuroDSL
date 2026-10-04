#!/bin/bash
# Gemma-2-2B : fin du téléchargement (reprise si interrompu) -> construction NeuroDSL -> référence transformers -> parité.
cd /c/Users/Nevermind/Desktop/NeuroDSL/notebook
LOG=gemma_setup.log; PY=/c/Users/Nevermind/anaconda3/envs/neurodsl_llm_check/python.exe
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "[$(ts)] attente du téléchargement en cours (PID $1)" >> $LOG
while [ "$(powershell -NoProfile -Command "(Get-Process -Id $1 -ErrorAction SilentlyContinue | Measure-Object).Count")" != "0" ]; do sleep 30; done
echo "[$(ts)] téléchargement initial terminé ; vérification / reprise" >> $LOG
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-2-2b', local_dir='gemma-2-2b', allow_patterns=['*.json', '*.safetensors', 'tokenizer.model'])" >> gemma_download.log 2>&1
echo "[$(ts)] fichiers : $(ls gemma-2-2b/*.safetensors 2>/dev/null | wc -l)/3 morceaux" >> $LOG
cd .. && julia --project=. notebook/gemma_build.jl >> notebook/gemma_build.log 2>&1 && cd notebook
echo "[$(ts)] construction NeuroDSL : $(tail -1 gemma_build.log)" >> $LOG
$PY gemma_reference_logits.py >> gemma_reference.log 2>&1
echo "[$(ts)] référence transformers : $(tail -1 gemma_reference.log)" >> $LOG
cd .. && julia --project=. notebook/gemma_parity_check.jl >> notebook/gemma_parity.log 2>&1; cd notebook
echo "[$(ts)] $(grep 'PORTE DE PARITÉ' gemma_parity.log | tail -1) -- FIN" >> $LOG
