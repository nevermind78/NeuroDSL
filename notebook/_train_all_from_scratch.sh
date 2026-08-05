#!/bin/bash
# Sequential from-scratch training of the 5 marker-task instances + the bidir
# instance, reusing notebook/marker_task_experiment.jl and
# notebook/bidir_recall_task_experiment.jl UNCHANGED (only ENV vars set).
# Existing checkpoints under marker_ckpt/_scratch/ and bidir_ckpt/_scratch/
# are used purely as a same-run resume cache (skipped if already trained by
# an earlier, interrupted attempt of THIS script) -- never as a silent
# external prerequisite.
set -e
cd "$(dirname "$0")"
mkdir -p marker_ckpt/_scratch bidir_ckpt/_scratch

run_marker() {
  local name="$1" init_seed="$2" train_seed="$3"
  local ckpt="marker_ckpt/_scratch/${name}"
  if [ -f "${ckpt}.json" ] && [ -f "${ckpt}.bin" ]; then
    echo "=== [$(date +%T)] $name already trained (resume cache hit), skipping ==="
    return 0
  fi
  echo "=== [$(date +%T)] Training $name from scratch (init_seed=$init_seed train_seed=$train_seed) ==="
  MARKER_INIT_SEED=$init_seed MARKER_TRAIN_SEED=$train_seed MARKER_SAVE="$ckpt" MARKER_SAVE_ALWAYS=1 \
    julia --project=.. marker_task_experiment.jl
  echo "=== [$(date +%T)] Finished $name ==="
}

run_marker inst2   1  123
run_marker seed_11 11 11123
run_marker seed_22 22 22123
run_marker seed_33 33 33123
run_marker seed_44 44 44123

echo "=== [$(date +%T)] Training bidir instance from scratch ==="
if [ -f bidir_ckpt/_scratch/bidir_task.json ] && [ -f bidir_ckpt/_scratch/bidir_task.bin ]; then
  echo "bidir already trained (resume cache hit), skipping"
else
  BIDIR_INIT_SEED=1 BIDIR_TRAIN_SEED=123 BIDIR_SAVE=bidir_ckpt/_scratch/bidir_task BIDIR_SAVE_ALWAYS=1 \
    julia --project=.. bidir_recall_task_experiment.jl
fi

echo "=== [$(date +%T)] ALL TRAINING DONE ==="
