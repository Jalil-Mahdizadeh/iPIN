#!/usr/bin/env bash
#SBATCH --job-name=plmi-retrain-v1
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v1/logs/slurm-%j.log
set -euo pipefail
RUN_ROOT=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v1
cd "$RUN_ROOT"
RUN_NAME=${1:?Pass the frozen run name}
RUN_DIR="$RUN_ROOT/runs/$RUN_NAME"
test -f "$RUN_DIR/config.json"
test -f "$RUN_DIR/code/train.py"
sha256sum --check --quiet "$RUN_DIR/frozen-files.sha256"
if test -f "$RUN_DIR/completed.json"; then
  echo "This frozen run already completed; nothing to resume."
  exit 0
fi
on_requeue() { touch "$RUN_DIR/REQUEST_REQUEUE"; }
trap on_requeue USR1
trap on_requeue TERM
if test -f "$RUN_DIR/REQUEST_STOP"; then
  echo "Manual stop is pending; retained checkpoints can be resumed after removing REQUEST_STOP."
  exit 0
fi
rm -f "$RUN_DIR/REQUEST_REQUEUE"
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
  bash "$RUN_DIR/code/container.sh" torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  "$RUN_DIR/code/train.py" --config "$RUN_DIR/config.json" --output "$RUN_DIR" &
TRAIN_PID=$!
# Bash wait is interrupted when USR1 arrives; keep waiting for the same Slurm step.
set +e
while true; do
  wait "$TRAIN_PID"
  TRAIN_RC=$?
  if kill -0 "$TRAIN_PID" 2>/dev/null; then continue; fi
  break
done
set -e
if test "$TRAIN_RC" -ne 0; then
  echo "Training process failed (exit $TRAIN_RC); complete checkpoints retained."
  exit "$TRAIN_RC"
fi
if test -f "$RUN_DIR/completed.json"; then
  echo "Planned training completed; validation-selected model is recorded in best.json."
  exit 0
fi
if test -f "$RUN_DIR/REQUEST_REQUEUE" && test -f "$RUN_DIR/stopped.json"; then
  if test "${SLURM_RESTART_COUNT:-0}" -ge 8; then
    echo "Automatic restart limit reached; checkpoint retained for manual resubmission."
    exit 78
  fi
  echo "Checkpoint committed; requeuing the same job for continuation."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
if test -f "$RUN_DIR/stopped.json"; then
  echo "Training stopped with a complete checkpoint; no automatic resume was requested."
  exit 0
fi
echo "Worker exited without a completion or stopped marker."
exit 79
