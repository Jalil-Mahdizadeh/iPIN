#!/usr/bin/env bash
#SBATCH --job-name=plmi-v3-final
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
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v3/logs/final-%j.log
set -euo pipefail
R=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v3
RELEASE_DIR=${1:?Pass the frozen final release directory}
RUN_NAME=clean-residue-mean-official-seed2
RUN_DIR="$R/runs/$RUN_NAME"
CONFIG="$RELEASE_DIR/configs/$RUN_NAME.json"
mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/.pipeline.lock"
flock -n 9 || { echo 'Another final training owns this run.'; exit 73; }
cd "$R"
on_requeue() { touch "$RUN_DIR/REQUEST_REQUEUE"; }
trap on_requeue USR1 TERM
if test -f "$RUN_DIR/REQUEST_STOP"; then
  echo 'Final training is manually stopped. Retained artifacts are unchanged.'
  exit 0
fi
rm -f "$RUN_DIR/REQUEST_REQUEUE"
python "$RELEASE_DIR/code/final_runtime.py" "$RELEASE_DIR"

run_worker() {
  srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 "$@" &
  WORKER=$!
  while true; do
    set +e
    wait "$WORKER"; RC=$?
    set -e
    if kill -0 "$WORKER" 2>/dev/null; then continue; fi
    break
  done
  return "$RC"
}

handle_stop() {
  local rc=$1
  if test "$rc" -ne 0 && test "$rc" -ne 75; then
    echo "Final training failed with exit $rc; resumable artifacts retained."
    exit "$rc"
  fi
  if test -f "$RUN_DIR/REQUEST_STOP"; then
    echo 'Final training stopped safely on request.'
    exit 0
  fi
  if test -f "$RUN_DIR/REQUEST_REQUEUE"; then
    if test "${SLURM_RESTART_COUNT:-0}" -ge 8; then
      echo 'Automatic requeue limit reached; state retained.'
      exit 78
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  if test "$rc" -ne 0; then exit "$rc"; fi
}

if ! test -f "$RUN_DIR/completed.json"; then
  echo 'PHASE: final full-data training'
  if run_worker bash "$RELEASE_DIR/code/container.sh" torchrun --standalone --nnodes=1 --nproc-per-node=4 \
      "$RELEASE_DIR/code/train.py" --production --config "$CONFIG" --output "$RUN_DIR"; then RC=0; else RC=$?; fi
  handle_stop "$RC"
  test -f "$RUN_DIR/completed.json" || { echo 'Training exited without full completion; benchmark blocked.'; exit 79; }
else
  echo 'Full-data training already completed; no further action.'
  exit 0
fi
handle_stop 0
echo 'FULL-DATA TRAINING COMPLETE. best.json identifies the selected checkpoint; benchmarking is deferred.'
