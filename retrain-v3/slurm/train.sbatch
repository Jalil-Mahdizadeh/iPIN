#!/usr/bin/env bash
#SBATCH --job-name=plmi-v3-adaptation
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
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v3/logs/production-%A_%a.log
set -euo pipefail
R=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v3
RELEASE_DIR=${1:?Pass a frozen release directory}
cd "$R"
# Array index 0..5 selects an immutable adaptation configuration. An optional
# explicit name supports resubmitting a single existing run with the same release.
RUN_NAME=$(python - "$RELEASE_DIR/release.json" "${2:-}" "${SLURM_ARRAY_TASK_ID:-}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));allowed=r['enabled_runs']
name=sys.argv[2] or allowed[int(sys.argv[3])]
assert name in allowed
print(name)
PY
)
RUN_DIR="$R/runs/$RUN_NAME"
CONFIG="$RELEASE_DIR/configs/$RUN_NAME.json"
test -f "$CONFIG"
if test -f "$RUN_DIR/completed.json"; then
  echo "Run $RUN_NAME has completed."
  exit 0
fi
mkdir -p "$RUN_DIR"
on_requeue() { touch "$RUN_DIR/REQUEST_REQUEUE"; }
trap on_requeue USR1 TERM
if test -f "$RUN_DIR/REQUEST_STOP"; then
  echo "Run $RUN_NAME is manually stopped; remove REQUEST_STOP only when continuation is intended."
  exit 0
fi
rm -f "$RUN_DIR/REQUEST_REQUEUE"
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
  bash "$RELEASE_DIR/code/container.sh" torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  "$RELEASE_DIR/code/train.py" --production --config "$CONFIG" --output "$RUN_DIR" &
WORKER=$!
set +e
while true; do
  wait "$WORKER"; RC=$?
  if kill -0 "$WORKER" 2>/dev/null; then continue; fi
  break
done
set -e
if test "$RC" -ne 0; then
  echo "Run $RUN_NAME failed (exit $RC); complete checkpoints retained."
  exit "$RC"
fi
if test -f "$RUN_DIR/completed.json"; then
  echo "Run $RUN_NAME completed; best.json identifies the validation-selected checkpoint."
  exit 0
fi
if test -f "$RUN_DIR/REQUEST_REQUEUE" && test -f "$RUN_DIR/stopped.json" && ! test -f "$RUN_DIR/REQUEST_STOP"; then
  if test "${SLURM_RESTART_COUNT:-0}" -ge 8; then
    echo "Automatic requeue limit reached; state retained for manual continuation."
    exit 78
  fi
  # Requeue this array task, never all other arms of the array.
  TASK_JOB_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
  if test -n "${SLURM_ARRAY_TASK_ID:-}"; then TASK_JOB_ID="${TASK_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
  scontrol requeue "$TASK_JOB_ID"
  exit 0
fi
if test -f "$RUN_DIR/stopped.json"; then
  echo "Run $RUN_NAME stopped with a committed checkpoint."
  exit 0
fi
echo "Worker exited without a completion or stopped marker."
exit 79
