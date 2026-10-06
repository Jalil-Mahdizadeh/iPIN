#!/usr/bin/env bash
#SBATCH --job-name=plmi-v5
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
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retraining-v5/logs/production-%j.log
set -euo pipefail
V5_ROOT=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retraining-v5/qualification/wrapper-fixture/stop-during-worker
V5_RELEASE=${1:?Pass the frozen release directory}
V5_RUN=${2:?Pass exactly one enabled run name}
V5_RUN_DIR="$V5_ROOT/runs/$V5_RUN"
case "$V5_RUN" in
  esm2-native-ilp-seed2) V5_BACKBONE=esm2 ;;
  esmc-native-ilp-seed2) V5_BACKBONE=esmc ;;
  *) echo 'Unsupported production run' >&2; exit 64 ;;
esac
mkdir -p "$V5_RUN_DIR"
exec 9>"$V5_RUN_DIR/.pipeline.lock"
flock -n 9 || { echo 'Another job owns this run.'; exit 73; }
cd "$V5_ROOT"
on_requeue() { touch "$V5_RUN_DIR/REQUEST_REQUEUE"; }
trap on_requeue USR1 TERM
if test -f "$V5_RUN_DIR/REQUEST_STOP"; then
  echo 'Manually stopped; retained training state is unchanged.'
  exit 0
fi
rm -f "$V5_RUN_DIR/REQUEST_REQUEUE"
python3 -B "$V5_RELEASE/code/runtime.py" "$V5_RELEASE" "$V5_RUN"

run_worker() {
  srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 "$@" &
  V5_WORKER=$!
  while true; do
    set +e
    wait "$V5_WORKER"; V5_RC=$?
    set -e
    if kill -0 "$V5_WORKER" 2>/dev/null; then continue; fi
    break
  done
  return "$V5_RC"
}

handle_stop() {
  local rc=$1
  if test "$rc" -ne 0 && test "$rc" -ne 75; then
    echo "Training failed with exit $rc; checkpoint artifacts retained."
    exit "$rc"
  fi
  if test -f "$V5_RUN_DIR/REQUEST_STOP"; then
    echo 'Training stopped safely on request.'
    exit 0
  fi
  if test -f "$V5_RUN_DIR/REQUEST_REQUEUE"; then
    if test "${SLURM_RESTART_COUNT:-0}" -ge 8; then
      echo 'Automatic requeue limit reached; training state retained.'
      exit 78
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  if test "$rc" -ne 0; then exit "$rc"; fi
}

if test -f "$V5_RUN_DIR/completed.json"; then
  echo 'Training already complete; no further action.'
  exit 0
fi
if run_worker bash "$V5_RELEASE/code/container.sh" "$V5_BACKBONE" \
    python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 "$V5_RELEASE/code/train.py" \
    --production --config "$V5_RELEASE/configs/$V5_RUN.json" --output "$V5_RUN_DIR"; then
  V5_RC=0
else
  V5_RC=$?
fi
handle_stop "$V5_RC"
test -f "$V5_RUN_DIR/completed.json" || { echo 'Worker exited without full completion.'; exit 79; }
echo 'TRAINING COMPLETE. best.json identifies the selected checkpoint. This job ends here.'
