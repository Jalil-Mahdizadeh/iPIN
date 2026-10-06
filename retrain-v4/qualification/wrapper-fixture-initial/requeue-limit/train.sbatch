#!/usr/bin/env bash
#SBATCH --job-name=plmi-v4
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
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v4/logs/production-%j.log
set -euo pipefail
V4_ROOT=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v4/qualification/wrapper-fixture/requeue-limit
V4_RELEASE=${1:?Pass the frozen release directory}
V4_RUN=${2:?Pass exactly one enabled run name}
V4_RUN_DIR="$V4_ROOT/runs/$V4_RUN"
case "$V4_RUN" in
  esm2-chain-aware-official-seed2) V4_BACKBONE=esm2 ;;
  esmc-standard-official-seed2|esmc-chain-aware-official-seed2) V4_BACKBONE=esmc ;;
  *) echo 'Unsupported production run' >&2; exit 64 ;;
esac
mkdir -p "$V4_RUN_DIR"
exec 9>"$V4_RUN_DIR/.pipeline.lock"
flock -n 9 || { echo 'Another job owns this run.'; exit 73; }
cd "$V4_ROOT"
on_requeue() { touch "$V4_RUN_DIR/REQUEST_REQUEUE"; }
trap on_requeue USR1 TERM
if test -f "$V4_RUN_DIR/REQUEST_STOP"; then
  echo 'Manually stopped; retained training state is unchanged.'
  exit 0
fi
rm -f "$V4_RUN_DIR/REQUEST_REQUEUE"
python3 -B "$V4_RELEASE/code/runtime.py" "$V4_RELEASE" "$V4_RUN"

run_worker() {
  srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 "$@" &
  V4_WORKER=$!
  while true; do
    set +e
    wait "$V4_WORKER"; V4_RC=$?
    set -e
    if kill -0 "$V4_WORKER" 2>/dev/null; then continue; fi
    break
  done
  return "$V4_RC"
}

handle_stop() {
  local rc=$1
  if test "$rc" -ne 0 && test "$rc" -ne 75; then
    echo "Training failed with exit $rc; checkpoint artifacts retained."
    exit "$rc"
  fi
  if test -f "$V4_RUN_DIR/REQUEST_STOP"; then
    echo 'Training stopped safely on request.'
    exit 0
  fi
  if test -f "$V4_RUN_DIR/REQUEST_REQUEUE"; then
    if test "${SLURM_RESTART_COUNT:-0}" -ge 8; then
      echo 'Automatic requeue limit reached; training state retained.'
      exit 78
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  if test "$rc" -ne 0; then exit "$rc"; fi
}

if test -f "$V4_RUN_DIR/completed.json"; then
  echo 'Training already complete; no further action.'
  exit 0
fi
if run_worker bash "$V4_RELEASE/code/container.sh" "$V4_BACKBONE" \
    torchrun --standalone --nnodes=1 --nproc-per-node=4 "$V4_RELEASE/code/train.py" \
    --production --config "$V4_RELEASE/configs/$V4_RUN.json" --output "$V4_RUN_DIR"; then
  V4_RC=0
else
  V4_RC=$?
fi
handle_stop "$V4_RC"
test -f "$V4_RUN_DIR/completed.json" || { echo 'Worker exited without full completion.'; exit 79; }
echo 'TRAINING COMPLETE. best.json identifies the selected checkpoint. This job ends here.'
