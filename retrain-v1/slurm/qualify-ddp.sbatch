#!/usr/bin/env bash
#SBATCH --job-name=plmi-resume-check
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v1/logs/qualification-%j.log
set -euo pipefail
R=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v1
cd "$R"
PREFIX=${1:-ddp-v1}
A="$R/qualification/$PREFIX-uninterrupted"
B="$R/qualification/$PREFIX-resumed"
mkdir -p "$A" "$B"
trap 'touch "$B/REQUEST_REQUEUE"' USR1
launch() {
  srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
    bash "$R/scripts/container.sh" torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    "$R/scripts/train.py" --config "$R/configs/qualification.json" --output "$1" \
    --max-updates 5 --train-limit 35 --val-limit 8 "${@:2}"
}
if ! test -f "$A/completed.json"; then launch "$A"; fi
if test "${SLURM_RESTART_COUNT:-0}" = 0; then
  launch "$B" --sleep-after-update 3 &
  WORKER=$!
  python - "$B/status.json" "$$" <<'PY' &
import json,os,signal,sys,time
from pathlib import Path
p=Path(sys.argv[1]);deadline=time.monotonic()+600
while time.monotonic()<deadline:
    if p.exists() and json.loads(p.read_text())['state']['update']>=1:
        os.kill(int(sys.argv[2]),signal.SIGUSR1);break
    time.sleep(.5)
else:raise RuntimeError('No first optimizer update to interrupt')
PY
  WATCHER=$!
  set +e
  while true; do
    wait "$WORKER";RC=$?
    if kill -0 "$WORKER" 2>/dev/null; then continue; fi
    break
  done
  set -e
  wait "$WATCHER"
  test "$RC" = 0
  test -f "$B/stopped.json"
  test -f "$B/REQUEST_REQUEUE"
  echo "Qualification interrupted through batch USR1; requesting a real Slurm requeue."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
rm -f "$B/REQUEST_REQUEUE"
launch "$B"
bash "$R/scripts/container.sh" python "$R/scripts/compare_resume.py" "$A" "$B" \
  --report "$R/qualification/$PREFIX-resume-comparison.json"
echo "Full 650M four-GPU optimizer/RNG/sampler continuation is bitwise identical."
