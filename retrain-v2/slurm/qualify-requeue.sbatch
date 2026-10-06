#!/usr/bin/env bash
#SBATCH --job-name=plmi-v2-REQUEUE-CHECK
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=00:20:00
#SBATCH --array=0-0
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v2/logs/requeue-qualification-%A_%a.log
set -euo pipefail
R=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v2
OUT="$R/qualification/ddp-experimental-run-requeued"
mkdir -p "$OUT"
trap 'touch "$OUT/REQUEST_REQUEUE"' USR1 TERM
launch() {
  srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
    bash "$R/scripts/container.sh" torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    "$R/scripts/train.py" --qualification --config "$R/qualification/ddp-experimental.json" \
    --output "$OUT" --max-updates 3 --train-limit 11 --val-limit 8 --qualification-max-tokens 384 "$@"
}
if test "${SLURM_RESTART_COUNT:-0}" = 0; then
  test ! -f "$OUT/contract.json"
  launch --sleep-after-update 3 &
  WORKER=$!
  python - "$OUT/status.json" "$$" <<'PY' &
import json,os,signal,sys,time
from pathlib import Path
path=Path(sys.argv[1]);deadline=time.monotonic()+300
while time.monotonic()<deadline:
    if path.exists() and json.loads(path.read_text())['state']['update']>=1:
        os.kill(int(sys.argv[2]),signal.SIGUSR1);break
    time.sleep(.05)
else:raise RuntimeError('No optimizer update before bounded signal deadline')
PY
  WATCHER=$!
  set +e
  while true; do
    wait "$WORKER"; RC=$?
    if kill -0 "$WORKER" 2>/dev/null; then continue; fi
    break
  done
  set -e
  wait "$WATCHER"
  test "$RC" = 0
  test -f "$OUT/stopped.json"
  test -f "$OUT/REQUEST_REQUEUE"
  python - "$OUT/stopped.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert 0<s['state']['update']<s['state']['total_steps'] and s['checkpoint_committed']
PY
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
test "$SLURM_RESTART_COUNT" = 1
rm -f "$OUT/REQUEST_REQUEUE"
launch
bash "$R/scripts/container.sh" python "$R/scripts/compare_resume.py" \
  "$R/qualification/ddp-experimental-run-full" "$OUT" \
  --report "$R/qualification/ddp-experimental-requeue-comparison.json"
python - "$R" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
r=Path(sys.argv[1])
report={'passed':True,'production_training':False,'real_slurm_array_task_requeue':True,
        'array_job_id':os.environ['SLURM_ARRAY_JOB_ID'],'array_task_id':os.environ['SLURM_ARRAY_TASK_ID'],
        'job_id':os.environ['SLURM_JOB_ID'],'restart_count':int(os.environ['SLURM_RESTART_COUNT']),
        'full_650m_model':'chain-aware + residue readout + clean classification + masked MLM pass',
        'optimizer_continuation_after_requeue':True,'comparison':'qualification/ddp-experimental-requeue-comparison.json',
        'code':{n:hashlib.sha256((r/'scripts'/n).read_bytes()).hexdigest()
                for n in ['train.py','model.py','data.py','state.py','release.py']}}
(r/'qualification/scheduler-requeue.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
