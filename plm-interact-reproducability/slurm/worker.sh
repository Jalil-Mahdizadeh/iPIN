#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/plm-interact-reproducability
exec bash "$TASK_ROOT/scripts/container.sh" python -u "$TASK_ROOT/scripts/infer.py" \
  --rank "$SLURM_PROCID" --world-size 8 --job-id "$SLURM_JOB_ID" \
  --tasks cross_mouse cross_fly cross_worm cross_yeast cross_ecoli \
  reverse_mouse_sample reverse_fly_sample reverse_worm_sample reverse_yeast_sample reverse_ecoli_sample \
  bernett_full bernett_1603 bernett_2196
