#!/usr/bin/env bash
set -euo pipefail
BASELINE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASELINE_PROJECT=$(cd -- "$BASELINE_ROOT/../.." && pwd)
BASELINE_IMAGE="$BASELINE_PROJECT/images/plm-interact/plm-interact-native-arm64-v1.sif"
exec env -u APPTAINER_BIND -u APPTAINER_BINDPATH -u SINGULARITY_BIND -u SINGULARITY_BINDPATH \
  apptainer exec --cleanenv --no-home --no-mount bind-paths,home,cwd,hostfs \
  --bind "$BASELINE_PROJECT:$BASELINE_PROJECT:ro" --bind "$BASELINE_ROOT:$BASELINE_ROOT:rw" \
  --pwd "$BASELINE_ROOT" \
  --env "PYTHONHASHSEED=0,PYTHONDONTWRITEBYTECODE=1,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=16,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MPLCONFIGDIR=$BASELINE_ROOT/cache/matplotlib" \
  "$BASELINE_IMAGE" "$@"
