#!/usr/bin/env bash
set -euo pipefail
DEGREE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DEGREE_PROJECT=$(cd -- "$DEGREE_ROOT/../.." && pwd)
DEGREE_IMAGE="$DEGREE_PROJECT/images/plm-interact/plm-interact-native-arm64-v1.sif"
exec env -u APPTAINER_BIND -u APPTAINER_BINDPATH -u SINGULARITY_BIND -u SINGULARITY_BINDPATH \
  apptainer exec --cleanenv --no-home --no-mount bind-paths,home,cwd,hostfs \
  --bind "$DEGREE_PROJECT:$DEGREE_PROJECT:ro" --bind "$DEGREE_ROOT:$DEGREE_ROOT:rw" \
  --pwd "$DEGREE_ROOT" --env "PYTHONHASHSEED=0,PYTHONDONTWRITEBYTECODE=1,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=2,MKL_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2" \
  "$DEGREE_IMAGE" python "$DEGREE_ROOT/scripts/assess.py"
