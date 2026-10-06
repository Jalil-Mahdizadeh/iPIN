#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
project=$(cd -- "$root/.." && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec apptainer exec --cleanenv \
  --bind "$project:$project" \
  --env OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1 \
  "$root/../images/plm-interact-esmc/plm-interact-esmc-arm64-v1.sif" \
  "$root/environment/venv/bin/python" "$@"
