#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BENCH_PROJECT=$(dirname "$BENCH_ROOT")
BENCH_EXTERNAL=$(dirname "$BENCH_PROJECT")/iPIN-OpenPPI
BENCH_MODEL=${1:?runtime name}; shift
BENCH_EXTRA=()
BENCH_PYTHONPATH="$BENCH_ROOT/scripts"
case "$BENCH_MODEL" in
  esm2|native|analysis) BENCH_IMAGE="$BENCH_PROJECT/images/plm-interact/plm-interact-native-arm64-v1.sif" ;;
  esmc) BENCH_IMAGE="$BENCH_PROJECT/images/plm-interact-esmc/plm-interact-esmc-arm64-v1.sif" ;;
  tuna) BENCH_IMAGE="$BENCH_EXTERNAL/benchmark/containers/images/tuna-arm64-v1.sif" ;;
  rapppid)
    BENCH_IMAGE="$BENCH_EXTERNAL/benchmark/containers/images/rapppid-native-arm64-v1.sif"
    BENCH_PYTHONPATH="$BENCH_PYTHONPATH:/opt/rapppid/upstream" ;;
  sprint) BENCH_IMAGE="$BENCH_EXTERNAL/benchmark/containers/images/sprint-native-arm64-v1.sif" ;;
  xpair)
    BENCH_IMAGE="$BENCH_EXTERNAL/benchmark/containers/images/plm-interact-native-arm64-v1.sif"
    BENCH_EXTRA+=(--bind "$BENCH_EXTERNAL/experiments/x_pair_test2_v1:/work:ro")
    BENCH_PYTHONPATH="$BENCH_PYTHONPATH:/work/sources/X-PAIR" ;;
  *) exit 64 ;;
esac
mkdir -p "$BENCH_ROOT/cache/$BENCH_MODEL"
exec env -u APPTAINER_BIND -u APPTAINER_BINDPATH -u SINGULARITY_BIND -u SINGULARITY_BINDPATH \
  APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  apptainer exec --nv --cleanenv --no-home --no-mount bind-paths,home,cwd,hostfs \
  --bind "$BENCH_PROJECT:$BENCH_PROJECT:ro" --bind "$BENCH_ROOT:$BENCH_ROOT:rw" \
  --bind "$BENCH_EXTERNAL:$BENCH_EXTERNAL:ro" "${BENCH_EXTRA[@]}" --pwd "$BENCH_ROOT" \
  --env "PYTHONPATH=$BENCH_PYTHONPATH,PYTHONDONTWRITEBYTECODE=1,PYTHONUNBUFFERED=1,SLURM_JOB_ID=${SLURM_JOB_ID:-local},HF_HOME=$BENCH_ROOT/cache/$BENCH_MODEL,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8,TOKENIZERS_PARALLELISM=false,CUBLAS_WORKSPACE_CONFIG=:4096:8,NVIDIA_TF32_OVERRIDE=0" \
  "$BENCH_IMAGE" "$@"
