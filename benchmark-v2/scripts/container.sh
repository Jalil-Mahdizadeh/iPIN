#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/benchmark-v2
WORK_ROOT=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN
mkdir -p "$BENCH_ROOT/cache"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec apptainer exec --nv --cleanenv \
  --bind "$WORK_ROOT:$WORK_ROOT" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID:-local},SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0},HF_HOME=$BENCH_ROOT/cache,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8,TOKENIZERS_PARALLELISM=false,CUBLAS_WORKSPACE_CONFIG=:4096:8" \
  "$WORK_ROOT/images/plm-interact/plm-interact-native-arm64-v1.sif" "$@"
