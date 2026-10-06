#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(dirname "$TASK_ROOT")
mkdir -p "$TASK_ROOT/cache/huggingface"
exec apptainer exec --nv --cleanenv \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},HF_HOME=$TASK_ROOT/cache/huggingface,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8,TOKENIZERS_PARALLELISM=false,CUBLAS_WORKSPACE_CONFIG=:4096:8" \
  "$PROJECT_ROOT/images/plm-interact/plm-interact-native-arm64-v1.sif" "$@"
