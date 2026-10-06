#!/usr/bin/env bash
set -euo pipefail
V4_ROOT=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v4
V4_WORKSPACE=$(dirname "$V4_ROOT")
V4_BACKBONE=${1:?Specify esm2 or esmc}
shift
case "$V4_BACKBONE" in
  esm2)
    V4_IMAGE="$V4_WORKSPACE/images/plm-interact/plm-interact-native-arm64-v1.sif"
    V4_IMAGE_HASH=e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2 ;;
  esmc)
    V4_IMAGE="$V4_WORKSPACE/images/plm-interact-esmc/plm-interact-esmc-arm64-v1.sif"
    V4_IMAGE_HASH=ed6aeac781502090632bc2a705300b161b6a65ccfbcf6c7cf64d8d61d8e5b5ef ;;
  *) echo 'Unsupported v4 backbone' >&2; exit 64 ;;
esac
test "$(sha256sum "$V4_IMAGE" | cut -d ' ' -f1)" = "$V4_IMAGE_HASH"
mkdir -p "$V4_ROOT/cache/$V4_BACKBONE"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec apptainer exec --nv --cleanenv --bind "$V4_WORKSPACE:$V4_WORKSPACE" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID:-local},SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0},PLMI_V4_IMAGE_SHA256=$V4_IMAGE_HASH,HF_HOME=$V4_ROOT/cache/$V4_BACKBONE,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8,TOKENIZERS_PARALLELISM=false,CUBLAS_WORKSPACE_CONFIG=:4096:8" \
  "$V4_IMAGE" "$@"
