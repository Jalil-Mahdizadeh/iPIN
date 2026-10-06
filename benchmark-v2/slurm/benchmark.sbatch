#!/usr/bin/env bash
#SBATCH --job-name=plmi-benchmark-v2
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/benchmark-v2/logs/slurm-%j.log
set -euo pipefail
WORK_ROOT=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN
cd "$WORK_ROOT"
sha256sum --check --quiet benchmark-v2/provenance/frozen-files.sha256
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
  bash benchmark-v2/scripts/container.sh torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  benchmark-v2/scripts/infer.py
