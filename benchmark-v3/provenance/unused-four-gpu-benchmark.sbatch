#!/usr/bin/env bash
#SBATCH --job-name=plmi-benchmark-v3
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/benchmark-v3/logs/slurm-%j.log
set -euo pipefail
WORK_ROOT=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN
cd "$WORK_ROOT"
exec 9>benchmark-v3/logs/pipeline.lock
flock -n 9 || { echo 'Another benchmark process owns this run.'; exit 73; }
sha256sum --check --quiet benchmark-v3/provenance/frozen-files.sha256
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
  bash benchmark-v3/scripts/container.sh python -B benchmark-v3/scripts/pipeline.py
sha256sum --check --quiet benchmark-v3/provenance/frozen-files.sha256
