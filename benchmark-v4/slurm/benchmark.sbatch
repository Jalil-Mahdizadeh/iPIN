#!/usr/bin/env bash
#SBATCH --job-name=plmi-benchmark-v4
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=00:20:00
#SBATCH --open-mode=append
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/benchmark-v4/logs/slurm-%j.log
set -euo pipefail
cd /nobackup/proj/disk/theo-storage/personal/jalil/iPIN
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
  bash benchmark-v4/scripts/run.sh
