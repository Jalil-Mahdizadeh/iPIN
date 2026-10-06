#!/usr/bin/env bash
#SBATCH --job-name=plmi-v4-QUAL
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=00:20:00
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v4/logs/qualification-%j.log
set -euo pipefail
V4_CODE=${1:?Pass the frozen qualification code directory}
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
    python3 -B "$V4_CODE/qualify_ddp.py"
