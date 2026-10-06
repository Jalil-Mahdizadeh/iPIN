#!/usr/bin/env bash
#SBATCH --job-name=plmi-v3-aligned-QUAL
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v3/logs/readout-aligned-qualification-%j.log
set -euo pipefail
R=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v3
cd "$R"
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
  bash "$R/scripts/container.sh" python -B "$R/scripts/qualify_readout_aligned_ddp.py"
