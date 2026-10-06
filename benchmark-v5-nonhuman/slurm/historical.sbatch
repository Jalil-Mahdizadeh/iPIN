#!/usr/bin/env bash
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/benchmark-v5-nonhuman/v1-v4-comparison/logs/slurm-%x-%j.log
set -euo pipefail
cd /nobackup/proj/disk/theo-storage/personal/jalil/iPIN
HISTORICAL_MODEL=${1:?model name}
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
 bash benchmark-v5-nonhuman/scripts/container.sh native python scripts/launch_workers.py scripts/historical_infer.py --model "$HISTORICAL_MODEL"
