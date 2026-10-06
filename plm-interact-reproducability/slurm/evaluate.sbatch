#!/usr/bin/env bash
#SBATCH --job-name=plmi-reproduce
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=8
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%j.log
set -euo pipefail
cd /nobackup/proj/disk/theo-storage/personal/jalil/iPIN/plm-interact-reproducability
srun --ntasks=8 --ntasks-per-node=4 --gpus=8 --gpus-per-node=4 --gpus-per-task=1 \
  --cpus-per-task=18 --gpu-bind=single:1 --cpu-bind=none --kill-on-bad-exit=1 \
  --output=logs/evaluation-%j-rank%t.log bash slurm/worker.sh
