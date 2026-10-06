#!/usr/bin/env bash
#SBATCH --job-name=nhv5-dscript
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/bechmark-v5-nonhuman/logs/slurm-%x-%j.log
set -euo pipefail
cd /nobackup/proj/disk/theo-storage/personal/jalil/iPIN
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
 bash bechmark-v5-nonhuman/scripts/container_dscript.sh python scripts/launch_workers.py scripts/dscript_benchmark.py --stage prepare
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
 bash bechmark-v5-nonhuman/scripts/container_dscript.sh python scripts/dscript_benchmark.py --stage pack
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
 bash bechmark-v5-nonhuman/scripts/container_dscript.sh python scripts/launch_workers.py scripts/dscript_benchmark.py --stage score
