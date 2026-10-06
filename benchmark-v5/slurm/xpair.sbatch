#!/usr/bin/env bash
#SBATCH --job-name=bv5-xpair-both
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/benchmark-v5/logs/slurm-%x-%j.log
set -euo pipefail
cd /nobackup/proj/disk/theo-storage/personal/jalil/iPIN
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
 bash benchmark-v5/scripts/container.sh xpair /work/runtime/venv/bin/python scripts/launch_workers.py \
 scripts/xpair_benchmark.py --stage embedding
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
 bash benchmark-v5/scripts/container.sh xpair /work/runtime/venv/bin/python scripts/xpair_benchmark.py --stage qualify
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
 bash benchmark-v5/scripts/container.sh xpair /work/runtime/venv/bin/python scripts/launch_workers.py \
 scripts/xpair_benchmark.py --stage score
