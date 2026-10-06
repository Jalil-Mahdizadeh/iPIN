#!/usr/bin/env bash
#SBATCH --job-name=benchmark-v5-nonhuman-pair
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/benchmark-v5-nonhuman/logs/slurm-%x-%j.log
set -euo pipefail
cd /nobackup/proj/disk/theo-storage/personal/jalil/iPIN
BENCH_NAME=${1:?model}; BENCH_RUNTIME=${2:?runtime}
[[ -f benchmark-v5-nonhuman/provenance/runtime-inputs.json ]]
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
 bash benchmark-v5-nonhuman/scripts/container.sh "$BENCH_RUNTIME" python scripts/pair_infer.py --model "$BENCH_NAME" --qualify
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none --kill-on-bad-exit=1 \
 bash benchmark-v5-nonhuman/scripts/container.sh "$BENCH_RUNTIME" python scripts/launch_workers.py \
 scripts/pair_infer.py --model "$BENCH_NAME"
