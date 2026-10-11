#!/usr/bin/env bash
#SBATCH --job-name=ipin-pilot-vx
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=05:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
vx_root=${1:?Pass the absolute pilot directory}
cd "$vx_root/.."
exec 9>"$vx_root/provenance/batch.lock"
flock -n 9 || { echo 'Another job owns this pilot.' >&2; exit 73; }
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=72 --cpu-bind=none \
    apptainer exec --nv "$vx_root/../images/msa-pairformer/msa-pairformer-arm64-v1.sif" \
    python -B "$vx_root/scripts/run.py"
