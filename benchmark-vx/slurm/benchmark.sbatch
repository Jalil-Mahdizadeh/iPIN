#!/usr/bin/env bash
#SBATCH --job-name=ipin-benchmark-vx-128
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
vx_benchmark_root=${1:?Pass the absolute benchmark-vx directory}
cd "$vx_benchmark_root/.."
exec 9>"$vx_benchmark_root/provenance/batch.lock"
flock -n 9 || { echo 'Another process owns this benchmark.' >&2; exit 73; }
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
exec python3 -B "$vx_benchmark_root/scripts/run.py"
