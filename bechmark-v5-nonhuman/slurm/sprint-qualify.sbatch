#!/usr/bin/env bash
#SBATCH --job-name=nhv5-sprint-qualify
#SBATCH --account=naiss2025-3-10-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/bechmark-v5-nonhuman/logs/slurm-%x-%j.log
set -euo pipefail
cd /nobackup/proj/disk/theo-storage/personal/jalil/iPIN/bechmark-v5-nonhuman
g++ -O3 -Wall scripts/sprint_sparse/predict_interactions.cpp -o bin/sprint_predict_sparse_cpu
python3 scripts/qualify_sprint_sparse.py
