#!/usr/bin/env bash
#SBATCH --job-name=nhv5-sprint-score
#SBATCH --account=naiss2025-3-10-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/bechmark-nonhuman-v5/logs/slurm-%x-%j.log
set -euo pipefail
cd /nobackup/proj/disk/theo-storage/personal/jalil/iPIN/bechmark-nonhuman-v5
exec 9>predictions/sprint-score.lock
flock -n 9
OMP_NUM_THREADS=1 python3 scripts/run_sprint_cpu.py
