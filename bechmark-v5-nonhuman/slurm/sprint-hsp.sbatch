#!/usr/bin/env bash
#SBATCH --job-name=nhv5-sprint-hsp
#SBATCH --account=naiss2025-3-10-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/bechmark-v5-nonhuman/logs/slurm-%x-%j.log
set -euo pipefail
cd /nobackup/proj/disk/theo-storage/personal/jalil/iPIN/bechmark-v5-nonhuman
exec 9>predictions/sprint-hsp.lock
flock -n 9
test -f provenance/sprint-input.json
g++ --version > provenance/sprint-cpu-compiler.txt
mkdir -p scripts/vendor/include
tar -xf scripts/vendor/boost-headers.tar -C scripts/vendor/include
g++ -O3 -fopenmp -Wall -Iscripts/vendor/include scripts/sprint_upstream/Src/compute_HSPs/compute_hsps.cpp -o bin/sprint_compute_hsps_cpu -DPARAL
g++ -O3 -Wall scripts/sprint_upstream/Src/predict_interactions/predict_interactions.cpp -o bin/sprint_predict_native_cpu
g++ -O3 -Wall scripts/sprint_requested/predict_interactions.cpp -o bin/sprint_predict_endpoints_cpu
sha256sum bin/sprint_*cpu > provenance/sprint-cpu-binaries.sha256
if [[ ! -f predictions/sprint/hsp-command-succeeded ]]; then
 OMP_NUM_THREADS=256 bin/sprint_compute_hsps_cpu -p data/sprint/proteins.fasta -h predictions/sprint/raw.hsp -Thit 15 -Tsim 35 -M 1 > logs/sprint-hsp.log 2>&1
 date -u -Is > predictions/sprint/hsp-command-succeeded
fi
