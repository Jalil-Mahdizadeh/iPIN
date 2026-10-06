#!/usr/bin/env bash
set -euo pipefail
R=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v2
cd "$R"
module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197
module load mmseqs2-gpu/18-8cc5c
mmseqs easy-search data/prepared/development-proteins.fasta data/prepared/development-proteins.fasta \
  data/development-homology.tsv cache/mmseqs-development --threads 16 -s 7.5 \
  --min-seq-id 0.4 -c 0.8 --cov-mode 0 --max-seqs 10000 -e 0.001 \
  --format-output query,target,fident,qcov,tcov,alnlen
sha256sum "$(command -v mmseqs)" > provenance/mmseqs-binary.sha256
