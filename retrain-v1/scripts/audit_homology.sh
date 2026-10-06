#!/usr/bin/env bash
set -euo pipefail
cd /nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v1
module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197
module load mmseqs2-gpu/18-8cc5c
mmseqs easy-search data/prepared/heldout.fasta data/prepared/train.fasta data/heldout-vs-train.tsv cache/mmseqs-heldout \
 --threads 16 -s 7.5 --min-seq-id 0.4 -c 0.8 --cov-mode 0 --max-seqs 10000 -e 0.001 \
 --format-output query,target,fident,qcov,tcov,alnlen
mmseqs easy-search data/prepared/val.fasta data/prepared/test.fasta data/val-vs-test.tsv cache/mmseqs-val \
 --threads 16 -s 7.5 --min-seq-id 0.4 -c 0.8 --cov-mode 0 --max-seqs 10000 -e 0.001 \
 --format-output query,target,fident,qcov,tcov,alnlen
sha256sum "$(command -v mmseqs)" > provenance/mmseqs-binary.sha256
