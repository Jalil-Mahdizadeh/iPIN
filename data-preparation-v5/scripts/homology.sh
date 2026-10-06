#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197
module load mmseqs2-gpu/18-8cc5c
mkdir -p work/mmseqs provenance
sha256sum "$(command -v mmseqs)" > provenance/mmseqs-current.sha256
cmp provenance/mmseqs-binary.sha256 provenance/mmseqs-current.sha256
mmseqs version > provenance/mmseqs-version.txt
mmseqs easy-search work/eligible-before-homology.fasta frozen-tests/original-sequences.fasta \
  work/eligible-vs-test.tsv work/mmseqs/eligible-test --threads 16 -s 7.5 \
  --min-seq-id 0.4 -c 0.8 --cov-mode 0 --alignment-mode 3 --max-seqs 100000 -e 0.001 \
  --format-output query,target,fident,qcov,tcov,alnlen,qlen,tlen,bits
mmseqs easy-search frozen-tests/original-sequences.fasta work/eligible-before-homology.fasta \
  work/test-vs-eligible.tsv work/mmseqs/test-eligible --threads 16 -s 7.5 \
  --min-seq-id 0.4 -c 0.8 --cov-mode 0 --alignment-mode 3 --max-seqs 100000 -e 0.001 \
  --format-output query,target,fident,qcov,tcov,alnlen,qlen,tlen,bits
bash scripts/python.sh scripts/prepare_groups.py filter
mmseqs easy-search work/eligible.fasta work/eligible.fasta \
  work/development-all-hits.tsv work/mmseqs/development-all --threads 16 -s 7.5 \
  --min-seq-id 0 -c 0 --alignment-mode 3 --max-seqs 100000 -e 0.001 \
  --format-output query,target,fident,qcov,tcov,alnlen,qlen,tlen,bits
bash scripts/python.sh scripts/prepare_groups.py group
