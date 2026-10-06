#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197
module load mmseqs2-gpu/18-8cc5c
sha256sum --check --status provenance/mmseqs-binary.sha256
mmseqs easy-search work/train.fasta work/val.fasta work/final-train-vs-val.tsv work/mmseqs/final-train-val \
  --threads 16 -s 7.5 --min-seq-id 0.4 -c 0.8 --cov-mode 0 --alignment-mode 3 --max-seqs 100000 -e 0.001 \
  --format-output query,target,fident,qcov,tcov,alnlen,qlen,tlen,bits
mmseqs easy-search work/val.fasta work/train.fasta work/final-val-vs-train.tsv work/mmseqs/final-val-train \
  --threads 16 -s 7.5 --min-seq-id 0.4 -c 0.8 --cov-mode 0 --alignment-mode 3 --max-seqs 100000 -e 0.001 \
  --format-output query,target,fident,qcov,tcov,alnlen,qlen,tlen,bits
bash scripts/python.sh - <<'PY'
import sys
sys.path.insert(0,'scripts')
from common import ROOT,write_json,mark
from prepare_groups import hits,qualifies
counts={p:sum(qualifies(h) for h in hits(ROOT/'work'/p)) for p in ['final-train-vs-val.tsv','final-val-vs-train.tsv']}
write_json(ROOT/'reports/final-development-homology.json',counts)
assert not any(counts.values()),counts
mark('boundary_verification','complete',qualifying_hits=counts)
PY
