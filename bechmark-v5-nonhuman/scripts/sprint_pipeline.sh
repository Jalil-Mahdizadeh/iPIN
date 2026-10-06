#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$BENCH_ROOT"
exec 9>predictions/sprint-pipeline.lock
flock -n 9
if [[ ! -e provenance/sprint-input.json ]]; then
  bash scripts/container.sh analysis python scripts/sprint_benchmark.py --stage prepare
fi
if [[ ! -e predictions/sprint/hsp-command-succeeded ]]; then
  date -Is
  bash scripts/container.sh sprint env OMP_NUM_THREADS=64 /opt/sprint/bin/compute_HSPs \
    -p "$BENCH_ROOT/data/sprint/proteins.fasta" -h "$BENCH_ROOT/predictions/sprint/raw.hsp" \
    -Thit 15 -Tsim 35 -M 1 > logs/sprint-hsp.log 2>&1
  date -Is > predictions/sprint/hsp-command-succeeded
fi
bash scripts/container.sh analysis python scripts/sprint_benchmark.py --stage canonical
if [[ ! -e predictions/sprint/prediction-command-succeeded ]]; then
  BENCH_SPRINT_BINARY=/opt/sprint/bin/predict_interactions
  if [[ -f qualification/sprint-requested/qualification.json ]]; then
    BENCH_SPRINT_BINARY="$BENCH_ROOT/bin/sprint_predict_requested"
  fi
  # The native scorer appends to its output. Preserve interrupted output and
  # restart this stage into a fresh file; never concatenate two attempts.
  BENCH_ATTEMPT=$(date -u +%Y%m%dT%H%M%S)
  for BENCH_SUFFIX in "" .pos .neg; do
    if [[ -e "predictions/sprint/scores.txt$BENCH_SUFFIX" ]]; then
      mv "predictions/sprint/scores.txt$BENCH_SUFFIX" "logs/sprint-partial-$BENCH_ATTEMPT.txt$BENCH_SUFFIX"
    fi
  done
  date -Is
  bash scripts/container.sh sprint env OMP_NUM_THREADS=1 "$BENCH_SPRINT_BINARY" \
    -p "$BENCH_ROOT/data/sprint/proteins.fasta" -h "$BENCH_ROOT/predictions/sprint/canonical.hsp" \
    -tr "$BENCH_ROOT/data/sprint/train-positive.txt" -pos "$BENCH_ROOT/data/sprint/pairs.txt" \
    -neg "$BENCH_ROOT/data/sprint/empty.txt" -o "$BENCH_ROOT/predictions/sprint/scores.txt" \
    -Thc 40 > logs/sprint-predict.log 2>&1
  date -Is > predictions/sprint/prediction-command-succeeded
fi
bash scripts/container.sh analysis python scripts/sprint_benchmark.py --stage collect
date -Is
