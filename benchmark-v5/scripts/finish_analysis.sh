#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$BENCH_ROOT"
exec 9>logs/analysis-pipeline.lock
flock -n 9
while [[ ! -f predictions/sprint/done.json || ! -f predictions/dscript/rank-03.done.json || ! -f predictions/dscript/rank-02.done.json || ! -f predictions/dscript/rank-01.done.json || ! -f predictions/dscript/rank-00.done.json ]]; do
  sleep 20
done
bash scripts/container.sh analysis python scripts/collect.py
bash scripts/container.sh analysis python scripts/analyze.py
