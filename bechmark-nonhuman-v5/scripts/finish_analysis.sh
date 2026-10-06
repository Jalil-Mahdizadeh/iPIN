#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$BENCH_ROOT"
exec 9>results/analysis.lock
flock -n 9
bash scripts/container.sh analysis python scripts/sprint_benchmark.py --stage collect
bash scripts/container.sh analysis python scripts/collect.py
bash scripts/container.sh analysis python scripts/analyze.py
bash scripts/container.sh analysis python scripts/report_nonhuman.py
bash scripts/container.sh analysis python scripts/verify_complete.py
