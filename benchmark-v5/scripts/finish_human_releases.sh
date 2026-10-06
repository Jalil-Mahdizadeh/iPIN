#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$BENCH_ROOT"
exec 9>logs/human-analysis.lock
flock -n 9
for BENCH_RANK in 00 01 02 03; do
  [[ -f "predictions/native-human/rank-${BENCH_RANK}.done.json" ]]
done
[[ -f predictions/tuna-human/done.json ]]
bash scripts/container.sh analysis python scripts/collect.py
bash scripts/container.sh analysis python scripts/analyze.py
bash scripts/container.sh analysis python scripts/plot_exposure_subsets.py
bash scripts/container.sh analysis python scripts/plot_combined_exposure.py
bash scripts/container.sh analysis python scripts/report.py
bash scripts/container.sh analysis python scripts/verify_complete.py --seal
bash scripts/container.sh analysis python scripts/verify_complete.py
