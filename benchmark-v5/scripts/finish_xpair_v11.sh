#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
for BENCH_SCRIPT in xpair_v11_exposure collect analyze plot_exposure_subsets plot_combined_exposure report; do
  bash "$BENCH_ROOT/scripts/container.sh" analysis python "scripts/$BENCH_SCRIPT.py"
done
bash "$BENCH_ROOT/scripts/container.sh" analysis python scripts/verify_complete.py --seal
bash "$BENCH_ROOT/scripts/container.sh" analysis python scripts/verify_complete.py
