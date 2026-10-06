#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
for BENCH_SCRIPT in xpair_v11_exposure collect analyze plot_with_historical report_nonhuman verify_complete; do
  bash "$BENCH_ROOT/scripts/container.sh" analysis python "scripts/$BENCH_SCRIPT.py"
done
