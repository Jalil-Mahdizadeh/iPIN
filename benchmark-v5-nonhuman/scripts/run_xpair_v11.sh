#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -f "$BENCH_ROOT/provenance/xpair-v11.json" ]]; then
  bash "$BENCH_ROOT/scripts/container.sh" xpair /work/runtime/venv/bin/python scripts/xpair_v11.py --stage freeze
fi
bash "$BENCH_ROOT/scripts/container.sh" xpair /work/runtime/venv/bin/python scripts/xpair_v11.py --stage qualify
bash "$BENCH_ROOT/scripts/container.sh" xpair /work/runtime/venv/bin/python scripts/xpair_v11.py --stage score
bash "$BENCH_ROOT/scripts/container.sh" xpair /work/runtime/venv/bin/python scripts/verify_xpair_v11_predictions.py
