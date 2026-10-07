#!/usr/bin/env bash
# The original human study is sealed; this entry point includes its extension.
set -euo pipefail
BASELINE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$BASELINE_ROOT/scripts/run_nonhuman.sh"
