#!/usr/bin/env bash
set -euo pipefail
BASELINE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$BASELINE_ROOT/logs"
exec 9>"$BASELINE_ROOT/logs/run.lock"
flock -n 9
module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197 mmseqs2-gpu/18-8cc5c
if [[ -f "$BASELINE_ROOT/provenance/prepared.json" ]]; then
  bash "$BASELINE_ROOT/scripts/container.sh" python scripts/prepare.py
else
  /usr/bin/time -v -o "$BASELINE_ROOT/logs/prepare.resources.txt" \
    bash "$BASELINE_ROOT/scripts/container.sh" python scripts/prepare.py > "$BASELINE_ROOT/logs/prepare.log" 2>&1
fi
python "$BASELINE_ROOT/scripts/search.py" >> "$BASELINE_ROOT/logs/search.log" 2>&1
if [[ -f "$BASELINE_ROOT/provenance/predictions-frozen.json" ]]; then
  bash "$BASELINE_ROOT/scripts/container.sh" python scripts/fit_predict.py
else
  /usr/bin/time -v -o "$BASELINE_ROOT/logs/fit-predict.resources.txt" \
    bash "$BASELINE_ROOT/scripts/container.sh" python scripts/fit_predict.py > "$BASELINE_ROOT/logs/fit-predict.log" 2>&1
fi
/usr/bin/time -v -o "$BASELINE_ROOT/logs/evaluate.resources.txt" \
  bash "$BASELINE_ROOT/scripts/container.sh" python scripts/evaluate.py > "$BASELINE_ROOT/logs/evaluate.log" 2>&1
bash "$BASELINE_ROOT/scripts/container.sh" python scripts/finalize.py
