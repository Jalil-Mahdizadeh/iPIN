#!/usr/bin/env bash
set -euo pipefail
BASELINE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
test -s "$BASELINE_ROOT/archive/before-nonhuman/results/summary.json"
mkdir -p "$BASELINE_ROOT/logs/nonhuman"
exec 9>"$BASELINE_ROOT/logs/nonhuman-run.lock"
flock -n 9
module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197 mmseqs2-gpu/18-8cc5c
bash "$BASELINE_ROOT/scripts/container.sh" python scripts/prepare_nonhuman.py
if [[ ${SLURM_CPUS_ON_NODE:-16} -ge 32 ]]; then
  python3 "$BASELINE_ROOT/scripts/search_nonhuman.py" v5 &
  baseline_v5_pid=$!
  python3 "$BASELINE_ROOT/scripts/search_nonhuman.py" bernett &
  baseline_bernett_pid=$!
  wait "$baseline_v5_pid"
  wait "$baseline_bernett_pid"
else
  python3 "$BASELINE_ROOT/scripts/search_nonhuman.py" v5
  python3 "$BASELINE_ROOT/scripts/search_nonhuman.py" bernett
fi
bash "$BASELINE_ROOT/scripts/container.sh" python scripts/predict_nonhuman.py
python3 "$BASELINE_ROOT/scripts/qualify_alignment_exports.py" v5
python3 "$BASELINE_ROOT/scripts/qualify_alignment_exports.py" bernett
python3 "$BASELINE_ROOT/scripts/check_identity_boundary.py"
bash "$BASELINE_ROOT/scripts/container.sh" python scripts/evaluate_nonhuman.py
bash "$BASELINE_ROOT/scripts/container.sh" python scripts/diagnose_nonhuman.py
bash "$BASELINE_ROOT/scripts/container.sh" python scripts/verify_nonhuman.py
bash "$BASELINE_ROOT/scripts/container.sh" python scripts/publish_nonhuman.py
bash "$BASELINE_ROOT/scripts/container.sh" python scripts/finalize_nonhuman.py
