#!/usr/bin/env bash
set -euo pipefail
BENCH_WORKSPACE=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN
cd "$BENCH_WORKSPACE"
exec 9>benchmark-v4/logs/pipeline.lock
flock -n 9 || { echo 'Another process owns benchmark-v4.'; exit 73; }
sha256sum --check --quiet benchmark-v4/provenance/frozen-files.sha256
bash benchmark-v4/scripts/container.sh python -B benchmark-v4/scripts/pipeline.py
sha256sum --check --quiet benchmark-v4/provenance/frozen-files.sha256
