#!/usr/bin/env bash
set -euo pipefail
WORK_ROOT=/nobackup/proj/disk/theo-storage/personal/jalil/iPIN
cd "$WORK_ROOT"
exec 9>benchmark-v3/logs/pipeline.lock
flock -n 9 || { echo 'Another benchmark process owns this run.'; exit 73; }
sha256sum --check --quiet benchmark-v3/provenance/frozen-files.sha256
bash benchmark-v3/scripts/container.sh python -B benchmark-v3/scripts/pipeline.py
sha256sum --check --quiet benchmark-v3/provenance/frozen-files.sha256
