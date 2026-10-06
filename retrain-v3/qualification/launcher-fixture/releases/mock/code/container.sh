#!/usr/bin/env bash
set -euo pipefail
exec python - "$@" <<'PY'
import json,os,signal,sys,time
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--output')+1]);kind=os.environ['QUAL_CASE']
if kind=='failure':sys.exit(42)
if kind=='requeue':
    os.kill(os.getppid(),signal.SIGUSR1)
    for _ in range(100):
        if (p/'REQUEST_REQUEUE').exists():break
        time.sleep(.02)
    assert (p/'REQUEST_REQUEUE').exists()
    (p/'stopped.json').write_text(json.dumps({'checkpoint_committed':True}))
else:(p/'completed.json').write_text('{}')
PY
