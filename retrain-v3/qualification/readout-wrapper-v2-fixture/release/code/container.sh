#!/usr/bin/env bash
set -euo pipefail
exec python - "$@" <<'PY'
import hashlib,json,os,signal,sys,time
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--output')+1]);kind=os.environ['QUAL_CASE']
if kind=='failure':sys.exit(42)
if kind=='missed-window':
    (p/'completed.json').write_text('{}');sys.exit(0)
if kind=='manual-before-validation':
    (p/'REQUEST_STOP').touch()
    state={'update':1,'last_validation_update':None,'pending_validation':False,'total_steps':6}
    (p/'stopped.json').write_text(json.dumps({'checkpoint_committed':True,'state':state}));sys.exit(0)
if kind=='requeue':
    os.kill(os.getppid(),signal.SIGUSR1)
    for _ in range(200):
        if (p/'REQUEST_REQUEUE').exists():break
        time.sleep(.02)
    assert (p/'REQUEST_REQUEUE').exists()
    state={'update':1,'last_validation_update':None,'pending_validation':False,'total_steps':6}
    (p/'stopped.json').write_text(json.dumps({'checkpoint_committed':True,'state':state}));sys.exit(0)
(p/'validation').mkdir();(p/'checkpoints').mkdir()
(p/'contract.json').write_text(json.dumps({'fingerprint':'fixture'}))
file=p/'checkpoints/update-000000002.pt';file.write_bytes(b'local synthetic checkpoint')
meta={'file':file.name,'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
      'fingerprint':'fixture','update':2,'best_update':2,'best_ap':.6}
for target in [p/'latest.json',p/'best.json',file.with_suffix('.pt.json')]:target.write_text(json.dumps(meta))
pred=p/'validation/update-000000002.npz';pred.write_bytes(b'synthetic prediction artifact')
(pred.with_suffix('.json')).write_text(json.dumps({'update':2,'fingerprint':'wrong' if kind=='monitor-error' else 'fixture',
    'sha256':hashlib.sha256(pred.read_bytes()).hexdigest()}))
for _ in range(500):
    if (p/'REQUEST_STOP').exists():break
    time.sleep(.02)
assert (p/'REQUEST_STOP').exists(),'Monitor failed to stop worker'
state={'update':2,'last_validation_update':2,'pending_validation':False,'best_update':2,'total_steps':6}
(p/'stopped.json').write_text(json.dumps({'checkpoint_committed':True,'state':state}))
PY
