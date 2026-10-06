"""Describe logged training batches; no new training or inference."""
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
rows = []
for p in sorted((ROOT / 'retrain-v2/analysis/completed-20260930T082601Z').glob('*-events.jsonl')):
    data = [json.loads(s) for s in p.open()]
    updates = [x for x in data if x.get('event') == 'update']
    validations = [x for x in data if x.get('event') == 'validation']
    windows = []
    for lo, hi in [(3000, 4000), (11745, 12745)]:
        batch = [x for x in updates if lo <= x['update'] <= hi]
        assert batch
        windows.append(dict(window=[lo, hi], n=len(batch),
            mean_bce=statistics.mean(x['mean_bce'] for x in batch),
            mean_grad_norm=statistics.mean(x['grad_norm'] for x in batch),
            fraction_grad_norm_gt1=statistics.mean(x['grad_norm'] > 1 for x in batch)))
    rows.append(dict(model=p.name.replace('-events.jsonl', ''), file=str(p.relative_to(ROOT)),
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(), logged_updates=len(updates),
        validation_example=validations[0], windows=windows))
out = dict(created_utc=datetime.now(timezone.utc).isoformat(),
    note='Periodically logged training batches; not full-dataset training loss or causal evidence',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), runs=rows)
(HERE / 'training-window-audit.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
print('Audited', len(rows), 'runs')
