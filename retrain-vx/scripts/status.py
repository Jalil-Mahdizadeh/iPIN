"""Read-only standard-library status; no models or datasets opened."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = {}
for name in ['preparation-progress', 'preparation-error', 'preflight-error', 'phase', 'execution-error', 'decision', 'verification', 'resources', 'COMPLETE']:
    p = root / 'results' / (name + '.json')
    if p.exists():
        value = json.loads(p.read_text())
        if name == 'verification':
            value = {k: value[k] for k in ['passed', 'rows_verified', 'eligible_rows', 'maximum_prediction_error']}
        out[name] = value
for name in ['submission', 'execution']:
    p = root / 'provenance' / (name + '.json')
    if p.exists():
        out[name] = json.loads(p.read_text())
workers = []
for rank in range(8):
    done = root / 'results' / f'worker-{rank}.done.json'; progress = root / 'results' / f'worker-{rank}.json'
    if done.exists() or progress.exists():
        workers.append(json.loads((done if done.exists() else progress).read_text()))
if workers:
    out['workers'] = workers
print(json.dumps(out, indent=2))
