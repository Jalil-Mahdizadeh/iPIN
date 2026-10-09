"""Read-only status; standard library only, no data or model loading."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = {}
for name in ['provenance/launch.json', 'provenance/execution-start.json', 'results/phase.json', 'results/worker.json',
             'results/worker.done.json', 'results/execution-error.json', 'results/decision.json',
             'results/verification.json', 'results/resources.json', 'results/COMPLETE.json']:
    p = root / name
    if p.exists():
        value = json.loads(p.read_text())
        if name.endswith('verification.json'):
            value = {k: value[k] for k in ['passed', 'required_pairs_verified', 'pairs_verified', 'maximum_prediction_reconstruction_error']}
        out[name] = value
start = out.get('provenance/execution-start.json')
if start and 'results/COMPLETE.json' not in out:
    try:
        os.kill(start['pid'], 0); out['controller_process_exists'] = True
    except ProcessLookupError:
        out['controller_process_exists'] = False
print(json.dumps(out, indent=2))
