"""Read-only compact status; does not launch, resume or edit anything."""
import json
import subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for name in ['provenance/submission.json', 'results/preparation-progress.json', 'results/phase.json',
             'results/preparation-error.json', 'results/preflight-error.json', 'results/execution-error.json', 'results/decision.json']:
    p = root / name
    if p.exists():
        d = json.loads(p.read_text()); print(name, json.dumps(d))
sub = root / 'provenance/submission.json'
if sub.exists():
    job = json.loads(sub.read_text())['job_id']
    subprocess.run(['sacct', '-j', job, '--format=JobID,State,ExitCode,Elapsed,NodeList', '-P'], check=False)
for rank in range(4):
    done = root / 'results' / f'worker-{rank}.done.json'; progress = root / 'results' / f'worker-{rank}.json'
    p = done if done.exists() else progress
    if p.exists():
        d = json.loads(p.read_text()); print(p.name, {k: d[k] for k in ['rows', 'processed', 'committed', 'total', 'seconds', 'elapsed_seconds', 'counts'] if k in d})
print('Verified completion:', (root / 'results/COMPLETE.json').exists())
