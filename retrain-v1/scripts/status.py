"""Read local run progress and Slurm state without loading the model."""
import json
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
active_attempt_ids=set()
for run in sorted((root/'runs').glob('*')):
    if not run.is_dir():continue
    print(run.name,flush=True)
    for filename in ['status.json','checkpoint-status.json','best.json','stopped.json','completed.json']:
        path=run/filename
        if path.exists():
            value=json.loads(path.read_text())
            print(f'  {filename}: {json.dumps(value)}',flush=True)
            attempt_id=value.get('attempt','').rsplit('-',1)[-1]
            if attempt_id.isdigit():active_attempt_ids.add(attempt_id)
jobs=root/'provenance/submitted-jobs.json'
if jobs.exists():
    ids=sorted(active_attempt_ids|{str(x['job_id']) for x in json.loads(jobs.read_text())})
    subprocess.run(['squeue','-j',','.join(ids),'-o','%.12i %.24j %.10T %.12M %R'],check=False)
