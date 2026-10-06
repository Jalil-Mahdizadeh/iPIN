"""Read scheduler and per-model coverage without loading models or predictions."""
import json
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / 'config.json').read_text())
job = json.loads((ROOT / 'provenance/submission.json').read_text())['job_id']
progress = {}
for path in (ROOT / 'logs').glob('progress-rank-*.json'):
    row = json.loads(path.read_text())
    progress[(row['task'], row['rank'])] = row
models = {}
for name in cfg['models']:
    task = name + '-test'
    ranks = []
    for rank in range(cfg['world_size']):
        path = ROOT / 'predictions' / task / f'rank-{rank:02d}.done.json'
        if path.exists():
            data = json.loads(path.read_text())
            ranks.append({'rank':rank,'complete':True,'rows':data['rows']})
        else:
            data = progress.get((task,rank),{})
            ranks.append({'rank':rank,'complete':False,'rows':data.get('rows_done',0),
                          'last_progress_utc':data.get('time_utc')})
    models[name] = {'rows_committed':sum(r['rows'] for r in ranks),
                    'rows_total':52048, 'ranks_complete':sum(r['complete'] for r in ranks), 'ranks':ranks}
result = subprocess.run(['squeue','-j',job,'--noheader','--format=%i|%T|%M|%N|%R'],text=True,capture_output=True)
print(json.dumps({'job_id':job,'scheduler':result.stdout.strip(),'scheduler_error':result.stderr.strip(),
    'models':models,'analysis_completed':(ROOT/'completed.json').exists()},indent=2))
