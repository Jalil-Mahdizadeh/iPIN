"""Record final scheduler state and checksum the compact reproducibility artifacts."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
coverage=json.loads((ROOT/'results/coverage.json').read_text())
assert coverage['complete']
accounting=subprocess.run(['sacct','-j','3110591','--parsable2','--format=JobID,State,ExitCode,Start,End,Elapsed,AllocTRES,NodeList'],capture_output=True,text=True,check=True).stdout
(ROOT/'provenance/slurm-accounting.txt').write_text(accounting)
record=dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_retraining=True,
            apptainer_version=subprocess.run(['apptainer','version'],capture_output=True,text=True,check=True).stdout.strip(),
            inference_code_sha256s=sorted({json.loads(p.read_text())['config']['code_sha256'] for p in (ROOT/'results/predictions').glob('*.json')}),
            evaluated_worker_gpu_hours=sum(x['summed_gpu_worker_seconds'] for x in coverage['tasks'])/3600,
            completed_tasks=len(coverage['tasks']),coverage_complete=True)
(ROOT/'provenance/completion.json').write_text(json.dumps(record,indent=2)+'\n')
paths=[]
for folder in ['scripts','tests','slurm','results','provenance']:
    paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.zip','.pyc'])
paths.extend(ROOT/f for f in ['README.md','PROTOCOL.md','REPORT.md'])
lines=[]
for path in sorted(set(paths)):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    lines.append(f'{h.hexdigest()}  {path.relative_to(ROOT)}')
(ROOT/'ARTIFACTS.sha256').write_text('\n'.join(lines)+'\n')
print('Checksummed',len(lines),'artifacts; checkpoint/data/image hashes are also in their provenance manifests.')
