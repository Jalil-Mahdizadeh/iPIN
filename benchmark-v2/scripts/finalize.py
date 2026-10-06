"""Wait for the submitted inference job, then verify and report all results."""
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
cfg = json.loads((ROOT/'config.json').read_text())
job = json.loads((ROOT/'provenance/submission.json').read_text())['job_id']
started = time.monotonic()
previous = None
while True:
    finished = {name:len(list((ROOT/'predictions'/f'{name}-test').glob('rank-*.done.json'))) for name in cfg['models']}
    if finished != previous:
        print(json.dumps({'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers_complete':finished}),flush=True)
        previous = finished
    q = subprocess.run(['squeue','-j',job,'--noheader'],text=True,capture_output=True,check=True)
    if not q.stdout.strip():
        accounting = subprocess.check_output(['sacct','-X','-j',job,'--format=JobID,State,ExitCode,Elapsed,ElapsedRaw,NodeList,AllocTRES%100','-P'],text=True)
        lines = accounting.strip().splitlines()
        rows = [dict(zip(lines[0].split('|'),line.split('|'))) for line in lines[1:]]
        if rows and rows[0]['State'] == 'COMPLETED':
            assert rows[0]['ExitCode'] == '0:0' and all(n==4 for n in finished.values()), (rows,finished)
            (ROOT/'provenance/slurm-accounting.txt').write_text(accounting)
            break
        assert not rows or rows[0]['State'] in ['RUNNING','COMPLETING','PENDING'], rows
    assert time.monotonic()-started < 4500, 'Inference did not finish within the monitoring window'
    time.sleep(20)
for script, log in [('analyze.py','analysis.log'),('write_report.py','report.log')]:
    with (ROOT/'logs'/log).open('w') as handle:
        subprocess.run(['bash',str(ROOT/'scripts/container.sh'),'python','-u',str(ROOT/'scripts'/script)],
                       cwd=WORK,stdout=handle,stderr=subprocess.STDOUT,check=True)
readme=ROOT/'README.md'
text=readme.read_text()
text=text.replace('Fresh inference for the four v2 models was submitted as SLURM job **3172100** on one four-GPU node.',
                  'Fresh inference for all four v2 models completed successfully as SLURM job **3172100** on one four-GPU node. The full seven-model comparison is in [REPORT.md](REPORT.md).')
text=text.replace('After inference completes, the following commands verify coverage, compute the complete comparison and write the report:',
                  'The following commands regenerate the verified comparison and report from the completed predictions:')
readme.write_text(text)
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()
files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in ['checkpoints','cache','__pycache__','logs'])
       and p.name!='artifact-sha256.txt']
(ROOT/'provenance/artifact-sha256.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(WORK)}\n' for p in sorted(files)))
summary=json.loads((ROOT/'results/benchmark-summary.json').read_text())
print(json.dumps({'report':str(ROOT/'REPORT.md'),'primary':summary['primary'],
                  'artifact_files_hashed':len(files),'completed':True},indent=2),flush=True)
