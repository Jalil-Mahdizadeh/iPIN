"""Launch the bounded CPU recovery on the current compute allocation."""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import read_json,write_json

assert os.environ.get('SLURM_JOB_ID'),'Use a compute allocation'
receipt=ROOT/'provenance/recovery-v1/launcher.json'
if receipt.exists():
    old=read_json(receipt)
    if old['host']==platform.node():
        try:
            if str(ROOT/'recovery-v1/run.py').encode() in Path(f"/proc/{old['pid']}/cmdline").read_bytes():
                print(json.dumps(dict(already_running=True,**old)));raise SystemExit(0)
        except FileNotFoundError:pass
if (ROOT/'completed.json').exists():
    assert read_json(ROOT/'completed.json')['complete']
    print('Already complete; no process launched.');raise SystemExit(0)
with (ROOT/'logs/recovery-v1-pipeline.log').open('a') as log:
    p=subprocess.Popen(['python',str(ROOT/'recovery-v1/run.py')],cwd=ROOT.parent,stdin=subprocess.DEVNULL,
                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,host=platform.node(),slurm_job_id=os.environ['SLURM_JOB_ID'],
            started_utc=datetime.now(timezone.utc).isoformat(),log='logs/recovery-v1-pipeline.log',
            resume_command='python data-preparation-v5/recovery-v1/launch.py')
write_json(receipt,record);print(json.dumps(record))
