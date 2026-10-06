"""Launch or resume preparation on the current allocation; return immediately."""
import json
import os
import platform
import subprocess
from datetime import datetime,timezone
from common import ROOT,read_json,write_json

assert os.environ.get('SLURM_JOB_ID'),'Use a compute allocation, not a login node'
receipt=ROOT/'provenance/launcher.json'
if receipt.exists():
    old=read_json(receipt)
    if old['host']==platform.node():
        try:
            command=open(f"/proc/{old['pid']}/cmdline",'rb').read()
            if str(ROOT/'scripts/run_pipeline.py').encode() in command:
                print(json.dumps(dict(already_running=True,**old)));raise SystemExit(0)
        except FileNotFoundError:pass
if (ROOT/'completed.json').exists() and read_json(ROOT/'completed.json').get('complete'):
    print('Preparation already complete; no process launched.');raise SystemExit(0)
with (ROOT/'logs/pipeline.log').open('a') as log:
    p=subprocess.Popen(['python',str(ROOT/'scripts/run_pipeline.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,host=platform.node(),slurm_job_id=os.environ['SLURM_JOB_ID'],started_utc=datetime.now(timezone.utc).isoformat(),
       log='logs/pipeline.log',status='pipeline-state.json',resume_command='python data-preparation-v5/scripts/launch.py')
write_json(receipt,r);print(json.dumps(r))
