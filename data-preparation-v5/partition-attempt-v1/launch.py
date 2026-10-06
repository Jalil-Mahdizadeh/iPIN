"""Single-attempt driver with an external deadline; no automatic retry."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from common import read_json,write_json

def main():
    lock=(HERE/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('SLURM_JOB_ID'),'An allocated compute node is required'
    assert not (HERE/'started.json').exists(),'The single production attempt has already started'
    assert not (HERE/'launcher.json').exists(),'No automatic second launch'
    environment=os.environ.copy()
    environment['APPTAINERENV_SLURM_JOB_ID']=os.environ['SLURM_JOB_ID']
    with (HERE/'attempt.log').open('x') as log:
        p=subprocess.Popen(['bash',str(ROOT/'scripts/python.sh'),str(HERE/'attempt.py')],
                           cwd=ROOT.parent,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True,env=environment)
        receipt=dict(driver_pid=os.getpid(),worker_pid=p.pid,slurm_job_id=os.environ['SLURM_JOB_ID'],
                     started_utc=datetime.now(timezone.utc).isoformat(),status='running',
                     maximum_solver_seconds=1800,watchdog_grace_seconds=60)
        write_json(HERE/'launcher.json',receipt)
        launched=time.monotonic();solve_seen=None;signalled=None
        while p.poll() is None:
            if (HERE/'started.json').exists() and solve_seen is None:solve_seen=time.monotonic()
            limit=(solve_seen+1802) if solve_seen is not None else launched+180
            if time.monotonic()>limit:
                if signalled is None:os.killpg(p.pid,signal.SIGTERM);signalled=time.monotonic()
                elif time.monotonic()-signalled>60:os.killpg(p.pid,signal.SIGKILL)
            time.sleep(2)
        receipt.update(status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode,
                       ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(HERE/'launcher.json',receipt)
        print(json.dumps(receipt),flush=True)
        return p.returncode

if __name__=='__main__':sys.exit(main())
