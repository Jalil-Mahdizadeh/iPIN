"""Run four independent GPU workers once, then the fixed CPU analysis."""
import json
import os
import subprocess
import sys
import time
from common import ROOT, atomic, now, verify_freeze


def main():
    fingerprint=verify_freeze();started=time.monotonic()
    (ROOT/'STATUS.md').write_text(f'# Pilot status\n\nGPU pilot running in SLURM job **{os.environ.get("SLURM_JOB_ID")}**. Four bounded workers process the frozen TRAIN/DEV sample. See `results/worker-*.json` and run the status script for progress.\n')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(visible)!=4 or any(not x for x in visible):raise RuntimeError('Expected exactly four allocated GPUs')
    processes=[];logs=[]
    try:
        for rank,gpu in enumerate(visible):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            log=(ROOT/f'logs/worker-{rank}.log').open('a');logs.append(log)
            p=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/features.py'),'--rank',str(rank),'--world','4','--seconds','16800'],env=env,stdout=log,stderr=subprocess.STDOUT)
            processes.append(p)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in [None,0] for p in processes):
                raise RuntimeError('A feature worker failed; stopping sibling workers')
            if time.monotonic()-started>17400:raise TimeoutError('Pilot orchestration deadline reached')
            time.sleep(5)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('Worker failed')
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/analyze.py')],check=True,timeout=400)
        atomic(ROOT/'results/batch-complete.json',{'at_utc':now(),'job_id':os.environ.get('SLURM_JOB_ID'),
               'fingerprint':fingerprint,'seconds':time.monotonic()-started,'production_started':False})
        decision=json.loads((ROOT/'results/decision.json').read_text())
        (ROOT/'STATUS.md').write_text(f'# Pilot status\n\nPilot batch finished. Decision: **{decision["status"]}**. Read [the report](results/REPORT.md) and `results/decision.json`. No TEST evaluation or production continuation was started.\n')
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        atomic(ROOT/'results/batch-error.json',{'at_utc':now(),'error':str(exc),'fingerprint':fingerprint})
        (ROOT/'STATUS.md').write_text('# Pilot status\n\nGPU batch stopped with an execution error. See `results/batch-error.json` and worker error records. Incomplete computation was not converted into biological fallback. No automatic retry is enabled.\n')
        raise
    finally:
        for log in logs:log.close()


if __name__=='__main__':main()
