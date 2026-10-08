"""Finish qualified pilot preparation and submit its bounded GPU stage once."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from common import ROOT, PROJECT, atomic, now, sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive-pid',type=int,required=True);args=ap.parse_args()
    deadline=json.loads((ROOT/'provenance/resource-start.json').read_text())['qualification_deadline_unix']
    image=PROJECT/'images/msa-pairformer/msa-pairformer-arm64-v1.sif'
    atomic(ROOT/'provenance/controller.json',{'at_utc':now(),'pid':os.getpid(),'archive_pid':args.archive_pid,'deadline_unix':deadline})
    def stage(name,gpu=False):
        remaining=deadline-time.time()-30
        if remaining<=0:raise TimeoutError('Qualification reserve exhausted')
        cmd=['apptainer','exec']+(['--nv'] if gpu else [])+[str(image),'python','-B',str(ROOT/'scripts'/name)]
        # Scheduler clients belong to the host; the image supplies numerical Python only.
        if name=='launch.py':cmd=[sys.executable,'-B',str(ROOT/'scripts'/name),'--submit']
        print(json.dumps({'at_utc':now(),'stage':name}),flush=True)
        subprocess.run(cmd,check=True,timeout=remaining)
    try:
        while not (ROOT/'results/archive-coverage.json').exists():
            if time.time()>deadline:raise TimeoutError('Archive preparation exceeded qualification reserve')
            try:
                stat=open(f'/proc/{args.archive_pid}/stat').read();state=stat[stat.rfind(')')+2:].split()[0]
                if state=='Z':raise ProcessLookupError('Archive process exited without completion')
                cmdline=open(f'/proc/{args.archive_pid}/cmdline','rb').read()
                if b'pilot-vx/scripts/archive.py' not in cmdline:raise ProcessLookupError('Archive PID no longer belongs to this task')
            except FileNotFoundError:raise ProcessLookupError('Archive process exited without completion')
            time.sleep(10)
        real_path=ROOT/'qualification/real-encoder.json'
        qualified=False
        if real_path.exists():
            real=json.loads(real_path.read_text())
            qualified=real.get('passed',False) and bool(real.get('code_sha256'))
            qualified=qualified and all(sha(ROOT/p)==h for p,h in real.get('code_sha256',{}).items())
            qualified=qualified and all(sha(ROOT/p)==h for c in real.get('cases',[]) for p,h in c['monomer_files'].items())
        if not qualified:stage('qualify_real.py',gpu=True)
        stage('freeze.py')
        stage('launch.py')
        job=json.loads((ROOT/'provenance/submission.json').read_text())['job_id']
        atomic(ROOT/'provenance/controller-complete.json',{'at_utc':now(),'job_id':job})
        (ROOT/'STATUS.md').write_text(f'# Pilot status\n\nGPU pilot submitted as SLURM job **{job}** after archive, parser, runtime and real-input qualification.\n\nThe job requests four GPUs for at most five hours (20 GPU-hours); the total pilot ceiling remains 24 GPU-hours including qualification. It runs only the fixed TRAIN/DEV sample and controls, then the predeclared analysis. No TEST evaluation or production continuation is enabled.\n\nRun `python -B pilot-vx/scripts/status.py` for live scheduler and artifact status. Results appear in `results/decision.json` and `results/REPORT.md`.\n')
    except BaseException as exc:
        atomic(ROOT/'provenance/controller-error.json',{'at_utc':now(),'error':str(exc),'traceback':traceback.format_exc()})
        (ROOT/'STATUS.md').write_text('# Pilot status\n\nPreparation stopped before successful GPU submission. See `provenance/controller-error.json` and the preparation logs. No automatic retry or budget extension is enabled.\n')
        raise


if __name__=='__main__':main()
