"""One bounded sequential run on an existing idle GPU; fail closed, no automatic retry."""
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from study import ROOT, config, read, atomic, now, verify_freeze, remaining, owned_bytes


def command(script,reserve):
    deadline=time.time()+remaining(reserve)
    with (ROOT/'logs'/script.replace('.py','.log')).open('xb') as log:
        p=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts'/script)],stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
        try:
            while p.poll() is None:
                if time.time()>deadline:raise TimeoutError('Phase deadline: '+script)
                time.sleep(3)
            if p.returncode:raise RuntimeError(f'{script} failed with exit {p.returncode}; inspect phase log')
        except BaseException:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise


def main():
    started=time.monotonic(); fp=verify_freeze(); cfg=config(); job=os.environ.get('SLURM_JOB_ID')
    resource=read(ROOT/'provenance/resource-start.json')
    if job!=resource['job_id']:raise ValueError('Different allocated job')
    lock=(ROOT/'provenance/execution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        for script,phase,reserve in [('native.py','native_train_scores',cfg['budgets']['analysis_reserve_seconds']),
            ('worker.py','contact_extraction',cfg['budgets']['analysis_reserve_seconds']),
            ('analyze.py','train_fitting_and_val_assessment',900),('verify.py','independent_verification',cfg['budgets']['shutdown_reserve_seconds'])]:
            atomic(ROOT/'results/phase.json',{'at_utc':now(),'phase':phase,'job_id':job,'fingerprint':fp})
            (ROOT/'STATUS.md').write_text(f'# V7 v1 status\n\nRunning **{phase}** on one GH200 in allocation **{job}**. All new fitting/calibration is TRAIN-only; no TEST access.\n')
            command(script,reserve)
        if not read(ROOT/'results/verification.json')['passed']:raise ValueError('Final verification missing')
        decision=read(ROOT/'results/decision.json'); elapsed=time.time()-resource['started_unix']
        atomic(ROOT/'results/COMPLETE.json',{'at_utc':now(),'fingerprint':fp,'job_id':job,'seconds':time.monotonic()-started,
            'active_allocation_interval_gpu_hours':elapsed/3600,'separate_gpu_hour_ceiling':cfg['budgets']['allocated_gpu_hours_ceiling'],
            'owned_bytes':owned_bytes(),'test_accessed':False,'production_started':False})
        (ROOT/'STATUS.md').write_text(f'# V7 v1 status\n\nCompleted and verified in allocation **{job}**. Decision: **{decision["status"]}**. See [results](results/REPORT.md). No TEST access or automatic continuation.\n')
    except BaseException as exc:
        error={'at_utc':now(),'fingerprint':fp,'job_id':job,'error':str(exc),'traceback':traceback.format_exc()}
        atomic(ROOT/'results/execution-error.json',error)
        decision_path=ROOT/'results/decision.json'
        prior=read(decision_path) if decision_path.exists() else None
        atomic(decision_path,{**error,'status':'inconclusive_execution_error','unverified_analysis_decision':prior,
            'production_authorized':False,'test_accessed':False})
        report_path=ROOT/'results/REPORT.md'
        previous=report_path.read_text() if report_path.exists() else ''
        report_path.write_text('# Execution incomplete\n\nThis run stopped with an execution or verification error. Any analysis below is unverified and cannot support a success decision. See results/execution-error.json.\n\n'+previous)
        (ROOT/'STATUS.md').write_text('# V7 v1 status\n\nStopped with an execution error. See results/execution-error.json and logs. No automatic retry or conversion of computation errors to biological fallback.\n')
        raise


if __name__=='__main__':main()
