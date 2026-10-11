"""Four bounded workers followed by complete-sample CPU analysis, once."""
import json
import os
import subprocess
import sys
import time
import traceback
from study import ROOT, config, atomic, read, now, verify_freeze


def main():
    started = time.monotonic(); cfg = config(); fp = verify_freeze()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 4 or any(not g for g in visible): raise RuntimeError('Expected four allocated GPUs')
    job = os.environ.get('SLURM_JOB_ID')
    atomic(ROOT / 'provenance/batch-start.json', {'at_utc': now(), 'job_id': job, 'fingerprint': fp, 'gpu_count': 4})
    (ROOT / 'STATUS.md').write_text(f'# Vx v2 status\n\nRunning four feature workers in SLURM job **{job}**. Analysis will run only after verifying all 8,000 computational records.\n')
    processes = []; logs = []
    try:
        for rank, gpu in enumerate(visible):
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = gpu
            log = (ROOT / 'logs' / f'worker-{rank}.log').open('a'); logs.append(log)
            processes.append(subprocess.Popen([sys.executable, '-B', str(ROOT / 'scripts/worker.py'), '--rank', str(rank)], env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in [None, 0] for p in processes): raise RuntimeError('A worker failed; stop sibling workers')
            if time.monotonic() - started > cfg['budgets']['worker_wall_seconds'] + 180: raise TimeoutError('Worker orchestration deadline')
            time.sleep(5)
        if any(p.returncode != 0 for p in processes): raise RuntimeError('Worker failed')
        remaining = cfg['budgets']['batch_wall_seconds'] - (time.monotonic() - started) - 60
        if remaining <= 0: raise TimeoutError('No budget remains for analysis')
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/analyze_v2.py')], check=True, timeout=remaining)
        decision = read(ROOT / 'results/decision.json')
        elapsed = time.monotonic() - started
        atomic(ROOT / 'results/batch-complete.json', {'at_utc': now(), 'job_id': job, 'fingerprint': fp,
            'seconds': elapsed, 'batch_gpu_hours_lower_bound': 4 * elapsed / 3600,
            'scheduler_accounting_required': True, 'production_started': False})
        (ROOT / 'STATUS.md').write_text(f'# Vx v2 status\n\nCompleted in SLURM job **{job}**. Decision: **{decision["status"]}**. Read [the report](results/REPORT.md). Final allocated resources should be reconciled against scheduler accounting. No TEST evaluation or production continuation was started.\n')
    except BaseException as exc:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try: p.wait(timeout=20)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
        error = {'at_utc': now(), 'job_id': job, 'error': str(exc), 'fingerprint': fp, 'traceback': traceback.format_exc()}
        atomic(ROOT / 'results/batch-error.json', error)
        atomic(ROOT / 'results/decision.json', {**error, 'status': 'inconclusive_execution_error', 'production_authorized': False, 'test_accessed': False})
        (ROOT / 'results/REPORT.md').write_text('# Vx v2\n\nInconclusive: execution failed. See batch-error.json and worker error records. Computational failure was not converted into biological fallback. No automatic retry is enabled.\n')
        (ROOT / 'STATUS.md').write_text('# Vx v2 status\n\nStopped with an execution error. See results/batch-error.json and worker error records. No automatic retry or biological fallback for missing computation.\n')
        raise
    finally:
        for log in logs: log.close()


if __name__ == '__main__': main()
