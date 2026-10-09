"""Detached bounded orchestration: all features, analysis, independent audit, ledger."""
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from study import ROOT, config, read, sha, atomic, now, verify_freeze, check_budget


def resources(state):
    start = read(ROOT / 'provenance/resource-start.json'); ended = time.time(); elapsed = ended - start['started_unix']
    cfg = config(); gpu = elapsed * start['allocated_gpu_count'] / 3600; cpu = elapsed * start['allocated_cpus'] / 3600
    size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    out = {'at_utc': now(), 'state': state, 'job_id': start['job_id'], 'study_started_unix': start['started_unix'],
        'study_ended_unix': ended, 'elapsed_seconds_including_preparation': elapsed, 'allocated_gpus': start['allocated_gpu_count'],
        'allocated_cpus': start['allocated_cpus'], 'allocated_gpu_hours_charged': gpu, 'allocated_cpu_core_hours_charged': cpu,
        'gpu_hour_ceiling': cfg['budgets']['allocated_gpu_hours'], 'cpu_core_hour_ceiling': cfg['budgets']['allocated_cpu_core_hours'],
        'additional_bytes': size, 'additional_byte_ceiling': cfg['budgets']['additional_bytes'],
        'within_limits': gpu <= cfg['budgets']['allocated_gpu_hours'] and cpu <= cfg['budgets']['allocated_cpu_core_hours'] and size <= cfg['budgets']['additional_bytes'],
        'accounting_scope': start['charging'], 'original_allocation_not_cancelled': True, 'separate_from_vx': True}
    atomic(ROOT / 'results/resources.json', out)
    return out


def main():
    lock = (ROOT / 'provenance/execution.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fp = verify_freeze(); cfg = config(); proc = None
    def stopped(signum, frame):
        raise RuntimeError('VZ controller received signal ' + str(signum))
    signal.signal(signal.SIGTERM, stopped); signal.signal(signal.SIGINT, stopped)
    if (ROOT / 'provenance/execution-start.json').exists():
        raise RuntimeError('Execution already started; no automatic retry')
    atomic(ROOT / 'provenance/execution-start.json', {'at_utc': now(), 'pid': os.getpid(), 'job_id': os.environ.get('SLURM_JOB_ID'), 'fingerprint': fp})
    try:
        for phase, script in [('extraction', 'worker.py'), ('analysis', 'analyze.py'), ('verification', 'verify.py')]:
            (ROOT / 'STATUS.md').write_text(f'# VZ status\n\nRunning **{phase}** in existing SLURM allocation {os.environ.get("SLURM_JOB_ID")}, one GPU. Fixed TRAIN/DEV only; no TEST access. The controller performs all remaining phases automatically.\n')
            atomic(ROOT / 'results/phase.json', {'at_utc': now(), 'phase': phase, 'fingerprint': fp})
            with (ROOT / 'logs' / f'{phase}.log').open('x') as log:
                proc = subprocess.Popen([sys.executable, '-B', str(ROOT / 'scripts' / script)], stdout=log, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    check_budget(cfg['budgets']['shutdown_reserve_seconds'])
                    time.sleep(2)
                if proc.returncode:
                    raise RuntimeError(f'{phase} failed with exit {proc.returncode}; inspect its log')
            proc = None
        verification = read(ROOT / 'results/verification.json')
        if not verification['passed']:
            raise ValueError('Independent verification did not pass')
        ledger = resources('completed')
        if not ledger['within_limits']:
            raise RuntimeError('Resource ceiling exceeded; cannot mark the study complete')
        decision = read(ROOT / 'results/decision.json')
        paths = ['heads.json', 'metrics.json', 'decision.json', 'verification.json', 'dev_predictions.npz', 'dev_evidence.npz', 'resources.json', 'REPORT.md']
        atomic(ROOT / 'results/COMPLETE.json', {'at_utc': now(), 'fingerprint': fp, 'verified': True,
            'required_pairs': verification['required_pairs_verified'], 'pairs': verification['pairs_verified'],
            'result_hashes': {name: sha(ROOT / 'results' / name) for name in paths}, 'test_accessed': False, 'production_started': False})
        (ROOT / 'STATUS.md').write_text(f'# VZ status\n\nCompleted and independently verified. Decision: **{decision["status"]}**. Read [the report](results/REPORT.md) and [verification](results/verification.json).\n\nCharged {ledger["allocated_gpu_hours_charged"]:.4f}/8 allocated GPU-hours, including preparation. No TEST evaluation or production continuation. The user\'s interactive allocation remains available.\n')
        print({'completed': True, 'decision': decision['status'], 'gpu_hours': ledger['allocated_gpu_hours_charged']}, flush=True)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
        error = {'at_utc': now(), 'fingerprint': fp, 'error': str(exc), 'traceback': traceback.format_exc(),
                 'test_accessed': False, 'production_started': False}
        atomic(ROOT / 'results/execution-error.json', error); resources('failed_or_incomplete')
        # Preserve any already-written scientific decision; the execution state takes precedence.
        atomic(ROOT / 'results/execution-decision.json', {**error, 'status': 'inconclusive_execution_failure'})
        (ROOT / 'STATUS.md').write_text('# VZ status\n\nStopped: execution failed or the study is incomplete. See [execution error](results/execution-error.json). No incomplete cohort is accepted as a completed experiment. No automatic retry.\n')
        raise


if __name__ == '__main__':
    main()
