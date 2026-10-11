"""One full four-GPU node, complete TEST extraction, frozen scoring and audit."""
import os
import signal
import time
import traceback
from runtime import ROOT, PROJECT, read, now, atomic, sha, verify_files, topology, run_command, owned_bytes


def main():
    cfg = read(ROOT / 'config.json'); started = time.time(); job = os.environ['SLURM_JOB_ID']
    if int(os.environ['SLURM_JOB_NUM_NODES']) != 1 or int(os.environ['SLURM_NTASKS']) != 4:
        raise RuntimeError('Exactly one node and four workers are required')
    if int(os.environ['SLURM_CPUS_PER_TASK']) != 72:
        raise RuntimeError('Expected 72 allocated CPUs per GPU worker')
    if (ROOT / 'provenance/execution.json').exists():
        raise RuntimeError('Batch execution already exists; no automatic retry')
    fp = verify_files(); deadline = started + cfg['budgets']['batch_wall_hours'] * 3600 - 120
    if sha(PROJECT / cfg['image']) != cfg['image_sha256']:
        raise ValueError('Qualified SIF changed while queued')
    if not cfg['test_authorized'] or not read(ROOT / 'provenance/authorization.json')['test_authorized']:
        raise ValueError('Explicit TEST authorization absent')
    atomic(ROOT / 'provenance/execution.json', {'at_utc': now(), 'started_unix': started, 'job_id': job,
        'nodes': os.environ['SLURM_JOB_NODELIST'], 'gpus': 4, 'cpus': 288, 'fingerprint': fp,
        'wall_hours_ceiling': cfg['budgets']['batch_wall_hours'], 'pid': os.getpid()})
    image = str(PROJECT / cfg['image'])
    runtime = ['apptainer', 'exec', '--env', 'OPENBLAS_NUM_THREADS=8', '--env', 'OMP_NUM_THREADS=8', image, 'python', '-B']
    command = ['srun', '--nodes=1', '--ntasks=4', '--ntasks-per-node=4', '--gpus-per-task=1', '--cpus-per-task=72',
        '--gpu-bind=single:1', '--cpu-bind=cores', '--kill-on-bad-exit=1', 'apptainer', 'exec', '--nv',
        '--env', 'OPENBLAS_NUM_THREADS=8', '--env', 'OMP_NUM_THREADS=8', image, 'python', '-B', str(ROOT / 'scripts/worker.py')]
    state = 'failed'; approved = False; last_storage = 0.; extraction_start = time.time()
    def check(process):
        nonlocal approved, last_storage
        if not approved:
            paths = [ROOT / 'provenance' / f'gpu-rank-{i}.json' for i in range(4)]
            if all(p.exists() for p in paths):
                records = [read(p) for p in paths]; topology(records, job, fp)
                atomic(ROOT / 'provenance/topology-approved.json', {'at_utc': now(), 'job_id': job, 'fingerprint': fp, 'workers': records})
                approved = True
            elif time.time() - extraction_start > 900:
                raise TimeoutError('Four GPU workers did not start')
        if time.time() - last_storage > 300:
            if owned_bytes() > cfg['budgets']['additional_bytes']:
                raise RuntimeError('New storage ceiling exceeded')
            last_storage = time.time()
    def interrupted(signum, frame):
        raise RuntimeError('Batch interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        atomic(ROOT / 'results/phase.json', {'at_utc': now(), 'phase': 'extraction', 'fingerprint': fp})
        (ROOT / 'STATUS.md').write_text(f'# Vx TEST benchmark status\n\nRunning extraction in job **{job}**, one full node/four GPUs. Fixed TRAIN/VAL heads; no fitting or shuffled arm.\n')
        run_command(command, started + cfg['budgets']['batch_wall_hours'] * 3600 - cfg['budgets']['analysis_reserve_seconds'], ROOT / 'logs/workers.log', check)
        if not approved:
            raise RuntimeError('Four-GPU topology was not verified')
        for phase, script in [('analysis', 'analyze.py'), ('verification', 'verify.py')]:
            atomic(ROOT / 'results/phase.json', {'at_utc': now(), 'phase': phase, 'fingerprint': fp})
            run_command(runtime + [str(ROOT / 'scripts' / script)], deadline, ROOT / 'logs' / (phase + '.log'))
        verification = read(ROOT / 'results/verification.json')
        assert verification['passed'] and verification['rows_verified'] == cfg['test_pairs'] and verification['fingerprint'] == fp
        for name, digest in verification['result_hashes'].items():
            assert sha(ROOT / 'results' / name) == digest
        state = 'completed'
    except BaseException:
        atomic(ROOT / 'results/execution-error.json', {'at_utc': now(), 'job_id': job, 'traceback': traceback.format_exc()})
        (ROOT / 'STATUS.md').write_text('# Vx TEST benchmark status\n\nExecution stopped with an error. No incomplete-cohort result is accepted. See results/execution-error.json. No automatic retry.\n')
        raise
    finally:
        elapsed = time.time() - started; submission = read(ROOT / 'provenance/submission.json'); used = owned_bytes()
        resources = {'at_utc': now(), 'state': state, 'job_id': job, 'batch_seconds': elapsed,
            'batch_allocated_gpu_hours': elapsed * 4 / 3600, 'batch_allocated_cpu_core_hours': elapsed * 288 / 3600,
            'batch_gpu_hour_ceiling': cfg['budgets']['batch_wall_hours'] * 4,
            'preflight_allocated_gpu_hours': submission['preflight_seconds'] / 3600,
            'preflight_allocated_cpu_core_hours': submission['preflight_seconds'] * 72 / 3600,
            'additional_bytes': used, 'within_limits': elapsed <= cfg['budgets']['batch_wall_hours'] * 3600 and used <= cfg['budgets']['additional_bytes'],
            'accounting_scope': 'All allocated GPUs/CPUs including idle phases; preflight separate; queued time uncharged; prior studies excluded'}
        atomic(ROOT / 'results/resources.json', resources)
    if not resources['within_limits']:
        raise RuntimeError('Registered resource ceiling exceeded')
    files = ['metrics.json', 'decision.json', 'verification.json', 'test_predictions.npz', 'test_evidence.npz', 'REPORT.md', 'resources.json']
    atomic(ROOT / 'results/COMPLETE.json', {'at_utc': now(), 'verified': True, 'fingerprint': fp,
        'rows': cfg['test_pairs'], 'heads_sha256': cfg['heads_sha256'], 'result_hashes': {n: sha(ROOT / 'results' / n) for n in files},
        'test_authorized': True, 'test_accessed': True, 'heads_fitted': False, 'calibration_performed': False, 'shuffled_arm': False})
    atomic(ROOT / 'results/phase.json', {'at_utc': now(), 'phase': 'completed', 'fingerprint': fp})
    decision = read(ROOT / 'results/decision.json')
    (ROOT / 'STATUS.md').write_text(f"# Vx TEST benchmark status\n\nCompleted and independently verified. Fixed practical criteria: **{decision['status']}**. See [report](results/REPORT.md).\n\nNo TEST fitting, calibration or shuffled arm. This benchmark ends here.\n")


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if not (ROOT / 'results/execution-error.json').exists():
            atomic(ROOT / 'results/execution-error.json', {'at_utc': now(), 'job_id': os.environ.get('SLURM_JOB_ID'), 'traceback': traceback.format_exc()})
        raise
