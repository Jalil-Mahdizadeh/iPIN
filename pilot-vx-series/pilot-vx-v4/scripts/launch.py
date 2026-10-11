"""One detached launch inside the already allocated idle GPU; never resubmit a job."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# This entry point runs on the host, whose Python has no NumPy. Container nesting is unnecessary.
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def read(path):
    return json.loads(Path(path).read_text())


def config():
    return read(ROOT / 'config.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def verify_freeze():
    path = ROOT / 'provenance/freeze.json'
    for rel, digest in read(path)['files'].items():
        if sha(PROJECT / rel) != digest:
            raise ValueError('Frozen input changed: ' + rel)
    return sha(path)


def check_budget(reserve=0):
    remaining = read(ROOT / 'provenance/resource-start.json')['deadline_unix'] - time.time() - reserve
    if remaining <= 0:
        raise TimeoutError('Separate Vx v4 resource deadline exhausted')
    return remaining


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--start', action='store_true'); args = ap.parse_args()
    cfg = config(); fp = verify_freeze(); source = read(ROOT / 'provenance/inputs.json')
    qualification = read(ROOT / 'qualification/real.json'); resource = read(ROOT / 'provenance/resource-start.json')
    if not qualification['passed'] or not read(ROOT / 'qualification/unit.json')['passed']:
        raise ValueError('Qualification incomplete')
    if os.environ.get('SLURM_JOB_ID') != resource['job_id']:
        raise ValueError('Current allocation differs from the charged resource interval')
    if qualification['conservative_total_seconds'] > check_budget(cfg['budgets']['shutdown_reserve_seconds']):
        raise TimeoutError('Full qualified workload no longer fits the remaining budget')
    if sha(PROJECT / source['image']) != source['image_sha256']:
        raise ValueError('Container checksum mismatch')
    job = subprocess.check_output(['scontrol', 'show', 'job', resource['job_id']], text=True)
    if 'JobState=RUNNING' not in job or 'gres/gpu=1' not in job:
        raise RuntimeError('Expected active one-GPU allocation')
    time_left = subprocess.check_output(['squeue', '--noheader', '--jobs', resource['job_id'], '--format=%L'], text=True).strip()
    days, clock = (time_left.split('-', 1) if '-' in time_left else ('0', time_left))
    fields = list(map(int, clock.split(':')))
    remaining_allocation = int(days) * 86400 + sum(value * scale for value, scale in zip(reversed(fields), [1, 60, 3600]))
    if remaining_allocation < qualification['conservative_total_seconds'] + cfg['budgets']['shutdown_reserve_seconds']:
        raise TimeoutError('Existing allocation expires before the qualified workload fits')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if apps:
        raise RuntimeError('Allocated GPU is not idle; do not interfere with another computation')
    size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    if size >= cfg['budgets']['additional_bytes']:
        raise RuntimeError('Storage ceiling exhausted')
    command = ['apptainer', 'exec', '--nv', '--env', 'OPENBLAS_NUM_THREADS=8', '--env', 'OMP_NUM_THREADS=8',
               str(PROJECT / source['image']), 'python', '-B', str(ROOT / 'scripts/run.py')]
    plan = {'at_utc': now(), 'fingerprint': fp, 'command': command, 'job_id': resource['job_id'],
            'new_scheduler_submission': False, 'budget_hours': cfg['budgets']['allocated_gpu_hours'], 'gpu_idle_verified': True,
            'additional_bytes_at_launch': size, 'remaining_allocation_seconds': remaining_allocation, 'no_automatic_retry': True}
    atomic(ROOT / 'provenance/launch-plan.json', plan)
    if not args.start:
        print(json.dumps(plan, indent=2)); return
    with (ROOT / 'provenance/execution-intent.json').open('x') as f:
        json.dump(plan, f, indent=2); f.flush(); os.fsync(f.fileno())
    with (ROOT / 'logs/controller.log').open('x') as log:
        proc = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
    atomic(ROOT / 'provenance/launch.json', {**plan, 'controller_pid': proc.pid})
    print({'started': True, 'controller_pid': proc.pid, 'job_id': resource['job_id'], 'fingerprint': fp}, flush=True)


if __name__ == '__main__':
    main()
