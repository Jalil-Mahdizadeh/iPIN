"""Bounded preparation wait, qualification, freeze and one authorized submission."""
import os
import socket
import time
import traceback
from pathlib import Path
from runtime import ROOT, PROJECT, read, now, atomic, sha, run_command


def alive(pid):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[0]
        return state not in ['Z', 'X']
    except FileNotFoundError:
        return False


def main():
    if (ROOT / 'provenance/preflight-start.json').exists():
        raise RuntimeError('Preflight already attempted; no automatic retry')
    cfg = read(ROOT / 'config.json'); authorization = read(ROOT / 'provenance/authorization.json')
    deadline = authorization['preflight_start_unix'] + cfg['budgets']['preflight_wall_hours'] * 3600
    paths = [ROOT / 'README.md', ROOT / 'config.json']
    paths += list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'tests').glob('*.py')) + list((ROOT / 'slurm').glob('*.sbatch'))
    pins = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    atomic(ROOT / 'provenance/preflight-start.json', {'at_utc': now(), 'pid': os.getpid(), 'host': socket.gethostname(), 'code': pins})
    preparation = read(ROOT / 'provenance/preparation-process.json')
    assert preparation['host'] == socket.gethostname()
    while not (ROOT / 'provenance/preparation.json').exists():
        if time.time() >= deadline:
            raise TimeoutError('Preparation/qualification deadline reached')
        if (ROOT / 'results/preparation-error.json').exists():
            raise RuntimeError('Preparation failed; inspect results/preparation-error.json')
        if not alive(preparation['pid']):
            raise RuntimeError('Preparation exited without successful completion')
        time.sleep(5)
    for name, digest in pins.items():
        assert sha(ROOT / name) == digest, name
    image = str(PROJECT / cfg['image'])
    base = ['apptainer', 'exec', '--env', 'OPENBLAS_NUM_THREADS=8', '--env', 'OMP_NUM_THREADS=8', image, 'python', '-B']
    gpu = ['apptainer', 'exec', '--nv', '--env', 'OPENBLAS_NUM_THREADS=8', '--env', 'OMP_NUM_THREADS=8', image, 'python', '-B']
    run_command(gpu + [str(ROOT / 'scripts/qualify.py')], deadline, ROOT / 'logs/qualification.log')
    run_command(base + [str(ROOT / 'scripts/freeze.py')], deadline, ROOT / 'logs/freeze.log')
    for name, digest in pins.items():
        assert sha(ROOT / name) == digest, name
    run_command(['python3', '-B', str(ROOT / 'scripts/launch.py')], deadline, ROOT / 'logs/launch.log')
    atomic(ROOT / 'provenance/preflight-complete.json', {'at_utc': now(), 'passed': True,
        'job_id': read(ROOT / 'provenance/submission.json')['job_id'], 'fingerprint': sha(ROOT / 'provenance/freeze.json')})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        atomic(ROOT / 'results/preflight-error.json', {'at_utc': now(), 'traceback': traceback.format_exc()})
        (ROOT / 'STATUS.md').write_text('# Vx TEST benchmark status\n\nPreflight stopped with an error. See results/preflight-error.json. No automatic retry; no new TEST performance has been accepted.\n')
        raise
