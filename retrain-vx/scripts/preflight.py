"""Finish preparation, qualify, freeze and submit once; no automatic retry."""
import os
import time
import traceback
from runtime import ROOT, PROJECT, read, now, atomic, sha, run_command


def main():
    cfg = read(ROOT / 'config.json'); auth = read(ROOT / 'provenance/authorization.json')
    deadline = auth['preflight_start_unix'] + cfg['budgets']['preflight_wall_hours'] * 3600
    pins = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'README.md', ROOT / 'config.json', *sorted((ROOT / 'scripts').glob('*.py')), *sorted((ROOT / 'tests').glob('*.py')), *sorted((ROOT / 'slurm').glob('*.sbatch'))]}
    atomic(ROOT / 'provenance/preflight-start.json', {'at_utc': now(), 'pid': os.getpid(), 'code_sha256': pins})
    def unchanged():
        for name, digest in pins.items():
            if sha(ROOT / name) != digest:
                raise ValueError('Code changed during preflight: ' + name)
    for marker, process, error in [('provenance/preparation.json', 'provenance/preparation-process.json', 'results/preparation-error.json'),
                                   ('qualification/baseline.json', 'provenance/baseline-process.json', None)]:
        while not (ROOT / marker).exists():
            if time.time() > deadline or (error and (ROOT / error).exists()):
                raise RuntimeError('Preparation failed or exceeded preflight deadline')
            try:
                pid = read(ROOT / process)['pid']
                os.kill(pid, 0)
                from pathlib import Path
                stat = Path(f'/proc/{pid}/stat')
                if stat.exists() and stat.read_text().split(') ', 1)[1].split()[0] == 'Z':
                    raise ProcessLookupError('Preparation process is a zombie')
            except ProcessLookupError as exc:
                raise RuntimeError('Preparation process exited before completion: ' + marker) from exc
            time.sleep(5)
    unchanged()
    runtime = ['apptainer', 'exec', '--env', 'OPENBLAS_NUM_THREADS=8', '--env', 'OMP_NUM_THREADS=8', str(PROJECT / cfg['image']), 'python', '-B']
    gpu = ['apptainer', 'exec', '--nv', *runtime[2:]]
    run_command(gpu + [str(ROOT / 'scripts/qualify.py')], deadline, ROOT / 'logs/qualification.log')
    unchanged()
    run_command(runtime + [str(ROOT / 'scripts/freeze.py')], deadline, ROOT / 'logs/freeze.log')
    unchanged()
    run_command(['python3', '-B', str(ROOT / 'scripts/launch.py')], deadline, ROOT / 'logs/submission.log')
    atomic(ROOT / 'provenance/preflight-complete.json', {'at_utc': now(), 'passed': True,
        'fingerprint': sha(ROOT / 'provenance/freeze.json'), 'job_id': read(ROOT / 'provenance/submission.json')['job_id']})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        atomic(ROOT / 'results/preflight-error.json', {'at_utc': now(), 'traceback': traceback.format_exc()})
        (ROOT / 'STATUS.md').write_text('# Vx full TRAIN/VAL status\n\nPreflight stopped; the full run is not qualified. See results/preflight-error.json. No automatic retry or TEST evaluation.\n')
        raise
