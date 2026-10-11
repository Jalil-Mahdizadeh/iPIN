"""Submit the single authorized four-GPU benchmark only after all checks."""
import fcntl
import subprocess
import time
from runtime import ROOT, PROJECT, read, now, sha, atomic, verify_files, owned_bytes


def main():
    lock = (ROOT / 'provenance/launch.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((ROOT / 'provenance' / n).exists() for n in ['submission-intent.json', 'submission.json', 'execution.json']):
        raise RuntimeError('Submission already attempted; no automatic duplicate/retry')
    cfg = read(ROOT / 'config.json'); authorization = read(ROOT / 'provenance/authorization.json')
    if not cfg['test_authorized'] or not authorization['test_authorized']:
        raise ValueError('Explicit TEST authorization missing')
    fp = verify_files()
    assert sha(PROJECT / cfg['image']) == cfg['image_sha256']
    assert read(ROOT / 'qualification/unit.json')['passed'] and read(ROOT / 'qualification/encoder.json')['passed']
    assert owned_bytes() <= cfg['budgets']['additional_bytes']
    seconds = time.time() - authorization['preflight_start_unix']
    if seconds > cfg['budgets']['preflight_wall_hours'] * 3600:
        raise TimeoutError('Preparation/qualification allocation ceiling exceeded')
    command = ['sbatch', '--parsable', '--chdir=' + str(PROJECT),
        '--output=' + str(ROOT / 'logs/benchmark-%j.log'), str(ROOT / 'slurm/benchmark.sbatch'), str(ROOT)]
    check = subprocess.run(command[:1] + ['--test-only'] + command[1:], check=True, capture_output=True, text=True)
    atomic(ROOT / 'provenance/submission-intent.json', {'at_utc': now(), 'fingerprint': fp,
        'command': command, 'test_only_stdout': check.stdout, 'test_only_stderr': check.stderr,
        'nodes': 1, 'gpus': 4, 'cpus': 288, 'wall_hours': cfg['budgets']['batch_wall_hours']})
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(';')[0]
    if not job_id.isdigit():
        raise RuntimeError('Submission result unclear; inspect Slurm before any further action')
    atomic(ROOT / 'provenance/submission.json', {'at_utc': now(), 'job_id': job_id, 'fingerprint': fp,
        'nodes': 1, 'gpus': 4, 'cpus': 288, 'wall_hours': cfg['budgets']['batch_wall_hours'],
        'preflight_seconds': time.time() - authorization['preflight_start_unix'], 'command': command,
        'test_authorized': True, 'heads_fitted': False, 'calibration_performed': False, 'shuffled_arm': False})
    (ROOT / 'STATUS.md').write_text(f'# Vx TEST benchmark status\n\nSubmitted job **{job_id}**: one exclusive node, four GPUs, 24-hour limit. All 52,048 original TEST pairs; frozen TRAIN/VAL heads and fusion weights. No fitting, calibration or shuffled arm.\n')
    print({'submitted': job_id, 'fingerprint': fp, 'nodes': 1, 'gpus': 4}, flush=True)


if __name__ == '__main__':
    main()
