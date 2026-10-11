"""One explicit two-node/eight-GPU submission after every preflight passes."""
import fcntl
import subprocess
import time
from runtime import ROOT, PROJECT, read, now, atomic, verify_files, sha, owned_bytes


def main():
    lock = (ROOT / 'provenance/launch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((ROOT / 'provenance' / n).exists() for n in ['submission-intent.json', 'submission.json', 'execution.json']):
        raise RuntimeError('Already submitted or launch intent exists; no automatic retry')
    cfg = read(ROOT / 'config.json'); fp = verify_files()
    for name in ['unit.json', 'encoder.json', 'baseline.json']:
        assert read(ROOT / 'qualification' / name)['passed']
    assert cfg['world_size'] == 8 and cfg['nodes'] == 2 and cfg['gpus_per_node'] == 4 and cfg['no_shuffle'] and not cfg['test_authorized']
    assert sha(PROJECT / cfg['image']) == cfg['image_sha256']
    preflight_seconds = time.time() - read(ROOT / 'provenance/authorization.json')['preflight_start_unix']
    if preflight_seconds > cfg['budgets']['preflight_wall_hours'] * 3600 or owned_bytes() > cfg['budgets']['additional_bytes']:
        raise RuntimeError('Preflight/storage ceiling exceeded')
    sbatch = ['sbatch', '--parsable', '--output=' + str(ROOT / 'logs/trainval-%j.log'), str(ROOT / 'slurm/trainval.sbatch'), str(ROOT)]
    preview = subprocess.run([sbatch[0], '--test-only', *sbatch[2:]], capture_output=True, text=True, check=True)
    atomic(ROOT / 'provenance/submission-intent.json', {'at_utc': now(), 'command': sbatch, 'fingerprint': fp,
        'nodes': 2, 'gpus_per_node': 4, 'allocated_cpu_cores': 576, 'no_shuffle': True, 'test_authorized': False,
        'scheduler_preview': preview.stdout + preview.stderr})
    response = subprocess.run(sbatch, capture_output=True, text=True, check=True)
    job = response.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Unexpected scheduler submission response')
    atomic(ROOT / 'provenance/submission.json', {'at_utc': now(), 'job_id': job, 'fingerprint': fp, 'command': sbatch,
        'nodes': 2, 'gpus': 8, 'preflight_seconds': time.time() - read(ROOT / 'provenance/authorization.json')['preflight_start_unix'],
        'no_automatic_requeue': True, 'test_authorized': False})
    # A very fast scheduler can start the job before sbatch returns; avoid
    # overwriting its newer extraction status with a stale queued message.
    if not (ROOT / 'provenance/execution.json').exists():
        (ROOT / 'STATUS.md').write_text(f"# Vx full TRAIN/VAL status\n\nSubmitted as job **{job}**: two complete nodes, four GPUs each, {cfg['budgets']['batch_wall_hours']}-hour limit. Preflight passed. No shuffled arm or TEST evaluation.\n")
    print({'submitted_job': job, 'nodes': 2, 'gpus': 8, 'fingerprint': fp}, flush=True)


if __name__ == '__main__':
    main()
