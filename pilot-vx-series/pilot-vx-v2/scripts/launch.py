"""Exactly one bounded SLURM submission; no inherited interactive GPU mask."""
import argparse
import json
import os
import subprocess
import time
from study import ROOT, BASE, PROJECT, config, read, sha, atomic, now, verify_freeze


def submission_environment():
    return {k: v for k, v in os.environ.items() if not (k.startswith(('SBATCH_', 'SRUN_')) or
        (k.startswith('SLURM_') and k not in ['SLURM_CONF', 'SLURM_CONF_SERVER']) or
        k in ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--submit', action='store_true'); args = ap.parse_args()
    fp = verify_freeze(); cfg = config(); budget = cfg['budgets']
    assert budget['prior_gpu_hours_charged'] + budget['v2_qualification_gpu_hours_reserved'] + budget['v2_batch_gpu_hours_maximum'] <= budget['cumulative_gpu_hours_ceiling']
    assert budget['prior_cpu_core_hours_conservatively_reserved'] + budget['v2_cpu_core_hours_reserved'] <= budget['cumulative_cpu_core_hours_ceiling']
    for rel in ['qualification/real.json', 'qualification/unit.json']: assert read(ROOT / rel)['passed']
    resource = read(ROOT / 'provenance/resource-start.json')
    if time.time() > resource['qualification_deadline_unix']: raise TimeoutError('Qualification reservation expired')
    frozen = read(ROOT / 'provenance/freeze.json')
    assert sha(PROJECT / frozen['image']) == frozen['image_sha256']
    size = sum(p.stat().st_size for folder in [BASE, ROOT, PROJECT / 'images/msa-pairformer']
               for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
    if size > budget['additional_bytes_ceiling']: raise RuntimeError('Storage budget exhausted')
    command = ['sbatch', '--parsable', f'--output={ROOT}/logs/batch-%j.log', str(ROOT / 'slurm/pilot.sbatch'), str(ROOT)]
    plan = {'at_utc': now(), 'fingerprint': fp, 'command': command, 'image_verified': True,
        'additional_bytes_at_launch': size, 'maximum_cumulative_gpu_hours': budget['prior_gpu_hours_charged'] + 1 + 10,
        'maximum_batch_gpu_hours': 10, 'qualification_gpu_hours_charged': 1, 'no_automatic_requeue': True}
    atomic(ROOT / 'provenance/launch-plan.json', plan)
    if not args.submit: print(json.dumps(plan, indent=2)); return
    if time.time() > resource['qualification_deadline_unix']: raise TimeoutError('Qualification reservation expired during checks')
    with (ROOT / 'provenance/submission-intent.json').open('x') as f:
        json.dump(plan, f, indent=2); f.flush(); os.fsync(f.fileno())
    result = subprocess.run(command, capture_output=True, text=True, env=submission_environment())
    if result.returncode:
        atomic(ROOT / 'provenance/submission-error.json', {'at_utc': now(), 'stderr': result.stderr, 'returncode': result.returncode})
        raise RuntimeError(result.stderr)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit(): raise RuntimeError('Ambiguous scheduler reply; intent retained')
    atomic(ROOT / 'provenance/submission.json', {**plan, 'job_id': job, 'scheduler_response': result.stdout.strip()})
    (ROOT / 'STATUS.md').write_text(f'# Vx v2 status\n\nQualified and frozen. Submitted as SLURM job **{job}**, four GPUs for at most 2h30m (ten GPU-hours). The job automatically extracts all features, verifies complete coverage, fits the four small heads, and writes the assessment report. No TEST access or automatic retry/continuation is enabled.\n')
    print(json.dumps({'submitted': job, 'maximum_batch_gpu_hours': 10, 'fingerprint': fp}), flush=True)


if __name__ == '__main__': main()
