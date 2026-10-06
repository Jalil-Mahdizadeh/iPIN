"""Dry-run the three v4 jobs by default; explicit --submit is required to launch."""
import argparse
import datetime
import fcntl
import json
import os
import shlex
import subprocess
from pathlib import Path
from contracts import RUNS, sha256, verify_inputs
from runtime import ROOT, verify_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release'); p.add_argument('--run', choices=RUNS)
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    lock = (ROOT / 'provenance/.launch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    release = Path(args.release).resolve() if args.release else ROOT / 'releases' / (ROOT / 'releases/CURRENT').read_text().strip()
    runs = [args.run] if args.run else RUNS
    active = subprocess.run(['squeue', '--me', '--noheader', '--format=%j'],
                            capture_output=True, text=True, check=True).stdout.splitlines()
    active = {line.strip() for line in active}
    commands = []
    for name in runs:
        verify_run(release, name, verify_assets=False)
        cfg = json.loads((release / 'configs' / (name + '.json')).read_text())
        out = ROOT / 'runs' / name
        assert not (out / 'completed.json').exists(), (name, 'Already complete; benchmark separately')
        assert not (out / 'REQUEST_STOP').exists(), (name, 'Manually stopped; retain state until intentional continuation')
        job_name = 'plmi-v4-' + cfg['arm']
        assert job_name not in active, (job_name, 'Already queued or running')
        commands.append(['sbatch', '--parsable', '--job-name=' + job_name,
                         str(release / 'slurm/train.sbatch'), str(release), name])
    verify_inputs(ROOT)
    plan = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'dry_run': not args.submit, 'release': str(release),
            'release_sha256': sha256(release / 'release.json'), 'runs': runs,
            'commands': commands, 'shell_commands': [shlex.join(c) for c in commands],
            'gpus_per_run': 4, 'maximum_concurrent_runs': len(runs),
            'production_jobs_submitted_by_this_call': 0,
            'automatic_benchmark': False, 'reused_control_retrained': False,
            'qualification_passed': True, 'slurm_job_ids': []}
    destination = ROOT / 'provenance/production-launch-plan.json'
    if args.submit:
        destination = ROOT / 'provenance' / ('submission-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')
        for command in commands:
            job = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
            plan['slurm_job_ids'].append(job)
            plan['production_jobs_submitted_by_this_call'] += 1
            with destination.open('w') as f:
                json.dump(plan, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    destination.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__': main()
