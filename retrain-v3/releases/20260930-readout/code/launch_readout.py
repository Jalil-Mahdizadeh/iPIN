"""Verify and launch only the frozen readout screen; default is a dry run."""
import argparse
import datetime
import fcntl
import json
import shlex
import subprocess
from pathlib import Path
from launch import sha, verify

ROOT = Path('/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v3')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release')
    p.add_argument('--run')
    p.add_argument('--max-concurrent', type=int, choices=range(1, 7), default=6)
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    release = Path(args.release).resolve() if args.release else ROOT / 'releases' / (ROOT / 'releases/CURRENT_READOUT').read_text().strip()
    manifest = verify(release)
    assert manifest['stage'] == 'readout' and manifest['screen_stop_update'] == 5000
    runs = [args.run] if args.run else manifest['enabled_runs']
    assert all(name in manifest['enabled_runs'] for name in runs)
    for name in runs:
        out = ROOT / 'runs' / name
        assert not (out / 'SCREEN_COMPLETE.json').exists(), f'{name}: screen already complete'
        assert not (out / 'completed.json').exists(), f'{name}: full schedule already complete'
        assert not (out / 'REQUEST_STOP').exists(), f'{name}: stopped; inspect before intentional continuation'
        assert not (out / 'WINDOW_MONITOR_ERROR.json').exists(), f'{name}: investigate the monitor failure'
        if (out / 'contract.json').exists():
            contract = json.loads((out / 'contract.json').read_text())
            assert contract['configuration'] == json.loads((release / 'configs' / (name + '.json')).read_text())
            assert contract['code'] == {n: sha(release / 'code' / n) for n in contract['code']}
    command = ['sbatch', '--parsable']
    if not args.run:
        command += [f'--array=0-{len(runs)-1}%{args.max_concurrent}']
    command += [str(release / 'slurm/train.sbatch'), str(release)]
    if args.run:
        command += [args.run]
    plan = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'stage': 'readout',
            'release': str(release), 'runs': runs, 'command': command, 'shell_command': shlex.join(command),
            'dry_run': not args.submit, 'qualified': True, 'gpus_per_run': 4,
            'maximum_concurrent_runs': 1 if args.run else args.max_concurrent,
            'screen_stop_update': 5000, 'optimizer_schedule_updates': 6000,
            'launcher_path': str(Path(__file__).resolve()), 'launcher_sha256': sha(Path(__file__).resolve()),
            'production_jobs_submitted_by_this_call': 0,
            'authorization': 'User authorized ending the LR screen and moving to the next step; prior six-way concurrency preference retained.'}
    if args.submit:
        lock = (ROOT / 'provenance/readout-submission.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.run(['squeue', '--me', '--noheader', '--format=%j'], text=True, capture_output=True, check=True).stdout
        assert not any(name in active for name in ['plmi-v3-readout', 'plmi-v3-adaptation']), 'A production v3 campaign is already active'
        job = subprocess.run(command, text=True, capture_output=True, check=True).stdout.strip()
        plan.update(slurm_job_id=job, production_jobs_submitted_by_this_call=len(runs))
        destination = ROOT / 'provenance' / f'submission-{job.split(";")[0]}.json'
        assert not destination.exists()
    else:
        destination = ROOT / 'provenance/readout-launch-plan.json'
    destination.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
