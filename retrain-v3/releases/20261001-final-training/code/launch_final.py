"""Submit/resume one final training job; no automatic benchmarking."""
import argparse
import datetime
import fcntl
import json
import shlex
import subprocess
from pathlib import Path
from final_runtime import ROOT, RUN, verify_final
from launch import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    lock = (ROOT / 'provenance/.final-submit.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    release = Path(args.release).resolve() if args.release else ROOT / 'releases' / (ROOT / 'releases/CURRENT_FINAL').read_text().strip()
    verify_final(release)
    run = ROOT / 'runs' / RUN
    assert not (run / 'completed.json').exists(), 'Final training is already complete; benchmark separately later'
    assert not (run / 'REQUEST_STOP').exists(), 'Run manually stopped; inspect before intentional continuation'
    if (run / 'contract.json').exists():
        contract = json.loads((run / 'contract.json').read_text())
        assert contract['configuration'] == json.loads((release / 'configs' / f'{RUN}.json').read_text())
        assert contract['code'] == {n: sha(release / 'code' / n) for n in contract['code']}
    active = subprocess.run(['squeue', '--me', '--noheader', '--format=%j'], text=True, capture_output=True, check=True).stdout.splitlines()
    assert 'plmi-v3-final' not in [x.strip() for x in active], 'Final pipeline is already queued or running'
    command = ['sbatch', '--parsable', str(release / 'slurm/train.sbatch'), str(release)]
    plan = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'dry_run': not args.submit,
        'release': str(release), 'release_sha256': sha(release / 'release.json'), 'runs': [RUN],
        'command': command, 'shell_command': shlex.join(command), 'gpus': 4,
        'training_models': 1, 'automatic_final_benchmark': False, 'resume_from_complete_checkpoint': True,
        'benchmark_deferred_at_user_request': True, 'qualified': True}
    path = ROOT / 'provenance/final-launch-plan.json'
    if args.submit:
        job = subprocess.run(command, text=True, capture_output=True, check=True).stdout.strip()
        plan['slurm_job_id'] = job
        path = ROOT / 'provenance' / f'final-submission-{job.split(";")[0]}.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__': main()
