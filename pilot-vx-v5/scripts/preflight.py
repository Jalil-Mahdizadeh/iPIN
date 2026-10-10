"""Finish active preparation, freeze/test, and launch once in the existing allocation."""
import argparse
import fcntl
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path
from launch import ROOT, PROJECT, read, sha, atomic, now, check_budget


def process_stamp(pid):
    try:
        # Fields after comm start at field 3; starttime is field 22.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def validate_preparation(inputs, cfg):
    expected = cfg['expected']
    if (inputs['required_pairs'] != expected['eligible_train'] + expected['eligible_dev']
            or inputs['counts'] != expected or not 0 < inputs['increased_depth_pairs'] <= inputs['required_pairs']
            or any(inputs[k] for k in ('test_accessed', 'outcome_heads_fitted', 'R_evaluated', 'labels_used_for_sampling'))):
        raise ValueError('Preparation did not satisfy the frozen design')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preparation-pid', type=int, required=True)
    args = parser.parse_args(); cfg = read(ROOT / 'config.json'); resource = read(ROOT / 'provenance/resource-start.json')
    lock = (ROOT / 'provenance/preflight.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / 'provenance/preflight-start.json').exists():
        raise RuntimeError('Preflight already started; no automatic retry')
    if os.environ.get('SLURM_JOB_ID') != resource['job_id']:
        raise ValueError('Wrong allocation')
    stamp = process_stamp(args.preparation_pid)
    if not (ROOT / 'provenance/inputs.json').exists():
        command = Path(f'/proc/{args.preparation_pid}/cmdline').read_bytes().replace(b'\0', b' ')
        if stamp is None or b'pilot-vx-v5/scripts/prepare.py' not in command:
            raise ValueError('Wrong preparation process')
    paths = [ROOT / 'README.md', ROOT / 'config.json', ROOT / 'provenance/proposal-at-start.md',
             *sorted((ROOT / 'scripts').glob('*.py')), *sorted((ROOT / 'tests').glob('*.py'))]
    pins = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    atomic(ROOT / 'provenance/preflight-start.json', {'at_utc': now(), 'pid': os.getpid(), 'job_id': resource['job_id'],
        'preparation_pid': args.preparation_pid, 'preparation_process_start_ticks': stamp,
        'code_and_protocol_sha256': pins, 'R_evaluated': False, 'test_accessed': False})
    def unchanged():
        for rel, digest in pins.items():
            if sha(ROOT / rel) != digest:
                raise ValueError('Protocol/code changed during preparation: ' + rel)
    def stopped(signum, frame):
        raise RuntimeError('Preflight interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, stopped); signal.signal(signal.SIGINT, stopped)
    proc = None
    try:
        while not (ROOT / 'provenance/inputs.json').exists():
            remaining = check_budget(cfg['budgets']['shutdown_reserve_seconds'])
            if process_stamp(args.preparation_pid) != stamp:
                raise RuntimeError('Preparation exited without complete verified inputs; inspect logs/preparation.log')
            count = len(list((ROOT / 'data/depth_inputs').glob('*.npz')))
            phase = {'at_utc': now(), 'phase': 'preparation', 'prepared_inputs': count, 'required_inputs': 4453,
                     'gpu_qualification_passed': False, 'automatic_freeze_then_launch': True}
            atomic(ROOT / 'results/phase.json', phase); print(phase, flush=True)
            time.sleep(15)
        unchanged()
        inputs = read(ROOT / 'provenance/inputs.json')
        validate_preparation(inputs, cfg)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if apps: raise RuntimeError('GPU is busy; do not interfere')
        atomic(ROOT / 'results/phase.json', {'at_utc': now(), 'phase': 'GPU_qualification'})
        command = ['apptainer', 'exec', '--nv', '--env', 'OPENBLAS_NUM_THREADS=8', '--env', 'OMP_NUM_THREADS=8',
                   str(PROJECT / inputs['image']), 'python', '-B', str(ROOT / 'scripts/qualify.py')]
        with (ROOT / 'logs/qualification.log').open('x') as log:
            proc = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                check_budget(cfg['budgets']['shutdown_reserve_seconds']); time.sleep(2)
            if proc.returncode: raise RuntimeError('GPU qualification failed; inspect logs/qualification.log')
        proc = None; unchanged()
        atomic(ROOT / 'results/phase.json', {'at_utc': now(), 'phase': 'freeze_and_unit_tests'})
        command = ['apptainer', 'exec', '--env', 'OPENBLAS_NUM_THREADS=8', '--env', 'OMP_NUM_THREADS=8',
                   str(PROJECT / inputs['image']), 'python', '-B', str(ROOT / 'scripts/freeze.py')]
        with (ROOT / 'logs/freeze.log').open('x') as log:
            proc = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                check_budget(cfg['budgets']['shutdown_reserve_seconds']); time.sleep(2)
            if proc.returncode:
                raise RuntimeError('Freeze/tests failed; inspect logs/freeze.log and qualification/unit.log')
        proc = None; unchanged()
        with (ROOT / 'logs/launch.log').open('x') as log:
            subprocess.run(['python', '-B', str(ROOT / 'scripts/launch.py'), '--start'], cwd=PROJECT,
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=min(600, check_budget(cfg['budgets']['shutdown_reserve_seconds'])))
        launch = read(ROOT / 'provenance/launch.json')
        atomic(ROOT / 'provenance/preflight-complete.json', {'at_utc': now(), 'fingerprint': launch['fingerprint'],
            'controller_pid': launch['controller_pid'], 'all_preflight_checks_passed': True})
        print({'preflight_complete': True, 'controller_pid': launch['controller_pid'], 'fingerprint': launch['fingerprint']}, flush=True)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        if stamp is not None and process_stamp(args.preparation_pid) == stamp and not (ROOT / 'provenance/inputs.json').exists():
            os.kill(args.preparation_pid, signal.SIGTERM)
        error = {'at_utc': now(), 'error': str(exc), 'traceback': traceback.format_exc(), 'test_accessed': False,
                 'R_evaluated': False, 'automatic_retry': False, 'status': 'inconclusive_preflight_failure'}
        atomic(ROOT / 'results/preflight-error.json', error)
        atomic(ROOT / 'results/phase.json', {'at_utc': now(), 'phase': 'preflight_failed'})
        elapsed = time.time() - resource['started_unix']
        atomic(ROOT / 'results/preflight-resources.json', {'at_utc': now(), 'elapsed_seconds_including_preparation': elapsed,
            'allocated_gpu_hours_charged': elapsed * resource['allocated_gpu_count'] / 3600,
            'allocated_cpu_core_hours_charged': elapsed * resource['allocated_cpus'] / 3600,
            'additional_bytes': sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())})
        (ROOT / 'STATUS.md').write_text('# Vx v5 status\n\nPreflight stopped without a completed study. See [the error](results/preflight-error.json). No automatic retry or incomplete-cohort fitting. R remains deferred.\n')
        raise


if __name__ == '__main__':
    main()
