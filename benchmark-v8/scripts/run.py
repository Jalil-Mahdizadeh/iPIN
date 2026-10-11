"""Bounded CPU-only execution inside the existing allocation and pinned SIF."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
HISTORY = PROJECT.parent / 'iPIN-OpenPPI'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def status(message):
    (ROOT / 'STATUS.md').write_text('# V8 status\n\n' + message + '\n')


def sources():
    vx = PROJECT / 'retrain-vx'; pilot = PROJECT / 'pilot-vx-series/pilot-vx'
    assay = HISTORY / 'experiments/independent_human_binary_v1'
    return {
        'vx_complete.json': vx / 'results/COMPLETE.json',
        'vx_freeze.json': vx / 'provenance/freeze.json',
        'vx_pairs.json': vx / 'data/pairs.json',
        'vx_proteins.json': vx / 'data/proteins.json',
        'vx_groups.json': vx / 'data/dev-groups.json',
        'vx_catalog.json': vx / 'data/monomer-catalog.json',
        'vx_heads.json': vx / 'results/heads.json',
        'vx_predictions.npz': vx / 'results/val_predictions.npz',
        'vx_evidence.npz': vx / 'results/val_evidence.npz',
        'pilot_freeze.json': pilot / 'provenance/freeze.json',
        'test_sequence_hashes.json': pilot / 'data/test-sequence-hashes.json',
        'assay_source_freeze.json': assay / 'SOURCE_FREEZE.json',
        'assay_candidates.npz': assay / 'data/candidates.npz',
        'assay_truth.npz': assay / 'private/truth.npz',
        'assay_exposure.npz': assay / 'private/exposure.npz',
        'assay_sequences.json': assay / 'data/sequences.json',
    }


def execute(command, phase, deadline, cfg):
    path = ROOT / 'logs' / (phase + '.log')
    with path.open('xb', buffering=0) as log:
        p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while p.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Four-hour CPU work ceiling reached')
                size = sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file())
                if size > cfg['additional_bytes']:
                    raise RuntimeError('Storage ceiling reached')
                time.sleep(2)
            if p.returncode:
                raise RuntimeError(f'{phase} failed ({p.returncode}); see {path}')
        except BaseException:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
            raise


def main():
    for folder in ['provenance', 'logs', 'results', 'data', 'work', 'qualification']:
        (ROOT / folder).mkdir(exist_ok=True)
    lock = (ROOT / 'provenance/run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / 'provenance/freeze.json').exists():
        raise RuntimeError('This study execution is already frozen; no automatic rerun')
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use the existing allocation, not a login node')
    cfg = read(ROOT / 'config.json'); parent = read(PROJECT / 'retrain-vx/config.json')
    started = time.monotonic(); deadline = started + cfg['wall_seconds']
    image = PROJECT / parent['image']
    status('Verifying the pinned container and input manifests. CPU-only; no new GPU allocation requested.')
    print({'phase': 'pin_inputs', 'job': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname()}, flush=True)
    image_sha = sha(image)
    if image_sha != parent['image_sha256']:
        raise RuntimeError('Parent scientific container hash changed')
    if not read(ROOT / 'qualification/metrics.json')['passed']:
        raise RuntimeError('Synthetic metric qualification has not passed')
    for p in (ROOT / 'scripts').glob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    records = {}
    source_paths = sources()
    for alias, p in source_paths.items():
        record = {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
        if p.is_relative_to(PROJECT):
            record['project_relative'] = str(p.relative_to(PROJECT))
        records[alias] = record
    code_paths = [ROOT / 'config.json', ROOT / 'README.md', ROOT / 'qualification/metrics.json',
                  *sorted((ROOT / 'scripts').glob('*.py'))]
    freeze = {'at_utc': now(), 'proposal_sha256': sha(PROJECT / 'improvment-proposal-v8.md'),
              'inputs': records, 'implementation': {str(p.relative_to(ROOT)): sha(p) for p in code_paths},
              'image': {'path': str(image), 'sha256': image_sha},
              'allocation': {'job': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
                             'cpu_threads': cfg['cpu_threads'], 'new_gpu_requests': 0},
              'test_access': False, 'no_new_model_fits': True,
              'completed_before_new_diagnostic_scores_or_external_support_counts': True}
    write(ROOT / 'provenance/freeze.json', freeze)
    command = ['apptainer', 'exec', '--cleanenv', '--containall', '--no-home', '--pwd', '/study',
               '--bind', f'{ROOT}:/study:rw', '--bind', f'{PROJECT / "improvment-proposal-v8.md"}:/proposal.md:ro']
    for alias, p in source_paths.items():
        command += ['--bind', f'{p}:/inputs/{alias}:ro']
    command += ['--env', f'OPENBLAS_NUM_THREADS={cfg["cpu_threads"]},OMP_NUM_THREADS={cfg["cpu_threads"]},PYTHONDONTWRITEBYTECODE=1', str(image), 'python', '-B']
    phase = 'prepare'
    try:
        for phase in ['prepare', 'external_audit', 'evaluate', 'verify', 'report']:
            status(f'Running `{phase}` in existing allocation {os.environ["SLURM_JOB_ID"]}, CPU-only. Inputs and scientific rules are frozen.')
            print({'phase': phase, 'started': now()}, flush=True)
            execute(command + [f'/study/scripts/{phase}.py'], phase, deadline, cfg)
            print({'phase': phase, 'complete': True}, flush=True)
        for alias, p in source_paths.items():
            if sha(p) != records[alias]['sha256']:
                raise RuntimeError('Historical input changed during execution: ' + alias)
        answer = read(ROOT / 'results/decision.json')
        resources = {'at_utc': now(), 'wall_seconds': time.monotonic() - started,
                     'cpu_thread_limit': cfg['cpu_threads'], 'allocation_id': os.environ['SLURM_JOB_ID'],
                     'no_new_gpu_allocation': True, 'new_encoder_calls': 0,
                     'execution': 'CPU-only processes in an already running interactive GPU allocation; container hashing and idle validation time included',
                     'additional_bytes': sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
                     'source_files_unchanged': True}
        write(ROOT / 'results/resources.json', resources)
        complete = not answer['further_external_audit_required']
        files = ['REPORT.md', 'validation_diagnostics.json', 'external_feasibility.json',
                 'verification.json', 'decision.json', 'resources.json']
        write(ROOT / 'results/COMPLETE.json', {'at_utc': now(), 'verified': True,
              'status': 'complete_external_gate_no_go' if complete else 'validation_complete_external_audit_pending',
              'phase_AB_early_checks_complete': True, 'all_required_work_complete': complete,
              'fingerprint': sha(ROOT / 'provenance/freeze.json'),
              'result_hashes': {name: sha(ROOT / 'results' / name) for name in files},
              'test_accessed': False, 'models_fitted': 0, 'external_inference_performed': False})
        status(('Completed and independently verified. External inference stopped at its prespecified panel-support gate.' if complete
                else 'Validation diagnostics are verified; external class/length gates pass, so further MSA and component qualification remains.')
               + '\n\nSee [the report](results/REPORT.md). No training or TEST evaluation was performed.')
        print({'complete': complete, 'decision': answer, 'wall_seconds': resources['wall_seconds']}, flush=True)
    except BaseException as exc:
        write(ROOT / 'results/ERROR.json', {'at_utc': now(), 'phase': phase,
              'exception': str(exc), 'traceback': traceback.format_exc(), 'accepted_completion': False})
        status(f'Execution stopped in `{phase}`. No incomplete result is accepted. See `results/ERROR.json` and `logs/{phase}.log`.')
        raise


if __name__ == '__main__':
    main()
