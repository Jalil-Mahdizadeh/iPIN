"""Stop a screening run after its committed validation window; keep trainer state intact."""
import argparse
import datetime
import hashlib
import json
import os
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validated(out, target):
    val = out / 'validation' / f'update-{target:09d}.json'
    latest = out / 'latest.json'
    if not val.exists() or not latest.exists():
        return False
    meta, checkpoint = json.loads(val.read_text()), json.loads(latest.read_text())
    contract = json.loads((out / 'contract.json').read_text())
    assert meta['fingerprint'] == checkpoint['fingerprint'] == contract['fingerprint']
    assert meta['update'] == target
    if checkpoint['update'] < target:
        return False
    assert sha(val.with_suffix('.npz')) == meta['sha256'], 'Validation artifact changed'
    return True


def watch(out, target, worker_pid):
    try:
        while True:
            try:
                os.kill(worker_pid, 0)
            except ProcessLookupError:
                return
            if (out / 'REQUEST_STOP').exists():
                return  # Preserve a user's stop; never clear or replace it.
            if validated(out, target):
                write(out / 'WINDOW_STOP_REQUEST.json', {
                    'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'target_validation_update': target,
                    'reason': 'Prespecified common comparison window completed; keep the original optimizer schedule.'})
                (out / 'REQUEST_STOP').write_text(f'Automatic readout screen stop after validation {target}.\n')
                return
            time.sleep(.5)
    except Exception as error:
        write(out / 'WINDOW_MONITOR_ERROR.json', {'error': repr(error)})
        (out / 'REQUEST_STOP').write_text('Window monitor failed; checkpoint and stop for inspection.\n')
        raise


def finish(out, target):
    if (out / 'WINDOW_MONITOR_ERROR.json').exists():
        raise RuntimeError('The validation-window monitor failed; inspect its error report')
    stopped = out / 'stopped.json'
    if not stopped.exists():
        return False
    stop = json.loads(stopped.read_text())
    state = stop['state']
    if (state.get('last_validation_update') or 0) < target or state['pending_validation']:
        return False
    assert stop['checkpoint_committed'] and validated(out, target)
    assert target <= state['update'] <= target + 5, 'Unexpected overshoot of the screening budget'
    latest, best = [json.loads((out / f'{name}.json').read_text()) for name in ['latest', 'best']]
    assert latest['update'] == state['update'] and best['update'] == state['best_update'] <= target
    assert best['update'] % 1000 == 0 or target < 1000  # Small fixtures/qualification are explicit.
    for pointer in [latest, best]:
        path = out / 'checkpoints' / pointer['file']
        assert path.is_file() and path.stat().st_size == pointer['bytes']
        assert json.loads(path.with_suffix(path.suffix + '.json').read_text()) == pointer
    write(out / 'SCREEN_COMPLETE.json', {
        'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'screen_completed': True, 'comparison_window_end': target,
        'actual_stopped_update': state['update'], 'declared_schedule_updates': state['total_steps'],
        'fingerprint': latest['fingerprint'], 'checkpoint_committed': True,
        'completed_full_schedule': False, 'best_update': best['update'],
        'best_ap': best['best_ap'], 'best_checkpoint_sha256': best['sha256'],
        'latest_checkpoint_sha256': latest['sha256']})
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['watch', 'finish'])
    p.add_argument('--release', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--worker-pid', type=int)
    args = p.parse_args()
    release = Path(args.release).resolve()
    manifest = json.loads((release / 'release.json').read_text())
    assert args.run in manifest['enabled_runs']
    cfg = json.loads((release / 'configs' / (args.run + '.json')).read_text())
    out = Path(cfg['root']) / 'runs' / args.run
    target = manifest['screen_stop_update']
    assert 0 < target < cfg['total_updates'] and target % cfg['validate_every_updates'] == 0
    if args.mode == 'watch':
        assert args.worker_pid and args.worker_pid > 1
        watch(out, target, args.worker_pid)
        return 0
    return 0 if finish(out, target) else 3


if __name__ == '__main__':
    raise SystemExit(main())
