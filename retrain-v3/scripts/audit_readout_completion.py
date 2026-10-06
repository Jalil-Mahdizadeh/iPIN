"""Audit finished readout workers and all retained checkpoint payloads, without inference."""
import concurrent.futures
import datetime
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    destination = ROOT / 'provenance/readout-completion-3196827.json'
    assert not destination.exists(), 'Preserve prior audit records'
    release = ROOT / 'releases/20260930-readout'
    manifest = json.loads((release / 'release.json').read_text())
    raw = subprocess.run(['sacct', '-j', '3196827', '-X',
        '--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES', '--parsable2'],
        capture_output=True, text=True, check=True).stdout
    (ROOT / 'provenance/readout-accounting-3196827.txt').write_text(raw)
    accounting = {a[0]: dict(state=a[1], exit_code=a[2], elapsed_seconds=int(a[3]), resources=a[4])
                  for a in [line.split('|') for line in raw.splitlines()[1:] if line]}
    runs, payloads = [], []
    for index, name in enumerate(manifest['enabled_runs']):
        job = f'3196827_{index}'
        job_state = accounting[job]
        assert job_state['state'] == 'COMPLETED' and job_state['exit_code'] == '0:0'
        assert 'gres/gpu=4' in job_state['resources'].split(',')
        out = ROOT / 'runs' / name
        cfg = json.loads((release / 'configs' / (name + '.json')).read_text())
        contract = json.loads((out / 'contract.json').read_text())
        stop = json.loads((out / 'stopped.json').read_text())
        screen = json.loads((out / 'SCREEN_COMPLETE.json').read_text())
        state = stop['state']
        assert cfg == contract['configuration'] and not contract['qualification'] and contract['world_size'] == 4
        assert contract['code'] == {n: manifest['files']['code/' + n] for n in contract['code']}
        assert screen['screen_completed'] and not screen['completed_full_schedule']
        assert screen['comparison_window_end'] == state['last_validation_update'] == 5000
        assert state['update'] == screen['actual_stopped_update'] and 5000 <= state['update'] <= 5005
        assert state['total_steps'] == screen['declared_schedule_updates'] == cfg['total_updates'] == 6000
        assert state['examples_seen'] == state['update'] * cfg['global_pairs_per_update']
        assert not state['pending_validation'] and stop['checkpoint_committed']
        assert screen['fingerprint'] == contract['fingerprint']
        assert not (out / 'completed.json').exists() and not (out / 'WINDOW_MONITOR_ERROR.json').exists()
        request = json.loads((out / 'WINDOW_STOP_REQUEST.json').read_text())
        assert request['target_validation_update'] == 5000 and (out / 'REQUEST_STOP').exists()
        events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
        initial = [e for e in events if e['event'] in ['initialized', 'resumed']]
        assert len(initial) == 1 and initial[0]['event'] == 'initialized' and initial[0]['checkpoint'] is None
        assert initial[0]['world_size'] == 4 and len({d['uuid'] for d in initial[0]['devices']}) == 4
        fold = int(cfg['partition'].split('-')[-1])
        assert (initial[0]['train_rows'], initial[0]['val_rows']) == [(73691, 17490), (71698, 18285), (71986, 18515)][fold]
        assert state['cursor']['cycle'] * initial[0]['train_rows'] + state['cursor']['offset'] == state['examples_seen']
        updates = [e for e in events if e['event'] == 'update']
        assert updates and updates[-1]['update'] == 5000
        for e in updates:
            assert all(math.isfinite(e[k]) for k in ['mean_loss', 'mean_bce', 'grad_norm', 'lr'])
            assert e['global_pairs'] == 64 and e['mean_mlm'] == 0
        validations = [e for e in events if e['event'] == 'validation']
        assert [e['update'] for e in validations] == [1000, 2000, 3000, 4000, 5000]
        assert events[-1]['event'] == 'stopped' and events[-1]['state'] == state
        for e in validations:
            file = out / 'validation' / f'update-{e["update"]:09d}.npz'
            meta = json.loads(file.with_suffix('.json').read_text())
            assert sha(file) == meta['sha256'] and meta['fingerprint'] == contract['fingerprint']
            assert meta['metrics'] == e['metrics'] and meta['rows'] == initial[0]['val_rows']
        log = ROOT / 'logs' / f'production-{job}.log'
        text = log.read_text()
        assert all(error not in text for error in ['Traceback', 'CUDA out of memory', 'ChildFailedError', 'AssertionError', 'NCCL WARN'])
        assert text.splitlines()[-1] == f'Run {name} finished its validation window; checkpoints retained.'
        for pointer_name in ['best', 'latest']:
            pointer = json.loads((out / (pointer_name + '.json')).read_text())
            checkpoint = out / 'checkpoints' / pointer['file']
            assert pointer == json.loads(checkpoint.with_suffix('.pt.json').read_text())
            assert pointer['fingerprint'] == contract['fingerprint']
            assert pointer['sha256'] == screen[pointer_name + '_checkpoint_sha256']
            assert pointer['update'] == (state['best_update'] if pointer_name == 'best' else state['update'])
        for checkpoint in sorted((out / 'checkpoints').glob('*.pt')):
            pointer = json.loads(checkpoint.with_suffix('.pt.json').read_text())
            assert pointer['fingerprint'] == contract['fingerprint'] and checkpoint.stat().st_size == pointer['bytes']
            payloads.append((checkpoint, pointer))
        evidence = {str(p.relative_to(ROOT)): sha(p) for p in [out / 'contract.json', out / 'stopped.json',
                    out / 'SCREEN_COMPLETE.json', out / 'best.json', out / 'latest.json', out / 'events.jsonl', log]}
        runs.append({'job_id': job, 'run': name, 'accounting': job_state, 'state': state,
                     'all_logged_training_metrics_finite': True, 'validation_updates': [e['update'] for e in validations],
                     'restarts': 0, 'maximum_logged_gpu_allocated_gib': max(e['gpu_peak_allocated_gib'] for e in updates),
                     'best_validation_metrics': next(e['metrics'] for e in validations if e['update'] == state['best_update']),
                     'final_validation_metrics': validations[-1]['metrics'], 'evidence_sha256': evidence})
    def verify_payload(item):
        path, pointer = item
        digest = sha(path)
        assert digest == pointer['sha256'], ('Checkpoint payload changed', str(path))
        print('Verified', path.relative_to(ROOT), flush=True)
        return {'path': str(path.relative_to(ROOT)), 'sha256': digest, 'bytes': pointer['bytes'], 'update': pointer['update']}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        verified = list(pool.map(verify_payload, payloads))
    report = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True,
              'array_id': '3196827', 'runs': runs, 'all_retained_payloads': verified,
              'payload_bytes_verified': sum(v['bytes'] for v in verified),
              'allocated_gpu_hours': sum(v['accounting']['elapsed_seconds'] * 4 / 3600 for v in runs),
              'screen_window_complete': True, 'full_6000_update_schedule_complete': False,
              'scope': 'All six job exits, contracts, logs, validation metadata and hashes, automatic safe stops, and SHA-256 of every retained checkpoint. Scientific prediction/selection recomputation is recorded separately.',
              'source_sha256': sha(Path(__file__)), 'release_sha256': sha(release / 'release.json')}
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print('All six jobs and', len(verified), 'retained checkpoint payloads passed the completion audit.', flush=True)


if __name__ == '__main__':
    main()
