"""Read-only startup audit; never submits, stops or modifies a training run."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
JOBS = {'esm2-native-ilp-seed2': '3301121', 'esmc-native-ilp-seed2': '3301122'}
RELEASE = ROOT / 'releases/20261002-native-ilp'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    tmp.write_text(value)
    os.replace(tmp, path)


def inspect_checkpoint(name):
    import torch
    torch.set_num_threads(2)
    out = ROOT / 'runs' / name
    contract = read(out / 'contract.json')
    meta = read(out / 'latest.json')
    assert meta['update'] >= 10 and meta['world_size'] == 4
    assert not contract['qualification'] and not contract['tiny']
    assert contract['configuration'] == read(RELEASE / 'configs' / (name + '.json'))
    path = out / 'checkpoints' / meta['file']
    assert path.stat().st_size == meta['bytes']
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024**2), b''):
            digest.update(chunk)
    assert digest.hexdigest() == meta['sha256']
    # This is our own hash-verified local checkpoint, not an external pickle.
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert payload['fingerprint'] == meta['fingerprint'] == contract['fingerprint']
    assert payload['world_size'] == len(payload['rng_by_rank']) == 4
    state = payload['training_state']
    assert state['update'] == meta['update']
    assert state['total_steps'] == 54748
    assert state['examples_seen'] == state['update'] * 64
    assert state['cursor']['cycle'] * 700764 + state['cursor']['offset'] == state['examples_seen']
    optimizer = payload['optimizer']['state']
    assert len(optimizer) >= 370
    assert all(int(value['step'].item()) == state['update'] for value in optimizer.values())
    head_keys = [key for key in payload['model'] if key.startswith('classifier.')]
    decoder_keys = [key for key in payload['model'] if '.lm_head.' in key or '.sequence_head.' in key]
    assert head_keys and decoder_keys
    assert all(bool(torch.isfinite(payload['model'][key]).all()) for key in head_keys + decoder_keys)
    assert all({'python', 'numpy', 'torch', 'cuda'} <= set(rng) for rng in payload['rng_by_rank'])
    print(json.dumps({'passed': True, 'checked_at_utc': now(), 'manifest': meta,
                      'training_state': state, 'optimizer_parameter_states': len(optimizer),
                      'rng_ranks': 4, 'classifier_tensors': len(head_keys),
                      'active_decoder_tensors': len(decoder_keys)}))


def publish(report):
    report['updated_at_utc'] = now()
    atomic(ROOT / 'provenance/startup-health.json', json.dumps(report, indent=2) + '\n')
    lines = ['# V5 production startup health', '',
             'Updated (UTC): ' + report['updated_at_utc'], '',
             'Both jobs were submitted at **2026-10-02 22:17:50 Stockholm time**, without dependencies. Each requests a separate node with four GH200 GPUs.', '',
             '| Model | Job | Scheduler state | Latest observed update | Startup audit |',
             '| --- | --- | --- | --- | --- |']
    for name, item in report['runs'].items():
        lines.append('| %s | %s | %s | %s | %s |' % (
            name, JOBS[name], item.get('scheduler_state', 'unknown'),
            item.get('last_update', 'pending'), item['status']))
    lines += ['', 'Monitor status: **' + report['status'] + '**.', '',
              'A passing audit requires four distinct GPUs, all 700,764 TRAIN and 165,742 DEV rows, at least 20 production updates with finite BCE/MLM losses and gradients, progress across observations, and a hash-verified checkpoint containing model, optimizer, four ranks of RNG and the correct dataset cursor.', '',
              'The four-GPU resume qualification passed for both models before submission. Production execution health remains pending until this report records a pass. No production validation performance is claimed; the first full DEV evaluation is at update **2,738**.', '',
              'This read-only monitor runs in the existing interactive allocation, polls once per minute, and ends after both startup audits pass, a failure is detected, or 2026-10-03 09:00 UTC. It launches no additional SLURM jobs and does not change training. Earlier termination of the interactive allocation also ends the monitor. Detailed evidence: [startup-health.json](provenance/startup-health.json).', '']
    atomic(ROOT / 'HEALTH_REPORT.md', '\n'.join(lines))


def monitor(once=False):
    lock = (ROOT / 'provenance/.startup-watch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = {'status': 'waiting for startup', 'monitor_pid': os.getpid(),
              'monitor_host': os.uname().nodename, 'started_at_utc': now(),
              'runs': {name: {'status': 'pending'} for name in JOBS}}
    prior = {}
    deadline = datetime.datetime(2026, 10, 3, 9, tzinfo=datetime.timezone.utc)
    while True:
        try:
            result = subprocess.run(['squeue', '-h', '-j', ','.join(JOBS.values()), '-o', '%i|%T|%N|%R'],
                                    capture_output=True, text=True, check=True, timeout=30)
            states = {fields[0]: fields[1:] for line in result.stdout.splitlines()
                      if (fields := line.split('|')) and len(fields) == 4}
            for name, job in JOBS.items():
                item = report['runs'][name]
                if item['status'] == 'passed':
                    continue
                scheduler = states.get(job, ['not in queue', '', ''])
                item.update(scheduler_state=scheduler[0], node=scheduler[1], reason=scheduler[2])
                out = ROOT / 'runs' / name
                if scheduler[0] == 'not in queue':
                    raise RuntimeError('Job ' + job + ' left the queue before its startup audit passed; inspect sacct and logs')
                if not (out / 'events.jsonl').exists():
                    continue
                events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines() if line.endswith('}')]
                initialized = [e for e in events if e['event'] in ['initialized', 'resumed']]
                updates = [e for e in events if e['event'] == 'update']
                if not initialized or not updates:
                    continue
                init = initialized[-1]
                assert init['world_size'] == 4 and len({d['uuid'] for d in init['devices']}) == 4
                assert init['train_rows'] == 700764 and init['val_rows'] == 165742
                assert all(e['global_pairs'] == 64 and e['masked_tokens'] > 0 for e in updates)
                assert all(math.isfinite(e[key]) for e in updates for key in ['mean_loss', 'mean_bce', 'mean_mlm', 'grad_norm'])
                assert all(e['mean_bce'] > 0 and e['mean_mlm'] > 0 and e['grad_norm'] > 0 for e in updates)
                last = updates[-1]
                item['last_update'] = last['update']
                item['last_update_utc'] = last['time_utc']
                progressed = name in prior and last['update'] > prior[name]
                prior[name] = last['update']
                if last['update'] < 20 or not progressed or scheduler[0] != 'RUNNING':
                    continue
                log = (ROOT / 'logs' / ('production-' + job + '.log')).read_text()
                assert not any(error in log for error in ['Traceback (most recent call last)', 'CUDA out of memory', 'DistBackendError', 'ChildFailedError'])
                child = subprocess.run(['bash', str(RELEASE / 'code/container.sh'), name.split('-')[0],
                    'python', str(Path(__file__).resolve()), '--checkpoint', name],
                    capture_output=True, text=True, check=True, timeout=180)
                checkpoint = json.loads(child.stdout.strip().splitlines()[-1])
                assert checkpoint['passed']
                item.update(status='passed', checked_at_utc=now(), initialized=init,
                            observed_updates=updates, checkpoint=checkpoint)
            if all(item['status'] == 'passed' for item in report['runs'].values()):
                report['status'] = 'both startup audits passed'
                publish(report)
                return
            if datetime.datetime.now(datetime.timezone.utc) >= deadline:
                report['status'] = 'monitor deadline reached; incomplete startup audit'
                publish(report)
                return
            publish(report)
            if once:
                return
            time.sleep(60)
        except Exception as error:
            report['status'] = 'inspection required'
            report['error'] = repr(error)
            if isinstance(error, subprocess.CalledProcessError):
                report['command_stderr'] = error.stderr[-6000:] if error.stderr else ''
            publish(report)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=list(JOBS))
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint) if args.checkpoint else monitor(args.once)
