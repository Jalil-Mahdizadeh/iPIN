"""One of four GPU shards; fixed TEST inputs, no labels or fitted heads."""
import fcntl
import os
import socket
import time
import traceback
from collections import Counter
import numpy as np
from shared import ROOT, read, config, now, atomic, verify_freeze
from inputs import Inputs
import chunks


def main():
    import torch
    rank = int(os.environ['SLURM_PROCID']); world = int(os.environ['SLURM_NTASKS'])
    if world != 4 or not 0 <= rank < 4 or torch.cuda.device_count() != 1:
        raise RuntimeError('Four workers with exactly one visible GPU each are required')
    torch.cuda.set_device(0); execution = read(ROOT / 'provenance/execution.json')
    device = {'rank': rank, 'world': world, 'local_rank': int(os.environ['SLURM_LOCALID']),
        'host': socket.gethostname(), 'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
        'gpu_name': torch.cuda.get_device_name(0), 'job_id': os.environ['SLURM_JOB_ID'], 'fingerprint': execution['fingerprint']}
    atomic(ROOT / 'provenance' / f'gpu-rank-{rank}.json', device)
    deadline = time.monotonic() + 600
    while not (ROOT / 'provenance/topology-approved.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Four-GPU topology approval absent')
        time.sleep(2)
    topology = read(ROOT / 'provenance/topology-approved.json')
    if topology['job_id'] != os.environ['SLURM_JOB_ID'] or topology['fingerprint'] != execution['fingerprint']:
        raise ValueError('Stale topology approval')
    lock = (ROOT / 'provenance' / f'worker-{rank}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fp = verify_freeze(); cfg = config()
    if execution['fingerprint'] != fp or execution['job_id'] != os.environ['SLURM_JOB_ID']:
        raise ValueError('Wrong frozen allocation')
    all_rows = read(ROOT / 'data/pairs.json'); ids = read(ROOT / 'data/shards.json')[rank]
    if any('label' in r or r['split'] != 'test' for r in all_rows):
        raise ValueError('Feature workers require a label-free TEST manifest')
    rows = [all_rows[i] for i in ids]; inputs = Inputs(reuse=False); encoder = None
    counters = Counter(); committed = processed = 0; started = time.monotonic()
    deadline = execution['started_unix'] + cfg['budgets']['batch_wall_hours'] * 3600 - cfg['budgets']['analysis_reserve_seconds']
    try:
        for number, begin in enumerate(range(0, len(rows), cfg['chunk_rows'])):
            part = rows[begin:begin + cfg['chunk_rows']]
            base = ROOT / 'features' / f'rank-{rank:02d}' / f'chunk-{number:04d}'
            existing = chunks.load(base, part, fp)
            if existing is None:
                records = []; true = []; quality = []
                for row in part:
                    if time.time() >= deadline:
                        raise TimeoutError('Extraction budget ended; partial-cohort evaluation forbidden')
                    before = time.monotonic(); tokens, q, info = inputs.materialize(row)
                    if tokens is not None:
                        if encoder is None:
                            from guarded_encoder import GuardedEncoder
                            encoder = GuardedEncoder()
                        t, timing = encoder.encode(tokens, info['msa']['length_a'])
                        info['kind'] = 'encoded'; info['timing'] = timing
                    else:
                        t = np.zeros(128); info['kind'] = 'fallback'; info['timing'] = None
                    info['elapsed_seconds'] = time.monotonic() - before
                    records.append(info); true.append(t); quality.append(q); processed += 1
                    if processed % 16 == 0:
                        atomic(ROOT / 'results' / f'worker-{rank}.json', {'at_utc': now(), 'rank': rank,
                            'processed': processed, 'committed': committed, 'total': len(rows),
                            'current_length': row['length'], 'elapsed_seconds': time.monotonic() - started,
                            'fingerprint': fp, 'host': socket.gethostname()})
                arrays = {'indices': np.asarray([r['index'] for r in part], dtype=np.int64),
                    'true': np.asarray(true), 'quality': np.asarray(quality),
                    'gate': np.asarray([r['gate'] for r in records]),
                    'available': np.asarray([r['available'] for r in records], dtype=bool)}
                chunks.save(base, arrays, records, part, fp); chunks.load(base, part, fp)
            else:
                arrays, records = existing; processed += len(part)
            counters.update(r['kind'] for r in records); committed += len(part)
            print({'rank': rank, 'committed': committed, 'total': len(rows), 'counts': dict(counters)}, flush=True)
        if committed != len(rows):
            raise RuntimeError('Incomplete worker')
        atomic(ROOT / 'results' / f'worker-{rank}.done.json', {'at_utc': now(), **device,
            'rows': committed, 'counts': dict(counters), 'seconds': time.monotonic() - started,
            'test_authorized': True, 'shuffle_encoded': False, 'heads_fitted': False})
    except BaseException:
        atomic(ROOT / 'results' / f'worker-{rank}-error.json', {'at_utc': now(), 'rank': rank,
            'committed': committed, 'uid': row['uid'] if 'row' in locals() else None, 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
