"""Every original TEST row and four successful GPU workers are mandatory."""
import numpy as np
from shared import ARMS, read
import chunks


def collect(root, rows, shards, fingerprint, chunk_rows, require_workers=True):
    if len(shards) != 4 or sorted(i for shard in shards for i in shard) != list(range(len(rows))):
        raise ValueError('Exactly four complete disjoint shards required')
    if any(r['index'] != i or r['split'] != 'test' or 'label' in r for i, r in enumerate(rows)):
        raise ValueError('Invalid label-free TEST manifest')
    out = {k: np.zeros((len(rows), d)) for k, d in ARMS.items()}
    out.update(gate=np.zeros(len(rows)), available=np.zeros(len(rows), dtype=bool))
    records = [None] * len(rows); seen = np.zeros(len(rows), dtype=bool); workers = []
    execution = read(root / 'provenance/execution.json') if require_workers else None
    for rank, ids in enumerate(shards):
        if require_workers:
            p = root / 'results' / f'worker-{rank}.done.json'
            if not p.exists():
                raise ValueError('Worker not completed: ' + str(rank))
            w = read(p)
            if (w['fingerprint'] != fingerprint or w['rank'] != rank or w['world'] != 4 or
                w['rows'] != len(ids) or w['job_id'] != execution['job_id']):
                raise ValueError('Wrong worker coverage/fingerprint/allocation')
            if not w['test_authorized'] or w['shuffle_encoded'] or w['heads_fitted']:
                raise ValueError('Worker violated benchmark scope')
            workers.append(w)
        expected = (len(ids) + chunk_rows - 1) // chunk_rows
        actual = list((root / 'features' / f'rank-{rank:02d}').glob('chunk-*.npz'))
        if len(actual) != expected:
            raise ValueError('Missing or extra feature chunks')
        for number, begin in enumerate(range(0, len(ids), chunk_rows)):
            part = [rows[i] for i in ids[begin:begin + chunk_rows]]
            saved = chunks.load(root / 'features' / f'rank-{rank:02d}' / f'chunk-{number:04d}', part, fingerprint)
            if saved is None:
                raise ValueError('Missing feature chunk')
            arrays, meta = saved; indices = arrays['indices']
            if seen[indices].any():
                raise ValueError('Duplicate feature row')
            for record in meta:
                if record['kind'] not in ['encoded', 'fallback']:
                    raise ValueError('Unexpected TEST feature source')
                if not record['available'] and record['reason'] not in ['missing_exact_query', 'long_query', 'low_quality_coverage', 'shallow_paired_alignment']:
                    raise ValueError('Computational failure disguised as biological fallback')
            for key in out:
                out[key][indices] = arrays[key]
            for i, record in zip(indices, meta):
                records[int(i)] = record
            seen[indices] = True
    if not seen.all() or any(r is None for r in records):
        raise ValueError('Incomplete TEST coverage')
    if require_workers:
        if len({w['gpu_uuid'] for w in workers}) != 4 or len({w['host'] for w in workers}) != 1:
            raise ValueError('Four distinct GPUs on one node not demonstrated')
        approved = read(root / 'provenance/topology-approved.json')
        if approved['job_id'] != execution['job_id'] or approved['fingerprint'] != fingerprint:
            raise ValueError('Stale topology evidence')
        for w in workers:
            record = next(r for r in approved['workers'] if r['rank'] == w['rank'])
            if record['host'] != w['host'] or record['gpu_uuid'] != w['gpu_uuid']:
                raise ValueError('Worker changed device or node')
    return out, records, workers
