"""Require every frozen row and every successful worker before any fit."""
import numpy as np
from shared import read, ARMS
import chunks


def collect(root, rows, shards, fingerprint, chunk_rows, require_workers=True):
    if sorted(i for shard in shards for i in shard) != list(range(len(rows))):
        raise ValueError('Shard omission or duplication')
    out = {k: np.zeros((len(rows), dim)) for k, dim in ARMS.items()}
    out.update(gate=np.zeros(len(rows)), available=np.zeros(len(rows), dtype=bool))
    records = [None] * len(rows); seen = np.zeros(len(rows), dtype=bool); workers = []
    for rank, ids in enumerate(shards):
        if require_workers:
            p = root / 'results' / f'worker-{rank}.done.json'
            if not p.exists():
                raise ValueError('Worker not completed: ' + str(rank))
            w = read(p)
            if w['fingerprint'] != fingerprint or w['rank'] != rank or w['world'] != 8 or w['rows'] != len(ids):
                raise ValueError('Wrong worker coverage/fingerprint')
            if w['test_accessed'] or w['shuffle_encoded']:
                raise ValueError('Worker violated experiment scope')
            workers.append(w)
        expected_chunks = (len(ids) + chunk_rows - 1) // chunk_rows
        actual = list((root / 'features' / f'rank-{rank:02d}').glob('chunk-*.npz'))
        if len(actual) != expected_chunks:
            raise ValueError('Missing or extra feature chunks')
        for number, begin in enumerate(range(0, len(ids), chunk_rows)):
            part = [rows[i] for i in ids[begin:begin + chunk_rows]]
            saved = chunks.load(root / 'features' / f'rank-{rank:02d}' / f'chunk-{number:04d}', part, fingerprint)
            if saved is None:
                raise ValueError('Missing feature chunk')
            arrays, meta = saved; indices = arrays['indices']
            if seen[indices].any():
                raise ValueError('Duplicate feature row')
            for k in out:
                out[k][indices] = arrays[k]
            for i, m in zip(indices, meta):
                records[int(i)] = m
            seen[indices] = True
    if not seen.all() or any(r is None for r in records):
        raise ValueError('Incomplete feature coverage')
    if require_workers:
        if len({w['gpu_uuid'] for w in workers}) != 8:
            raise ValueError('Eight distinct GPUs not demonstrated')
        hosts = {w['host'] for w in workers}
        if len(hosts) != 2 or any(sum(w['host'] == h for w in workers) != 4 for h in hosts):
            raise ValueError('Expected four GPU workers on each of two nodes')
    return out, records, workers
