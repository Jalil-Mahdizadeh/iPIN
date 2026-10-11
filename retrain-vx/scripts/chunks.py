"""Atomic bounded feature chunks. Partial/corrupt output never becomes fallback."""
import gzip
import json
import os
from pathlib import Path
import numpy as np
from shared import atomic, read, sha, save_npz, ARMS


def validate(arrays, records, rows):
    if set(arrays) != {'indices', 'gate', 'available', *ARMS} or len(records) != len(rows):
        raise ValueError('Chunk arms/count mismatch')
    np.testing.assert_array_equal(arrays['indices'], [r['index'] for r in rows])
    for arm, dim in ARMS.items():
        x = arrays[arm]
        if x.shape != (len(rows), dim) or not np.isfinite(x).all():
            raise ValueError('Invalid feature dimensions/numerics')
    gate = arrays['gate']; available = arrays['available']
    if gate.shape != (len(rows),) or available.shape != gate.shape or available.dtype != np.bool_:
        raise ValueError('Invalid gate/availability shape')
    if not np.isfinite(gate).all() or np.any((gate < 0) | (gate > 1)):
        raise ValueError('Invalid gate')
    np.testing.assert_array_equal(gate > 0, available)
    if arrays['true'][~available].any():
        raise ValueError('Unavailable true evidence must be zero')
    for i, (record, row) in enumerate(zip(records, rows)):
        if record['uid'] != row['uid'] or record['index'] != row['index'] or [record['a'], record['b']] != sorted([row['a'], row['b']]):
            raise ValueError('Chunk row identity mismatch')
        if record['gate'] != gate[i] or record['available'] != bool(available[i]) or record['length'] != row['length']:
            raise ValueError('Chunk metadata mismatch')
        if available[i] and (not record['token_sha256'] or not 9 <= record['depth'] <= 128):
            raise ValueError('Missing/invalid MSA provenance')
        if record['kind'] not in ['encoded', 'reused_original', 'fallback']:
            raise ValueError('Unknown feature source')
        if available[i] and record['kind'] == 'fallback':
            raise ValueError('Computed evidence replaced by fallback')
        if record['kind'] == 'encoded':
            t = record['timing']
            if t['orientation_depths'] != [record['depth']] * 2 or t['layer_indices'] != list(range(16)) * 2:
                raise ValueError('Encoder execution audit mismatch')


def paths(base):
    return base.with_suffix('.npz'), base.with_suffix('.metadata.json.gz'), base.with_suffix('.json')


def save(base, arrays, records, rows, fingerprint):
    base = Path(base); base.parent.mkdir(parents=True, exist_ok=True); a, m, c = paths(base)
    if any(p.exists() for p in [a, m, c]):
        raise RuntimeError('Refuse to overwrite existing feature chunk')
    validate(arrays, records, rows); save_npz(a, **arrays)
    temp = m.with_name(m.name + f'.tmp-{os.getpid()}')
    with temp.open('wb') as f:
        with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as gz:
            gz.write(json.dumps(records, separators=(',', ':'), allow_nan=False).encode())
        f.flush(); os.fsync(f.fileno())
    temp.replace(m)
    atomic(c, {'fingerprint': fingerprint, 'rows': len(rows), 'arrays_sha256': sha(a), 'metadata_sha256': sha(m)})


def load(base, rows, fingerprint):
    a, m, c = paths(Path(base)); exists = [p.exists() for p in [a, m, c]]
    if not any(exists):
        return None
    if not all(exists):
        raise ValueError('Partial feature chunk')
    meta = read(c)
    if meta['fingerprint'] != fingerprint or meta['rows'] != len(rows) or sha(a) != meta['arrays_sha256'] or sha(m) != meta['metadata_sha256']:
        raise ValueError('Corrupt/incompatible feature chunk')
    with np.load(a, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    with gzip.open(m, 'rt') as f:
        records = json.load(f)
    validate(arrays, records, rows)
    return arrays, records
