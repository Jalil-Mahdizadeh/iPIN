"""Isolated, hash-checked TRAIN/VAL artifacts for the frozen-contact pilot."""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
BASE = PROJECT / 'pilot-vx-series/pilot-vx'
V2 = PROJECT / 'pilot-vx-series/pilot-vx-v2'
sys.path.append(str(BASE / 'scripts'))


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def config():
    return read(ROOT / 'config.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def save_npz(path, **arrays):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def code_hashes():
    return {str(p.relative_to(PROJECT)): sha(p) for folder in ['scripts', 'tests']
            for p in sorted((ROOT / folder).glob('*.py'))}


def verify_freeze():
    p = ROOT / 'provenance/freeze.json'
    for rel, digest in read(p)['files'].items():
        if sha(PROJECT / rel) != digest:
            raise ValueError('Frozen input changed: ' + rel)
    return sha(p)


def remaining(reserve=0):
    seconds = read(ROOT / 'provenance/resource-start.json')['deadline_unix'] - time.time() - reserve
    if seconds <= 0:
        raise TimeoutError('V7 resource deadline exhausted')
    return seconds


def owned_bytes():
    return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())


def fuse(base, evidence, gate, alpha):
    import numpy as np
    b, e, g = [np.asarray(v, dtype=float) for v in (base, evidence, gate)]
    if b.shape != e.shape or b.shape != g.shape or not np.isfinite(b).all():
        raise ValueError('Invalid fusion dimensions/baseline')
    if not np.isfinite(g).all() or np.any((g < 0) | (g > 1)) or not np.isfinite(e[g > 0]).all():
        raise ValueError('Invalid gate/evidence, including at alpha zero')
    z = b.copy()
    if alpha:
        z[g > 0] += alpha * g[g > 0] * e[g > 0]
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite prediction')
    return z


def evidence(state, x):
    import numpy as np
    z = (np.asarray(x) - np.asarray(state['standard_mean'])) / np.asarray(state['standard_scale'])
    if 'pca_components' in state:
        z = (z - np.asarray(state['pca_mean'])) @ np.asarray(state['pca_components']).T
    return (z @ np.asarray(state['coef']).T + state['intercept']).ravel()


def load_record(uid, fingerprint, maps=False):
    import numpy as np
    p = ROOT / 'features' / (uid + '.json'); side = p.with_suffix('.sha.json')
    if not p.exists() and not side.exists():
        return None
    if not p.exists() or not side.exists():
        raise ValueError('Incomplete record: ' + uid)
    s, v = read(side), read(p)
    if s['sha256'] != sha(p) or s['fingerprint'] != fingerprint or v['fingerprint'] != fingerprint or v['uid'] != uid:
        raise ValueError('Record provenance mismatch: ' + uid)
    for arm in config()['arms']:
        a = np.asarray(v['features'][arm])
        if a.shape != (6,) or not np.isfinite(a).all() or (not v['available'] and a.any()):
            raise ValueError('Invalid contact features: ' + uid)
    if v['available'] != (v['gate'] > 0):
        raise ValueError('Invalid eligibility/gate: ' + uid)
    if maps and v['available']:
        m = ROOT / v['map_path']
        if sha(m) != v['map_sha256']:
            raise ValueError('Corrupt contact map: ' + uid)
    return v


def save_record(value, fingerprint):
    p = ROOT / 'features' / (value['uid'] + '.json')
    if p.exists() or p.with_suffix('.sha.json').exists():
        raise RuntimeError('Refuse to overwrite a feature record')
    atomic(p, {**value, 'fingerprint': fingerprint})
    atomic(p.with_suffix('.sha.json'), {'fingerprint': fingerprint, 'sha256': sha(p)})
