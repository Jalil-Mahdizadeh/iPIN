"""Isolated Vx v5 artifacts; original Vx scientific modules are read-only dependencies."""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
BASE = PROJECT / 'pilot-vx'
SOURCE = PROJECT / 'pilot-vx-v2'
# Append, so Vx v5's controller/analysis modules cannot be shadowed by older scripts.
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


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype=np.uint8).tobytes()).hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def atomic_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def check_budget(reserve=0):
    remaining = read(ROOT / 'provenance/resource-start.json')['deadline_unix'] - time.time() - reserve
    if remaining <= 0:
        raise TimeoutError('Separate Vx v5 resource deadline exhausted')
    return remaining


def verify_freeze():
    p = ROOT / 'provenance/freeze.json'
    for rel, digest in read(p)['files'].items():
        if sha(PROJECT / rel) != digest:
            raise ValueError('Frozen input changed: ' + rel)
    return sha(p)


def fuse(base, evidence, gate, alpha):
    base, evidence, gate = [np.asarray(a, dtype=float) for a in (base, evidence, gate)]
    if base.shape != gate.shape or gate.shape != evidence.shape or not np.isfinite(base).all():
        raise ValueError('Invalid fusion shapes/baseline')
    if not np.isfinite(gate).all() or np.any((gate < 0) | (gate > 1)):
        raise ValueError('Invalid reliability gate')
    active = gate > 0
    if not np.isfinite(evidence[active]).all():
        raise FloatingPointError('Nonfinite computed evidence, including at alpha zero')
    z = base.copy()
    if alpha:
        z[active] += alpha * gate[active] * evidence[active]
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite fused prediction')
    return z


DIMENSIONS = {'T256': 128, 'S256': 128, 'P256': 68}


def validate_features(features, expected):
    if set(features) != set(DIMENSIONS):
        raise ValueError('Missing or extra depth arm')
    for arm, size in DIMENSIONS.items():
        x = np.asarray(features[arm], dtype=float)
        if x.shape != (size,) or not np.isfinite(x).all():
            raise ValueError('Invalid feature: ' + arm)
        if not expected['available'] and x.any():
            raise ValueError('Unavailable pair has evidence')


def save_record(uid, features, fingerprint, expected, timing=None):
    path = ROOT / 'features' / (uid + '.json'); side = path.with_suffix('.sha.json')
    if path.exists() or side.exists(): raise RuntimeError('Refuse to overwrite feature: ' + uid)
    validate_features(features, expected)
    atomic(path, {**expected, 'features': {k: np.asarray(v).tolist() for k,v in features.items()},
                  'fingerprint': fingerprint, 'at_utc': now(), 'timing': timing})
    atomic(side, {'sha256': sha(path), 'fingerprint': fingerprint})


def load_record(uid, fingerprint, expected):
    path = ROOT / 'features' / (uid + '.json'); side = path.with_suffix('.sha.json')
    if not path.exists() and not side.exists(): return None
    if not path.exists() or not side.exists(): raise ValueError('Partial feature artifact: ' + uid)
    check, record = read(side), read(path)
    if check.get('fingerprint') != fingerprint or record.get('fingerprint') != fingerprint or sha(path) != check.get('sha256'):
        raise ValueError('Corrupt feature artifact: ' + uid)
    if record.get('uid') != uid or any(record.get(k) != v for k,v in expected.items()):
        raise ValueError('Feature input identity changed: ' + uid)
    validate_features(record['features'], expected)
    return record

