"""VY provenance, explicit budgets and fail-closed compact artifacts."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SOURCE = PROJECT / 'pilot-vx-v2'
BASE = PROJECT / 'pilot-vx'


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


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


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
    resource = read(ROOT / 'provenance/resource-start.json')
    remaining = resource['deadline_unix'] - time.time() - reserve
    if remaining <= 0:
        raise TimeoutError('Frozen VY resource deadline exhausted')
    return remaining


def verify_freeze():
    path = ROOT / 'provenance/freeze.json'; manifest = read(path)
    for rel, digest in manifest['files'].items():
        if sha(PROJECT / rel) != digest:
            raise ValueError('Frozen input changed: ' + rel)
    return sha(path)


def fuse(base, evidence, gate, alpha):
    base, evidence, gate = map(lambda x: np.asarray(x, dtype=float), (base, evidence, gate))
    if base.shape != gate.shape or gate.shape != evidence.shape or not np.isfinite(base).all():
        raise ValueError('Invalid fusion shapes/baseline')
    if not np.isfinite(gate).all() or np.any((gate < 0) | (gate > 1)):
        raise ValueError('Invalid reliability gate')
    active = gate > 0
    if not np.isfinite(evidence[active]).all():
        raise FloatingPointError('Nonfinite computed evidence')
    z = base.copy()
    if alpha:
        z[active] += alpha * gate[active] * evidence[active]
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite fused prediction')
    return z


def feature_paths(pid):
    path = ROOT / 'features' / f'{pid}.npz'
    return path, path.with_suffix('.json')


def save_feature(pid, M, S, fingerprint, metadata):
    path, side = feature_paths(pid)
    if path.exists() or side.exists():
        raise RuntimeError('Refuse to overwrite feature: ' + str(pid))
    n = config()['encoder']['summary_dimensions']
    for x in (M, S):
        if np.asarray(x).shape != (n,) or not np.isfinite(x).all():
            raise ValueError('Invalid monomer representation')
    atomic_npz(path, M=np.asarray(M, dtype=np.float32), S=np.asarray(S, dtype=np.float32))
    atomic(side, {**metadata, 'pid': str(pid), 'fingerprint': fingerprint, 'sha256': sha(path)})


def load_feature(pid, fingerprint, expected=None):
    path, side = feature_paths(pid)
    if not path.exists() and not side.exists():
        return None
    if not path.exists() or not side.exists():
        raise ValueError('Partial feature artifact: ' + str(pid))
    info = read(side)
    if info['pid'] != str(pid) or info['fingerprint'] != fingerprint or sha(path) != info['sha256']:
        raise ValueError('Feature integrity error: ' + str(pid))
    if expected is not None:
        for key in ('query_sha256', 'tokens_sha256'):
            if info[key] != expected[key]:
                raise ValueError('Feature input identity changed: ' + str(pid))
    with np.load(path, allow_pickle=False) as f:
        if set(f.files) != {'M', 'S'}:
            raise ValueError('Unexpected feature arrays')
        result = {k: np.asarray(f[k], dtype=float) for k in f.files}
    n = config()['encoder']['summary_dimensions']
    if any(x.shape != (n,) or not np.isfinite(x).all() for x in result.values()):
        raise ValueError('Invalid cached neural summary')
    return result
