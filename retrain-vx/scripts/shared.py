"""Local artifacts and immutable original Vx dependencies; no TEST inputs."""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
BASE = PROJECT / 'pilot-vx-series/pilot-vx'
sys.path.append(str(BASE / 'scripts'))
ARMS = {'true': 128, 'quality': 68}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def config():
    return read(ROOT / 'config.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype=np.uint8).tobytes()).hexdigest()


def fuse(base, evidence, gate, alpha):
    base, evidence, gate = [np.asarray(x, dtype=float) for x in (base, evidence, gate)]
    if base.shape != evidence.shape or base.shape != gate.shape or not np.isfinite(base).all():
        raise ValueError('Invalid baseline/fusion dimensions')
    active = gate > 0
    if not np.isfinite(gate).all() or np.any((gate < 0) | (gate > 1)) or not np.isfinite(evidence[active]).all():
        raise ValueError('Invalid gate/evidence, including at alpha zero')
    z = base.copy()
    if alpha:
        z[active] += alpha * gate[active] * evidence[active]
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite fusion')
    return z


def verify_freeze():
    path = ROOT / 'provenance/freeze.json'; f = read(path)
    for rel, digest in f['files'].items():
        if sha(PROJECT / rel) != digest:
            raise ValueError('Frozen file changed: ' + rel)
    return sha(path)


def mono_path(entry):
    return PROJECT / entry['path']


def quality_from_profiles(pa, pb, meta):
    return np.r_[(pa + pb) / 2, np.abs(pa - pb), np.log1p(meta.get('common_keys', 0)),
                 np.log1p(meta.get('neff', 0)), (meta['quality_a'] + meta['quality_b']) / 2,
                 abs(meta['quality_a'] - meta['quality_b'])]
