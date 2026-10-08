"""V2 identities and fail-closed I/O; imports only the original TRAIN/DEV path."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
BASE = PROJECT / 'pilot-vx'
sys.path.insert(0, str(BASE / 'scripts'))
from common import atomic, now, sha, text_hash


def config():
    return json.loads((ROOT / 'config.json').read_text())


def read(path):
    return json.loads(Path(path).read_text())


def verify_freeze():
    path = ROOT / 'provenance/freeze.json'
    manifest = read(path)
    for name, digest in manifest['files'].items():
        if sha(PROJECT / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return sha(path)


def source_record_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save_record(uid, value, fingerprint):
    p = ROOT / 'features' / (uid + '.json')
    atomic(p, {**value, 'fingerprint': fingerprint})
    atomic(p.with_suffix('.sha.json'), {'sha256': sha(p), 'fingerprint': fingerprint})


def load_record(uid, fingerprint):
    import numpy as np
    p = ROOT / 'features' / (uid + '.json')
    side = p.with_suffix('.sha.json')
    if not p.exists() and not side.exists():
        return None
    if not p.exists() or not side.exists():
        raise ValueError('Incomplete atomic record: ' + uid)
    check = read(side)
    if check['fingerprint'] != fingerprint or check['sha256'] != sha(p):
        raise ValueError('Corrupt or incompatible record: ' + uid)
    x = read(p)
    if x['fingerprint'] != fingerprint or x['uid'] != uid:
        raise ValueError('Record identity mismatch: ' + uid)
    for arm in config()['arms']:
        a = np.asarray(x[arm])
        if a.shape != (128,) or not np.isfinite(a).all():
            raise ValueError('Invalid features: ' + uid + ':' + arm)
    if not x['available'] and (x['gate'] != 0 or any(any(x[a]) for a in config()['arms'])):
        raise ValueError('Invalid unavailable record: ' + uid)
    return x


def fuse(base, evidence, gate, alpha):
    import numpy as np
    z = np.array(base, dtype=float, copy=True)
    active = np.asarray(gate) > 0
    if alpha:
        if not np.isfinite(np.asarray(evidence)[active]).all():
            raise FloatingPointError('Nonfinite available evidence')
        z[active] += alpha * np.asarray(gate)[active] * np.asarray(evidence)[active]
    return z
