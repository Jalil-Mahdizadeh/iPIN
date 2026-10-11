"""Small strict artifact helpers; all runtime writes stay under /study."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/study')
INPUT = Path('/inputs')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('Output outside the study')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def save(path, **values):
    import numpy as np
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('Output outside the study')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **values)


def guard():
    frozen = read(ROOT / 'provenance/freeze.json')
    for name, record in frozen['inputs'].items():
        if sha(INPUT / name) != record['sha256']:
            raise RuntimeError('Changed input: ' + name)
    for name, digest in frozen['implementation'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Changed implementation: ' + name)
    if sha('/proposal.md') != frozen['proposal_sha256']:
        raise RuntimeError('Changed proposal')
    if Path('/repo').exists() or Path('/nobackup').exists():
        raise RuntimeError('Unexpected broad repository mount')
    return frozen
