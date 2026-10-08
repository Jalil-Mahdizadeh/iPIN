"""Shared provenance and atomic artifact helpers for the bounded Vx pilot."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def text_hash(s):
    return hashlib.sha256(s.encode()).hexdigest()


def atomic(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def config():
    return json.loads((ROOT/'config.json').read_text())


def fasta(path):
    name = None; seq = []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(seq)
                name = line[1:].strip(); seq = []
            else:
                seq.append(line.strip())
        if name is not None:
            yield name, ''.join(seq)


def verify_freeze():
    manifest = json.loads((ROOT/'provenance/freeze.json').read_text())
    for rel, digest in manifest['files'].items():
        if sha(PROJECT/rel) != digest:
            raise ValueError(f'Frozen input changed: {rel}')
    return sha(ROOT/'provenance/freeze.json')
