"""Benchmark paths; reuse immutable Vx numerical libraries with local inputs."""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
TRAIN = PROJECT / 'retrain-vx'
BASE = PROJECT / 'pilot-vx-series/pilot-vx'
# These unchanged modules import shared for paths: inputs/chunks/guarded_encoder.
# Numerical behavior and original msa/encoder/common are retained byte-for-byte.
sys.path.extend([str(TRAIN / 'scripts'), str(BASE / 'scripts')])
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
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype=np.uint8).tobytes()).hexdigest()


def mono_path(entry):
    return PROJECT / entry['path']


def quality_from_profiles(pa, pb, meta):
    return np.r_[(pa + pb) / 2, np.abs(pa - pb), np.log1p(meta.get('common_keys', 0)),
                 np.log1p(meta.get('neff', 0)), (meta['quality_a'] + meta['quality_b']) / 2,
                 abs(meta['quality_a'] - meta['quality_b'])]


def fuse(base, evidence, gate, alpha):
    base, evidence, gate = [np.asarray(x, dtype=float) for x in (base, evidence, gate)]
    if base.shape != evidence.shape or base.shape != gate.shape or not np.isfinite(base).all():
        raise ValueError('Invalid baseline/fusion dimensions')
    active = gate > 0
    if not np.isfinite(gate).all() or np.any((gate < 0) | (gate > 1)) or not np.isfinite(evidence[active]).all():
        raise ValueError('Invalid gate/evidence')
    z = base.copy()
    if alpha:
        z[active] += alpha * gate[active] * evidence[active]
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite fusion')
    return z


def verify_freeze():
    path = ROOT / 'provenance/freeze.json'; frozen = read(path)
    for name, digest in frozen['files'].items():
        if sha(PROJECT / name) != digest:
            raise ValueError('Frozen file changed: ' + name)
    return sha(path)


def source_contract():
    cfg = config(); complete = read(TRAIN / 'results/COMPLETE.json')
    if not complete['verified'] or complete['rows'] != 222452 or complete['test_accessed'] or complete['shuffled_arm']:
        raise ValueError('Wrong completed TRAIN/VAL experiment')
    if sha(TRAIN / 'provenance/freeze.json') != cfg['retrain_fingerprint'] or complete['fingerprint'] != cfg['retrain_fingerprint']:
        raise ValueError('TRAIN/VAL source fingerprint changed')
    for name, digest in complete['result_hashes'].items():
        if sha(TRAIN / 'results' / name) != digest:
            raise ValueError('Completed TRAIN/VAL result changed: ' + name)
    if sha(PROJECT / cfg['heads_path']) != cfg['heads_sha256']:
        raise ValueError('Frozen head parameters changed')
    heads = read(PROJECT / cfg['heads_path'])
    if set(heads) != set(ARMS) or {k: h['alpha'] for k, h in heads.items()} != cfg['fixed_alphas']:
        raise ValueError('Frozen arms/alphas changed')
    old = read(TRAIN / 'config.json')
    keys = ['seed', 'maximum_paired_rows_including_query', 'maximum_combined_residues',
        'maximum_cached_homologs_per_protein', 'minimum_homolog_query_coverage', 'minimum_paired_homologs',
        'minimum_quality_fraction_per_chain', 'redundancy_identity', 'minimum_identity_comparison_coverage',
        'windows', 'local_search_rescue', 'encoder', 'archive', 'baseline', 'image', 'image_sha256']
    if any(cfg[k] != old[k] for k in keys):
        raise ValueError('Benchmark changed the original scientific configuration')
    return heads
