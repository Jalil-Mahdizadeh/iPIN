"""Shared, deterministic data and provenance helpers for this standalone audit."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import time
import resource

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent.parent
PREVIOUS = PROJECT / 'literature/string-v11-train-degree-baseline'
NONHUMAN = PROJECT / 'bechmark-nonhuman-v5'
SPECIES = ['mouse', 'fly', 'worm', 'yeast', 'ecoli']
DATASETS = ['human-validation'] + SPECIES
BASELINES = ['exact-degree', 'homology-degree', 'sequence-propensity', 'interolog']
REFERENCES = ['native-human', 'tuna-human']
NAMES = BASELINES + REFERENCES
LABELS = {'exact-degree': 'Exact TRAIN degree (reference)',
          'homology-degree': 'Homology-transferred degree',
          'sequence-propensity': 'Sequence-only propensity', 'interolog': 'Interolog lookup',
          'native-human': 'PLM-interact humanV11', 'tuna-human': 'TUnA human seed 47'}
SET_LABELS = {'human-validation': 'Human validation', 'mouse': 'Mouse', 'fly': 'Fly',
              'worm': 'Worm', 'yeast': 'Yeast', 'ecoli': 'E. coli'}
CONFIG = {
    'seed': 47, 'threads': 16, 'smoothing': 1.0,
    'search': {'sensitivity': 7.5, 'evalue': 1e-5, 'min_identity': 0.25,
               'min_coverage': 0.5, 'max_prefilter': 1000, 'top_k': 5, 'gpu': 0,
               'weight': 'fident * sqrt(qcov * tcov)', 'no_hit': 'mean TRAIN protein log ratio'},
    'regressor': {'loss': 'squared_error', 'max_iter': 300, 'learning_rate': 0.05,
                  'max_leaf_nodes': 15, 'min_samples_leaf': 20, 'l2_regularization': 10.0,
                  'max_bins': 255, 'early_stopping': False, 'random_state': 47},
    'features': 'log length; 20 AA fractions; 400 ordered dipeptide fractions; unknown fraction',
    'bootstrap': {'replicates': 500, 'seed': 20261006, 'unit': 'protein endpoints'},
}

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)

def verify(item):
    p = Path(item['path'])
    assert p.is_file() and sha(p) == item['sha256'], str(p)
    return p

def verify_protocol():
    frozen = read(ROOT / 'provenance/protocol-freeze.json')
    assert frozen['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md')
    assert frozen['config'] == CONFIG
    return frozen

def stage_stat(start):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {'wall_seconds': time.monotonic() - start, 'user_seconds': usage.ru_utime,
            'system_seconds': usage.ru_stime, 'max_rss_kib': usage.ru_maxrss,
            'at_utc': now()}

def degrees(pairs, labels, count):
    pos, neg = np.zeros(count, dtype=np.int64), np.zeros(count, dtype=np.int64)
    for label, out in [(1, pos), (0, neg)]:
        p = pairs[labels == label]
        np.add.at(out, p[:, 0], 1)
        np.add.at(out, p[p[:, 0] != p[:, 1], 1], 1)
    return pos, neg

class Ranking:
    """Grouped AP and trapezoidal AUROC, independently checked against sklearn."""
    def __init__(self, labels, scores):
        self.order = np.argsort(-scores, kind='stable')
        self.y = labels[self.order]
        s = scores[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]

    def compute(self, weights):
        w = weights[self.order]
        tp = np.cumsum(w * self.y)[self.ends]
        fp = np.cumsum(w * (1 - self.y))[self.ends]
        p, n = tp[-1], fp[-1]
        if p <= 0 or n <= 0:
            return np.array([np.nan, np.nan])
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
        ap = np.dot(np.diff(np.r_[0.0, tp]), precision) / p
        auc = np.dot(np.diff(np.r_[0.0, fp]), (tp + np.r_[0.0, tp[:-1]]) / 2) / (p * n)
        return np.array([ap, auc])
