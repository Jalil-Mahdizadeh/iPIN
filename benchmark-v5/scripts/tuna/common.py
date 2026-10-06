"""Small, benchmark-local I/O and numerical helpers; no repository mutations."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import numpy as np

SEEDS = (20260803, 20260817, 20260831)
CELLS = ('C1_test', 'C2_test', 'C3_test')
P_COUNTS = {'C1_test':3187, 'C2_test':13446, 'C3_test':2379}
ORIGINAL_SHA = 'bf2dd42af75d98324798b76837ca738e4ed230b096ee538ccdbaa001dcc88500'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data, *, exclusive=False):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w') as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')

def now():
    return datetime.now(timezone.utc).isoformat()

def record(path):
    path=Path(path)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}

def arrays(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}

def concordance(scores, positive, weights):
    scores=np.asarray(scores,np.float64)
    positive=np.asarray(positive,bool)
    weights=np.asarray(weights,np.float64)
    if not np.isfinite(scores).all() or not positive.any() or positive.all():
        raise ValueError('Invalid scores or PU classes')
    u=scores[~positive]; w=weights[~positive]
    if not np.isfinite(w).all() or (w<=0).any():
        raise ValueError('Invalid U weights')
    order=np.argsort(u,kind='stable')
    u=u[order]
    prefix=np.concatenate(([0.],np.cumsum(w[order],dtype=np.float64)))
    lo=np.searchsorted(u,scores[positive],side='left')
    hi=np.searchsorted(u,scores[positive],side='right')
    return float(np.mean((prefix[lo]+prefix[hi])*.5)/prefix[-1])

def cuda(seed=20260803):
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA GPU is required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    return torch.device('cuda')
