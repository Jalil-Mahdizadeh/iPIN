"""Exact frozen v3 forward with the existing full-coverage benchmark row order."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from frozen_data import PairData
from training_data import PairData as TrainingData
from frozen_model import PairModel

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def configure(local_rank=0):
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(8)
    torch.manual_seed(20260929)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.device('cuda', local_rank)


def stop_requested():
    # Benchmarking has its own lifecycle; the training stop marker stays in place.
    return any((ROOT / name).exists() for name in ['REQUEST_STOP', 'REQUEST_REQUEUE'])


def require_continue():
    if stop_requested():
        raise SystemExit(75)


def load_checkpoint(entry, base, device):
    path = ROOT / entry['path']
    assert sha256(path) == entry['sha256'], 'Checkpoint hash mismatch'
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert saved['training_state']['update'] == entry['update']
    assert saved['fingerprint'] == entry['source_manifest']['fingerprint']
    model = PairModel(base, entry['configuration'])
    model.load_state_dict(saved['model'], strict=True)
    assert all(torch.equal(v, saved['model'][k]) for k, v in model.state_dict().items())
    model.esm_mask.gradient_checkpointing_disable()
    model.eval().requires_grad_(False).to(device)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return model


def load_model(name, device, backend='efficient'):
    selection = json.loads((ROOT / 'provenance/selection.json').read_text())
    assert name == 'v3-residue-mlp' and backend == 'efficient'
    return load_checkpoint(selection['models'][name], selection['base_model_path'], device)


def batch_features(data, indices, device):
    # Use the byte-identical training batch implementation, including chain IDs
    # and residue masks. The old benchmark loader provides only rows/sequences.
    return TrainingData.batch(data, indices, np.zeros(len(indices), dtype=np.int64),
                              2, False, device)


@torch.inference_mode()
def predict(model, data, indices, device, bf16=True):
    features = batch_features(data, indices, device)
    assert features['attention_mask'].sum(1).cpu().tolist() == [
        int(data.lengths[i]) for i in indices for _ in range(2)]
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        scores = model(**features, compute_loss=False)
    scores = scores.float().cpu().numpy().astype(np.float64)
    assert scores.shape == (len(indices), 2) and np.isfinite(scores).all()
    return np.column_stack((indices, data.rows[indices, 2], scores))


def prediction_fingerprint():
    files = ['config.json', 'provenance/selection.json']
    files += [f'scripts/{name}' for name in ['common.py', 'infer.py', 'frozen_data.py',
                                           'training_data.py', 'frozen_model.py', 'container.sh']]
    return hashlib.sha256(json.dumps({p: sha256(ROOT / p) for p in files},
                                    sort_keys=True).encode()).hexdigest()
