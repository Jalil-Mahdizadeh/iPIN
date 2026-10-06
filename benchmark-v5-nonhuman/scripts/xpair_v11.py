"""Frozen V11 interaction-only X-PAIR; one-GPU, full-length, resumable inference."""
import argparse
import fcntl
import hashlib
import io
import os
import socket
import time
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from bench_utils import ROOT, atomic, cuda, load_npz, now, read, record, save_npz, sha
import xpair_benchmark as base
from xpair.model import XPairModel
from xpair.utils.embedding_menager import pad_and_mask_emb_batch

NAME = 'xpair-v11'
FREEZE = ROOT / 'provenance/xpair-v11.json'
QUALIFICATION = ROOT / 'qualification/xpair-v11.json'
CODE = ['scripts/xpair_v11.py', 'scripts/xpair_benchmark.py', 'scripts/xpair/native.py',
        'scripts/xpair/io_utils.py', 'scripts/container.sh', 'scripts/bench_utils.py']


def checkpoint(path):
    allowed = [argparse.Namespace, np.dtype,
               (np.core.multiarray.scalar, 'numpy._core.multiarray.scalar'),
               type(np.dtype('float64')), type(np.dtype('float32'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


def freeze():
    assert not FREEZE.exists(), 'Checkpoint and code are already frozen'
    path = base.SOURCE / 'sources/X-PAIR/pretrained_models/interaction_dscript.ckpt'
    c = checkpoint(path)
    h = c['hyper_parameters']
    assert h['model_task'] == 'interaction' and h['plm'] == 'ankh_large'
    assert h['train_path_interface'] is None and h['val_path_interface'] is None
    assert 'train_human_dscript' in h['train_path_interaction']
    assert 'val_human_dscript' in h['val_path_interaction']
    prepared = read(ROOT / 'provenance/prepared.json')
    for rel, expected in prepared['data_files'].items():
        assert sha(ROOT / rel) == expected, rel
    image = read(ROOT / 'provenance/runtime-inputs.json')['items']['xpair-runtime']
    assert Path(image['path']).stat().st_size == image['bytes'] and sha(image['path']) == image['sha256']
    human_release = read(ROOT.parent / 'benchmark-v5/provenance/xpair-v11.json')
    assert record(path) == human_release['checkpoint']
    feature_records = []
    for rank in range(4):
        feature_file = ROOT / 'features/xpair' / f'rank-{rank:02d}.done.json'
        feature_run = read(feature_file)
        assert feature_run['passed'] and feature_run['provenance'] == base.provenance()
        feature_records.append(record(feature_file))
    atomic(FREEZE, {
        'at_utc': now(), 'name': NAME, 'checkpoint': record(path),
        'epoch': int(c['epoch']), 'global_step': int(c['global_step']),
        'hyper_parameters': h, 'addendum': record(ROOT / 'XPAIR-V11-ADDENDUM.md'),
        'inference_code': {p: sha(ROOT / p) for p in CODE},
        'prepared': record(ROOT / 'provenance/prepared.json'),
        'prior_collection': record(ROOT / 'archive/before-xpair-v11/results/collection.json'),
        'archive_receipt': record(ROOT / 'provenance/xpair-v11-archive.json'),
        'image': image, 'runtime': record(base.SOURCE / 'runtime/requirements.freeze.txt'),
        'encoder_provenance': base.provenance(),
        'existing_feature_runs': feature_records,
        'same_checkpoint_as_human_benchmark': record(ROOT.parent / 'benchmark-v5/provenance/xpair-v11.json'),
        'cache_loading_adjustment': record(ROOT / 'provenance/xpair-v11-cache-loading-adjustment.json'),
        'selection_uses_test_performance': False, 'retraining': False,
        'training_source': 'Human STRING v11 D-SCRIPT/Sledzieski TRAIN and validation',
    })
    atomic(ROOT / 'xpair-v11-status.json', {'at_utc': now(), 'complete': False, 'phase': 'frozen'})
    print({'frozen': NAME, 'checkpoint': record(path), 'epoch': c['epoch']}, flush=True)


def signature():
    item = read(FREEZE)
    for rel, expected in item['inference_code'].items():
        assert sha(ROOT / rel) == expected, rel
    assert sha(item['checkpoint']['path']) == item['checkpoint']['sha256']
    assert sha(ROOT / 'provenance/prepared.json') == item['prepared']['sha256']
    assert base.provenance() == item['encoder_provenance']
    assert sha(item['addendum']['path']) == item['addendum']['sha256']
    return {'freeze_sha256': sha(FREEZE), 'code': item['inference_code'],
            'precision': 'fp32_no_tf32', 'world_size': 1}


def load_model(device):
    signature()
    c = checkpoint(read(FREEZE)['checkpoint']['path'])
    model = XPairModel(c['hyper_parameters']).to(device).eval().requires_grad_(False)
    model.load_state_dict(c['state_dict'], strict=True)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return model


@torch.inference_mode()
def qualify(device):
    sig = signature()
    if QUALIFICATION.exists():
        q = read(QUALIFICATION)
        assert q['passed'] and q['signature'] == sig
        print('Existing V11 qualification verified', flush=True)
        return
    meta = read(ROOT / 'data/sequences.json')
    lengths = np.array(meta['length'])
    order = np.argsort(lengths, kind='stable')
    ids = [int(order[j]) for j in [0, 1, len(order)//4, len(order)//2, 3*len(order)//4, -1]]
    rows = np.load(ROOT / 'data/union.npy')
    longest = tuple(map(int, rows[np.argmax(lengths[rows[:, 0]] + lengths[rows[:, 1]]), :2]))
    pairs = [(ids[0], ids[1]), (ids[0], ids[-1]), (ids[-1], ids[2]),
             (ids[2], ids[3]), (ids[3], ids[4]), (ids[-1], ids[-1]), longest]
    features = {i: torch.load(base.verified_feature(i, meta), map_location=device, weights_only=True)
                for i in set(sum(([a, b] for a, b in pairs), []))}
    model = load_model(device)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    projected = {i: model.embedding_projection(f) for i, f in features.items()}
    gold, fast, reverse = [], [], []
    for a, b in pairs:
        m1 = torch.ones(1, len(features[a]), dtype=torch.bool, device=device)
        m2 = torch.ones(1, len(features[b]), dtype=torch.bool, device=device)
        gold.append(model({'input1': (features[a][None], m1), 'input2': (features[b][None], m2)}, task='interaction')[0])
        base.clear_attention(model)
        fast.append(base.projected_forward(model, projected[a][None], projected[b][None], m1, m2))
        reverse.append(base.projected_forward(model, projected[b][None], projected[a][None], m2, m1))
    gold, fast, reverse = map(torch.cat, (gold, fast, reverse))
    pp = [[pairs[0], pairs[3], pairs[4]][i % 3] for i in range(base.MAX_BATCH)]
    x1, m1 = pad_and_mask_emb_batch([features[a] for a, b in pp])
    x2, m2 = pad_and_mask_emb_batch([features[b] for a, b in pp])
    raw = model({'input1': (x1, m1), 'input2': (x2, m2)}, task='interaction')[0]
    base.clear_attention(model)
    z1, _ = pad_and_mask_emb_batch([projected[a] for a, b in pp])
    z2, _ = pad_and_mask_emb_batch([projected[b] for a, b in pp])
    result = base.projected_forward(model, z1, z2, m1, m2)
    scalar = torch.stack([gold[pairs.index(p)] for p in pp])
    errors = {'cache_logit': float((gold-fast).abs().max()), 'swap_logit': float((gold-reverse).abs().max()),
              'padded_cache_logit': float((raw-result).abs().max()),
              'batch_singleton_logit': float((scalar-result).abs().max()),
              'padded_probability': float((raw.sigmoid()-result.sigmoid()).abs().max())}
    assert torch.isfinite(gold).all() and torch.isfinite(result).all()
    assert max(errors.values()) < 2e-4 and errors['padded_probability'] < 1e-5, errors
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    atomic(QUALIFICATION, {'at_utc': now(), 'passed': True, 'signature': sig, 'errors': errors,
        'cases': [{'a': a, 'b': b, 'length_a': int(lengths[a]), 'length_b': int(lengths[b])} for a, b in pairs],
        'state_unchanged': True, 'no_truncation': True, 'test_metrics_read': False,
        'checkpoint': read(FREEZE)['checkpoint'], 'native_forward': True, 'padded_batch_size': base.MAX_BATCH})
    print({'qualified': NAME, 'errors': errors}, flush=True)


@torch.inference_mode()
def projection_cache(model, meta, device):
    """Verify every existing raw feature once and cache this model's native projection."""
    mapping = np.load(ROOT / 'data/xpair-cache-indices.npy')
    lengths = np.asarray(meta['length'], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths[:-1])]
    dimension = read(FREEZE)['hyper_parameters']['reduced_dim']
    table = torch.empty((int(lengths.sum()), dimension), dtype=torch.float32, device=device)
    expected = base.provenance()
    digest = hashlib.sha256()
    started = time.monotonic()
    def read_feature(i):
        old = mapping[i]
        path = base.SOURCE / 'features' / f'{int(old):05d}.pt' if old >= 0 else ROOT / 'features/xpair' / f'{i:05d}.pt'
        item = read(path.with_suffix('.json'))
        assert item['sequence_sha256'] == meta['sha256'][i] and item['length'] == int(lengths[i])
        assert item['dimension'] == 1536 and item['dtype'] == 'float32'
        raw = path.read_bytes()
        assert len(raw) == item['file']['bytes'] and hashlib.sha256(raw).hexdigest() == item['file']['sha256']
        assert all(item[key] == value for key, value in expected.items())
        feature = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        assert feature.shape == (int(lengths[i]), 1536) and feature.dtype == torch.float32
        return feature, item['file']['sha256']
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = deque(pool.submit(read_feature, i) for i in range(min(16, len(mapping))))
        for i in range(len(mapping)):
            feature, feature_sha = pending.popleft().result()
            if i+16 < len(mapping): pending.append(pool.submit(read_feature, i+16))
            feature = feature.to(device)
            assert torch.isfinite(feature).all()
            projected = model.embedding_projection(feature)
            table[int(offsets[i]):int(offsets[i]+lengths[i])] = projected
            digest.update(f"{i}:{meta['sha256'][i]}:{feature_sha}\n".encode())
            if i % 1000 == 0 or i == len(mapping)-1:
                progress = {'at_utc': now(), 'complete': False, 'phase': 'verifying_and_projecting_existing_features',
                            'features': i+1, 'total_features': len(mapping), 'seconds': time.monotonic()-started}
                atomic(ROOT / 'xpair-v11-status.json', progress)
                print(progress, flush=True)
    assert torch.isfinite(table).all()
    atomic(ROOT / 'provenance/xpair-v11-features.json', {
        'at_utc': now(), 'features_verified_and_reused': len(mapping), 'features_computed': 0,
        'residues': int(lengths.sum()), 'dimension': dimension, 'ordered_identity_sha256': digest.hexdigest(),
        'checkpoint': read(FREEZE)['checkpoint'], 'encoder_provenance': expected,
        'all_sequence_and_file_hashes_checked': True, 'seconds': time.monotonic()-started,
    })
    return table, torch.as_tensor(offsets, device=device), torch.as_tensor(lengths, device=device)


@torch.inference_mode()
def score(device):
    sig = signature()
    q = read(QUALIFICATION)
    assert q['passed'] and q['signature'] == sig
    output = ROOT / 'predictions' / NAME
    output.mkdir(exist_ok=True)
    with (output / 'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'done.json').exists():
            done = read(output / 'done.json')
            assert done['signature'] == sig and sha(done['file']['path']) == done['file']['sha256']
            print('Completed V11 predictions verified', flush=True)
            return
        meta = read(ROOT / 'data/sequences.json')
        lengths = np.array(meta['length'])
        rows = np.load(ROOT / 'data/union.npy')
        a, b = rows[:, 0], rows[:, 1]
        order = np.lexsort((lengths[b]//64, lengths[a]//64))
        model = load_model(device)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        started = time.monotonic()
        table, offsets, glengths = projection_cache(model, meta, device)
        print({'features_verified_and_reused': len(lengths), 'seconds': time.monotonic()-started}, flush=True)
        files = []
        scores = np.full(len(rows), np.nan)
        for start in range(0, len(order), base.CHUNK):
            ids = order[start:start+base.CHUNK]
            path = output / f'chunk-{start:06d}.npz'
            side = path.with_suffix('.json')
            if side.exists():
                item = read(side)
                assert item['signature'] == sig and sha(path) == item['file']['sha256']
                saved = load_npz(path)
                assert np.array_equal(saved['indices'], ids)
                z = saved['scores']
            else:
                logits = []
                for selected, la, lb in base.batches(ids, a, b, lengths):
                    x1, m1 = base.padded(table, offsets, glengths, a[selected], la)
                    x2, m2 = base.padded(table, offsets, glengths, b[selected], lb)
                    logits.append(base.projected_forward(model, x1, x2, m1, m2).cpu().numpy())
                z = np.concatenate(logits).astype(np.float64)
                assert z.shape == (len(ids),) and np.isfinite(z).all()
                save_npz(path, indices=ids, scores=z)
                atomic(side, {'at_utc': now(), 'signature': sig, 'file': record(path)})
            assert z.shape == (len(ids),) and np.isfinite(z).all()
            scores[ids] = z
            files.append(record(path))
            progress = {'at_utc': now(), 'complete': False, 'phase': 'scoring', 'rows': start+len(ids),
                        'total': len(order), 'seconds': time.monotonic()-started}
            atomic(ROOT / 'xpair-v11-status.json', progress)
            print(progress, flush=True)
        assert np.isfinite(scores).all() and all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
        from scipy.special import expit
        path = output / 'union.npz'
        save_npz(path, scores=scores, probabilities=expit(scores))
        atomic(output / 'done.json', {'at_utc': now(), 'rows': len(rows), 'signature': sig,
            'file': record(path), 'chunks': files, 'qualification': record(QUALIFICATION),
            'checkpoint': read(FREEZE)['checkpoint'], 'state_unchanged': True, 'no_truncation': True,
            'features_reused': len(lengths), 'features_computed': 0, 'seconds': time.monotonic()-started,
            'hostname': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'gpu': torch.cuda.get_device_name(device), 'world_size': 1})
        print({'complete_inference': NAME, 'rows': len(rows)}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['freeze', 'qualify', 'score'], required=True)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    else:
        device = cuda(0)
        if args.stage == 'qualify': qualify(device)
        else: score(device)


if __name__ == '__main__': main()
