"""Snapshot verified completed pilot inputs and diagnostics, never TEST data."""
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from study import ROOT, PROJECT, BASE, V2, config, read, atomic, save_npz, sha, now, text_hash


def main():
    if (ROOT / 'provenance/inputs.json').exists():
        raise RuntimeError('Inputs already prepared')
    for folder in ['data', 'features', 'cache/maps', 'results', 'qualification', 'provenance', 'logs']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    cfg = config(); original = read(BASE / 'provenance/freeze.json'); prior = read(V2 / 'provenance/freeze.json')
    if sha(BASE / 'provenance/freeze.json') != cfg['source_fingerprint']:
        raise ValueError('Unexpected original Vx fingerprint')
    sources = {}
    def snapshot(source, target, expected=None):
        digest = sha(source)
        if expected is not None and digest != expected:
            raise ValueError('Source hash mismatch: ' + str(source))
        sources[str(source.relative_to(PROJECT))] = digest
        shutil.copyfile(source, ROOT / 'data' / target)
    for name in ['sample.json', 'proteins.json', 'baseline.json', 'dev-groups.json', 'monomer-catalog.json']:
        snapshot(BASE / 'data' / name, name, original['files']['pilot-vx/data/' + name])
    for name in ['source.npz', 'source_metadata.json', 'dev_family_witnesses.json']:
        snapshot(V2 / 'data' / name, name, prior['files']['pilot-vx-v2/data/' + name])
    for name in ['heads.json', 'metrics.json']:
        snapshot(V2 / 'data' / ('source_' + name), 'source_' + name, prior['files']['pilot-vx-v2/data/source_' + name])
    for base, manifest, prefix in [(BASE, original, 'pilot-vx'), (V2, prior, 'pilot-vx-v2')]:
        for path in [base / 'config.json', *sorted((base / 'scripts').glob('*.py'))]:
            expected = manifest['files'][prefix + '/' + str(path.relative_to(base))]
            if sha(path) != expected:
                raise ValueError('Original scientific dependency changed')
            sources[str(path.relative_to(PROJECT))] = expected
        sources[str((base / 'provenance/freeze.json').relative_to(PROJECT))] = sha(base / 'provenance/freeze.json')
    sample = read(ROOT / 'data/sample.json'); proteins = read(ROOT / 'data/proteins.json')
    metadata = read(ROOT / 'data/source_metadata.json')
    if len(sample) != 8000 or {r['split'] for r in sample} != {'train', 'val'} or {p['split'] for p in proteins.values()} != {'train', 'val'}:
        raise ValueError('Only original TRAIN/VAL inputs allowed')
    with np.load(ROOT / 'data/source.npz', allow_pickle=False) as arrays:
        if arrays['uids'].tolist() != [r['uid'] for r in sample]:
            raise ValueError('Source order mismatch')
        np.testing.assert_array_equal(arrays['gate'], [m['gate'] for m in metadata])
    prior_fp = sha(V2 / 'provenance/freeze.json')
    def check(item):
        r, m = item; p = V2 / 'features' / (r['uid'] + '.json')
        side = read(p.with_suffix('.sha.json')); v = read(p)
        if sha(p) != side['sha256'] or side['fingerprint'] != prior_fp or v['fingerprint'] != prior_fp:
            raise ValueError('Corrupt Vx-v2 diagnostic record')
        if any(v[k] != m[k] for k in ['uid', 'a', 'b', 'available', 'gate', 'reason']):
            raise ValueError('Source identity/eligibility mismatch')
        if sorted([r['a'], r['b']]) != [m['a'], m['b']]:
            raise ValueError('Pair identity mismatch')
        if any(proteins[str(r[k])]['split'] != r['split'] for k in ['a', 'b']):
            raise ValueError('Protein cohort mismatch')
        if r['length'] != sum(len(proteins[str(r[k])]['sequence']) for k in ['a', 'b']):
            raise ValueError('Length mismatch')
        return {k: v.get(k) for k in ['diagnostics', 'tokens_sha256', 'null_tokens_sha256', 'elapsed_seconds']}, side['sha256']
    with ThreadPoolExecutor(max_workers=8) as ex:
        checked = list(ex.map(check, zip(sample, metadata)))
    atomic(ROOT / 'data/diagnostics.json', [d for d, _ in checked])
    atomic(ROOT / 'provenance/diagnostic-source-digests.json', {r['uid']: s for r, (_, s) in zip(sample, checked)})
    train = np.array([r['split'] == 'train' and m['available'] for r, m in zip(sample, metadata)])
    eligible = np.flatnonzero(train); y = np.array([r['label'] for r in sample])
    groups = np.array([text_hash(':'.join(sorted(proteins[str(r[k])]['sha256'] for k in ['a', 'b']))) for r in sample])
    for g in set(groups[train]):
        if len(set(y[train & (groups == g)])) != 1:
            raise ValueError('Conflicting labels for duplicate sequence pairs')
    folds = np.full(len(sample), -1)
    splitter = StratifiedGroupKFold(n_splits=cfg['fitting']['folds'], shuffle=True, random_state=cfg['seed'])
    for fold, (fit, held) in enumerate(splitter.split(eligible, y[eligible], groups[eligible])):
        fi, hi = eligible[fit], eligible[held]
        if set(groups[fi]) & set(groups[hi]):
            raise ValueError('OOF duplicate pair leakage')
        if min(np.bincount(y[fi], minlength=2)) < cfg['fitting']['minimum_fold_class_count']:
            raise ValueError('Insufficient OOF fitting classes')
        folds[hi] = fold
    if np.any(folds[train] < 0) or np.any(folds[~train] >= 0) or train.sum() != 1825:
        raise ValueError('Unexpected eligible TRAIN folds')
    atomic(ROOT / 'data/folds.json', {'folds': folds.tolist(), 'groups': groups.tolist(),
        'policy': cfg['fitting']['fold_policy'], 'protein_disjoint': False, 'family_disjoint': False})
    # Qualification cases chosen without labels: longest eligible TRAIN example per length band.
    cases = []
    for low, high in [(0, 512), (512, 1024), (1024, 1536)]:
        ids = [i for i in eligible if low < sample[i]['length'] <= high]
        cases.append(max(ids, key=lambda i: (sample[i]['length'], sample[i]['uid'])))
    ids = list(eligible)
    masked = min(ids, key=lambda i: (metadata[i]['msa']['quality_a'] * metadata[i]['msa']['quality_b'], sample[i]['uid']))
    if masked not in cases:
        cases.append(masked)
    atomic(ROOT / 'data/qualification-cases.json', [int(i) for i in cases])
    # Recreate three audited native VAL microbatches exactly, only for numeric parity.
    old_reuse = read(BASE / 'provenance/baseline-reuse.json')
    native_rows = np.load(PROJECT / 'benchmark-v1/data/val.npy', allow_pickle=False)
    sources['benchmark-v1/data/val.npy'] = sha(PROJECT / 'benchmark-v1/data/val.npy')
    native_checks = []
    for entry in old_reuse['sources']:
        if '/rank-00-' not in entry['path']:
            continue
        p = PROJECT / entry['path']
        if sha(p) != entry['sha256']:
            raise ValueError('Corrupt native VAL reference chunk')
        with np.load(p, allow_pickle=False) as f:
            predictions = f['predictions'].copy()
        batches = []; batch = []; maximum = 0
        for row in predictions:
            idx = int(row[0]); a, b = native_rows[idx, :2]
            length = len(proteins[str(a)]['sequence']) + len(proteins[str(b)]['sequence']) + 3
            if batch and (len(batch) >= 8 or 2*(len(batch)+1)*max(maximum, length) > 16384):
                batches.append(batch); batch = []; maximum = 0
            batch.append(row); maximum = max(maximum, length)
        if batch:
            batches.append(batch)
        for batch in batches:
            indices = [int(r[0]) for r in batch]
            length = max(sum(len(proteins[str(p)]['sequence']) for p in native_rows[i, :2])+3 for i in indices)
            if len(native_checks) < 3 and length >= [450, 650, 1024][len(native_checks)]:
                native_checks.append({'rows': native_rows[indices].tolist(), 'orientations': np.array(batch)[:, 2:].tolist(),
                    'source_indices': indices, 'length': length, 'source': entry['path'], 'source_sha256': entry['sha256']})
                sources[entry['path']] = entry['sha256']
        if len(native_checks) == 3:
            break
    if len(native_checks) != 3:
        raise ValueError('Insufficient native qualification references')
    atomic(ROOT / 'data/native-qualification.json', native_checks)
    for name in ['common.py', 'frozen_data.py', 'frozen_model.py']:
        p = PROJECT / 'benchmark-v1/scripts' / name; sources[str(p.relative_to(PROJECT))] = sha(p)
    for name in ['benchmark-v1/provenance/selection.json', 'retrain-v1/assets/esm2/vocab.txt', 'retrain-v1/assets/esm2/config.json']:
        sources[name] = sha(PROJECT / name)
    atomic(ROOT / 'provenance/inputs.json', {'at_utc': now(), 'source_files': sources,
        'source_fingerprint': cfg['source_fingerprint'], 'v2_fingerprint': prior_fp,
        'records': 8000, 'eligible_train': int(train.sum()), 'eligible_val': sum(m['available'] and r['split']=='val' for r,m in zip(sample,metadata)),
        'prior_v2_extraction_seconds_sum': sum(d['elapsed_seconds'] for d,_ in checked),
        'test_accessed': False, 'new_heads_fitted': False,
        'native_training_logits_required': int(train.sum()), 'native_reference_qualification_only': True})
    print(read(ROOT / 'provenance/inputs.json') | {'source_files': len(sources)}, flush=True)


if __name__ == '__main__':
    main()
