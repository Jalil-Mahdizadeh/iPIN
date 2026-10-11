"""GPU/CPU reproduction and label-free timing before the TEST batch."""
import math
import time
import numpy as np
import torch
from shared import ROOT, PROJECT, TRAIN, BASE, ARMS, read, config, now, atomic, sha, source_contract
from inputs import Inputs
from guarded_encoder import GuardedEncoder
from msa import load_monomer, pair, quality_features
from predict import score, prohibit_training


def main():
    started = time.monotonic(); cfg = config(); heads = source_contract(); prohibit_training()
    if (ROOT / 'qualification/encoder.json').exists():
        raise RuntimeError('Encoder already qualified; do not replace evidence')
    assert read(ROOT / 'provenance/preparation.json')['complete']
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    encoder = GuardedEncoder(); catalog = read(TRAIN / 'data/monomer-catalog.json')
    reference = {r['uid']: r for r in read(BASE / 'data/sample.json')}
    with np.load(TRAIN / 'results/val_predictions.npz', allow_pickle=False) as f:
        prior_scores = {k: f[k] for k in ['baseline', 'true', 'quality']}
    with np.load(TRAIN / 'results/val_evidence.npz', allow_pickle=False) as f:
        prior_evidence = {k: f[k] for k in ARMS}
    cases = []
    # Fixed before TEST feature extraction: short, boundary and upper-length examples.
    for uid in ['val-025136', 'val-057993', 'train-012468', 'val-001182', 'train-047928', 'train-027500']:
        row = reference[uid]; a, b = sorted([row['a'], row['b']])
        ma, mb = [load_monomer(PROJECT / catalog[str(p)]['path']) for p in [a, b]]
        before = time.monotonic(); tokens, meta = pair(ma, mb, cfg); input_seconds = time.monotonic() - before
        expected = read(BASE / 'features' / (uid + '.json'))
        assert tokens is not None and expected['msa'] == meta
        q = quality_features(ma, mb, meta); np.testing.assert_array_equal(q, expected['quality'])
        value, timing = encoder.encode(tokens, meta['length_a']); np.testing.assert_array_equal(value, expected['true'])
        error = None
        if row['split'] == 'val':
            features = {'true': value[None, :], 'quality': q[None, :], 'gate': np.asarray([meta['gate']]), 'available': np.asarray([True])}
            scores, evidence = score(features, prior_scores['baseline'][[row['row']]], heads, cfg['fixed_alphas'])
            error = 0.
            for arm in ARMS:
                np.testing.assert_allclose(scores[arm], prior_scores[arm][[row['row']]], rtol=0, atol=1e-10)
                np.testing.assert_allclose(evidence[arm], prior_evidence[arm][[row['row']]], rtol=0, atol=1e-10)
                error = max(error, float(abs(scores[arm][0] - prior_scores[arm][row['row']])))
        cases.append({'uid': uid, 'split': row['split'], 'length': row['length'], 'depth': len(tokens),
            'timing': timing, 'input_seconds': input_seconds, 'original_feature_exact': True,
            'original_quality_exact': True, 'frozen_val_prediction_max_error': error})
    inputs = Inputs(reuse=False); rows = read(ROOT / 'data/pairs.json')
    chosen = []
    for lo, hi in [(0, 512), (1280, 1536)]:
        candidates = sorted([r for r in rows if lo < r['length'] <= hi], key=lambda r: (-r['length'], r['uid']))
        for row in candidates:
            before = time.monotonic(); tokens, q, info = inputs.materialize(row); elapsed = time.monotonic() - before
            if tokens is not None:
                chosen.append((row, tokens, q, info, elapsed)); break
        else:
            raise ValueError('No qualified TEST case in the fixed length bin')
    for row, tokens, q, info, input_seconds in chosen:
        a, b = sorted([row['a'], row['b']]); np.testing.assert_array_equal(q, quality_features(inputs.monomer(a), inputs.monomer(b), info['msa']))
        value, timing = encoder.encode(tokens, info['msa']['length_a'])
        repeat, repeat_timing = encoder.encode(tokens, info['msa']['length_a']); np.testing.assert_array_equal(value, repeat)
        boundary = info['msa']['length_a']; reverse = np.concatenate([tokens[:, boundary:], tokens[:, :boundary]], axis=1)
        swapped, swap_timing = encoder.encode(reverse, tokens.shape[1] - boundary); np.testing.assert_array_equal(value, swapped)
        cases.append({'uid': row['uid'], 'split': 'test', 'length': row['length'], 'depth': len(tokens),
            'timing': timing, 'input_seconds': input_seconds, 'original_quality_exact': True,
            'repeat_and_symmetry_exact': True, 'token_sha256': info['token_sha256']})
    # Qualified source timings remain a floor; a faster rerun cannot lower the bound.
    source_q = read(TRAIN / 'qualification/encoder.json')
    measured = list(source_q['upper_length_timing_buckets'])
    measured += [(c['length'], c['input_seconds'] + c['timing']['seconds']) for c in cases]
    buckets = []; bound = 0.
    for length in sorted({int(x[0]) for x in measured}):
        bound = max(bound, max(t for l, t in measured if l == length)); buckets.append((length, bound))
    assert buckets[-1][0] == cfg['maximum_combined_residues']
    def cost(row):
        entries = [inputs.catalog.get(str(row[k])) for k in ['a', 'b']]
        candidate = row['length'] <= 1536 and all(e and e['quality_fraction'] >= .5 for e in entries)
        if not candidate:
            return .01
        return next(seconds for length, seconds in buckets if row['length'] <= length) + .02
    row_costs = [cost(r) for r in rows]
    estimated = [sum(row_costs[i] for i in shard) for shard in read(ROOT / 'data/shards.json')]
    conservative = math.ceil((max(estimated) * cfg['budgets']['timing_safety_factor'] + 600 + cfg['budgets']['analysis_reserve_seconds']) / 36) / 100
    timing = {'at_utc': now(), 'upper_length_timing_buckets': buckets, 'estimated_shard_seconds': estimated,
        'estimated_allocated_gpu_hours_extraction': max(estimated) * 4 / 3600,
        'conservative_wall_hours': conservative, 'registered_wall_hours': cfg['budgets']['batch_wall_hours'],
        'labels_used': False, 'source_timing_floor': 'retrain-vx/qualification/encoder.json'}
    atomic(ROOT / 'qualification/timing-estimate.json', timing)
    if conservative > cfg['budgets']['batch_wall_hours']:
        raise TimeoutError('Conservative forecast exceeds the registered allocation')
    pinned = [ROOT / 'scripts' / n for n in ['qualify.py', 'predict.py', 'shared.py']]
    pinned += [TRAIN / 'scripts' / n for n in ['inputs.py', 'guarded_encoder.py', 'heads.py', 'chunks.py']]
    pinned += [BASE / 'scripts' / n for n in ['common.py', 'msa.py', 'encoder.py']]
    atomic(ROOT / 'qualification/encoder.json', {'at_utc': now(), 'passed': True, 'cases': cases,
        'gpu': torch.cuda.get_device_name(0), 'conservative_wall_hours': conservative,
        'heads_sha256': cfg['heads_sha256'], 'heads_fitted': False, 'test_scores_computed': False,
        'shuffle_called': False, 'forbidden_heads_called': False,
        'code_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in pinned}, 'seconds': time.monotonic() - started})
    print({'qualified': True, 'cases': len(cases), 'conservative_wall_hours': conservative}, flush=True)


if __name__ == '__main__':
    main()
