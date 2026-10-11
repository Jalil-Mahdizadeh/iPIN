"""Independent numerical/source audit without another classifier fit."""
import hashlib
import time
from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from shared import ROOT, PROJECT, BASE, config, read, atomic, sha, now, verify_freeze, ARMS
from collect import collect
from inputs import Inputs
from statistics_rx import intervals
from msa import pair, quality_features


def main():
    started = time.monotonic(); cfg = config(); fp = verify_freeze()
    rows = read(ROOT / 'data/pairs.json'); shards = read(ROOT / 'data/shards.json')
    x, records, workers = collect(ROOT, rows, shards, fp, cfg['chunk_rows'])
    y = np.asarray([r['label'] for r in rows]); train = np.asarray([r['split'] == 'train' for r in rows]); dev = ~train
    cal = np.asarray([r['role'] == 'calibration' for r in rows]); available = x['available']; gate = x['gate']; fit = train & available
    groups = read(ROOT / 'data/dev-groups.json'); old_ids = {r['uid'] for r in read(BASE / 'data/sample.json')}
    for split, expected in [('train', cfg['train_pairs']), ('val', cfg['dev_pairs'])]:
        subset = [r for r in rows if r['split'] == split]
        source = np.load(PROJECT / f'retrain-v1/data/prepared/{split}.raw.npy', allow_pickle=False)
        assert len(subset) == expected
        np.testing.assert_array_equal([[r['a'], r['b'], r['label'], r['row']] for r in subset], source)
    for r in rows:
        if r['pilot_row'] != (r['uid'] in old_ids):
            raise ValueError('Pilot/new-row assignment changed')
        if r['split'] == 'val':
            a, b = [groups['assignments'][str(r[k])] for k in ['a', 'b']]
            if r['role'] != (a if a == b else 'crossing'):
                raise ValueError('DEV assignment changed')
    with np.load(ROOT / 'data/baseline.npz', allow_pickle=False) as f:
        baseline = f['baseline']
    base = np.zeros(len(rows)); base[dev] = baseline
    heads = read(ROOT / 'results/heads.json'); metrics = read(ROOT / 'results/metrics.json')
    if set(heads) != set(ARMS):
        raise ValueError('Unexpected head/arm')
    with np.load(ROOT / 'results/val_predictions.npz', allow_pickle=False) as f:
        saved = {k: f[k] for k in f.files}
    if set(saved) != {'uid', 'y', 'gate', 'baseline', *ARMS}:
        raise ValueError('Unexpected prediction arms')
    np.testing.assert_array_equal(saved['uid'], [r['uid'] for r in rows if r['split'] == 'val'])
    np.testing.assert_array_equal(saved['y'], y[dev]); np.testing.assert_array_equal(saved['gate'], gate[dev])
    scores = {'baseline': base}; errors = {}; alphas = {}; evidence = {}
    for arm, h in heads.items():
        values = x[arm]; scaler = StandardScaler().fit(values[fit]); sx = scaler.transform(values[fit])
        np.testing.assert_allclose(scaler.mean_, h['standard_mean'], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(scaler.scale_, h['standard_scale'], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(sx.mean(0), h['pca_mean'], atol=1e-12, rtol=1e-12)
        pc = np.asarray(h['pca_components'])
        np.testing.assert_allclose(pc @ pc.T, np.eye(32), atol=1e-10, rtol=1e-10)
        if pc.shape != (32, ARMS[arm]) or h['classes'] != [0, 1] or h['fit_rows'] != int(fit.sum()):
            raise ValueError('Wrong head shape/classes/TRAIN coverage')
        # Reconstruct directly, without calling analysis fitting/fusion helpers.
        centered = (values - np.asarray(h['standard_mean'])) / np.asarray(h['standard_scale']) - np.asarray(h['pca_mean'])
        ev = (centered @ pc.T @ np.asarray(h['coef']).T).ravel() + h['intercept'][0]
        if not np.isfinite(ev).all():
            raise FloatingPointError('Nonfinite reconstructed evidence')
        evidence[arm] = ev; choices = []
        for alpha in cfg['fusion']['alpha_grid']:
            z = base.copy(); z[available] += alpha * gate[available] * ev[available]
            choices.append((alpha, float(average_precision_score(y[cal], z[cal]))))
        best = max(v for a, v in choices); alpha = min(a for a, v in choices if v == best)
        if alpha != h['alpha']:
            raise ValueError('Calibration-selected alpha differs')
        np.testing.assert_allclose([v for a, v in choices], [v['ap'] for v in h['calibration_grid']], atol=1e-12, rtol=1e-12)
        alphas[arm] = alpha; z = base.copy(); z[available] += alpha * gate[available] * ev[available]; scores[arm] = z
        np.testing.assert_array_equal(z[dev & ~available], base[dev & ~available])
    for arm, values in scores.items():
        np.testing.assert_allclose(saved[arm], values[dev], atol=1e-10, rtol=1e-10)
        errors[arm] = float(np.max(np.abs(saved[arm] - values[dev])))
    with np.load(ROOT / 'results/val_evidence.npz', allow_pickle=False) as f:
        for arm, ev in evidence.items():
            np.testing.assert_allclose(f[arm], ev[dev], atol=1e-10, rtol=1e-10)
    role = np.asarray([r['role'] for r in rows]); old = np.asarray([r['pilot_row'] for r in rows])
    populations = {'assessment_new_rows': (role == 'assessment') & ~old, 'assessment_all': role == 'assessment',
        'calibration': role == 'calibration', 'crossing': role == 'crossing', 'full_val': dev, 'pilot_val': dev & old}
    checked_intervals = {}
    for name, mask in populations.items():
        result = metrics['results'][name]
        if result['rows'] != int(mask.sum()) or result['eligible'] != int((mask & available).sum()):
            raise ValueError('Reported population differs')
        for arm, values in scores.items():
            for key, fn in [('ap', average_precision_score), ('auroc', roc_auc_score)]:
                if abs(fn(y[mask], values[mask]) - result['metrics'][arm][key]) > 1e-12:
                    raise ValueError('Reported metric differs')
        if name in ['assessment_new_rows', 'assessment_all', 'full_val']:
            subset = [r for r, keep in zip(rows, mask) if keep]
            ci = intervals(subset, y[mask], {k: v[mask] for k, v in scores.items()}, cfg['fusion']['bootstrap_replicates'], cfg['seed'], independent=True)
            for contrast, v in ci.items():
                ref = result['intervals'][contrast]
                np.testing.assert_allclose([v['low'], v['high']], [ref['low'], ref['high']], atol=1e-10, rtol=1e-10)
                assert v['replicates'] == ref['replicates']
            checked_intervals[name] = ci
    inputs = Inputs(); buckets = defaultdict(list); source_reuses = 0
    for i, (row, record) in enumerate(zip(rows, records)):
        if record['available']:
            m = record['msa']; expected = min(1, m['neff'] / 32) * min(m['quality_a'], m['quality_b'])
            assert expected == gate[i] and record['depth'] == m['retained_homologs'] + 1
            assert len(m['genome_keys']) == len(m['taxonomy']) == m['retained_homologs']
        if record['kind'] == 'reused_original':
            old_feature = inputs.old_feature(row, record, x['quality'][i])
            np.testing.assert_array_equal(old_feature, x['true'][i]); source_reuses += 1
        key = (record['reason'], min(3, (row['length'] - 1) // 512), record['depth'] // 32)
        buckets[key].append(i)
    checked_inputs = []
    for key in sorted(buckets):
        selected = sorted(buckets[key], key=lambda i: hashlib.sha256(f"{cfg['seed']}:audit:{rows[i]['uid']}".encode()).hexdigest())[:4]
        for i in selected:
            row = rows[i]; tokens, quality, info = inputs.materialize(row); stored = records[i]
            for k, value in info.items():
                if stored[k] != value:
                    raise ValueError('Replayed input differs: ' + k)
            np.testing.assert_array_equal(quality, x['quality'][i])
            if info['msa'] is not None:
                a, b = inputs.monomer(info['a']), inputs.monomer(info['b'])
                original_tokens, original_meta = pair(a, b, cfg)
                assert original_meta == info['msa']
                if tokens is not None:
                    np.testing.assert_array_equal(tokens, original_tokens)
                np.testing.assert_array_equal(quality_features(a, b, original_meta), quality)
            checked_inputs.append(row['uid'])
    primary = metrics['results'][cfg['primary']]; m = primary['metrics']; ci = primary['intervals']
    baseline_gain = m['true']['ap'] - m['baseline']['ap'] >= .010 and ci['true_minus_baseline']['low'] > 0
    quality_gain = ci['true_minus_quality']['low'] > 0; auroc_ok = m['true']['auroc'] >= m['baseline']['auroc'] - .005
    answer = read(ROOT / 'results/decision.json')
    expected_status = 'useful_predictive_gain_supported' if baseline_gain and quality_gain and auroc_ok else 'no_go_or_inconclusive'
    assert answer['status'] == expected_status and answer['metrics_sha256'] == sha(ROOT / 'results/metrics.json')
    assert not answer['test_authorized'] and not metrics['shuffled_arm']
    names = ['heads.json', 'metrics.json', 'decision.json', 'val_predictions.npz', 'val_evidence.npz', 'REPORT.md']
    result = {'at_utc': now(), 'passed': True, 'fingerprint': fp, 'rows_verified': len(rows),
        'eligible_rows': int(available.sum()), 'source_reuses_verified': source_reuses,
        'source_replay_uids': checked_inputs, 'source_replay_policy': 'Four deterministic rows per reason/length/depth bin; no outcome selection',
        'TRAIN_only_transforms_verified': True, 'classifier_refitted': False, 'native_fallback_exact': True,
        'alphas': alphas, 'maximum_prediction_error': max(errors.values()), 'independent_intervals': checked_intervals,
        'result_hashes': {n: sha(ROOT / 'results' / n) for n in names}, 'test_accessed': False,
        'shuffled_arm': False, 'seconds': time.monotonic() - started}
    atomic(ROOT / 'results/verification.json', result); print({'verified': True, 'rows': len(rows)}, flush=True)


if __name__ == '__main__':
    main()
