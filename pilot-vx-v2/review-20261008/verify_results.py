"""Independent CPU reconstruction of completed v2 results; no fitting or TEST access."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
from study import ROOT, atomic, read, sha, now, config, verify_freeze, load_record


def predict(h, x):
    z = (x - np.array(h['standard_mean'])) / np.array(h['standard_scale'])
    p = (z - np.array(h['pca_mean'])) @ np.array(h['pca_components']).T
    return (p @ np.array(h['coef']).T + np.array(h['intercept'])).ravel()


def summary(values):
    a = np.asarray(values, float)
    return {'mean': float(a.mean()), **dict(zip(['min', 'q25', 'median', 'q75', 'max'],
        map(float, np.quantile(a, [0, .25, .5, .75, 1]))))}


def main():
    started = time.monotonic(); cfg = config(); fp = verify_freeze()
    metrics_path = ROOT / 'results/metrics.json'; heads_path = ROOT / 'results/heads.json'
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in [metrics_path, heads_path,
        ROOT / 'results/decision.json', ROOT / 'results/dev_predictions.npz']}
    result = read(metrics_path); heads = read(heads_path); decision = read(ROOT / 'results/decision.json')
    assert decision['metrics_sha256'] == sha(metrics_path)
    assert result['fingerprint'] == fp
    sample = read(ROOT / 'data/sample.json'); previous = read(ROOT / 'data/source_metadata.json')
    assert len(sample) == 8000 and {r['split'] for r in sample} == {'train', 'val'}
    with ThreadPoolExecutor(max_workers=8) as executor:
        data = list(executor.map(lambda r: load_record(r['uid'], fp), sample))
    assert all(d is not None for d in data)
    # NPZ members are lazily decompressed on every lookup; materialize once.
    with np.load(ROOT / 'data/source.npz', allow_pickle=False) as snapshot:
        old = {name: snapshot[name] for name in snapshot.files}
    assert list(old['uids']) == [r['uid'] for r in sample]
    gate = np.array([r['gate'] for r in data]); active = gate > 0
    y = np.array([r['label'] for r in sample]); train = np.array([r['split'] == 'train' for r in sample]); dev = ~train
    ass = np.array([r['role'] == 'assessment' for r in sample]); cal = np.array([r['role'] == 'calibration' for r in sample])
    length = np.array([r['length'] for r in sample]); fit = train & active
    for i, (r, d, p) in enumerate(zip(sample, data, previous)):
        assert d['uid'] == r['uid'] and [d['a'], d['b']] == sorted([r['a'], r['b']])
        assert (d['available'], d['gate'], d['reason']) == (p['available'], p['gate'], p['reason'])
        for arm in cfg['arms']:
            kind = 'shuffled' if arm.endswith('shuffled') else 'true'
            np.testing.assert_array_equal(d[arm][:64], old[kind][i, :64])
        if active[i]:
            assert d['global_feature_max_absolute_error'] == {'true': 0., 'shuffled': 0.}
            assert d['pooling']['blocks_qualifying'] >= 1 and 1 <= d['pooling']['k_used'] <= 4
    baseline = read(ROOT / 'data/baseline.json')
    base = np.array([baseline[r['uid']]['score'] if r['split'] == 'val' else 0. for r in sample])
    scores = {'baseline': base}; evidence = {}; features = {}
    def fuse(ev, alpha):
        z = base.copy(); z[active] += alpha * gate[active] * ev[active]
        return z
    old_heads = read(ROOT / 'data/source_heads.json')
    for kind, name in [('true', 'global_true'), ('shuffled', 'global_shuffled'), ('quality', 'quality')]:
        scores[name] = fuse(predict(old_heads[kind], old[kind]), old_heads[kind]['alpha'])
    for arm, h in heads.items():
        x = np.array([d[arm] for d in data]); features[arm] = x
        np.testing.assert_allclose(h['standard_mean'], x[fit].mean(0), atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(h['standard_scale'], x[fit].std(0), atol=1e-11, rtol=1e-11)
        ev = predict(h, x); evidence[arm] = ev
        scores[arm] = fuse(ev, h['alpha'])
        grid = []
        for choice in h['calibration_grid']:
            ap = float(average_precision_score(y[cal], fuse(ev, choice['alpha'])[cal]))
            assert abs(ap - choice['ap']) < 1e-12
            grid.append((ap, choice['alpha']))
        assert h['alpha'] == min(a for ap, a in grid if ap == max(v for v, _ in grid))
    for prefix in ['local', 'scrambled']:
        h = heads[prefix + '_true']
        scores[prefix + '_true_on_shuffled'] = fuse(predict(h, features[prefix + '_shuffled']), h['alpha'])
    with np.load(ROOT / 'results/dev_predictions.npz', allow_pickle=False) as snapshot:
        saved = {name: snapshot[name] for name in snapshot.files}
    np.testing.assert_array_equal(saved['uids'], old['uids'][dev]); np.testing.assert_array_equal(saved['labels'], y[dev])
    max_score_error = 0.
    for name, z in scores.items():
        np.testing.assert_allclose(z[dev], saved[name], atol=1e-12, rtol=1e-12)
        max_score_error = max(max_score_error, float(np.max(np.abs(z[dev] - saved[name]))))
        np.testing.assert_array_equal(z[dev & ~active], base[dev & ~active])
        for pop, mask in [('assessment', ass), ('calibration', cal), ('fixed_dev_sample', dev)]:
            recorded = result['results'][pop]['metrics'][name]
            assert abs(average_precision_score(y[mask], z[mask]) - recorded['ap']) < 1e-12
            assert abs(roc_auc_score(y[mask], z[mask]) - recorded['auroc']) < 1e-12
    # Recheck the primary interval using sklearn's AP rather than the custom fast routine.
    ids = sorted({r[k] for r, keep in zip(sample, dev) if keep for k in ['a', 'b']}); lookup = {p: i for i, p in enumerate(ids)}
    ia = np.array([lookup[r['a']] for r, keep in zip(sample, ass) if keep])
    ib = np.array([lookup[r['b']] for r, keep in zip(sample, ass) if keep])
    rng = np.random.default_rng(cfg['seed']); primary = []; null_gap = []
    for _ in range(1000):
        counts = rng.poisson(1, len(ids)); weights = counts[ia] * counts[ib]
        ap = {name: average_precision_score(y[ass], scores[name][ass], sample_weight=weights)
              for name in ['local_true', 'local_shuffled', 'global_true', 'global_shuffled']}
        primary.append(ap['local_true'] - ap['local_shuffled'] - ap['global_true'] + ap['global_shuffled'])
        null_gap.append(ap['local_true'] - ap['local_shuffled'])
    for name, values in [('pairing_gap_improvement', primary), ('local_minus_shuffled', null_gap)]:
        interval = result['results']['assessment']['ap_intervals'][name]
        np.testing.assert_allclose(np.quantile(values, [.025, .975]), [interval['low'], interval['high']], atol=1e-12, rtol=1e-12)
    # Descriptive sensitivity only: no coefficient is selected or substituted from this table.
    same_alpha = {}
    for pop, mask in [('assessment', ass), ('fixed_dev_sample', dev), ('calibration', cal)]:
        same_alpha[pop] = []
        for alpha in cfg['fusion']['alpha_grid']:
            aps = {arm: float(average_precision_score(y[mask], fuse(ev, alpha)[mask])) for arm, ev in evidence.items()}
            same_alpha[pop].append({'alpha': alpha, 'ap': aps,
                'local_true_minus_shuffled': aps['local_true'] - aps['local_shuffled'],
                'locality_pairing_gap': aps['local_true'] - aps['local_shuffled'] - aps['scrambled_true'] + aps['scrambled_shuffled']})
    diagnostics = {}
    for pop, mask in [('train', fit), ('dev', dev & active), ('assessment', ass & active),
                      ('short_assessment', ass & active & (length <= 512))]:
        selected = [d for d, keep in zip(data, mask) if keep]
        diagnostics[pop] = {'rows': len(selected),
            'actual_sequence_changed_fraction': summary([d['diagnostics']['actual_sequence_changed_fraction'] for d in selected]),
            'actual_token_changed_fraction': summary([d['diagnostics']['actual_token_changed_fraction'] for d in selected]),
            'fraction_pairs_at_least_half_sequences_changed': float(np.mean([d['diagnostics']['actual_sequence_changed_fraction'] >= .5 for d in selected])),
            'neff80': summary([d['diagnostics']['paired_diversity']['neff80'] for d in selected]),
            'fraction_joint_cells_at_least_half_depth': summary([d['diagnostics']['true_support']['fraction_cells_at_least_half_rows'] for d in selected]),
            'qualifying_blocks': summary([d['pooling']['blocks_qualifying'] for d in selected]),
            'identical_local_true_and_spatial_scramble': sum(d['local_true'] == d['scrambled_true'] for d in selected)}
    for name, digest in source_hashes.items(): assert sha(ROOT / name) == digest
    report = {'at_utc': now(), 'fingerprint': fp, 'script_sha256': sha(__file__), 'source_sha256': source_hashes,
        'verified_records': len(data), 'verified_eligible_records': int(active.sum()),
        'original_global_error_zero_all_eligible': True, 'retained_global_statistics_exact': True,
        'train_only_scaler_verified': True, 'calibration_grid_and_selection_verified': True,
        'prediction_and_metric_reconstruction_verified': True, 'maximum_prediction_absolute_error': max_score_error,
        'primary_and_pairing_intervals_independently_reproduced': True, 'exact_baseline_fallback_verified': True,
        'diagnostics': diagnostics, 'equal_alpha_sensitivity': same_alpha,
        'sensitivity_interpretation': 'Posthoc descriptive full predefined alpha grid; no assessment-based selection, refitting, or change to primary results.',
        'new_heads_fitted': False, 'gpu_used': False, 'test_accessed': False, 'original_results_modified': False,
        'seconds': time.monotonic() - started}
    atomic(HERE / 'verification.json', report)
    print(json.dumps({'verified_records': len(data), 'eligible': int(active.sum()), 'max_prediction_error': max_score_error,
        'assessment_equal_alpha_sensitivity': same_alpha['assessment'], 'assessment_diagnostics': diagnostics['assessment'],
        'seconds': report['seconds']}, indent=2), flush=True)


if __name__ == '__main__': main()
