"""Independent CPU prediction and sklearn interval reconstruction; no classifier refit."""
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from study import ROOT, config, read, atomic, now, sha, verify_freeze
from analyze import load_complete
from statistics_vz import CONTRASTS
from decision import decide


def manual(h, x):
    x = np.asarray(x, dtype=float)
    centered = (x - np.asarray(h['standard_mean'])) / np.asarray(h['standard_scale'])
    projected = (centered - np.asarray(h['pca_mean'])).dot(np.asarray(h['pca_components']).T)
    return projected.dot(np.asarray(h['coef']).ravel()) + h['intercept'][0]


def main():
    started = time.monotonic(); cfg = config(); fp = verify_freeze()
    sample, pairs, x = load_complete(fp); y = np.array([r['label'] for r in sample])
    gate = np.array([p['gate'] for p in pairs]); available = gate > 0
    train = np.array([r['split'] == 'train' for r in sample]); dev = ~train; fit = train & available
    cal = np.array([r['role'] == 'calibration' for r in sample]); ass = np.array([r['role'] == 'assessment' for r in sample])
    model = read(ROOT / 'results/heads.json'); metrics = read(ROOT / 'results/metrics.json'); decision = read(ROOT / 'results/decision.json')
    if any(item['fingerprint'] != fp for item in (model, metrics, decision)):
        raise ValueError('Results fingerprint differs')
    if model['fit_pair_uids'] != [r['uid'] for r, keep in zip(sample, fit) if keep] or model['TRAIN_baseline_used']:
        raise ValueError('Q TRAIN population differs')
    h = model['Q']; sc = StandardScaler().fit(x[fit]); sx = sc.transform(x[fit]); pc = np.asarray(h['pca_components'])
    np.testing.assert_allclose(sc.mean_, h['standard_mean'], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(sc.scale_, h['standard_scale'], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(sx.mean(0), h['pca_mean'], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(pc @ pc.T, np.eye(len(pc)), atol=1e-10, rtol=1e-10)
    ratios = ((sx - sx.mean(0)) @ pc.T).var(0, ddof=1) / sx.var(0, ddof=1).sum()
    np.testing.assert_allclose(ratios, h['pca_explained_variance_ratio'], atol=1e-10, rtol=1e-10)
    if pc.shape != (cfg['fusion']['pca_components'], 128) or h['fit_rows'] != int(fit.sum()) or h['classes'] != [0, 1]:
        raise ValueError('Wrong Q head dimensions/classes')
    baseline = read(ROOT / 'data/baseline.json')
    base = np.array([baseline[r['uid']]['score'] if r['split'] == 'val' else 0. for r in sample])
    scores = {'baseline': base}; evidence = {}; alphas = {}
    with np.load(ROOT / 'data/source.npz', allow_pickle=False) as f:
        old = {k: f[k].copy() for k in f.files}
    if old['uids'].tolist() != [r['uid'] for r in sample]:
        raise ValueError('Source row mismatch')
    np.testing.assert_array_equal(old['gate'], gate)
    heads = {'Q': h, **read(ROOT / 'data/source_heads.json')}
    for arm, hh in heads.items():
        ev = manual(hh, x if arm == 'Q' else old[arm]); evidence[arm] = ev; alphas[arm] = hh['alpha']
        if not np.isfinite(ev[available]).all():
            raise ValueError('Nonfinite head evidence')
        z = base.copy(); z[available] += hh['alpha'] * gate[available] * ev[available]; scores[arm] = z
        np.testing.assert_array_equal(z[~available], base[~available])
    choices = []
    for alpha in cfg['fusion']['alpha_grid']:
        z = base.copy(); z[available] += alpha * gate[available] * evidence['Q'][available]
        choices.append((alpha, float(average_precision_score(y[cal], z[cal]))))
    best = max(ap for _, ap in choices); selected = min(a for a, ap in choices if ap == best)
    if selected != h['alpha']:
        raise ValueError('Incorrect calibration selection')
    np.testing.assert_allclose([ap for _, ap in choices], [v['ap'] for v in h['calibration_grid']], atol=1e-12, rtol=1e-12)
    with np.load(ROOT / 'results/dev_predictions.npz', allow_pickle=False) as f:
        saved = {k: f[k].copy() for k in f.files}
    if saved['uids'].tolist() != [r['uid'] for r, keep in zip(sample, dev) if keep]:
        raise ValueError('Saved prediction row mismatch')
    np.testing.assert_array_equal(saved['labels'], y[dev]); np.testing.assert_array_equal(saved['gate'], gate[dev])
    errors = {}
    for arm, z in scores.items():
        errors[arm] = float(np.max(np.abs(z[dev] - saved[arm])))
        np.testing.assert_allclose(z[dev], saved[arm], atol=1e-10, rtol=1e-10)
    with np.load(ROOT / 'results/dev_evidence.npz', allow_pickle=False) as f:
        for arm, ev in evidence.items():
            np.testing.assert_allclose(f[arm], ev[dev], atol=1e-10, rtol=1e-10)
    with np.load(ROOT / 'data/vy_dev_predictions.npz', allow_pickle=False) as f:
        for arm in ('baseline', 'true', 'shuffled', 'quality'):
            np.testing.assert_allclose(scores[arm][dev], f['baseline' if arm == 'baseline' else 'vx_' + arm], atol=1e-10, rtol=1e-10)
    scopes = {'assessment': ass, 'assessment/eligible': ass & available, 'assessment/fallback': ass & ~available,
              'calibration': cal, 'fixed_dev_sample': dev, 'crossing_descriptive': np.array([r['role'] == 'crossing' for r in sample])}
    for pop, mask in scopes.items():
        group = metrics['results'][pop]
        if group['rows'] != int(mask.sum()) or group['positive'] != int(y[mask].sum()):
            raise ValueError('Incorrect reported population')
        if len(np.unique(y[mask])) < 2:
            continue
        for arm, z in scores.items():
            for metric, fn in [('ap', average_precision_score), ('auroc', roc_auc_score)]:
                if abs(fn(y[mask], z[mask]) - group['metrics'][arm][metric]) > 1e-12:
                    raise ValueError('Incorrect reported metric')
    # Match the ORIGINAL assessment-only universe, independently using sklearn AP.
    rows = [r for r, keep in zip(sample, ass) if keep]; yy = y[ass]
    ids = sorted({r[k] for r in rows for k in ('a', 'b')}); lookup = {p: i for i, p in enumerate(ids)}
    ia = np.array([lookup[r['a']] for r in rows]); ib = np.array([lookup[r['b']] for r in rows]); rng = np.random.default_rng(cfg['seed'])
    subsets = {'assessment': np.ones(len(rows), dtype=bool), 'assessment/eligible': available[ass]}
    values = {pop: {name: [] for name in CONTRASTS} for pop in subsets}
    for _ in range(cfg['fusion']['bootstrap_replicates']):
        counts = rng.poisson(1, len(ids)); w = counts[ia] * counts[ib]
        for pop, m in subsets.items():
            if any(w[m & (yy == c)].sum() == 0 for c in (0, 1)):
                continue
            aps = {arm: average_precision_score(yy[m], z[ass][m], sample_weight=w[m]) for arm, z in scores.items()}
            for name, terms in CONTRASTS.items():
                values[pop][name].append(float(sum(c * aps[k] for k, c in terms.items())))
    intervals = {}
    for pop, results in values.items():
        intervals[pop] = {}
        for name, data in results.items():
            bounds = np.quantile(data, [.025, .975]); ref = metrics['results'][pop]['ap_intervals'][name]
            np.testing.assert_allclose(bounds, [ref['low'], ref['high']], atol=1e-10, rtol=1e-10)
            if len(data) != ref['replicates']:
                raise ValueError('Bootstrap replicate counts differ')
            intervals[pop][name] = {'low': float(bounds[0]), 'high': float(bounds[1]), 'replicates': len(data)}
    old_metrics = read(ROOT / 'data/source_metrics.json')
    for name, oldname in [('true_minus_baseline', 'true-minus-baseline'), ('true_minus_shuffled', 'true-minus-shuffled')]:
        c = intervals['assessment'][name]; ref = old_metrics['results']['assessment']['ap_intervals'][oldname]
        np.testing.assert_allclose([c['low'], c['high']], [ref['low'], ref['high']], atol=1e-10, rtol=1e-10)
    reproduced = decide(metrics['results'], cfg)
    if any(decision[k] != v for k, v in reproduced.items()) or decision['metrics_sha256'] != sha(ROOT / 'results/metrics.json'):
        raise ValueError('Decision does not match frozen criteria and metrics')
    atomic(ROOT / 'results/verification.json', {'at_utc': now(), 'passed': True, 'fingerprint': fp, 'pairs_verified': len(sample),
        'required_pairs_verified': int(available.sum()), 'complete_coverage': True, 'Q_alpha_independently_reproduced': selected,
        'maximum_prediction_reconstruction_error': max(errors.values()), 'prediction_errors': errors,
        'independent_sklearn_intervals': intervals, 'TRAIN_only_transforms_verified': True, 'exact_baseline_fallback': True,
        'original_predictions_and_intervals_reproduced': True, 'heads_refitted': False, 'test_accessed': False,
        'scientific_result_hashes': {name: sha(ROOT / 'results' / name) for name in ['heads.json', 'metrics.json', 'decision.json', 'dev_predictions.npz', 'dev_evidence.npz']},
        'script_sha256': sha(ROOT / 'scripts/verify.py'), 'seconds': time.monotonic() - started})
    print({'independent_verification_passed': True, 'seconds': time.monotonic() - started}, flush=True)


if __name__ == '__main__':
    main()
