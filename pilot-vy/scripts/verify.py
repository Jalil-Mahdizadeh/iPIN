"""Independent CPU reconstruction and sklearn bootstrap; no outcome refitting."""
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from study import ROOT, config, read, sha, atomic, now, verify_freeze
from analyze import load_complete, baseline_and_references
from statistics_vy import CONTRASTS


def apply_transform(h, x):
    centered = (x - np.asarray(h['mean'])) / np.asarray(h['scale'])
    centered -= np.asarray(h['pca_mean'])
    return centered.dot(np.asarray(h['components']).T)


def main():
    started = time.monotonic(); cfg = config(); fp = verify_freeze()
    sample, monomers, pairs, ids, available, gate, raw = load_complete(fp)
    model = read(ROOT / 'results/heads.json'); metrics = read(ROOT / 'results/metrics.json'); decision = read(ROOT / 'results/decision.json')
    if any(x['fingerprint'] != fp for x in [model, metrics, decision]):
        raise ValueError('Result fingerprint mismatch')
    y = np.array([r['label'] for r in sample]); train = np.array([r['split'] == 'train' for r in sample]); dev = ~train
    fit = train & available; cal = np.array([r['role'] == 'calibration' for r in sample]); ass = np.array([r['role'] == 'assessment' for r in sample])
    fitids = {str(r[k]) for r, keep in zip(sample, fit) if keep for k in ('a', 'b')}
    indices = np.array([i for i, p in enumerate(ids) if p in fitids]); lookup = {p: i for i, p in enumerate(ids)}
    if model['fit_pair_uids'] != [r['uid'] for r, keep in zip(sample, fit) if keep] or model['TRAIN_baseline_used']:
        raise ValueError('Invalid recorded fitting population')
    pair_x = {}; transform_checks = {}
    for arm in ['M', 'S', 'P']:
        h = model['transforms'][arm]
        if h['fit_protein_ids'] != [ids[i] for i in indices] or len(h['fit_protein_ids']) != len(set(h['fit_protein_ids'])):
            raise ValueError('Transform fitted on wrong/repeated protein population')
        sc = StandardScaler().fit(raw[arm][indices])
        np.testing.assert_allclose(h['mean'], sc.mean_, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(h['scale'], sc.scale_, atol=1e-12, rtol=1e-12)
        sx = sc.transform(raw[arm][indices]); comp = np.asarray(h['components'])
        np.testing.assert_allclose(h['pca_mean'], sx.mean(0), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(comp @ comp.T, np.eye(len(comp)), atol=1e-10, rtol=1e-10)
        ratios = ((sx - sx.mean(0)) @ comp.T).var(0, ddof=1) / sx.var(0, ddof=1).sum()
        np.testing.assert_allclose(h['explained_variance_ratio'], ratios, atol=1e-10, rtol=1e-10)
        z = apply_transform(h, raw[arm]); x = np.zeros((len(sample), 3 * z.shape[1]))
        for i in np.flatnonzero(available):
            a, b = [z[lookup[str(sample[i][k])]] for k in ('a', 'b')]
            x[i] = np.r_[(a + b) / 2, np.abs(a - b), a * b]
        pair_x[arm] = x; transform_checks[arm] = {'unique_train_proteins': len(indices), 'retained_variance': float(ratios.sum())}
    pair_x['SP'] = np.concatenate([pair_x['S'], pair_x['P']], axis=1)
    pair_x['U'] = pair_x['M'][:, :cfg['fusion']['protein_pca_components']]
    z = apply_transform(model['transforms']['M'], raw['S'])
    cf = np.zeros_like(pair_x['M'])
    for i in np.flatnonzero(available):
        a, b = [z[lookup[str(sample[i][k])]] for k in ('a', 'b')]
        cf[i] = np.r_[(a + b) / 2, np.abs(a - b), a * b]
    pair_x['M_on_S'] = cf
    base, scores, oldavailable = baseline_and_references(sample)
    evidence = {}; calibration_checks = {}
    for arm in [*cfg['arms'], 'M_on_S']:
        h = model['heads']['M' if arm == 'M_on_S' else arm]; x = pair_x[arm]
        if arm != 'M_on_S':
            sc = StandardScaler().fit(x[fit])
            np.testing.assert_allclose(h['mean'], sc.mean_, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(h['scale'], sc.scale_, atol=1e-12, rtol=1e-12)
            assert h['dimensions'] == x.shape[1] and h['fit_rows'] == int(fit.sum())
        ev = ((x - h['mean']) / h['scale']).dot(np.asarray(h['coef']).ravel()) + h['intercept'][0]
        evidence[arm] = ev
        z = base.copy(); z[available] = base[available] + h['alpha'] * gate[available] * ev[available]
        np.testing.assert_array_equal(z[~available], base[~available]); scores[arm] = z
        if arm != 'M_on_S':
            choices = []
            for alpha in cfg['fusion']['alpha_grid']:
                zz = base.copy(); zz[available] += alpha * gate[available] * ev[available]
                ap = float(average_precision_score(y[cal], zz[cal])); choices.append((alpha, ap))
            best = max(ap for _, ap in choices); selected = min(a for a, ap in choices if ap == best)
            if selected != h['alpha']:
                raise ValueError('Fusion alpha was not selected on calibration')
            np.testing.assert_allclose([ap for _, ap in choices], [c['ap'] for c in h['calibration_grid']], atol=1e-12, rtol=1e-12)
            calibration_checks[arm] = selected
    with np.load(ROOT / 'results/dev_predictions.npz', allow_pickle=False) as f:
        saved = {k: f[k].copy() for k in f.files}
    assert saved['uids'].tolist() == [r['uid'] for r, keep in zip(sample, dev) if keep]
    np.testing.assert_array_equal(saved['labels'], y[dev]); np.testing.assert_array_equal(saved['gate'], gate[dev])
    errors = {}
    for name, z in scores.items():
        errors[name] = float(np.max(np.abs(saved[name] - z[dev])))
        np.testing.assert_allclose(saved[name], z[dev], atol=1e-10, rtol=1e-10)
    with np.load(ROOT / 'results/dev_evidence.npz', allow_pickle=False) as f:
        for name, ev in evidence.items():
            np.testing.assert_allclose(f[name], ev[dev], atol=1e-10, rtol=1e-10)
    scopes = {'assessment': ass, 'calibration': cal, 'fixed_dev_sample': dev,
              'crossing_descriptive': np.array([r['role'] == 'crossing' for r in sample])}
    for pop in ['assessment', 'fixed_dev_sample']:
        for name, group in [('both', available & oldavailable), ('vy_only', available & ~oldavailable),
                            ('vx_only', ~available & oldavailable), ('neither', ~available & ~oldavailable)]:
            scopes[pop + '/availability_' + name] = scopes[pop] & group
    for name, mask in scopes.items():
        g = metrics['results'][name]
        if g['rows'] != int(mask.sum()) or g['positive'] != int(y[mask].sum()):
            raise ValueError('Wrong metric population')
        if len(np.unique(y[mask])) < 2:
            continue
        for arm, z in scores.items():
            for metric, fn in [('ap', average_precision_score), ('auroc', roc_auc_score)]:
                if abs(fn(y[mask], z[mask]) - g['metrics'][arm][metric]) > 1e-12:
                    raise ValueError('Stored metric differs from independent reconstruction')
    # Reconstruct the joint all-DEV protein multipliers, but use sklearn directly for each AP.
    rows = [r for r, keep in zip(sample, dev) if keep]; yy = y[dev]; mask = ass[dev]
    pids = sorted({r[k] for r in rows for k in ('a', 'b')}); index = {p: i for i, p in enumerate(pids)}
    ia = np.array([index[r['a']] for r in rows]); ib = np.array([index[r['b']] for r in rows])
    rng = np.random.default_rng(cfg['seed']); values = {k: [] for k in CONTRASTS}
    for _ in range(cfg['fusion']['bootstrap_replicates']):
        count = rng.poisson(1, len(pids)); weights = (count[ia] * count[ib])[mask]
        if sum(weights[yy[mask] == 0]) == 0 or sum(weights[yy[mask] == 1]) == 0:
            continue
        ap = {name: average_precision_score(yy[mask], saved[name][mask], sample_weight=weights) for name in scores}
        for name, terms in CONTRASTS.items():
            values[name].append(float(sum(c * ap[k] for k, c in terms.items())))
    intervals = {}
    for name, data in values.items():
        bound = np.quantile(data, [.025, .975]); previous = metrics['results']['assessment']['ap_intervals'][name]
        np.testing.assert_allclose(bound, [previous['low'], previous['high']], atol=1e-10, rtol=1e-10)
        assert len(data) == previous['replicates']
        intervals[name] = {'low': float(bound[0]), 'high': float(bound[1]), 'replicates': len(data)}
    point = metrics['results']['assessment']['contrasts']['M_minus_SP']
    if decision['primary_point'] != point or decision['metrics_sha256'] != sha(ROOT / 'results/metrics.json'):
        raise ValueError('Decision/result integrity failure')
    atomic(ROOT / 'results/verification.json', {'at_utc': now(), 'passed': True, 'fingerprint': fp,
        'required_monomers_verified': len(ids), 'pairs_verified': len(sample), 'complete_coverage': True,
        'transforms': transform_checks, 'calibration_alphas_independently_reproduced': calibration_checks,
        'maximum_prediction_reconstruction_error': max(errors.values()), 'prediction_errors': errors,
        'independent_sklearn_assessment_intervals': intervals, 'metric_populations_reproduced': list(scopes),
        'heads_refitted': False, 'test_accessed': False, 'exact_baseline_fallback': True,
        'scientific_result_hashes': {name: sha(ROOT / 'results' / name) for name in ['heads.json', 'metrics.json', 'decision.json', 'dev_predictions.npz', 'dev_evidence.npz']},
        'script_sha256': sha(ROOT / 'scripts/verify.py'), 'seconds': time.monotonic() - started})
    print({'independent_verification_passed': True, 'seconds': time.monotonic() - started}, flush=True)


if __name__ == '__main__':
    main()
