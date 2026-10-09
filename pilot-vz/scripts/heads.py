"""One Q classifier matching the original Vx pipeline; historical heads never refit."""
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from study import read, fuse


def evidence(h, x):
    return ((((x - h['standard_mean']) / h['standard_scale'] - h['pca_mean'])
             @ np.asarray(h['pca_components']).T) @ np.asarray(h['coef']).T + h['intercept']).ravel()


def fit_head(x, y, fit, cal, base, gate, cfg):
    if np.any(fit & cal) or np.any(fit & (gate <= 0)) or not np.isfinite(x).all():
        raise ValueError('Invalid Q fitting population/features')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipe = make_pipeline(StandardScaler(), PCA(n_components=cfg['fusion']['pca_components'], svd_solver='full'),
            LogisticRegression(C=cfg['fusion']['C'], max_iter=cfg['fusion']['max_iter'], solver='lbfgs', random_state=cfg['seed']))
        pipe.fit(x[fit], y[fit]); ev = pipe.decision_function(x)
    choices = [{'alpha': a, 'ap': float(average_precision_score(y[cal], fuse(base, ev, gate, a)[cal]))} for a in cfg['fusion']['alpha_grid']]
    best = max(c['ap'] for c in choices); alpha = min(c['alpha'] for c in choices if c['ap'] == best)
    sc, pc, cl = [s[1] for s in pipe.steps]
    h = {'standard_mean': sc.mean_.tolist(), 'standard_scale': sc.scale_.tolist(), 'pca_mean': pc.mean_.tolist(),
         'pca_components': pc.components_.tolist(), 'pca_explained_variance_ratio': pc.explained_variance_ratio_.tolist(),
         'coef': cl.coef_.tolist(), 'intercept': cl.intercept_.tolist(), 'classes': cl.classes_.tolist(),
         'alpha': alpha, 'calibration_grid': choices, 'fit_rows': int(fit.sum()), 'dimensions': x.shape[1],
         'supervised_parameters': int(cl.coef_.size + cl.intercept_.size), 'iterations': cl.n_iter_.tolist()}
    np.testing.assert_allclose(evidence(h, x), ev, atol=1e-10, rtol=1e-10)
    return h, ev, fuse(base, ev, gate, alpha)


def references(root, sample):
    y = np.array([r['label'] for r in sample]); dev = np.array([r['split'] == 'val' for r in sample])
    baseline = read(root / 'data/baseline.json'); base = np.zeros(len(sample))
    for i in np.flatnonzero(dev):
        r = sample[i]; b = baseline[r['uid']]
        if b['label'] != r['label'] or sorted([b['a'], b['b']]) != sorted([r['a'], r['b']]):
            raise ValueError('Baseline identity mismatch')
        base[i] = b['score']
    with np.load(root / 'data/source.npz', allow_pickle=False) as f:
        if f['uids'].tolist() != [r['uid'] for r in sample]:
            raise ValueError('Historical feature ordering differs')
        old = {k: f[k].copy() for k in ('true', 'shuffled', 'quality', 'gate')}
    heads = read(root / 'data/source_heads.json'); scores = {'baseline': base}; ev = {}
    for arm in ('true', 'shuffled', 'quality'):
        h = heads[arm]; ev[arm] = evidence(h, old[arm]); scores[arm] = fuse(base, ev[arm], old['gate'], h['alpha'])
    previous = read(root / 'data/source_metrics.json')
    for name, mask in [('assessment', np.array([r['role'] == 'assessment' for r in sample])),
                       ('calibration', np.array([r['role'] == 'calibration' for r in sample])), ('fixed_dev_sample', dev)]:
        for arm, z in scores.items():
            ref = previous['results'][name]['metrics'][arm]
            for metric, fn in [('ap', average_precision_score), ('auroc', roc_auc_score)]:
                if abs(fn(y[mask], z[mask]) - ref[metric]) > 1e-12:
                    raise ValueError('Original metric failed to reproduce')
    with np.load(root / 'data/vy_dev_predictions.npz', allow_pickle=False) as f:
        if f['uids'].tolist() != [r['uid'] for r, keep in zip(sample, dev) if keep]:
            raise ValueError('Verified VY reference ordering differs')
        np.testing.assert_array_equal(f['labels'], y[dev])
        for arm, z in scores.items():
            np.testing.assert_allclose(z[dev], f['baseline' if arm == 'baseline' else 'vx_' + arm], atol=1e-10, rtol=1e-10)
    return base, scores, ev, old['gate']
