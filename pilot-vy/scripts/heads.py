"""Distinct-TRAIN-protein transforms; symmetric low-capacity pair classifiers."""
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.exceptions import ConvergenceWarning
from study import fuse


def transform(h, x):
    return ((np.asarray(x, dtype=float) - h['mean']) / h['scale'] - h['pca_mean']) @ np.asarray(h['components']).T


def fit_transform(x, fit_indices, ids, components):
    fit_indices = np.asarray(fit_indices, dtype=int)
    if len(np.unique(fit_indices)) != len(fit_indices) or len(fit_indices) <= components:
        raise ValueError('Representation fitting must use distinct TRAIN proteins')
    sc = StandardScaler().fit(x[fit_indices]); standardized = sc.transform(x)
    pc = PCA(n_components=components, svd_solver='full').fit(standardized[fit_indices])
    h = {'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(), 'pca_mean': pc.mean_.tolist(),
         'components': pc.components_.tolist(), 'explained_variance_ratio': pc.explained_variance_ratio_.tolist(),
         'explained_variance_ratio_sum': float(pc.explained_variance_ratio_.sum()),
         'fit_protein_ids': [str(ids[i]) for i in fit_indices], 'fit_count': len(fit_indices)}
    z = transform(h, x)
    np.testing.assert_allclose(z, pc.transform(standardized), atol=1e-10, rtol=1e-10)
    return h, z


def symmetric(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError('Endpoint latent shapes differ')
    return np.concatenate([(a + b) / 2, np.abs(a - b), a * b], axis=-1)


def pair_features(z, rows, ids, available):
    lookup = {str(pid): i for i, pid in enumerate(ids)}
    x = np.zeros((len(rows), z.shape[1] * 3), dtype=float)
    for i in np.flatnonzero(available):
        a, b = (lookup[str(rows[i][k])] for k in ('a', 'b'))
        x[i] = symmetric(z[a], z[b])
        if not np.array_equal(x[i], symmetric(z[b], z[a])):
            raise ValueError('Pair feature lost symmetry')
    return x


def evidence(h, x):
    return (((x - h['mean']) / h['scale']) @ np.asarray(h['coef']).T + h['intercept']).ravel()


def fit_head(x, y, fit, calibration, baseline, gate, cfg):
    if np.any(fit & calibration) or np.any(fit & (gate <= 0)):
        raise ValueError('Invalid TRAIN/calibration fitting masks')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        sc = StandardScaler().fit(x[fit])
        cl = LogisticRegression(C=cfg['C'], max_iter=cfg['max_iter'], solver='lbfgs').fit(sc.transform(x[fit]), y[fit])
    h = {'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(), 'coef': cl.coef_.tolist(),
         'intercept': cl.intercept_.tolist(), 'classes': cl.classes_.tolist(), 'dimensions': x.shape[1],
         'supervised_parameters': x.shape[1] + 1, 'iterations': cl.n_iter_.tolist(), 'fit_rows': int(fit.sum())}
    ev = evidence(h, x)
    np.testing.assert_allclose(ev, cl.decision_function(sc.transform(x)), atol=1e-10, rtol=1e-10)
    choices = [{'alpha': a, 'ap': float(average_precision_score(y[calibration], fuse(baseline, ev, gate, a)[calibration]))}
               for a in cfg['alpha_grid']]
    best = max(c['ap'] for c in choices); alpha = min(c['alpha'] for c in choices if c['ap'] == best)
    h.update({'alpha': alpha, 'calibration_grid': choices})
    return h, ev, fuse(baseline, ev, gate, alpha)


def legacy_evidence(h, x):
    return ((((x - h['standard_mean']) / h['standard_scale'] - h['pca_mean']) @ np.asarray(h['pca_components']).T)
            @ np.asarray(h['coef']).T + h['intercept']).ravel()
