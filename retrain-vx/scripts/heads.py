"""The original two fixed TRAIN-only transforms/classifiers and calibration grid."""
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from shared import fuse


def fit_head(x, y, fit, base, gate, calibration, cfg):
    if min(np.sum(fit & (y == c)) for c in [0, 1]) < 100:
        raise ValueError('Insufficient eligible TRAIN classes')
    if np.any(fit & calibration) or min(np.sum(calibration & (y == c)) for c in [0, 1]) < 50:
        raise ValueError('Invalid calibration partition')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipe = make_pipeline(StandardScaler(), PCA(n_components=32, svd_solver='full'),
            LogisticRegression(C=.1, max_iter=1000, solver='lbfgs', random_state=cfg['seed']))
        pipe.fit(x[fit], y[fit])
    evidence = pipe.decision_function(x)
    choices = [(float(average_precision_score(y[calibration], fuse(base, evidence, gate, a)[calibration])), a)
               for a in cfg['fusion']['alpha_grid']]
    best = max(v for v, a in choices); alpha = min(a for v, a in choices if v == best)
    predictions = fuse(base, evidence, gate, alpha)
    sc, pc, cl = [p for _, p in pipe.steps]
    state = {'standard_mean': sc.mean_.tolist(), 'standard_scale': sc.scale_.tolist(),
        'pca_mean': pc.mean_.tolist(), 'pca_components': pc.components_.tolist(),
        'coef': cl.coef_.tolist(), 'intercept': cl.intercept_.tolist(), 'classes': cl.classes_.tolist(),
        'alpha': alpha, 'calibration_grid': [{'alpha': a, 'ap': v} for v, a in choices],
        'fit_rows': int(fit.sum()), 'fit_classes': [int(np.sum(fit & (y == c))) for c in [0, 1]]}
    return state, evidence, predictions


def reconstruct(x, state):
    scaled = (x - np.asarray(state['standard_mean'])) / np.asarray(state['standard_scale'])
    components = np.asarray(state['pca_components'])
    return ((scaled - np.asarray(state['pca_mean'])) @ components.T @ np.asarray(state['coef']).T).ravel() + state['intercept'][0]
