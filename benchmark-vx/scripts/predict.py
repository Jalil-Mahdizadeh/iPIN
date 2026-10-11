"""Inference-only application of the frozen TRAIN transforms and VAL weights."""
import numpy as np
from shared import ARMS, fuse
from heads import reconstruct


def prohibit_training():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    def forbidden(*args, **kwargs):
        raise RuntimeError('Fitting or calibration is forbidden in the TEST benchmark')
    for cls in [PCA, LogisticRegression, StandardScaler]:
        for name in ['fit', 'fit_transform', 'partial_fit']:
            if hasattr(cls, name):
                setattr(cls, name, forbidden)


def score(features, baseline, heads, fixed_alphas):
    if set(heads) != set(ARMS) or set(features) != {'gate', 'available', *ARMS}:
        raise ValueError('Unexpected arm or feature schema')
    n = len(baseline); available = features['available']; gate = features['gate']
    if available.shape != (n,) or available.dtype != np.bool_ or gate.shape != (n,):
        raise ValueError('Invalid availability/gate')
    np.testing.assert_array_equal(gate > 0, available)
    scores = {'baseline': np.asarray(baseline, dtype=float).copy()}; evidence = {}
    for arm, dimension in ARMS.items():
        h = heads[arm]; x = features[arm]
        if x.shape != (n, dimension) or not np.isfinite(x).all() or h['classes'] != [0, 1]:
            raise ValueError('Invalid frozen head input/classes')
        shapes = {'standard_mean': (dimension,), 'standard_scale': (dimension,),
                  'pca_mean': (dimension,), 'pca_components': (32, dimension), 'coef': (1, 32), 'intercept': (1,)}
        for key, shape in shapes.items():
            v = np.asarray(h[key])
            if v.shape != shape or not np.isfinite(v).all():
                raise ValueError('Invalid frozen parameter: ' + key)
        if np.any(np.asarray(h['standard_scale']) <= 0) or h['alpha'] != fixed_alphas[arm]:
            raise ValueError('Frozen scaling/fusion changed')
        evidence[arm] = reconstruct(x, h)
        scores[arm] = fuse(scores['baseline'], evidence[arm], gate, h['alpha'])
        np.testing.assert_array_equal(scores[arm][~available], scores['baseline'][~available])
    return scores, evidence
