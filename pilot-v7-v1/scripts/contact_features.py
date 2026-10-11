"""Six fixed summaries; geometry remains in the original query coordinates."""
import numpy as np
from study import config, text_hash


def window_sums(a, width):
    s = np.pad(np.asarray(a, dtype=np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return s[width:, width:] - s[:-width, width:] - s[width:, :-width] + s[:-width, :-width]


def entropy(mass, n):
    total = mass.sum()
    if n <= 1 or total <= 0:
        return 0.
    p = mass[mass > 0] / total
    return float(-(p * np.log(p)).sum() / np.log(n))


def summarize(probability, valid_a, valid_b, excluded=None):
    cfg = config()['features']
    p = np.asarray(probability, dtype=np.float64)
    a, b = np.asarray(valid_a, dtype=bool), np.asarray(valid_b, dtype=bool)
    if p.shape != (len(a), len(b)) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid contact probability map')
    valid = a[:, None] & b[None, :]
    if excluded is not None:
        if np.shape(excluded) != p.shape:
            raise ValueError('Invalid exclusion mask')
        valid &= ~np.asarray(excluded, dtype=bool)
    n = int(valid.sum())
    if not n:
        return np.zeros(6), {'valid_cells': 0, 'supported_windows': 0, 'maximum_probability': 0., 'empty': True}
    values = p[valid]; k = min(cfg['top_k'], n)
    high = (p >= cfg['threshold']) & valid
    ca = float(high.any(1).sum() / max(1, a.sum()))
    cb = float(high.any(0).sum() / max(1, b.sum()))
    mass = np.where(valid, p, 0.)
    ent = (entropy(mass.sum(1), int(a.sum())) + entropy(mass.sum(0), int(b.sum()))) / 2
    w = cfg['window']; supported = 0; peak = 0.
    if min(p.shape) >= w:
        counts = window_sums(valid, w)
        enough = counts >= np.ceil(w * w * cfg['minimum_window_valid_fraction'])
        supported = int(enough.sum())
        if supported:
            peak = float((window_sums(mass, w)[enough] / counts[enough]).max())
    features = np.array([np.partition(values, n-k)[n-k:].mean(), high.sum()/n, min(ca, cb), max(ca, cb), ent, peak])
    if not np.isfinite(features).all() or np.any(features < -1e-9) or np.any(features > 1+1e-9):
        raise ValueError('Nonfinite/out-of-range summary')
    return features, {'valid_cells': n, 'supported_windows': supported,
        'maximum_probability': float(values.max()), 'empty': False}


def coordinate_indices(valid, uid, protein):
    idx = np.arange(len(valid)); good = np.flatnonzero(valid)
    seed = int(text_hash(f"v7-coordinate:{config()['seed']}:{uid}:{protein}")[:16], 16)
    idx[good] = np.random.default_rng(seed).permutation(good)
    return idx


def boundary_mask(shape):
    w = config()['features']['boundary_width']; a, b = np.indices(shape)
    # Both concatenation orientations: A-end/B-start and A-start/B-end.
    return ((a >= shape[0]-w) & (b < w)) | ((a < w) & (b >= shape[1]-w))


def all_views(probability, valid_a, valid_b, uid, a, b):
    pa, pb = coordinate_indices(valid_a, uid, a), coordinate_indices(valid_b, uid, b)
    standard, sm = summarize(probability, valid_a, valid_b)
    coordinate, cm = summarize(probability[np.ix_(pa, pb)], valid_a, valid_b)
    boundary, bm = summarize(probability, valid_a, valid_b, boundary_mask(probability.shape))
    np.testing.assert_allclose(standard[:5], coordinate[:5], atol=1e-12, rtol=1e-12)
    return {'standard': standard, 'coordinate': coordinate, 'boundary': boundary}, {'standard': sm, 'coordinate': cm, 'boundary': bm}
