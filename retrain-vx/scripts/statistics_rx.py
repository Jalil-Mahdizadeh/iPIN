"""Fixed-model matched protein bootstrap and fixed full-population summaries."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CONTRASTS = {'true_minus_baseline': ('true', 'baseline'), 'true_minus_quality': ('true', 'quality'),
             'quality_minus_baseline': ('quality', 'baseline')}


def ap_setup(y, z):
    order = np.argsort(-z, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(z[order])), len(z) - 1]
    return order, ends, y[order]


def weighted_ap(setup, weights):
    order, ends, y = setup; w = weights[order]
    tp = np.cumsum(w * y)[ends]; total = np.cumsum(w)[ends]
    if tp[-1] == 0 or tp[-1] == total[-1]:
        return np.nan
    precision = np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)
    return float(np.sum(np.diff(np.r_[0, tp]) * precision) / tp[-1])


def intervals(rows, y, scores, repeats, seed, independent=False):
    ids = sorted({r[k] for r in rows for k in ['a', 'b']}); lookup = {p: i for i, p in enumerate(ids)}
    ia = np.asarray([lookup[r['a']] for r in rows]); ib = np.asarray([lookup[r['b']] for r in rows])
    setup = {k: ap_setup(y, z) for k, z in scores.items()}
    rng = np.random.default_rng(seed); values = {k: [] for k in CONTRASTS}
    for _ in range(repeats):
        count = rng.poisson(1, len(ids)); weights = count[ia] * count[ib]
        if min(weights[y == c].sum() for c in [0, 1]) <= 0:
            continue
        aps = {k: float(average_precision_score(y, z, sample_weight=weights)) if independent else weighted_ap(setup[k], weights)
               for k, z in scores.items()}
        for k, (a, b) in CONTRASTS.items():
            values[k].append(aps[a] - aps[b])
    if any(len(v) < .95 * repeats for v in values.values()):
        raise ValueError('Insufficient valid protein-bootstrap replicates')
    return {k: {'low': float(np.quantile(v, .025)), 'high': float(np.quantile(v, .975)), 'replicates': len(v)} for k, v in values.items()}


def summarize(y, scores, available):
    result = {'rows': len(y), 'positive': int(y.sum()), 'eligible': int(available.sum()),
              'eligible_positive': int(y[available].sum())}
    if len(np.unique(y)) != 2:
        return {**result, 'status': 'insufficient_classes'}
    counts = np.bincount(y, minlength=2); weights = .5 / counts[y]
    result['metrics'] = {k: {'ap': float(average_precision_score(y, z)), 'auroc': float(roc_auc_score(y, z)),
        'ap_at_50_percent_prevalence': float(average_precision_score(y, z, sample_weight=weights))} for k, z in scores.items()}
    result['contrasts'] = {k: result['metrics'][a]['ap'] - result['metrics'][b]['ap'] for k, (a, b) in CONTRASTS.items()}
    return result


def decision(primary):
    m = primary['metrics']; ci = primary['intervals']; point = primary['contrasts']['true_minus_baseline']
    baseline_gain = point >= .010 and ci['true_minus_baseline']['low'] > 0
    beyond_quality = ci['true_minus_quality']['low'] > 0
    auroc_ok = m['true']['auroc'] >= m['baseline']['auroc'] - .005
    return {'status': 'useful_predictive_gain_supported' if baseline_gain and beyond_quality and auroc_ok else 'no_go_or_inconclusive',
        'primary_ap_change': point, 'baseline_gain_supported': baseline_gain, 'beyond_quality_supported': beyond_quality,
        'auroc_noninferiority': auroc_ok, 'pairing_specific_claim': False, 'test_authorized': False, 'test_accessed': False}
