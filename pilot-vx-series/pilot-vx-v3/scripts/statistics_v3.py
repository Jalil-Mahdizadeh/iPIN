"""Matched protein bootstrap; fast AP independently checked against sklearn."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CONTRASTS = {
    'C_minus_shuffled': {'C': 1, 'shuffled': -1}, 'C_minus_true': {'C': 1, 'true': -1},
    'C_minus_Q': {'C': 1, 'Q': -1}, 'C_minus_baseline': {'C': 1, 'baseline': -1}, 'C_minus_quality': {'C': 1, 'quality': -1},
    'true_minus_C': {'true': 1, 'C': -1}, 'shuffled_minus_C': {'shuffled': 1, 'C': -1},
    'true_minus_shuffled': {'true': 1, 'shuffled': -1}, 'true_minus_baseline': {'true': 1, 'baseline': -1},
    'shuffled_minus_baseline': {'shuffled': 1, 'baseline': -1}, 'Q_minus_shuffled': {'Q': 1, 'shuffled': -1},
    'Q_minus_true': {'Q': 1, 'true': -1}, 'Q_minus_baseline': {'Q': 1, 'baseline': -1},
    'Q_minus_quality': {'Q': 1, 'quality': -1}, 'true_minus_Q': {'true': 1, 'Q': -1}, 'shuffled_minus_Q': {'shuffled': 1, 'Q': -1}}


def contrasts(aps):
    return {name: float(sum(c * aps[k] for k, c in terms.items())) for name, terms in CONTRASTS.items()}


def ap_setup(y, z):
    order = np.argsort(-z, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(z[order])), len(z) - 1]
    return order, ends, y[order]


def weighted_ap(setup, weights):
    order, ends, y = setup; w = weights[order]
    tp = np.cumsum(w * y)[ends]; total = np.cumsum(w)[ends]
    if tp[-1] == 0 or total[-1] == tp[-1]:
        return np.nan
    precision = np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)
    return float(np.sum(np.diff(np.r_[0, tp]) * precision) / tp[-1])


def bootstrap(rows, y, scores, masks, cfg, seed):
    ids = sorted({r[k] for r in rows for k in ('a', 'b')}); lookup = {p: i for i, p in enumerate(ids)}
    ia = np.array([lookup[r['a']] for r in rows]); ib = np.array([lookup[r['b']] for r in rows])
    accepted = {k: m for k, m in masks.items() if min(np.sum(m & (y == c)) for c in (0, 1)) >= cfg['minimum_subgroup_per_class']}
    setups = {k: {name: ap_setup(y[m], z[m]) for name, z in scores.items()} for k, m in accepted.items()}
    samples = {k: {name: [] for name in CONTRASTS} for k in accepted}; rng = np.random.default_rng(seed)
    for _ in range(cfg['bootstrap_replicates']):
        count = rng.poisson(1, len(ids)); w = count[ia] * count[ib]
        for group, m in accepted.items():
            ap = {name: weighted_ap(setup, w[m]) for name, setup in setups[group].items()}
            for name, value in contrasts(ap).items():
                if np.isfinite(value):
                    samples[group][name].append(value)
    result = {}
    for group, values in samples.items():
        result[group] = {}
        for name, data in values.items():
            value = {'replicates': len(data)}
            if len(data) >= .95 * cfg['bootstrap_replicates']:
                value.update({'low': float(np.quantile(data, .025)), 'high': float(np.quantile(data, .975))})
            else:
                value['status'] = 'insufficient_valid_replicates'
            result[group][name] = value
    return result


def summarize(mask, y, scores, available):
    result = {'rows': int(mask.sum()), 'positive': int(y[mask].sum()), 'negative': int(np.sum(mask & (y == 0))),
              'eligible': int(np.sum(mask & available)), 'eligible_positive': int(y[mask & available].sum())}
    if len(np.unique(y[mask])) != 2:
        return {**result, 'status': 'insufficient_classes'}
    yy = y[mask]; counts = np.bincount(yy, minlength=2); weights = .5 / counts[yy]
    result['metrics'] = {name: {'ap': float(average_precision_score(yy, z[mask])),
        'auroc': float(roc_auc_score(yy, z[mask])),
        'ap_at_50_percent_prevalence': float(average_precision_score(yy, z[mask], sample_weight=weights))} for name, z in scores.items()}
    result['contrasts'] = contrasts({name: m['ap'] for name, m in result['metrics'].items()})
    return result
