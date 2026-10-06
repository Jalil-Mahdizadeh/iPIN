"""Fixed development endpoints; protein AP counts each incident pair once."""
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(rows, scores):
    y = rows[:, 2]
    assert len(scores) == len(rows) and np.isfinite(scores).all()
    assert set(np.unique(y)) == {0, 1}
    per_protein = []
    for protein in np.unique(rows[:, :2]):
        keep = (rows[:, :2] == protein).any(axis=1)
        labels = y[keep]
        if np.count_nonzero(labels == 1) >= 2 and np.count_nonzero(labels == 0) >= 2:
            per_protein.append(average_precision_score(labels, scores[keep]))
    return {
        'rows': len(rows), 'prevalence': float(y.mean()),
        'ap': float(average_precision_score(y, scores)),
        'auroc': float(roc_auc_score(y, scores)),
        'brier': float(np.mean((expit(scores) - y) ** 2)),
        'macro_ap': float(np.mean(per_protein)) if per_protein else None,
        'macro_eligible_proteins': len(per_protein),
        'macro_rule': 'At least 2 positive and 2 negative incident pairs; equal protein weight; self-pair counted once.'}


def promotion(deltas, rules):
    assert len(deltas) == 3, 'All three completed, matched folds are required'
    assert all(all(np.isfinite(d[k]) for k in ['ap', 'auroc', 'macro_ap']) for d in deltas)
    mean = {k: float(np.mean([d[k] for d in deltas])) for k in ['ap', 'auroc', 'macro_ap']}
    checks = {
        'mean_ap_gain': mean['ap'] >= rules['minimum_mean_ap_delta'],
        'positive_folds': sum(d['ap'] > 0 for d in deltas) >= rules['minimum_positive_folds'],
        'no_large_fold_ap_drop': min(d['ap'] for d in deltas) >= -rules['maximum_single_fold_ap_drop'],
        'mean_auroc_guard': mean['auroc'] >= -rules['maximum_mean_auroc_drop'],
        'mean_macro_ap_guard': mean['macro_ap'] >= -rules['maximum_mean_macro_ap_drop']}
    return {'promoted': bool(all(checks.values())), 'checks': {k: bool(v) for k, v in checks.items()},
            'mean_deltas': mean, 'positive_folds': sum(d['ap'] > 0 for d in deltas),
            'interpretation': 'Prospective development decision rule, not a significance test or proof of native superiority.'}
