"""Coverage-checked comparison and paired protein-aware uncertainty."""
import csv
import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             roc_curve, confusion_matrix, matthews_corrcoef)
from threadpoolctl import threadpool_limits

from common import ROOT, PairData, atomic_json, prediction_fingerprint, sha256

NAMES = ['native-bernett', 'reference-seed2', 'symmetric-seed2']
LABELS = {'native-bernett': 'Native PLM-interact', 'reference-seed2': 'Reference (update 4,000)',
          'symmetric-seed2': 'Symmetric (update 7,000)'}
COLORS = {'native-bernett': '#444444', 'reference-seed2': '#2766a5', 'symmetric-seed2': '#c66717'}
METRICS = ['ap', 'auroc', 'brier']


def save_csv(path, rows):
    assert rows
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge_task(name, split, cfg, contract):
    data = PairData(ROOT / 'data', split)
    directory = ROOT / 'predictions' / f'{name}-{split}'
    pieces = []
    records = []
    for rank in range(cfg['world_size']):
        manifest = json.loads((directory / f'rank-{rank:02d}.done.json').read_text())
        assert manifest['fingerprint'] == contract and manifest['rank'] == rank
        assert manifest['world_size'] == cfg['world_size']
        count = 0
        for item in manifest['chunks']:
            path = directory / item['file']
            assert sha256(path) == item['sha256']
            with np.load(path, allow_pickle=False) as saved:
                a = saved['predictions']
            assert a.shape == (item['rows'], 4)
            count += len(a)
            pieces.append(a)
        assert count == manifest['rows']
        records.append(manifest)
    assert len({m['gpu_uuid'] for m in records}) == cfg['world_size']
    a = np.concatenate(pieces)
    a = a[np.argsort(a[:, 0])]
    assert a.shape == (len(data), 4)
    assert np.array_equal(a[:, 0], np.arange(len(data))), (name, split, 'row coverage')
    assert np.array_equal(a[:, 1], data.rows[:, 2]), (name, split, 'labels')
    assert np.isfinite(a).all()
    np.savez_compressed(ROOT / 'results' / f'{name}-{split}.npz', predictions=a)
    rows = []
    for i, prediction in enumerate(a):
        pooled = float(prediction[2:4].mean())
        rows.append(dict(row_id=i, source_row_id=int(data.rows[i, 3]),
            protein_a=int(data.rows[i, 0]), protein_b=int(data.rows[i, 1]), label=int(prediction[1]),
            combined_residues=int(data.lengths[i] - 3), tokens=int(data.lengths[i]),
            logit_ab=float(prediction[2]), logit_ba=float(prediction[3]),
            pooled_logit=pooled, pooled_probability=float(expit(pooled))))
    save_csv(ROOT / 'results' / f'{name}-{split}-predictions.csv', rows)
    return a, {'rows': len(a), 'complete_unique_coverage': True, 'labels_match': True,
               'finite': True, 'worker_manifests': records}


def threshold_on_validation(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-30)
    highest = np.flatnonzero(f1 == f1.max())[-1]
    return {'logit_threshold': float(thresholds[highest]),
            'probability_threshold': float(expit(thresholds[highest])),
            'validation_f1': float(f1[highest]), 'validation_rows': len(labels),
            'selection': 'maximum validation F1; highest threshold resolves exact ties'}


def metrics(labels, scores, threshold=0.):
    probability = expit(scores)
    tn, fp, fn, tp = confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel()
    return {'rows': len(labels), 'positive': int(labels.sum()), 'prevalence': float(labels.mean()),
            'ap': float(average_precision_score(labels, scores)),
            'auroc': float(roc_auc_score(labels, scores)),
            'brier': float(np.mean((probability - labels) ** 2)),
            'logit_threshold': float(threshold), 'probability_threshold': float(expit(threshold)),
            'precision': float(tp / max(tp + fp, 1)), 'recall': float(tp / max(tp + fn, 1)),
            'f1': float(2 * tp / max(2 * tp + fp + fn, 1)),
            'mcc': float(matthews_corrcoef(labels, scores >= threshold)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


class WeightedRanking:
    """Cached score ordering, including tied-score groups, for weighted AP/AUROC."""
    def __init__(self, labels, scores):
        self.order = np.argsort(-scores, kind='stable')
        self.y = labels[self.order]
        ordered = scores[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
        self.error = (expit(scores) - labels) ** 2

    def compute(self, weights):
        w = weights[self.order]
        tp = np.cumsum(w * self.y)[self.ends]
        fp = np.cumsum(w * (1 - self.y))[self.ends]
        positive, negative = tp[-1], fp[-1]
        assert positive > 0 and negative > 0
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        ap = np.dot(np.diff(np.r_[0., tp]), precision) / positive
        auc = np.dot(np.diff(np.r_[0., fp]), (tp + np.r_[0., tp[:-1]]) / 2) / (positive * negative)
        brier = np.dot(weights, self.error) / weights.sum()
        return np.array([ap, auc, brier])


def bootstrap(data, scores, groups, cfg):
    _, inverse = np.unique(data.rows[:, :2], return_inverse=True)
    endpoints = inverse.reshape(len(data), 2)
    proteins = int(endpoints.max() + 1)
    self_pair = endpoints[:, 0] == endpoints[:, 1]
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    ranks = {group: [WeightedRanking(data.rows[mask, 2], scores[name][mask]) for name in NAMES]
             for group, mask in groups.items()}
    # Check the optimized implementation against independent sklearn weighted metrics.
    verification_weights = np.random.default_rng(71).integers(0, 4, len(data)).astype(float)
    for group, mask in groups.items():
        labels = data.rows[mask, 2]
        w = verification_weights[mask]
        for name, ranking in zip(NAMES, ranks[group]):
            value = ranking.compute(w)
            expected = [average_precision_score(labels, scores[name][mask], sample_weight=w),
                        roc_auc_score(labels, scores[name][mask], sample_weight=w),
                        np.average((expit(scores[name][mask]) - labels) ** 2, weights=w)]
            assert np.allclose(value, expected, atol=1e-12, rtol=1e-12), (group, name, value, expected)
    samples = {group: np.empty((cfg['bootstrap_replicates'], len(NAMES), 3)) for group in groups}
    for rep in range(cfg['bootstrap_replicates']):
        counts = np.bincount(rng.integers(0, proteins, proteins), minlength=proteins)
        weights = counts[endpoints[:, 0]].astype(float) * counts[endpoints[:, 1]]
        weights[self_pair] = counts[endpoints[self_pair, 0]]
        for group, mask in groups.items():
            for j, ranking in enumerate(ranks[group]):
                samples[group][rep, j] = ranking.compute(weights[mask])
    np.savez_compressed(ROOT / 'results/bootstrap-replicates.npz', **samples)
    intervals = []
    differences = []
    for group, sample in samples.items():
        mask = groups[group]
        estimates = np.array([r.compute(np.ones(int(mask.sum()))) for r in ranks[group]])
        for j, name in enumerate(NAMES):
            for k, metric in enumerate(METRICS):
                low, high = np.quantile(sample[:, j, k], [.025, .975])
                intervals.append(dict(group=group, model=name, metric=metric,
                    estimate=float(estimates[j, k]), low=float(low), high=float(high)))
        for j, other in ((1, 0), (2, 0), (2, 1)):
            for k, metric in enumerate(METRICS):
                delta = sample[:, j, k] - sample[:, other, k]
                low, high = np.quantile(delta, [.025, .975])
                differences.append(dict(group=group, comparison=f'{NAMES[j]} minus {NAMES[other]}',
                    metric=metric, difference=float(estimates[j, k] - estimates[other, k]),
                    low=float(low), high=float(high)))
    return intervals, differences, {'replicates': cfg['bootstrap_replicates'],
        'seed': cfg['bootstrap_seed'], 'unique_test_protein_sequences': proteins,
        'self_pairs': int(self_pair.sum()), 'sklearn_weighted_metric_verification_passed': True,
        'method': 'Paired resampling of protein identities; edge weight is product of endpoint multiplicities, one multiplicity for self-pairs'}


def plots(data, scores, summary, intervals):
    y = data.rows[:, 2]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    calibration = []
    for name in NAMES:
        score = scores[name]
        p, r, _ = precision_recall_curve(y, score)
        fpr, tpr, _ = roc_curve(y, score)
        axes[0].plot(r, p, color=COLORS[name], label=f'{LABELS[name]}: AP {summary[name]["ap"]:.4f}')
        axes[1].plot(fpr, tpr, color=COLORS[name], label=f'{LABELS[name]}: AUC {summary[name]["auroc"]:.4f}')
        probability = expit(score)
        bins = np.minimum((probability * 10).astype(int), 9)
        points = []
        for bin_id in range(10):
            mask = bins == bin_id
            if not mask.any():
                continue
            predicted, observed = float(probability[mask].mean()), float(y[mask].mean())
            points.append((predicted, observed))
            calibration.append(dict(model=name, bin=bin_id, rows=int(mask.sum()),
                                    mean_probability=predicted, observed_fraction=observed))
        points = np.asarray(points)
        axes[2].plot(points[:, 0], points[:, 1], '-o', markersize=4, color=COLORS[name], label=LABELS[name])
    axes[0].axhline(y.mean(), linestyle='--', color='#aaaaaa', linewidth=1)
    axes[1].plot([0, 1], [0, 1], '--', color='#aaaaaa', linewidth=1)
    axes[2].plot([0, 1], [0, 1], '--', color='#aaaaaa', linewidth=1)
    for ax, title, xlabel, ylabel in zip(axes, ['Precision–recall', 'ROC', 'Probability calibration (10 fixed bins)'],
            ['Recall', 'False positive rate', 'Mean predicted probability'], ['Precision', 'True positive rate', 'Observed positive fraction']):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7, loc='best', frameon=False)
        ax.grid(alpha=.15)
    fig.suptitle('Bernett test: 52,048 pairs | full sequences | identical pooled scoring | BF16 compute')
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(ROOT / 'results' / f'test-curves.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    save_csv(ROOT / 'results/calibration-bins.csv', calibration)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, key, title, baseline in zip(axes, METRICS, ['AP (higher is better)', 'AUROC (higher is better)', 'Brier (lower is better)'], [.5, .5, .25]):
        for j, name in enumerate(NAMES):
            item = next(x for x in intervals if x['group'] == 'all' and x['model'] == name and x['metric'] == key)
            # Percentile intervals need not contain the point estimate, so plot endpoints directly.
            ax.plot([item['low'], item['high']], [j, j], color=COLORS[name], linewidth=2)
            ax.plot(item['estimate'], j, 'o', color=COLORS[name])
        ax.set_yticks(range(len(NAMES)), [LABELS[n] if ax is axes[0] else '' for n in NAMES])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Metric value')
        ax.axvline(baseline, color='#aaaaaa', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('95% paired protein-bootstrap intervals | 1,000 replicates | one training seed')
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(ROOT / 'results' / f'test-metrics.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    cfg = json.loads((ROOT / 'config.json').read_text())
    contract = prediction_fingerprint()
    test = PairData(ROOT / 'data', 'test')
    val = PairData(ROOT / 'data', 'val')
    assert len(test) == 52048 and int(test.rows[:, 2].sum()) == 26024
    assert len(val) == 59258 and int(val.rows[:, 2].sum()) == 29628
    test_predictions = {}
    val_predictions = {}
    coverage = {}
    for name in NAMES:
        test_predictions[name], coverage[name + '-test'] = merge_task(name, 'test', cfg, contract)
        if name == 'native-bernett':
            val_predictions[name], coverage[name + '-val'] = merge_task(name, 'val', cfg, contract)
        else:
            with np.load(ROOT / 'predictions' / f'{name}-selected-validation.npz', allow_pickle=False) as saved:
                a = saved['predictions']
            assert a.shape == (len(val), 4) and np.array_equal(a[:, 0], np.arange(len(val)))
            assert np.array_equal(a[:, 1], val.rows[:, 2]) and np.isfinite(a).all()
            val_predictions[name] = a
            coverage[name + '-val'] = {'rows': len(a), 'complete_unique_coverage': True,
                                       'labels_match': True, 'finite': True, 'source': 'saved selected-checkpoint validation'}
    groups = {'all': np.ones(len(test), dtype=bool),
              'combined_residues_le_2193': test.lengths <= 2196,
              'combined_residues_gt_2193': test.lengths > 2196}
    score_views = {'ab': lambda a: a[:, 2], 'pooled': lambda a: a[:, 2:4].mean(1)}
    all_metrics = []
    validation_metrics = []
    thresholds = []
    orientation = []
    primary = {}
    for name in NAMES:
        a = test_predictions[name]
        orientation.append(dict(model=name, mean_absolute_logit_gap=float(np.abs(a[:, 2] - a[:, 3]).mean()),
                                decisions_changed_on_reversal=int(((a[:, 2] >= 0) != (a[:, 3] >= 0)).sum()),
                                rows=len(a), final_pooled_score_is_order_invariant=True))
        for view, fn in score_views.items():
            scores = fn(a)
            rule = threshold_on_validation(val.rows[:, 2], fn(val_predictions[name]))
            validation_metrics.append({'model': name, 'view': view,
                'prediction_source': 'fresh native evaluation' if name == 'native-bernett' else 'saved selected-checkpoint validation',
                **metrics(val.rows[:, 2], fn(val_predictions[name]))})
            thresholds.append({'model': name, 'view': view, **rule})
            for group, mask in groups.items():
                for threshold_name, threshold in [('fixed_0.5', 0.), ('validation_max_f1', rule['logit_threshold'])]:
                    value = metrics(test.rows[mask, 2], scores[mask], threshold)
                    all_metrics.append({'model': name, 'view': view, 'group': group,
                                        'threshold_rule': threshold_name, **value})
                    if view == 'pooled' and group == 'all' and threshold_name == 'fixed_0.5':
                        primary[name] = value
    scores = {name: test_predictions[name][:, 2:4].mean(1) for name in NAMES}
    with threadpool_limits(limits=1):
        intervals, differences, bootstrap_meta = bootstrap(test, scores, groups, cfg)
    native_old = np.genfromtxt(ROOT.parent / 'plm-interact-reproducability/results/bernett_full.csv',
                               delimiter=',', names=True)
    native_old = native_old[np.argsort(native_old['row_id'])]
    assert np.array_equal(native_old['row_id'], test.rows[:, 3])
    assert np.array_equal(native_old['label'], test.rows[:, 2])
    assert np.array_equal(native_old['original_tokens'], test.lengths)
    original = native_old['logit']
    fresh = test_predictions['native-bernett'][:, 2]
    audit = {'comparison': 'Native original-order full-length: fresh common BF16 versus earlier eager FP32',
             'old_prediction_file_sha256': sha256(ROOT.parent / 'plm-interact-reproducability/results/bernett_full.csv'),
             'old_ap': float(average_precision_score(test.rows[:, 2], original)),
             'fresh_ap': float(average_precision_score(test.rows[:, 2], fresh)),
             'old_auroc': float(roc_auc_score(test.rows[:, 2], original)),
             'fresh_auroc': float(roc_auc_score(test.rows[:, 2], fresh)),
             'mean_absolute_probability_difference': float(np.abs(expit(fresh) - expit(original)).mean()),
             'maximum_probability_difference': float(np.abs(expit(fresh) - expit(original)).max()),
             'threshold_0_5_disagreements': int(((fresh >= 0) != (original >= 0)).sum())}
    context_path = ROOT.parent / 'plm-interact-reproducability/results/published-score-metrics.csv'
    with context_path.open() as handle:
        published_rows = [r for r in csv.DictReader(handle) if r['figure'] == '4' and r['model'] == 'PLM-interact']
    assert len(published_rows) == 1
    published = published_rows[0]
    context = {'paper_figure': 4, 'paper_rounded_ap': 0.69, 'paper_rounded_auroc': 0.70,
               'deposited_scores_ap': float(published['average_precision']),
               'deposited_scores_auroc': float(published['auroc']),
               'deposited_scores_f1_at_0_5': float(published['f1']),
               'source_file_sha256': sha256(context_path),
               'interpretation': 'Original-order published context; not the common full-length pooled comparison'}
    report = {'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'prediction_fingerprint': contract, 'primary_scoring': 'mean AB/BA logits',
              'primary': primary, 'metrics': all_metrics, 'validation_metrics': validation_metrics,
              'thresholds': thresholds,
              'orientation': orientation, 'confidence_intervals': intervals, 'paired_differences': differences,
              'bootstrap': bootstrap_meta, 'native_precision_audit': audit, 'published_context': context,
              'coverage': coverage,
              'qualification': json.loads((ROOT / 'provenance/qualification.json').read_text()),
              'selection': json.loads((ROOT / 'provenance/selection.json').read_text())}
    atomic_json(ROOT / 'results/benchmark-summary.json', report)
    for filename, rows in [('metrics.csv', all_metrics), ('validation-thresholds.csv', thresholds),
                            ('validation-metrics.csv', validation_metrics),
                            ('orientation-diagnostics.csv', orientation), ('confidence-intervals.csv', intervals),
                            ('paired-differences.csv', differences)]:
        save_csv(ROOT / 'results' / filename, rows)
    plots(test, scores, primary, intervals)
    print(json.dumps({'primary': primary, 'paired_differences_all': [d for d in differences if d['group'] == 'all'],
                      'native_precision_audit': audit, 'bootstrap': bootstrap_meta}, indent=2))


if __name__ == '__main__':
    main()
