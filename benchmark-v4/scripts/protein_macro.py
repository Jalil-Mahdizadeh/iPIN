"""Descriptive equal-protein AP; labels never alter checkpoint or thresholds."""
import json
import numpy as np
from sklearn.metrics import average_precision_score
from common import ROOT, PairData, atomic_json
from analyze import NAMES, save_csv


def main():
    data = PairData(ROOT / 'data', 'test')
    incident = {}
    for row, (a, b) in enumerate(data.rows[:, :2]):
        incident.setdefault(int(a), []).append(row)
        if a != b:
            incident.setdefault(int(b), []).append(row)
    eligible = {p: np.asarray(rows) for p, rows in incident.items()
                if len(np.unique(data.rows[rows, 2])) == 2}
    details, metrics = [], {}
    for name in NAMES:
        with np.load(ROOT / 'results' / f'{name}-test.npz', allow_pickle=False) as f:
            a = f['predictions']
        assert np.array_equal(a[:, 0], np.arange(len(data)))
        assert np.array_equal(a[:, 1], data.rows[:, 2])
        scores, values = a[:, 2:4].mean(1), []
        for protein, rows in sorted(eligible.items()):
            ap = float(average_precision_score(data.rows[rows, 2], scores[rows]))
            values.append(ap)
            details.append({'model': name, 'protein_sequence_id': protein,
                'pairs': len(rows), 'positives': int(data.rows[rows, 2].sum()), 'ap': ap})
        metrics[name] = float(np.mean(values))
    report = {'definition': 'Arithmetic mean of within-protein AP over test proteins with at least one positive and one negative incident pair. Self-pairs enter that protein once.',
        'eligible_proteins': len(eligible), 'total_test_proteins': len(incident),
        'excluded_single_class_proteins': len(incident) - len(eligible),
        'pooled_AB_BA_logits': True, 'descriptive_only': True,
        'used_for_selection': False, 'macro_ap': metrics}
    atomic_json(ROOT / 'results/protein-macro-summary.json', report)
    save_csv(ROOT / 'results/protein-macro-ap.csv', details)
    print(json.dumps({'event': 'protein_macro_complete', **report}), flush=True)


if __name__ == '__main__':
    main()
