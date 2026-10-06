"""Fixed cheap sequence baselines on existing internal folds; no neural retraining."""
import json
import warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import PairData
from metrics import evaluate
from state import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / 'diagnostics/cheap-baselines.json'
    assert not target.exists(), 'Preserve the fixed diagnostic; no repeated hyperparameter search'
    data = PairData(ROOT / 'data/prepared', 'official', 'train')
    proteins = np.unique(data.rows[:, :2])
    phi = np.zeros((len(data.offsets) - 1, 29), dtype=np.float64)
    for protein in proteins:
        seq = data.sequence(protein)
        assert len(seq) and not np.isin(seq, [0, 1, 2, 3, 32]).any()
        phi[protein, 0] = np.log1p(len(seq))
        phi[protein, 1:] = np.bincount(seq, minlength=33)[4:32] / len(seq)

    def features(rows, kind):
        a, b = phi[rows[:, 0]], phi[rows[:, 1]]
        if kind == 'length-only':
            a, b = a[:, :1], b[:, :1]
        if kind == 'additive-unary-composition':
            return a + b  # Linear head: f(A)+f(B)+bias, no interaction term.
        return np.concatenate([a + b, np.abs(a - b), a * b], axis=1)

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for fold in range(3):
            train = PairData(ROOT / 'data/prepared', f'fold-{fold}', 'train')
            val = PairData(ROOT / 'data/prepared', f'fold-{fold}', 'val')
            prevalence = float(train.rows[:, 2].mean())
            scores = np.full(len(val), np.log(prevalence / (1 - prevalence)))
            results.append({'fold': fold, 'baseline': 'constant-train-prevalence', 'metrics': evaluate(val.rows, scores)})
            for kind in ['length-only', 'additive-unary-composition', 'pair-composition']:
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000,
                                      solver='lbfgs', random_state=2, class_weight=None))
                model.fit(features(train.rows, kind), train.rows[:, 2])
                metrics = evaluate(val.rows, model.decision_function(features(val.rows, kind)))
                results.append({'fold': fold, 'baseline': kind, 'metrics': metrics,
                                'training_rows': len(train), 'feature_dimensions': features(val.rows[:1], kind).shape[1],
                                'solver_iterations': int(model[-1].n_iter_[0])})
                print(f'fold {fold} {kind}: AP={metrics["ap"]:.6f}, macro AP={metrics["macro_ap"]:.6f}', flush=True)
    atomic_json(target, {
        'passed': True, 'production_neural_training': False, 'test_used': False,
        'official_validation_used': False, 'fixed_C': 1., 'tuned_hyperparameters': False,
        'scaler_fit': 'Each fold training rows only', 'results': results,
        'purpose': 'Measure composition and additive endpoint shortcuts; these baselines do not demonstrate a neural ceiling.',
        'data_manifest_sha256': sha256(ROOT / 'data/prepared/manifest.json'),
        'source_sha256': {f'scripts/{n}': sha256(ROOT / 'scripts' / n) for n in ['diagnose_data.py', 'metrics.py']}})


if __name__ == '__main__':
    main()
