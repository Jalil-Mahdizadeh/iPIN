"""Synthetic full analysis/audit, including old predictions and interval preservation."""
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack, redirect_stdout
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import study
import analyze
import verify
from statistics_vz import bootstrap


def fixture(root, cfg):
    n = 640; rng = np.random.default_rng(92); y = np.arange(n) % 2
    base = rng.normal(size=n) + .3 * y; gate = rng.uniform(.5, 1., size=n); gate[::9] = 0
    x = rng.normal(size=(n, 128)); x[:, :4] += y[:, None] * .6; x[gate == 0] = 0
    rows = []; pairs = []
    for i in range(n):
        role = 'train' if i < 320 else 'calibration' if i < 440 else 'assessment' if i < 600 else 'crossing'
        r = {'uid': str(i), 'a': 2*i, 'b': 2*i+1, 'label': int(y[i]), 'split': 'train' if i < 320 else 'val', 'role': role, 'length': 600}
        rows.append(r); pairs.append({'uid': str(i), 'a': r['a'], 'b': r['b'], 'available': bool(gate[i]), 'gate': float(gate[i]), 'reason': 'available' if gate[i] else 'shallow_paired_alignment'})
    for name in ('data', 'results', 'features', 'scripts'): (root / name).mkdir()
    (root / 'scripts/verify.py').write_bytes(Path(verify.__file__).read_bytes())
    for name, value in [('sample', rows), ('pairs', pairs), ('diagnostics', [None]*n), ('dev_family_witnesses', {}),
                        ('v2_metrics', {'diagnostic_stratum_cuts_from_eligible_train': {}})]:
        study.atomic(root / 'data' / (name + '.json'), value)
    study.atomic(root / 'data/baseline.json', {r['uid']: {**r, 'score': float(base[i])} for i, r in enumerate(rows) if r['split'] == 'val'})
    def oldhead(dim):
        return {'standard_mean': [0.]*dim, 'standard_scale': [1.]*dim, 'pca_mean': [0.]*dim,
                'pca_components': np.eye(dim)[:32].tolist(), 'coef': [[0.]*32], 'intercept': [0.], 'alpha': 1}
    study.atomic(root / 'data/source_heads.json', {name: oldhead(dim) for name, dim in [('true', 128), ('shuffled', 128), ('quality', 68)]})
    study.atomic_npz(root / 'data/source.npz', uids=np.array([r['uid'] for r in rows]), gate=gate,
        true=np.zeros((n, 128)), shuffled=np.zeros((n, 128)), quality=np.zeros((n, 68)))
    train = np.arange(n) < 320; cal = (np.arange(n) >= 320) & (np.arange(n) < 440); ass = (np.arange(n) >= 440) & (np.arange(n) < 600)
    dev = ~train; old = {'results': {}, 'fit_counts': {name: [int(np.sum(m & (y == c))) for c in (0,1)] for name, m in [('fit', train & (gate > 0)), ('calibration', cal), ('assessment', ass)]}}
    for pop, m in [('assessment', ass), ('fixed_dev_sample', dev), ('calibration', cal)]:
        values = {'ap': float(average_precision_score(y[m], base[m])), 'auroc': float(roc_auc_score(y[m], base[m]))}
        old['results'][pop] = {'metrics': {name: values for name in ('baseline','true','shuffled','quality')}}
        if pop != 'calibration':
            ci = bootstrap([r for r, k in zip(rows,m) if k], y[m], {name: base[m] for name in ('Q','true','shuffled','quality','baseline')},
                           {pop: np.ones(m.sum(), dtype=bool)}, cfg['fusion'], cfg['seed'])[pop]
            old['results'][pop]['ap_intervals'] = {b: ci[a] for a,b in [('true_minus_baseline','true-minus-baseline'),('true_minus_shuffled','true-minus-shuffled')]}
    study.atomic(root / 'data/source_metrics.json', old)
    study.atomic_npz(root / 'data/vy_dev_predictions.npz', uids=np.array([r['uid'] for r in rows if r['split']=='val']), labels=y[dev],
        baseline=base[dev], vx_true=base[dev], vx_shuffled=base[dev], vx_quality=base[dev])
    with patch.object(study, 'ROOT', root):
        for p, xx in zip(pairs, x): study.save_record(p['uid'], xx, 'fixture', p)
    return rows, pairs, x, base, gate


class AnalysisFlow(unittest.TestCase):
    def patches(self, root, cfg):
        stack = ExitStack()
        for module in (study, analyze, verify):
            stack.enter_context(patch.object(module, 'ROOT', root))
            stack.enter_context(patch.object(module, 'config', return_value=cfg))
        for module in (analyze, verify): stack.enter_context(patch.object(module, 'verify_freeze', return_value='fixture'))
        stack.enter_context(patch.object(analyze, 'check_budget'))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def test_full_pipeline_independent_audit_and_tampering(self):
        cfg = copy.deepcopy(study.config()); cfg['expected']['pairs'] = 640; cfg['fusion']['bootstrap_replicates'] = 20
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows, pairs, x, base, gate = fixture(root, cfg)
            with self.patches(root, cfg):
                analyze.main(); verify.main()
                v = study.read(root / 'results/verification.json'); self.assertTrue(v['passed']); self.assertEqual(v['pairs_verified'], 640)
                h = study.read(root / 'results/heads.json')['Q']
                fit = np.array([r['split']=='train' and p['available'] for r,p in zip(rows,pairs)])
                np.testing.assert_allclose(h['standard_mean'], x[fit].mean(0)); self.assertEqual(h['fit_rows'], fit.sum())
                with np.load(root / 'results/dev_predictions.npz') as f: saved = {k: f[k].copy() for k in f.files}
                np.testing.assert_array_equal(saved['Q'][gate[320:]==0], base[320:][gate[320:]==0])
                saved['Q'][0] += .1; study.atomic_npz(root / 'results/dev_predictions.npz', **saved)
                with self.assertRaises(AssertionError): verify.main()
                with self.assertRaisesRegex(RuntimeError, 'already started'): analyze.main()

    def test_missing_output_aborts_before_fitting(self):
        cfg = copy.deepcopy(study.config()); cfg['expected']['pairs'] = 640; cfg['fusion']['bootstrap_replicates'] = 20
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root, cfg)
            (root / 'features/1.json').unlink(); (root / 'features/1.sha.json').unlink()
            with self.patches(root, cfg), patch.object(analyze, 'fit_head') as fit:
                with self.assertRaisesRegex(RuntimeError, 'Incomplete computational coverage'): analyze.main()
                fit.assert_not_called()
            self.assertFalse((root / 'results/fit-start.json').exists())


if __name__ == '__main__': unittest.main()
