"""Exercise the whole CPU analysis with synthetic, separated TRAIN/calibration/assessment."""
import copy
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze
import verify
from study import config, atomic, read


def fixture(root):
    rng = np.random.default_rng(37); n = 400; rows = []; pairs = []; monomers = {}; features = {}
    base = rng.normal(size=n); y = np.arange(n) % 2; gate = np.ones(n); gate[::9] = 0
    for i in range(n):
        role = 'train' if i < 200 else 'calibration' if i < 280 else 'assessment' if i < 380 else 'crossing'
        split = 'train' if i < 200 else 'val'; a, b = i * 2, i * 2 + 1; uid = str(i)
        r = {'uid': uid, 'a': a, 'b': b, 'label': int(y[i]), 'split': split, 'role': role, 'length': 600}
        rows.append(r); pairs.append({'uid': uid, 'a': a, 'b': b, 'gate': float(gate[i]), 'available': bool(gate[i]), 'vx_available': bool(gate[i])})
        for pid in [a, b]:
            monomers[str(pid)] = {'pid': str(pid), 'split': split, 'length': 300, 'available': bool(gate[i]),
                'quality_fraction': 1., 'neff80': 50., 'valid_residues': 300, 'occupancy_mean': .9,
                'fraction_good_positions_half_depth': .9, 'fraction_comparable': .9, 'retained_depth': 90,
                'filtered_depth': 200, 'taxonomic_effective_number': 30., 'ambiguous_fraction': .01,
                'profile': rng.normal(size=12).tolist()}
            features[str(pid)] = {'M': rng.normal(size=24), 'S': rng.normal(size=24)}
            features[str(pid)]['M'][:3] += y[i] * .5
    ids = sorted({str(r[k]) for r, p in zip(rows, pairs) if p['available'] for k in ('a', 'b')}, key=int)
    def oldhead(dim):
        return {'standard_mean': [0.] * dim, 'standard_scale': [1.] * dim, 'pca_mean': [0.] * dim,
                'pca_components': np.eye(dim)[:4].tolist(), 'coef': [[0.] * 4], 'intercept': [0.], 'alpha': 1}
    for directory in ['data', 'results']:
        (root / directory).mkdir()
    for name, value in [('sample', rows), ('monomers', monomers), ('pairs', pairs), ('required-proteins', ids), ('dev_family_witnesses', {})]:
        atomic(root / 'data' / f'{name}.json', value)
    atomic(root / 'data/baseline.json', {r['uid']: {**r, 'score': float(base[i])} for i, r in enumerate(rows) if r['split'] == 'val'})
    atomic(root / 'data/source_heads.json', {name: oldhead(dim) for name, dim in [('true', 128), ('shuffled', 128), ('quality', 68)]})
    np.savez(root / 'data/source.npz', uids=np.array([r['uid'] for r in rows]), true=np.zeros((n, 128)),
             shuffled=np.zeros((n, 128)), quality=np.zeros((n, 68)), gate=gate)
    metrics = {'results': {}}
    for pop in ['assessment', 'calibration', 'fixed_dev_sample']:
        m = np.array([r['split'] == 'val' if pop == 'fixed_dev_sample' else r['role'] == pop for r in rows])
        values = {'ap': float(average_precision_score(y[m], base[m])), 'auroc': float(roc_auc_score(y[m], base[m]))}
        metrics['results'][pop] = {'metrics': {name: values for name in ['baseline', 'true', 'shuffled', 'quality']}}
    atomic(root / 'data/source_metrics.json', metrics)
    return rows, pairs, ids, features, base, gate


class AnalysisFlow(unittest.TestCase):
    def test_complete_pipeline_and_train_only_transforms(self):
        cfg = copy.deepcopy(config()); cfg['fusion'].update({'bootstrap_replicates': 20, 'protein_pca_components': 4,
            'minimum_fit_per_class': 20, 'minimum_calibration_assessment_per_class': 10})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows, pairs, ids, features, base, gate = fixture(root)
            (root / 'scripts').mkdir()
            (root / 'scripts/verify.py').write_bytes(Path(verify.__file__).read_bytes())
            with patch.object(analyze, 'ROOT', root), patch.object(analyze, 'config', return_value=cfg), \
                 patch.object(analyze, 'verify_freeze', return_value='fixture'), patch.object(analyze, 'check_budget'), \
                 patch.object(analyze, 'load_feature', side_effect=lambda pid, fp, meta: features[pid]), redirect_stdout(io.StringIO()):
                analyze.main()
                with patch.object(verify, 'ROOT', root), patch.object(verify, 'config', return_value=cfg), \
                     patch.object(verify, 'verify_freeze', return_value='fixture'):
                    verify.main()
                    self.assertTrue(read(root / 'results/verification.json')['passed'])
                    with np.load(root / 'results/dev_predictions.npz') as f:
                        corrupted = {k: f[k].copy() for k in f.files}
                    corrupted['M'][0] += 1
                    np.savez_compressed(root / 'results/dev_predictions.npz', **corrupted)
                    with self.assertRaises(AssertionError):
                        verify.main()
                    corrupted['M'][0] -= 1
                    np.savez_compressed(root / 'results/dev_predictions.npz', **corrupted)
            result = read(root / 'results/heads.json'); fitids = {str(r[k]) for r, p in zip(rows, pairs) if p['available'] and r['split'] == 'train' for k in ('a', 'b')}
            h = result['transforms']['M']
            self.assertEqual(set(h['fit_protein_ids']), fitids)
            np.testing.assert_allclose(h['mean'], np.array([features[p]['M'] for p in ids if p in fitids]).mean(0))
            self.assertEqual(result['heads']['SP']['dimensions'], 24)
            self.assertEqual(result['heads']['U']['dimensions'], 4)
            with np.load(root / 'results/dev_predictions.npz') as f:
                for arm in ['M', 'S', 'P', 'SP', 'U', 'M_on_S']:
                    np.testing.assert_array_equal(f[arm][gate[200:] == 0], base[200:][gate[200:] == 0])
            metrics = read(root / 'results/metrics.json')
            self.assertIn('M_minus_SP', metrics['results']['assessment']['ap_intervals'])
            self.assertTrue(metrics['original_metrics_reproduced'])
            self.assertFalse(metrics['test_accessed'])

    def test_incomplete_computation_refused_before_any_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root)
            with patch.object(analyze, 'ROOT', root), patch.object(analyze, 'verify_freeze', return_value='fixture'), \
                 patch.object(analyze, 'check_budget'), patch.object(analyze, 'load_feature', return_value=None), \
                 patch.object(analyze, 'fit_transform') as fit:
                with self.assertRaisesRegex(RuntimeError, 'Incomplete computational coverage'):
                    analyze.main()
                fit.assert_not_called()
            self.assertFalse((root / 'results/fit-start.json').exists())


if __name__ == '__main__':
    unittest.main()
