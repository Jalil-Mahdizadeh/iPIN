"""Complete synthetic study checks governance, reconstruction and missing-data refusal."""
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
import analyze_v2 as analysis
from study import atomic, read, config
from launch import submission_environment


class AnalysisFlow(unittest.TestCase):
    def test_incomplete_study_never_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'data').mkdir(); (root / 'results').mkdir()
            atomic(root / 'data/sample.json', [{'uid': 'missing'}]); atomic(root / 'data/source_metadata.json', [{}])
            with patch.object(analysis, 'ROOT', root), patch.object(analysis, 'verify_freeze', return_value='fixture'), \
                 patch.object(analysis, 'load_record', return_value=None), patch.object(analysis, 'fit_head') as fit:
                analysis.main(); fit.assert_not_called()
            self.assertEqual(read(root / 'results/decision.json')['status'], 'inconclusive_incomplete')
            self.assertFalse((root / 'results/heads.json').exists())

    def test_full_fit_calibration_assessment_and_exact_fallback(self):
        rng = np.random.default_rng(53); cfg = config(); cfg['fusion']['bootstrap_replicates'] = 20
        n = 800; y = np.arange(n) % 2; gate = np.ones(n); gate[400::11] = 0
        sample = []; metadata = []; data = {}; base = rng.normal(size=n)
        xx = {arm: rng.normal(size=(n, 128)) for arm in cfg['arms']}
        for arm, x in xx.items():
            x[:, :5] += y[:, None] * (1. if arm.endswith('true') else .1)
            x[400:, 110:] += 8
            x[gate == 0] = 0
        diag = {'paired_diversity': {'neff80': 70., 'neff_per_valid_residue': .1}, 'minimum_quality': .8,
                'valid_interchain_cells': 10000, 'true_support': {'fraction_cells_at_least_half_rows': .9},
                'actual_token_changed_fraction': .2, 'length_asymmetry': 1.5,
                'candidates_after_coverage_and_conflict_filters': 1000, 'actual_sequence_changed_fraction': .95}
        for i in range(n):
            uid = str(i); role = 'train' if i < 400 else 'calibration' if i < 600 else 'assessment'
            r = {'uid': uid, 'split': 'train' if i < 400 else 'val', 'label': int(y[i]), 'role': role,
                 'a': i * 2, 'b': i * 2 + 1, 'length': [400, 900, 1300][i % 3]}
            sample.append(r)
            p = {'uid': uid, 'gate': float(gate[i]), 'available': bool(gate[i]), 'reason': 'available' if gate[i] else 'long_query'}
            metadata.append(p)
            data[uid] = {**p, 'a': r['a'], 'b': r['b'], 'diagnostics': diag,
                         **{arm: x[i].tolist() for arm, x in xx.items()}}
        def empty_head(dim):
            return {'standard_mean': [0.] * dim, 'standard_scale': [1.] * dim, 'pca_mean': [0.] * dim,
                'pca_components': np.eye(dim)[:32].tolist(), 'coef': [[0.] * 32], 'intercept': [0.], 'alpha': 1}
        old_heads = {k: empty_head(dim) for k, dim in [('true', 128), ('shuffled', 128), ('quality', 68)]}
        source_metrics = {'fit_counts': {'fit': [200, 200], 'calibration': [100, 100], 'assessment': [100, 100]}, 'results': {}}
        for pop, mask in [('assessment', np.arange(n) >= 600), ('calibration', (np.arange(n) >= 400) & (np.arange(n) < 600)), ('fixed_dev_sample', np.arange(n) >= 400)]:
            source_metrics['results'][pop] = {'metrics': {k: {'ap': float(average_precision_score(y[mask], base[mask]))}
                for k in ['baseline', 'true', 'shuffled', 'quality']}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'data').mkdir(); (root / 'results').mkdir()
            atomic(root / 'data/sample.json', sample); atomic(root / 'data/source_metadata.json', metadata)
            atomic(root / 'data/source_heads.json', old_heads); atomic(root / 'data/source_metrics.json', source_metrics)
            atomic(root / 'data/dev_family_witnesses.json', {})
            atomic(root / 'data/baseline.json', {r['uid']: {'score': float(base[i]), 'label': r['label'], 'a': r['a'], 'b': r['b']}
                for i, r in enumerate(sample) if i >= 400})
            np.savez(root / 'data/source.npz', uids=np.array([r['uid'] for r in sample]), true=np.zeros((n, 128)),
                     shuffled=np.zeros((n, 128)), quality=np.zeros((n, 68)), gate=gate)
            with patch.object(analysis, 'ROOT', root), patch.object(analysis, 'config', return_value=cfg), \
                 patch.object(analysis, 'verify_freeze', return_value='fixture'), \
                 patch.object(analysis, 'load_record', side_effect=lambda uid, fp: data[uid]), redirect_stdout(io.StringIO()):
                analysis.main()
            heads = read(root / 'results/heads.json'); metrics = read(root / 'results/metrics.json')
            for arm, h in heads.items(): np.testing.assert_allclose(h['standard_mean'], xx[arm][:400].mean(0), atol=1e-12)
            saved = np.load(root / 'results/dev_predictions.npz', allow_pickle=False)
            for name in saved.files:
                if name not in ['uids', 'labels']:
                    np.testing.assert_array_equal(saved[name][gate[400:] == 0], base[400:][gate[400:] == 0])
            self.assertEqual(metrics['results']['assessment']['rows'], 200)
            self.assertEqual(metrics['results']['fixed_dev_sample']['rows'], 400)
            self.assertFalse(metrics['test_accessed'])
            self.assertTrue(metrics['original_metrics_reproduced'])
            self.assertIn('pairing_gap_improvement', metrics['results']['assessment']['ap_intervals'])
            self.assertFalse(read(root / 'results/decision.json')['production_authorized'])

    def test_scheduler_environment_does_not_inherit_other_allocation(self):
        with patch.dict('os.environ', {'SLURM_JOB_ID': 'wrong', 'SBATCH_GRES': 'wrong', 'SRUN_CPU_BIND': 'wrong',
                        'CUDA_VISIBLE_DEVICES': 'wrong', 'NVIDIA_VISIBLE_DEVICES': 'wrong', 'SLURM_CONF': '/site/conf', 'PATH': '/bin'}, clear=True):
            env = submission_environment()
        self.assertEqual(env, {'SLURM_CONF': '/site/conf', 'PATH': '/bin'})


if __name__ == '__main__': unittest.main()
