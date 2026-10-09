import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import study
from heads import fit_transform, symmetric, fit_head
from statistics_vy import ap_setup, weighted_ap


class Heads(unittest.TestCase):
    def test_protein_transform_does_not_see_dev_or_edge_multiplicity(self):
        rng = np.random.default_rng(6); x = rng.normal(size=(70, 24)); ids = list(map(str, range(70)))
        train = np.arange(40)
        a, _ = fit_transform(x, train, ids, 4)
        altered = x.copy(); altered[40:] = 1e8
        b, _ = fit_transform(altered, train, ids, 4)
        self.assertEqual(a, b)
        np.testing.assert_allclose(a['mean'], x[:40].mean(0))
        with self.assertRaises(ValueError):
            fit_transform(x, np.r_[train, train[0]], ids, 4)

    def test_pair_terms_are_symmetric_but_not_unary(self):
        a, b = np.array([1., 3.]), np.array([3., 1.])
        np.testing.assert_array_equal(symmetric(a, b), symmetric(b, a))
        c = np.array([2., 2.])
        np.testing.assert_array_equal(symmetric(a, b)[:2], symmetric(c, c)[:2])
        self.assertFalse(np.array_equal(symmetric(a, b), symmetric(c, c)))

    def test_fit_and_alpha_do_not_see_assessment_labels_or_features(self):
        rng = np.random.default_rng(17); x = rng.normal(size=(240, 12)); y = np.arange(240) % 2
        train, cal = np.arange(240) < 120, (np.arange(240) >= 120) & (np.arange(240) < 180)
        base, gate = rng.normal(size=240), np.ones(240); gate[200:] = 0
        cfg = study.config()['fusion']
        h, _, pred = fit_head(x, y, train, cal, base, gate, cfg)
        other_y = y.copy(); other_y[180:] = 1 - other_y[180:]
        other_x = x.copy(); other_x[180:] *= 1000
        hh, _, _ = fit_head(other_x, other_y, train, cal, base, gate, cfg)
        self.assertEqual(h, hh)
        np.testing.assert_array_equal(pred[200:], base[200:])
        np.testing.assert_allclose(h['mean'], x[train].mean(0))

    def test_fast_ap_matches_sklearn_with_ties_zero_weights_and_imbalance(self):
        rng = np.random.default_rng(72)
        for _ in range(20):
            y = rng.integers(0, 2, 91); z = rng.integers(0, 6, len(y)).astype(float)
            weights = rng.poisson(1, len(y)) * rng.poisson(1, len(y))
            a = weighted_ap(ap_setup(y, z), weights)
            b = average_precision_score(y, z, sample_weight=weights)
            self.assertAlmostEqual(a, b, places=13)

    def test_corrupt_or_partial_cache_never_becomes_fallback(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(study, 'ROOT', Path(directory)), \
             patch.object(study, 'config', return_value={'encoder': {'summary_dimensions': 4}}):
            self.assertIsNone(study.load_feature('1', 'f'))
            study.save_feature('1', np.ones(4), np.zeros(4), 'f', {'query_sha256': 'q', 'tokens_sha256': 't'})
            self.assertIsNotNone(study.load_feature('1', 'f'))
            p, side = study.feature_paths('1'); side.unlink()
            with self.assertRaises(ValueError):
                study.load_feature('1', 'f')
            with self.assertRaises(RuntimeError):
                study.save_feature('1', np.ones(4), np.zeros(4), 'f', {})

    def test_nonfinite_computation_rejected_even_at_alpha_zero(self):
        with self.assertRaises(FloatingPointError):
            study.fuse([1.], [np.nan], [1.], 0)
        np.testing.assert_array_equal(study.fuse([1.], [np.nan], [0.], 1), [1.])


if __name__ == '__main__':
    unittest.main()
