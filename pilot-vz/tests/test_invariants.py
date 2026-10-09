import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import study
from inputs import query_only, reconstruct
from heads import fit_head
from statistics_vz import ap_setup, weighted_ap
from decision import decide


class Invariants(unittest.TestCase):
    def test_only_homolog_rows_are_removed(self):
        t = np.array([[0, 26, 2, 3, 26, 5], [6, 26, 8, 9, 26, 11]], dtype=np.uint8)
        q = query_only(t, 3)
        self.assertEqual(q.shape, (1, 6)); np.testing.assert_array_equal(q[0], t[0])
        t[1] = 12; np.testing.assert_array_equal(query_only(t, 3), q)
        q[0, 0] = 20; self.assertEqual(t[0, 0], 0)

    def test_query_coordinates_and_alphabet_fail_closed(self):
        for tokens, breakpoint in [(np.array([[0, 1]]), 0), (np.array([[26, 1]]), 1), (np.array([[0, 28]]), 1), (np.array([0, 1]), 1)]:
            with self.assertRaises(ValueError): query_only(tokens, breakpoint)

    def test_previously_unavailable_pair_never_encoded(self):
        with self.assertRaisesRegex(ValueError, 'previously unavailable'):
            reconstruct({'available': False}, {}, lambda x: None)

    def test_train_scaling_and_alpha_ignore_assessment_and_ineligible_training(self):
        rng = np.random.default_rng(42); x = rng.normal(size=(360, 128)); y = np.arange(360) % 2
        fit = np.arange(360) < 180; fit[0:20] = False
        cal = (np.arange(360) >= 180) & (np.arange(360) < 260)
        gate = np.ones(360); gate[:20] = 0; gate[330:] = 0
        base = rng.normal(size=360); cfg = study.config()
        h, _, z = fit_head(x, y, fit, cal, base, gate, cfg)
        xx = x.copy(); yy = y.copy(); bb = base.copy()
        xx[260:] *= 1000; xx[:20] *= 1000; yy[260:] = 1 - yy[260:]; bb[:180] += 500
        hh, _, _ = fit_head(xx, yy, fit, cal, bb, gate, cfg)
        self.assertEqual(h, hh)
        np.testing.assert_allclose(h['standard_mean'], x[fit].mean(0))
        np.testing.assert_array_equal(z[gate == 0], base[gate == 0])
        self.assertEqual(h['supervised_parameters'], 33)

    def test_alpha_ties_choose_zero(self):
        rng = np.random.default_rng(8); x = rng.normal(size=(220, 128)); y = np.arange(220) % 2
        fit = np.arange(220) < 160; cal = ~fit; gate = fit.astype(float)
        h, _, _ = fit_head(x, y, fit, cal, np.zeros(220), gate, study.config())
        self.assertEqual(h['alpha'], 0)

    def test_nonfinite_active_evidence_rejected_at_zero_alpha(self):
        with self.assertRaises(FloatingPointError): study.fuse([0.], [np.nan], [1.], 0)
        np.testing.assert_array_equal(study.fuse([.5], [np.nan], [0.], 1), [.5])

    def test_corruption_partial_cache_and_input_changes_fail(self):
        expected = {'uid': 'a', 'available': True, 'gate': .8, 'q_tokens_sha256': 'q'}
        with tempfile.TemporaryDirectory() as d, patch.object(study, 'ROOT', Path(d)):
            self.assertIsNone(study.load_record('a', 'f', expected))
            study.save_record('a', np.ones(128), 'f', expected)
            self.assertIsNotNone(study.load_record('a', 'f', expected))
            with self.assertRaises(ValueError): study.load_record('a', 'f', {**expected, 'gate': .7})
            p = Path(d) / 'features/a.json'; original = p.read_text(); p.write_text(original + ' ')
            with self.assertRaises(ValueError): study.load_record('a', 'f', expected)
            p.write_text(original); p.with_suffix('.sha.json').unlink()
            with self.assertRaises(ValueError): study.load_record('a', 'f', expected)
            with self.assertRaises(RuntimeError): study.save_record('a', np.ones(128), 'f', expected)

    def test_unavailable_evidence_cannot_be_nonzero(self):
        with tempfile.TemporaryDirectory() as d, patch.object(study, 'ROOT', Path(d)):
            with self.assertRaises(ValueError):
                study.save_record('a', np.ones(128), 'f', {'available': False, 'gate': 0})

    def test_fast_ap_matches_independent_sklearn(self):
        rng = np.random.default_rng(95)
        for _ in range(20):
            y = rng.integers(0, 2, 101); z = rng.integers(0, 10, len(y)); w = rng.poisson(1, len(y)) * rng.poisson(1, len(y))
            self.assertAlmostEqual(weighted_ap(ap_setup(y, z), w), average_precision_score(y, z, sample_weight=w), places=13)

    def test_no_significant_difference_is_not_equivalence(self):
        from statistics_vz import CONTRASTS
        cfg = study.config(); pts = {name: 0. for name in CONTRASTS}
        ci = {name: {'low': -.02, 'high': .02} for name in CONTRASTS}
        pts['Q_minus_baseline'] = .02; ci['Q_minus_baseline'] = {'low': .001, 'high': .04}
        m = {'contrasts': pts, 'ap_intervals': ci, 'metrics': {'Q': {'auroc': .66}, 'baseline': {'auroc': .65}}}
        d = decide({'assessment': m}, cfg)
        self.assertTrue(d['useful_Q_gain']); self.assertFalse(d['query_only_recovery_within_margin'])
        self.assertFalse(d['equivalence']['Q_minus_shuffled'])
        for name in ('Q_minus_shuffled', 'Q_minus_true'): ci[name] = {'low': -.005, 'high': .03}
        d = decide({'assessment': m}, cfg)
        self.assertTrue(d['query_only_recovery_within_margin']); self.assertFalse(d['all_Q_true_shuffled_equivalent'])


if __name__ == '__main__': unittest.main()
