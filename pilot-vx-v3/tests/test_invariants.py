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
import inputs
from inputs import column_shuffle, validate_columns, reconstruct, seed_for
from diagnostics import describe
from heads import fit_head
from statistics_v3 import ap_setup, weighted_ap
from decision import decide


class Invariants(unittest.TestCase):
    def test_column_histograms_query_PAD_and_source_preserved(self):
        rng = np.random.default_rng(123)
        t = rng.integers(0, 26, (65, 24), dtype=np.uint8)
        t[1:] = np.broadcast_to(np.arange(64, dtype=np.uint8)[:, None] % 26, (64, 24))
        t[:, 5] = 26; original = t.copy()
        c, digest = column_shuffle(t, 12, 9)
        validate_columns(t, c, 12)
        np.testing.assert_array_equal(t, original)
        np.testing.assert_array_equal(c[0], t[0])
        np.testing.assert_array_equal(np.sort(c[1:], axis=0), np.sort(t[1:], axis=0))
        self.assertTrue(np.all(c[:, 5] == 26))
        self.assertLess(np.mean(c[1:, 0] == c[1:, 1]), .3)
        self.assertGreater(len(np.unique(c[1:], axis=0)), len(np.unique(t[1:], axis=0)))
        again, dd = column_shuffle(t, 12, 9)
        np.testing.assert_array_equal(c, again); self.assertEqual(digest, dd)
        other, od = column_shuffle(t, 12, 10)
        self.assertFalse(np.array_equal(c, other)); self.assertNotEqual(digest, od)
        c[0, 0] = 25; self.assertEqual(t[0, 0], original[0, 0])

    def test_invalid_coordinates_alphabet_and_nonuniform_PAD_fail(self):
        invalid = [(np.array([[0, 1], [0, 1]]), 0), (np.array([[26, 1], [26, 1]]), 1),
                   (np.array([[0, 28], [0, 1]]), 1), (np.array([0, 1]), 1),
                   (np.array([[0, 1], [26, 1]]), 1), (np.array([[0., 1.], [0., 1.]]), 1)]
        for tokens, breakpoint in invalid:
            with self.assertRaises(ValueError): column_shuffle(tokens, breakpoint, 4)

    def test_constant_columns_are_valid_and_query_does_not_enter_histogram(self):
        t = np.array([[4, 3, 2, 1], [0, 25, 2, 8], [0, 25, 2, 8]], dtype=np.uint8)
        c, _ = column_shuffle(t, 2, 5)
        np.testing.assert_array_equal(c, t)
        c[0, 0] = 0
        with self.assertRaises(ValueError): validate_columns(t, c, 2)
        c = t.copy(); c[1, 0] = 1
        with self.assertRaises(ValueError): validate_columns(t, c, 2)

    def test_pair_seed_is_label_independent_and_namespaced(self):
        p = {'uid': 'train-123', 'a': 2, 'b': 3, 'label': 0}
        self.assertEqual(seed_for(p), seed_for({**p, 'label': 1}))
        self.assertNotEqual(seed_for(p), seed_for(p, 1))
        self.assertNotEqual(seed_for(p), seed_for({**p, 'uid': 'train-124'}))

    def test_previously_unavailable_pair_never_encoded(self):
        with self.assertRaisesRegex(ValueError, 'previously unavailable'):
            reconstruct({'available': False}, {}, lambda x: None)

    def test_independent_input_replay_rejects_histogram_preserving_wrong_shuffle(self):
        p = {'uid': 'p', 'a': 1, 'b': 2, 'available': True, 'length': 16, 'breakpoint': 8, 'paired_depth': 17}
        t = np.random.default_rng(45).integers(0, 26, (17, 16), dtype=np.uint8); t[:, 4] = 26
        seed = seed_for(p); c, pdigest = column_shuffle(t, 8, seed)
        with tempfile.TemporaryDirectory() as directory, patch.object(inputs, 'ROOT', Path(directory)):
            path = Path(directory) / 'p.npz'; study.atomic_npz(path, true=t, C=c)
            p.update({'c_seed': seed, 'paired_tokens_sha256': study.token_hash(t), 'q_tokens_sha256': study.token_hash(t[:1]),
                'c_tokens_sha256': study.token_hash(c), 'c_permutations_sha256': pdigest, 'c_input_file': 'p.npz', 'c_input_sha256': study.sha(path)})
            inputs.prepared(p, replay=True)
            with self.assertRaises(ValueError): inputs.prepared({**p, 'c_seed': seed + 1}, replay=True)
            # Same histograms, updated file/token checksums, but not the prescribed seed.
            wrong, _ = column_shuffle(t, 8, seed + 1); study.atomic_npz(path, true=t, C=wrong)
            pp = {**p, 'c_input_sha256': study.sha(path), 'c_tokens_sha256': study.token_hash(wrong)}
            inputs.prepared(pp, replay=False)
            with self.assertRaisesRegex(ValueError, 'permutation replay'): inputs.prepared(pp, replay=True)
            with path.open('ab') as f: f.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum'): inputs.prepared(pp)

    def test_manipulation_diagnostic_separates_row_and_column_shuffling(self):
        t = np.tile(np.arange(128, dtype=np.uint8)[:, None] % 2, (1, 12)); t[:, 2] = 26
        c, _ = column_shuffle(t, 6, 48)
        previous = {'null': {'permutation': [0] + list(range(127, 0, -1))}}
        d = describe(t, c, {'uid': 'diagnostic', 'breakpoint': 6}, previous, 20)
        for kind in ('within_A', 'within_B'):
            v = d['categorical_covariance_energy'][kind]
            self.assertAlmostEqual(v['true'], v['shuffled'], places=15)
            self.assertLess(v['C'], v['true'] / 10)
        for region in ('all', 'A', 'B'):
            v = d['row_structure'][region]
            self.assertAlmostEqual(v['C']['query_match_mean_including_gaps'], v['true']['query_match_mean_including_gaps'], places=15)

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
        from statistics_v3 import CONTRASTS
        cfg = study.config(); pts = {name: 0. for name in CONTRASTS}
        ci = {name: {'low': -.02, 'high': .02} for name in CONTRASTS}
        pts['C_minus_baseline'] = .02; ci['C_minus_baseline'] = {'low': .001, 'high': .04}
        m = {'contrasts': pts, 'ap_intervals': ci, 'metrics': {'C': {'auroc': .66}, 'baseline': {'auroc': .65}}}
        d = decide({'assessment': m}, cfg)
        self.assertTrue(d['useful_C_gain']); self.assertFalse(d['profile_recovery_within_margin'])
        self.assertFalse(d['equivalence']['C_minus_shuffled'])
        for name in ('C_minus_shuffled', 'C_minus_true'): ci[name] = {'low': -.005, 'high': .03}
        d = decide({'assessment': m}, cfg)
        self.assertTrue(d['profile_recovery_within_margin']); self.assertFalse(d['all_C_true_shuffled_equivalent'])
        self.assertFalse(d['profiles_recover_gain_beyond_Q'])
        pts['C_minus_Q'] = .015; ci['C_minus_Q'] = {'low': -.001, 'high': .03}
        self.assertFalse(decide({'assessment': m}, cfg)['profiles_recover_gain_beyond_Q'])
        ci['C_minus_Q']['low'] = .001
        self.assertTrue(decide({'assessment': m}, cfg)['profiles_recover_gain_beyond_Q'])


if __name__ == '__main__': unittest.main()
