"""Scientific invariants and numerical reconstruction; no real outcome fitting."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import study
from pooling import blocks, global_pool, local_pool, pool_views, spatial_indices
from msa_diagnostics import neff
from analyze_v2 import fit_head, evidence, ap_setup, weighted_ap, contrasts


class PoolingInvariants(unittest.TestCase):
    def test_local_patch_vs_spatial_null_with_identical_global_distribution(self):
        p = torch.zeros(64, 64, 32); p[:8, :8] = 10
        good = torch.ones(64, dtype=torch.bool)
        pa = spatial_indices(good.numpy(), 'fixture', 1); pb = spatial_indices(good.numpy(), 'fixture', 2)
        views, meta = pool_views(p, good, good, pa, pb)
        torch.testing.assert_close(views['global'], global_pool(p[pa][:, pb], good, good), atol=1e-7, rtol=1e-7)
        self.assertLess(float(views['scrambled'][64:96].mean()), float(views['local'][64:96].mean()))
        self.assertEqual(meta['blocks_qualifying'], 64)
        torch.testing.assert_close(views['local'][:64], views['scrambled'][:64], atol=0, rtol=0)

    def test_original_coordinates_masks_symmetry_and_small_blocks(self):
        torch.manual_seed(10); p = torch.randn(19, 11, 32)
        a = torch.tensor([True, False] * 9 + [True]); b = torch.tensor([True] * 6 + [False] * 5)
        f, m = local_pool(p, a, b)
        bad = p.clone(); bad[~a] = 1e20; bad[:, ~b] = -1e20
        torch.testing.assert_close(f, local_pool(bad, a, b)[0], atol=0, rtol=0)
        torch.testing.assert_close(f, local_pool(p.transpose(0, 1), b, a)[0], atol=2e-6, rtol=2e-6)
        self.assertEqual(blocks(19), [0, 7, 13, 19])
        one, om = local_pool(torch.ones(1, 3, 32), torch.tensor([True]), torch.ones(3, dtype=torch.bool))
        self.assertEqual(om['k_used'], 1); self.assertTrue(torch.isfinite(one).all())
        with self.assertRaises(FloatingPointError): local_pool(p, torch.zeros(19, dtype=torch.bool), b)

    def test_invalid_values_and_mask_changing_permutations_fail(self):
        p = torch.ones(9, 8, 32); a = torch.tensor([True] * 8 + [False]); b = torch.ones(8, dtype=torch.bool)
        with self.assertRaises(ValueError): pool_views(p, a, b, np.arange(9)[::-1].copy(), np.arange(8))
        p[0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError): local_pool(p, a, b)

    def test_spatial_permutation_is_deterministic_label_free_and_chain_specific(self):
        mask = np.array([True, False] * 40)
        p = spatial_indices(mask, 'uid', 42)
        np.testing.assert_array_equal(p, spatial_indices(mask, 'uid', 42))
        np.testing.assert_array_equal(mask[p], mask)
        np.testing.assert_array_equal(np.sort(p), np.arange(len(mask)))
        self.assertFalse(np.array_equal(p, spatial_indices(mask, 'uid', 43)))


class StatisticalInvariants(unittest.TestCase):
    def test_weighted_ap_matches_sklearn_with_ties_and_zero_weights(self):
        rng = np.random.default_rng(20); y = rng.integers(0, 2, 100); z = rng.integers(0, 6, 100)
        for _ in range(30):
            w = rng.poisson(1, 100)
            self.assertAlmostEqual(weighted_ap(ap_setup(y, z), w), average_precision_score(y, z, sample_weight=w), places=13)

    def test_train_only_fit_and_serialized_head(self):
        rng = np.random.default_rng(88); x = rng.normal(size=(400, 128)); y = np.arange(400) % 2
        x[:, :5] += y[:, None] * .8
        train = np.arange(400) < 240; cal = (np.arange(400) >= 240) & (np.arange(400) < 320)
        x[~train, 100:] += 20
        h, scores = fit_head(x, y, train, cal, np.zeros(400), np.ones(400), study.config())
        np.testing.assert_allclose(h['standard_mean'], x[train].mean(0), atol=1e-12)
        np.testing.assert_allclose(scores, evidence(h, x) * h['alpha'], atol=1e-10)
        self.assertIn(h['alpha'], study.config()['fusion']['alpha_grid'])

    def test_contrast_requires_pairing_specific_improvement(self):
        values = {'local_true': .7, 'local_shuffled': .69, 'global_true': .65, 'global_shuffled': .64,
            'baseline': .6, 'quality': .64, 'scrambled_true': .66, 'scrambled_shuffled': .65,
            'local_true_on_shuffled': .69, 'scrambled_true_on_shuffled': .65}
        self.assertAlmostEqual(contrasts(values)['pairing_gap_improvement'], 0.)
        self.assertAlmostEqual(contrasts(values)['locality_pairing_gap'], 0.)

    def test_neff80_is_not_deduplicated_depth_and_requires_comparison_coverage(self):
        t = np.array([[0]*5, [0]*5, [0, 0, 0, 0, 1], [1]*5], dtype=np.uint8)
        result = neff(t, np.ones(5, bool)); self.assertEqual(result['neff80'], 2.)
        t[2] = [0, 0, 25, 25, 25]
        self.assertEqual(neff(t, np.ones(5, bool))['neff80'], 3.)


class ExecutionInvariants(unittest.TestCase):
    def test_fallback_and_active_nan_refusal(self):
        base = np.array([.3, -1., 2.]); gate = np.array([0., 1., 0.])
        np.testing.assert_array_equal(study.fuse(base, [np.nan, 2., np.nan], gate, .5), [.3, 0., 2.])
        with self.assertRaises(FloatingPointError): study.fuse(base, [0., np.nan, 0.], gate, 1.)

    def test_corrupt_incomplete_and_mismatched_records_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(study, 'ROOT', Path(tmp)):
            (Path(tmp) / 'features').mkdir()
            cfg = {'arms': ['local_true', 'local_shuffled', 'scrambled_true', 'scrambled_shuffled']}
            with patch.object(study, 'config', return_value=cfg):
                self.assertIsNone(study.load_record('x', 'f'))
                value = {'uid': 'x', 'available': False, 'gate': 0, **{a: [0.] * 128 for a in cfg['arms']}}
                study.save_record('x', value, 'f'); self.assertEqual(study.load_record('x', 'f')['uid'], 'x')
                with self.assertRaises(ValueError): study.load_record('x', 'other')
                p = Path(tmp) / 'features/x.json'; p.write_text('{}')
                with self.assertRaises(ValueError): study.load_record('x', 'f')
                p.unlink()
                with self.assertRaises(ValueError): study.load_record('x', 'f')


if __name__ == '__main__': unittest.main()
