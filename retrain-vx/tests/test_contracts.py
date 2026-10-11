import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shared import fuse, quality_from_profiles
from heads import fit_head
from statistics_rx import ap_setup, weighted_ap, decision
from msa import pair, profile, quality_features


class Contracts(unittest.TestCase):
    def test_fallback_and_nonfinite_fail_closed_even_alpha_zero(self):
        base = np.array([.1, -.4]); ev = np.array([2., 9.]); gate = np.array([.5, 0.])
        np.testing.assert_array_equal(fuse(base, ev, gate, 1), [1.1, -.4])
        for alpha in [0, 1]:
            with self.assertRaises(ValueError):
                fuse(base, [np.nan, 9], gate, alpha)
        with self.assertRaises(ValueError):
            fuse(base, ev, [1.1, 0], 0)

    def test_precomputed_quality_is_exact_original_constructor(self):
        rng = np.random.default_rng(4); alphabet = np.asarray(list('ACDEFGHIKLMNPQRSTVWY'))
        def monomer(query):
            return {'query': query, 'mask': '*' * len(query), 'rows': [(f'g{i}', ''.join(rng.choice(alphabet, len(query))), 'g:f:o:c:p') for i in range(12)]}
        a, b = monomer('A' * 24), monomer('C' * 32)
        cfg = {'maximum_combined_residues': 1536, 'minimum_quality_fraction_per_chain': .5,
            'minimum_homolog_query_coverage': .5, 'minimum_paired_homologs': 8,
            'maximum_paired_rows_including_query': 128, 'redundancy_identity': .9, 'minimum_identity_comparison_coverage': .8}
        with patch('msa.shuffled', side_effect=AssertionError('Shuffle must not execute')):
            tokens, meta = pair(a, b, cfg)
        self.assertEqual(tokens.shape, (13, 56))
        np.testing.assert_array_equal(quality_from_profiles(profile(a), profile(b), meta), quality_features(a, b, meta))

    def test_fit_and_alpha_do_not_use_assessment_labels(self):
        rng = np.random.default_rng(7); x = rng.normal(size=(640, 68)); y = np.arange(640) % 2
        fit = np.arange(640) < 320; cal = (np.arange(640) >= 320) & (np.arange(640) < 440)
        base = rng.normal(size=640); gate = np.ones(640); cfg = {'seed': 20261007, 'fusion': {'alpha_grid': [0, .1, .25, .5, 1]}}
        a, ev, pred = fit_head(x, y, fit, base, gate, cal, cfg)
        changed = y.copy(); changed[~(fit | cal)] = 1 - changed[~(fit | cal)]
        b, ev2, pred2 = fit_head(x, changed, fit, base, gate, cal, cfg)
        self.assertEqual(a, b); np.testing.assert_array_equal(pred, pred2)
        np.testing.assert_allclose(a['standard_mean'], x[fit].mean(0), atol=1e-14)

    def test_weighted_ap_ties_and_zero_weights(self):
        y = np.asarray([1, 0, 1, 0, 1, 0]); z = np.asarray([.3, .3, -.2, .1, .3, .1])
        w = np.asarray([0, 2, 3, 1, 4, 0])
        self.assertAlmostEqual(weighted_ap(ap_setup(y, z), w), average_precision_score(y, z, sample_weight=w), places=14)

    def test_predictive_decision_requires_quality_and_auroc(self):
        p = {'metrics': {'true': {'ap': .67, 'auroc': .66}, 'baseline': {'ap': .64, 'auroc': .65}},
            'contrasts': {'true_minus_baseline': .03}, 'intervals': {'true_minus_baseline': {'low': .01, 'high': .04}, 'true_minus_quality': {'low': .002, 'high': .025}}}
        self.assertEqual(decision(p)['status'], 'useful_predictive_gain_supported')
        p['intervals']['true_minus_quality']['low'] = -.001
        self.assertEqual(decision(p)['status'], 'no_go_or_inconclusive')
        self.assertFalse(decision(p)['test_authorized'])


if __name__ == '__main__':
    unittest.main()
