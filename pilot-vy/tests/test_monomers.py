import copy
import hashlib
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from study import config, text_hash
from monomers import select, profile, neff80, encode, pair_gate, PAD, PROFILE_NAMES
from encoder import pool_query


def fixture():
    rng = np.random.default_rng(71); query = 'ARNDCQEGHILKMFPSTWYV' * 2
    rows = []
    for i in range(16):
        seq = ''.join(rng.choice(list('ARNDCQEGHILKMFPSTWYV'), len(query)))
        rows.append([f'g{i}', seq, f'genus:family{i % 5}:order:class:phylum'])
    return {'query': query, 'mask': '*' * len(query), 'query_sha256': text_hash(query),
            'rows': rows, 'stats': {'raw_rows': len(rows)}}


class Monomers(unittest.TestCase):
    def test_selection_is_independent_of_cache_order_and_external_partner(self):
        cfg = config()['monomer']; m = fixture()
        a, ma = select(m, cfg, 23)
        reversed_cache = copy.deepcopy(m); reversed_cache['rows'].reverse()
        b, mb = select(reversed_cache, cfg, 23)
        np.testing.assert_array_equal(a, b); self.assertEqual(ma, mb)
        self.assertTrue(ma['available']); self.assertEqual(len(a), 17)

    def test_coverage_cannot_be_supplied_only_by_masked_columns(self):
        m = fixture(); m['mask'] = '*' * 20 + '-' * 20
        m['rows'] = [[f'g{i}', '-' * 20 + row[1][20:], row[2]] for i, row in enumerate(m['rows'])]
        tokens, info = select(m, config()['monomer'], 4)
        self.assertIsNone(tokens); self.assertEqual(info['reason'], 'shallow_monomer_alignment')
        self.assertEqual(info['filtered_depth'], 0)

    def test_neff80_not_the_90_percent_dedup_depth(self):
        rows = ['A' * 20, 'A' * 20, 'A' * 17 + 'RRR', 'N' * 20]
        tokens = np.stack([encode(r) for r in rows])
        n = neff80(tokens, config()['monomer'])
        self.assertEqual(n['neff80'], 2.)
        self.assertEqual(n['fraction_comparable'], 1.)

    def test_profile_uses_homologs_and_keeps_gap_and_unknown_categories(self):
        m = fixture(); t, meta = select(m, config()['monomer'], 6)
        p = profile(t, meta)
        self.assertEqual(p.shape, (192,)); self.assertEqual(len(PROFILE_NAMES), 192)
        np.testing.assert_allclose(p[12:38].sum(), 1.)
        expected = np.mean(t[1:] == t[0])
        self.assertAlmostEqual(p[PROFILE_NAMES.index('global/query_agreement')], expected)
        self.assertNotEqual(expected, 1.)

    def test_block_pooling_preserves_original_coordinates(self):
        h = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        good = torch.tensor([True, True, False, False, True, True, False, False])
        out = pool_query(h, good).reshape(6, 2)
        np.testing.assert_allclose(out[2], h[:2].mean(0))
        np.testing.assert_array_equal(out[3], [0, 0])
        np.testing.assert_allclose(out[4], h[4:6].mean(0))
        np.testing.assert_array_equal(out[5], [0, 0])
        corrupted = h.clone(); corrupted[~good] = float('nan')
        np.testing.assert_array_equal(pool_query(h, good), pool_query(corrupted, good))
        corrupted[0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            pool_query(corrupted, good)

    def test_invalid_input_is_an_error_and_biological_absence_is_explicit(self):
        m = fixture(); m['rows'][0][1] = '?' * 40
        with self.assertRaises(ValueError):
            select(m, config()['monomer'], 1)
        m = fixture(); m['mask'] = '-' * 40
        t, info = select(m, config()['monomer'], 1)
        self.assertIsNone(t); self.assertEqual(info['reason'], 'low_quality_coverage')

    def test_reliability_is_symmetric_and_requires_both_monomers(self):
        a = {'available': True, 'neff80': 16, 'quality_fraction': .7}
        b = {'available': True, 'neff80': 64, 'quality_fraction': .9}
        self.assertEqual(pair_gate(a, b), .35); self.assertEqual(pair_gate(a, b), pair_gate(b, a))
        self.assertEqual(pair_gate(a, {'available': False}), 0.)


if __name__ == '__main__':
    unittest.main()
