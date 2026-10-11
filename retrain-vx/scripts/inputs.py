"""Exact original pairing and profile vectors, with verified optional cache reuse."""
from functools import lru_cache
import numpy as np
from shared import ROOT, BASE, read, config, sha, mono_path, token_hash, quality_from_profiles
from msa import load_monomer, pair


class Inputs:
    def __init__(self, reuse=True):
        self.cfg = config(); self.catalog = read(ROOT / 'data/monomer-catalog.json')
        self.proteins = read(ROOT / 'data/proteins.json')
        with np.load(ROOT / 'data/profiles.npz', allow_pickle=False) as f:
            ids, values = f['ids'], f['profiles']
        self.profiles = dict(zip(ids.tolist(), values))
        self.old_ids = {r['uid'] for r in read(BASE / 'data/sample.json')} if reuse else set()
        self.old_fp = sha(BASE / 'provenance/freeze.json')

    @lru_cache(maxsize=512)
    def monomer(self, pid):
        entry = self.catalog[str(pid)]; path = mono_path(entry)
        if sha(path) != entry['sha256']:
            raise ValueError('Corrupt monomer: ' + str(pid))
        m = load_monomer(path)
        if m['query'] != self.proteins[str(pid)]['sequence'] or m['query_sha256'] != entry['query_sha256']:
            raise ValueError('Monomer query identity mismatch')
        return m

    def materialize(self, row):
        a, b = sorted([row['a'], row['b']]); tokens = None; meta = None; gate = 0.
        quality = np.zeros(68); reason = 'missing_exact_query'
        if str(a) in self.catalog and str(b) in self.catalog:
            ea, eb = self.catalog[str(a)], self.catalog[str(b)]
            la, lb = len(self.proteins[str(a)]['sequence']), len(self.proteins[str(b)]['sequence'])
            # Original pair() returns before reading homologs in these two cases.
            # Use its exact metadata and precomputed profiles to avoid decompressing
            # very long monomer alignments repeatedly for known fallback pairs.
            meta = {'length_a': la, 'length_b': lb, 'quality_a': ea['quality_fraction'],
                'quality_b': eb['quality_fraction'], 'common_keys': 0, 'reused_homolog_rows': 0,
                'taxonomy_conflicts': 0, 'low_good_column_coverage_rows': 0, 'retained_homologs': 0}
            if la + lb > self.cfg['maximum_combined_residues']:
                meta['reason'] = 'long_query'
            elif min(meta['quality_a'], meta['quality_b']) < self.cfg['minimum_quality_fraction_per_chain']:
                meta['reason'] = 'low_quality_coverage'
            else:
                tokens, meta = pair(self.monomer(a), self.monomer(b), self.cfg)
            reason = meta['reason']; gate = meta.get('gate', 0.)
            quality = quality_from_profiles(self.profiles[str(a)], self.profiles[str(b)], meta)
        available = tokens is not None
        if available != (gate > 0) or quality.shape != (68,) or not np.isfinite(quality).all():
            raise ValueError('Invalid original input/gate/profile')
        info = {'uid': row['uid'], 'index': row['index'], 'a': a, 'b': b, 'reason': reason,
            'available': available, 'gate': gate, 'msa': meta,
            'token_sha256': token_hash(tokens) if available else None,
            'depth': len(tokens) if available else 0, 'length': row['length']}
        return tokens, quality, info

    def old_feature(self, row, info, quality):
        if row['uid'] not in self.old_ids:
            return None
        p = BASE / 'features' / (row['uid'] + '.json'); side = read(p.with_suffix('.sha.json'))
        if sha(p) != side['sha256'] or side['fingerprint'] != self.old_fp:
            raise ValueError('Corrupt original feature')
        value = read(p)
        if value['fingerprint'] != self.old_fp:
            raise ValueError('Original feature fingerprint mismatch')
        for key in ['uid', 'a', 'b', 'reason', 'available', 'gate']:
            if value[key] != info[key]:
                raise ValueError('Original feature input changed: ' + key)
        if value.get('msa') != info['msa']:
            raise ValueError('Original matched rows/metadata changed')
        np.testing.assert_array_equal(value['quality'], quality)
        true = np.asarray(value['true'], dtype=float)
        if true.shape != (128,) or not np.isfinite(true).all() or (not info['available'] and true.any()):
            raise ValueError('Invalid original true feature')
        return true
