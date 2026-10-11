"""Reproduce the exact original query, homolog rows, masks and taxonomic null."""
from functools import lru_cache
import numpy as np
from study import ROOT, BASE, read, sha, text_hash, config
from msa import load_monomer, pair, shuffled


class Inputs:
    def __init__(self):
        self.sample = read(ROOT / 'data/sample.json')
        self.meta = read(ROOT / 'data/source_metadata.json')
        self.diag = read(ROOT / 'data/diagnostics.json')
        self.catalog = read(ROOT / 'data/monomer-catalog.json')
        self.proteins = read(ROOT / 'data/proteins.json')
        with np.load(ROOT / 'data/source.npz', allow_pickle=False) as f:
            self.arrays = {k: f[k].copy() for k in f.files}
        if self.arrays['uids'].tolist() != [r['uid'] for r in self.sample]:
            raise ValueError('Input order mismatch')

    @lru_cache(maxsize=32)
    def monomer(self, pid):
        item = self.catalog[str(pid)]; p = BASE / item['path']
        if sha(p) != item['sha256']:
            raise ValueError('Monomer checksum mismatch')
        m = load_monomer(p)
        if m['query'] != self.proteins[str(pid)]['sequence'] or m['query_sha256'] != self.proteins[str(pid)]['sha256']:
            raise ValueError('Monomer query identity mismatch')
        return m

    def materialize(self, i):
        r, m = self.sample[i], self.meta[i]
        if r['split'] not in ['train', 'val'] or not r['uid'].startswith(r['split']+'-'):
            raise ValueError('Unauthorized cohort')
        if not m['available']:
            raise ValueError('No encoding for source fallback')
        a, b = sorted([r['a'], r['b']]); ma, mb = self.monomer(a), self.monomer(b)
        tokens, meta = pair(ma, mb)
        if tokens is None or meta != m['msa']:
            raise ValueError('Original paired-MSA changed: ' + r['uid'])
        seed = int(text_hash(f"{config()['source_seed']}:{r['uid']}")[:16], 16)
        null, nm = shuffled(tokens, len(ma['query']), meta['taxonomy'], seed)
        if nm != m['null']:
            raise ValueError('Original pairing null changed')
        import hashlib
        for array, key in [(tokens, 'tokens_sha256'), (null, 'null_tokens_sha256')]:
            if hashlib.sha256(array.tobytes()).hexdigest() != self.diag[i][key]:
                raise ValueError('Token bytes differ from Vx-v2')
        return tokens, null, len(ma['query'])
