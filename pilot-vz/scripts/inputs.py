"""The only input intervention: original masked paired tokens, first row only."""
from functools import lru_cache
import numpy as np
from study import ROOT, BASE, read, sha, token_hash
from msa import load_monomer, pair


def query_only(tokens, breakpoint):
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or len(tokens) < 1 or not 0 < breakpoint < tokens.shape[1]:
        raise ValueError('Invalid paired coordinates')
    if not np.isin(tokens, np.arange(27)).all():
        raise ValueError('Invalid MSA alphabet')
    q = tokens[:1].copy()
    if not (q[0, :breakpoint] != 26).any() or not (q[0, breakpoint:] != 26).any():
        raise ValueError('No valid positions in a chain')
    return q


def monomer_reader():
    catalog = read(ROOT / 'data/monomer-catalog.json')
    digests = read(ROOT / 'data/monomer-digests.json')
    proteins = read(ROOT / 'data/proteins.json')
    @lru_cache(maxsize=64)
    def get(pid):
        rec = catalog[str(pid)]; path = BASE / rec['path']
        if sha(path) != digests[str(pid)] or rec['sha256'] != digests[str(pid)]:
            raise ValueError('Original monomer changed: ' + str(pid))
        m = load_monomer(path)
        if m['query'] != proteins[str(pid)]['sequence'] or m['query_sha256'] != proteins[str(pid)]['sha256']:
            raise ValueError('Monomer/query identity differs')
        return m
    return get


def reconstruct(expected, previous, mono):
    if not expected['available']:
        raise ValueError('Never encode a previously unavailable pair')
    a, b = expected['a'], expected['b']; ma, mb = mono(a), mono(b)
    tokens, meta = pair(ma, mb, read(ROOT / 'data/original_config.json'))
    if tokens is None or meta != previous['msa'] or meta['gate'] != expected['gate']:
        raise ValueError('Original pairing/eligibility/gate failed to reproduce: ' + expected['uid'])
    if token_hash(tokens) != expected['paired_tokens_sha256']:
        raise ValueError('Original paired token hash differs: ' + expected['uid'])
    q = query_only(tokens, len(ma['query']))
    if token_hash(q) != expected['q_tokens_sha256'] or len(ma['query']) != expected['breakpoint']:
        raise ValueError('Query row/mask/breakpoint differs: ' + expected['uid'])
    return tokens, q, meta
