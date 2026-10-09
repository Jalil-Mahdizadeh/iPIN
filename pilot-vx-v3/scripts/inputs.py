"""Column-shuffle intervention and independently checkable frozen token inputs."""
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from study import ROOT, BASE, config, read, sha, token_hash
from msa import load_monomer, pair


def validate_tokens(tokens, breakpoint):
    t = np.asarray(tokens)
    if t.ndim != 2 or len(t) < 2 or not 0 < breakpoint < t.shape[1]:
        raise ValueError('Invalid paired MSA dimensions')
    if not np.issubdtype(t.dtype, np.integer) or not np.isin(t, np.arange(27)).all():
        raise ValueError('Invalid paired alphabet')
    pad = t[0] == 26
    if not np.array_equal(t == 26, np.broadcast_to(pad, t.shape)):
        raise ValueError('Original PAD mask is not constant across rows')
    if pad[:breakpoint].all() or pad[breakpoint:].all():
        raise ValueError('No valid positions in a chain')
    return t


def seed_for(p, replicate=0):
    cfg = config()
    text = f"{cfg['column_shuffle']['seed_namespace']}:{cfg['seed']}:{p['uid']}:{p['a']}:{p['b']}:{replicate}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def column_shuffle(tokens, breakpoint, seed):
    t = validate_tokens(tokens, breakpoint); out = t.copy()
    rng = np.random.Generator(np.random.PCG64(seed)); perm = np.empty((len(t) - 1, t.shape[1]), dtype=np.uint16)
    for j in range(t.shape[1]):
        perm[:, j] = rng.permutation(len(t) - 1)
        out[1:, j] = t[1 + perm[:, j], j]
    validate_columns(t, out, breakpoint)
    return out, hashlib.sha256(perm.astype('<u2').tobytes()).hexdigest()


def column_counts(tokens):
    return np.stack([(tokens[1:] == k).sum(axis=0) for k in range(27)]).astype('<u2')


def validate_columns(original, changed, breakpoint):
    t = validate_tokens(original, breakpoint); c = validate_tokens(changed, breakpoint)
    if t.shape != c.shape or not np.array_equal(t[0], c[0]) or not np.array_equal(t == 26, c == 26):
        raise ValueError('C changed depth, query coordinates, or PAD mask')
    # Sorting is independent of the histogram code used by diagnostics.
    if not np.array_equal(np.sort(t[1:], axis=0), np.sort(c[1:], axis=0)):
        raise ValueError('C changed a column residue/gap histogram')


def monomer_reader():
    catalog = read(ROOT / 'data/monomer-catalog.json'); digests = read(ROOT / 'data/monomer-digests.json')
    proteins = read(ROOT / 'data/proteins.json')
    @lru_cache(maxsize=256)
    def get(pid):
        rec = catalog[str(pid)]; path = BASE / rec['path']
        if sha(path) != digests[str(pid)] or rec['sha256'] != digests[str(pid)]:
            raise ValueError('Original monomer changed: ' + str(pid))
        m = load_monomer(path)
        if m['query'] != proteins[str(pid)]['sequence'] or m['query_sha256'] != proteins[str(pid)]['sha256']:
            raise ValueError('Monomer/query identity differs')
        return m
    return get


def reconstruct(p, previous, mono):
    if not p['available']:
        raise ValueError('Never encode a previously unavailable pair')
    ma, mb = mono(p['a']), mono(p['b'])
    tokens, meta = pair(ma, mb, read(ROOT / 'data/original_config.json'))
    if tokens is None or meta != previous['msa'] or meta['gate'] != p['gate']:
        raise ValueError('Original pairing/eligibility/gate failed to reproduce: ' + p['uid'])
    if token_hash(tokens) != p['paired_tokens_sha256'] or token_hash(tokens[:1]) != p['q_tokens_sha256']:
        raise ValueError('Original paired/query tokens changed: ' + p['uid'])
    if len(ma['query']) != p['breakpoint'] or tokens.shape[1] != p['length']:
        raise ValueError('Original coordinates changed')
    return tokens, meta


def prepared(p, replay=False):
    if not p['available']:
        raise ValueError('Never encode a previously unavailable pair')
    path = ROOT / p['c_input_file']
    if sha(path) != p['c_input_sha256']:
        raise ValueError('Prepared C input checksum mismatch: ' + p['uid'])
    with np.load(path, allow_pickle=False) as f:
        t, c = f['true'].copy(), f['C'].copy()
    validate_columns(t, c, p['breakpoint'])
    if t.shape != (p['paired_depth'], p['length']) or t.dtype != np.uint8 or c.dtype != np.uint8:
        raise ValueError('Prepared depth/length/dtype changed')
    for actual, expected in [(token_hash(t), p['paired_tokens_sha256']), (token_hash(t[:1]), p['q_tokens_sha256']),
                             (token_hash(c), p['c_tokens_sha256']), (seed_for(p), p['c_seed'])]:
        if actual != expected:
            raise ValueError('Prepared input identity or seed differs')
    if replay:
        # Independent replay directly from original columns; do not call column_shuffle.
        rng = np.random.Generator(np.random.PCG64(p['c_seed']))
        permutations = []
        for j in range(t.shape[1]):
            permutation = rng.permutation(t.shape[0] - 1)
            if not np.array_equal(c[1:, j], t[1:, j][permutation]):
                raise ValueError('Column permutation replay failed')
            permutations.append(permutation)
        digest = hashlib.sha256(np.asarray(permutations, dtype='<u2').T.tobytes()).hexdigest()
        if digest != p['c_permutations_sha256']:
            raise ValueError('Independent permutation digest differs')
    return t, c


def verify_input_population(pairs):
    selected = [p for p in pairs if p['available']]
    def one(p):
        prepared(p, replay=True)
        return 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        count = sum(pool.map(one, selected))
    return {'eligible_inputs_verified': count, 'exact_profiles_query_PAD_depth': True,
            'independent_permutation_replay': True, 'R_evaluated': False}
