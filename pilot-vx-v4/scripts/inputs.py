"""Independent intact-row selection; exact old depth/query/mask and source replay."""
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
from study import ROOT, BASE, config, read, sha, token_hash
from msa import load_monomer, encode, tax_group

GAP, PAD = 25, 26


def digest(p, chain, purpose, replicate=0):
    cfg = config()
    value = f"{cfg['independent_sampling']['seed_namespace']}:{cfg['seed']}:{p['uid']}:{p['a']}:{p['b']}:{replicate}:{chain}:{purpose}"
    return hashlib.sha256(value.encode()).hexdigest()


def validate_tokens(t, breakpoint):
    if t.ndim != 2 or len(t) < 2 or not 0 < breakpoint < t.shape[1] or t.dtype != np.uint8:
        raise ValueError('Invalid paired token dimensions/dtype')
    if np.any(t > PAD) or not np.array_equal(t == PAD, np.broadcast_to(t[0] == PAD, t.shape)):
        raise ValueError('Invalid alphabet or changing PAD mask')
    if (t[0, :breakpoint] == PAD).all() or (t[0, breakpoint:] == PAD).all():
        raise ValueError('Empty valid chain')
    return t


def monomer_reader():
    catalog = read(ROOT / 'data/monomer-catalog.json'); hashes = read(ROOT / 'data/monomer-digests.json')
    proteins = read(ROOT / 'data/proteins.json'); observed = {}
    @lru_cache(maxsize=256)
    def get(pid):
        item = catalog[str(pid)]; path = BASE / item['path']
        expected = hashes.get(str(pid), item['sha256'])
        if item['sha256'] != expected or sha(path) != expected:
            raise ValueError('Changed cached monomer: ' + str(pid))
        m = load_monomer(path)
        if m['query'] != proteins[str(pid)]['sequence'] or m['query_sha256'] != proteins[str(pid)]['sha256']:
            raise ValueError('Changed monomer/query identity')
        observed[str(pid)] = expected
        return m
    get.observed_digests = observed
    return get


def candidate_pool(m, threshold):
    good = np.array([c == '*' for c in m['mask']])
    if len(good) != len(m['query']) or set(m['mask']) - {'*', '-'} or not good.any():
        raise ValueError('Invalid monomer mask')
    if hashlib.sha256(m['query'].encode()).hexdigest() != m['query_sha256']:
        raise ValueError('Query checksum differs')
    rows = {}; keys = set()
    for key, sequence, taxonomy in m['rows']:
        if key in keys:
            raise ValueError('Duplicate accession in original cache')
        keys.add(key)
        if len(sequence) != len(good):
            raise ValueError('Invalid homolog coordinates')
        t = encode(sequence)
        if (t[good] != GAP).mean() >= threshold:
            rows[key] = (t, taxonomy)
    return rows, good


def select_chain(pool, count, p, chain, replicate=0):
    if count < 1 or count > len(pool):
        raise ValueError('Insufficient distinct independently valid homologs')
    salt = digest(p, chain, 'selection', replicate)
    def priority(value):
        return hashlib.sha256((salt + ':' + value).encode()).hexdigest()
    groups = defaultdict(list)
    for key, (_, taxonomy) in pool.items():
        groups[tax_group(taxonomy)].append(key)
    for group in groups.values():
        group.sort(key=lambda key: (priority('key:' + key), key))
    order = sorted(groups, key=lambda group: (priority('tax:' + group), group))
    candidates = [groups[g][j] for j in range(max(map(len, groups.values()))) for g in order if j < len(groups[g])]
    selected = candidates[:count]
    rng = np.random.Generator(np.random.PCG64(int(digest(p, chain, 'row-order', replicate)[:16], 16)))
    return [selected[i] for i in rng.permutation(count)]


def independent_pair(ma, mb, original, p, replicate=0):
    threshold = read(ROOT / 'data/original_config.json')['minimum_homolog_query_coverage']
    pa, ga = candidate_pool(ma, threshold); pb, gb = candidate_pool(mb, threshold)
    n = len(original) - 1
    ka = select_chain(pa, n, p, 'A', replicate); kb = select_chain(pb, n, p, 'B', replicate)
    a = np.vstack([encode(ma['query'])] + [pa[k][0] for k in ka])
    b = np.vstack([encode(mb['query'])] + [pb[k][0] for k in kb])
    a[:, ~ga] = PAD; b[:, ~gb] = PAD; result = np.concatenate([a, b], axis=1)
    validate_pair(original, result, p['breakpoint'])
    return result, {'A': ka, 'B': kb}, (pa, pb)


def validate_pair(original, changed, breakpoint):
    t = validate_tokens(original, breakpoint); i = validate_tokens(changed, breakpoint)
    if t.shape != i.shape or not np.array_equal(t[0], i[0]) or not np.array_equal(t == PAD, i == PAD):
        raise ValueError('I changed original depth/query/PAD/coordinates')


def prepared(p, replay=False, mono=None):
    if not p['available']:
        raise ValueError('Never encode a previously unavailable pair')
    path = ROOT / p['i_input_file']
    if sha(path) != p['i_input_sha256']:
        raise ValueError('Prepared I checksum mismatch')
    with np.load(path, allow_pickle=False) as f:
        t, i = f['true'].copy(), f['I'].copy()
    validate_pair(t, i, p['breakpoint'])
    if t.shape != (p['paired_depth'], p['length']) or token_hash(t) != p['paired_tokens_sha256'] or token_hash(t[:1]) != p['q_tokens_sha256'] or token_hash(i) != p['i_tokens_sha256']:
        raise ValueError('Prepared input identity differs')
    if replay:
        mono = mono or monomer_reader()
        ma, mb = mono(p['a']), mono(p['b'])
        threshold = read(ROOT / 'data/original_config.json')['minimum_homolog_query_coverage']
        for chain, m, start, end in [('A', ma, 0, p['breakpoint']), ('B', mb, p['breakpoint'], p['length'])]:
            pool, good = candidate_pool(m, threshold); keys = p['i_selected_keys'][chain]
            if len(keys) != len(t)-1 or len(set(keys)) != len(keys) or any(k not in pool for k in keys):
                raise ValueError('Invalid intact-source row selection')
            # Direct source-row membership is separate from the selection constructor.
            for row, key in enumerate(keys, 1):
                expected = pool[key][0].copy(); expected[~good] = PAD
                if not np.array_equal(expected, i[row, start:end]):
                    raise ValueError('I row is not its intact original homolog')
            # Independent priority/round-robin/order reconstruction.
            salt = digest(p, chain, 'selection')
            groups = {}
            for key, (_, taxonomy) in pool.items(): groups.setdefault(tax_group(taxonomy), []).append(key)
            priority = lambda text: hashlib.sha256((salt + ':' + text).encode()).hexdigest()
            for values in groups.values(): values.sort(key=lambda k:(priority('key:'+k),k))
            order = sorted(groups, key=lambda g:(priority('tax:'+g),g))
            chosen=[]
            for j in range(max(map(len, groups.values()))):
                for g in order:
                    if j < len(groups[g]) and len(chosen) < len(t)-1: chosen.append(groups[g][j])
            rng=np.random.Generator(np.random.PCG64(int(digest(p,chain,'row-order')[:16],16)))
            if [chosen[j] for j in rng.permutation(len(chosen))] != keys:
                raise ValueError('Independent selection replay differs')
    return t, i


def verify_input_population(pairs):
    mono = monomer_reader()
    def one(p): prepared(p, replay=True, mono=mono); return 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        count = sum(pool.map(one, [p for p in pairs if p['available']]))
    return {'eligible_inputs_verified': count, 'exact_query_PAD_depth': True,
            'intact_source_membership_verified': True, 'independent_selection_replay': True, 'R_evaluated': False}
