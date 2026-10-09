"""Label-free finite-depth manipulation diagnostics, never selection criteria."""
import hashlib
import numpy as np
from inputs import column_counts


def sampled_pairs(left, right, count, rng, same=False):
    possible = len(left) * (len(left) - 1) // 2 if same else len(left) * len(right)
    count = min(count, possible)
    if not count:
        return []
    chosen = set()
    while len(chosen) < count:
        i, j = int(rng.choice(left)), int(rng.choice(right))
        if same:
            if i == j:
                continue
            i, j = sorted((i, j))
        chosen.add((i, j))
    return sorted(chosen)


def covariance_energy(h, pairs):
    if not pairs:
        return None
    values = []
    for i, j in pairs:
        joint = np.bincount(h[:, i].astype(np.int64) * 26 + h[:, j], minlength=26**2).reshape(26, 26) / len(h)
        cov = joint - joint.sum(1)[:, None] * joint.sum(0)[None, :]
        values.append(float(np.square(cov).sum()))
    return float(np.mean(values))


def describe(tokens, c, p, previous, pair_limit):
    counts = column_counts(tokens); h = tokens[1:]; depth = len(h); bp = p['breakpoint']
    good = tokens[0] != 26; variable = good & (counts.max(0) < depth)
    freq = counts.astype(float) / depth
    null = tokens.copy(); permutation = np.asarray(previous['null']['permutation'])
    if not np.array_equal(np.sort(permutation), np.arange(len(tokens))) or permutation[0] != 0:
        raise ValueError('Invalid original row-shuffle permutation')
    null[1:, bp:] = tokens[permutation[1:], bp:]
    identity = {}; changes = {}
    for name, mask in [('all', good), ('A', good & (np.arange(len(good)) < bp)), ('B', good & (np.arange(len(good)) >= bp))]:
        changes[name] = {'actual_changed_fraction': float((h[:, mask] != c[1:, mask]).mean()),
                         'uniform_permutation_expected_changed_fraction': float((1 - (freq[:, mask]**2).sum(0)).mean()),
                         'variable_columns': int(variable[mask].sum()), 'valid_columns': int(mask.sum())}
        identity[name] = {}
        for arm, t in [('true', tokens), ('shuffled', null), ('C', c)]:
            match = (t[1:, mask] == t[0, mask]).mean(1); gaps = (t[1:, mask] == 25).mean(1)
            identity[name][arm] = {'query_match_mean_including_gaps': float(match.mean()), 'query_match_std': float(match.std()),
                                   'row_gap_fraction_mean': float(gaps.mean()), 'row_gap_fraction_std': float(gaps.std())}
    rng = np.random.Generator(np.random.PCG64(int(hashlib.sha256(('vx-v3:diagnostics:' + p['uid']).encode()).hexdigest()[:16], 16)))
    aa = np.flatnonzero(variable & (np.arange(len(good)) < bp)); bb = np.flatnonzero(variable & (np.arange(len(good)) >= bp))
    cov = {}
    for kind, left, right, same in [('within_A', aa, aa, True), ('within_B', bb, bb, True), ('cross_chain', aa, bb, False)]:
        pairs = sampled_pairs(left, right, pair_limit, rng, same)
        cov[kind] = {'sampled_column_pairs': len(pairs), 'column_pairs_sha256': hashlib.sha256(np.asarray(pairs, dtype='<i4').tobytes()).hexdigest(),
                     **{arm: covariance_energy(t[1:], pairs) for arm, t in [('true', tokens), ('shuffled', null), ('C', c)]}}
    return {'depth_including_query': len(tokens), 'column_histogram_sha256': hashlib.sha256(counts.tobytes()).hexdigest(),
            'exact_column_histograms': True, 'query_and_PAD_unchanged': True, 'changes': changes, 'row_structure': identity,
            'categorical_covariance_energy': cov, 'finite_sample_covariance_expected': True, 'labels_used': False}
