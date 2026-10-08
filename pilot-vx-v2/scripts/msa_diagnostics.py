"""Descriptive diversity and effective-support diagnostics; no feature/gate changes."""
from collections import Counter
import numpy as np
from msa import LOOKUP, PAD, encode, tax_group
from study import config


def summary(a):
    a = np.asarray(a, dtype=float).ravel()
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Empty/nonfinite diagnostic')
    return {'mean': float(a.mean()), **dict(zip(['min', 'q10', 'q25', 'median', 'q75', 'q90', 'max'],
            map(float, np.quantile(a, [0, .1, .25, .5, .75, .9, 1]))))}


def neff(tokens, good):
    cfg = config()['diagnostics']
    h = tokens[1:, good]
    n, length = h.shape
    counts = np.ones(n, dtype=int)
    comparisons = []
    comparable = 0
    for i in range(1, n):
        valid = (h[:i] != LOOKUP['-']) & (h[i] != LOOKUP['-'])
        coverage = valid.sum(1)
        identity_count = ((h[:i] == h[i]) & valid).sum(1)
        enough = coverage >= cfg['comparison_coverage'] * length
        related = enough & (identity_count >= cfg['identity'] * coverage)
        counts[i] += int(related.sum())
        counts[:i] += related.astype(int)
        comparable += int(enough.sum())
        comparisons.extend((coverage / length).tolist())
    return {'neff80': float((1 / counts).sum()), 'neighbors': summary(counts),
            'comparison_coverage': summary(comparisons),
            'fraction_comparable': comparable / (n * (n - 1) / 2),
            'neff_per_valid_residue': float((1 / counts).sum() / length)}


def joint_support(tokens, boundary):
    good = tokens[0] != PAD
    a = (tokens[1:, :boundary][:, good[:boundary]] != LOOKUP['-']).astype(np.float32)
    b = (tokens[1:, boundary:][:, good[boundary:]] != LOOKUP['-']).astype(np.float32)
    support = a.T @ b
    return {'a_occupancy': summary(a.mean(0)), 'b_occupancy': summary(b.mean(0)),
            'joint_nongap_rows': summary(support), 'fraction_cells_at_least_8_rows': float((support >= 8).mean()),
            'fraction_cells_at_least_half_rows': float((support >= len(a) / 2).mean())}


def candidate_count(a, b):
    da = {r[0]: r for r in a['rows']}; db = {r[0]: r for r in b['rows']}
    qa = np.array([c == '*' for c in a['mask']]); qb = np.array([c == '*' for c in b['mask']])
    count = 0
    for key in da.keys() & db.keys():
        _, sa, ta = da[key]; _, sb, tb = db[key]
        if (ta and tb and ta != tb) or sa.replace('-', '') == sb.replace('-', ''):
            continue
        ea, eb = encode(sa), encode(sb)
        if min(float((ea[qa] != LOOKUP['-']).mean()), float((eb[qb] != LOOKUP['-']).mean())) >= .5:
            count += 1
    return count


def diagnose(a, b, tokens, null, meta, null_meta):
    boundary = len(a['query']); good = tokens[0] != PAD
    qa, qb = good[:boundary], good[boundary:]
    changed = tokens[1:, boundary:][:, qb] != null[1:, boundary:][:, qb]
    groups = Counter(tax_group(t) for t in meta['taxonomy'])
    p = np.array(list(groups.values()), dtype=float) / (len(tokens) - 1)
    count = candidate_count(a, b)
    if count < meta['retained_homologs']:
        raise ValueError('Candidate count below retained depth')
    return {'cached_common_keys': meta['common_keys'], 'candidates_after_coverage_and_conflict_filters': count,
            'retained_depth': meta['retained_homologs'], 'depth_cap_reached': meta['retained_homologs'] == 127,
            'legacy_neff90': meta['neff'], 'monomer_depth': {k: [a['stats'].get(k, 0), b['stats'].get(k, 0)]
                for k in ['raw_rows', 'usable_before_cap', 'retained_rows']},
            'valid_residues_a': int(qa.sum()), 'valid_residues_b': int(qb.sum()),
            'valid_interchain_cells': int(qa.sum() * qb.sum()),
            'combined_length': tokens.shape[1], 'length_asymmetry': max(len(qa), len(qb)) / min(len(qa), len(qb)),
            'minimum_quality': float(min(qa.mean(), qb.mean())), 'gate': meta['gate'],
            'a_diversity': neff(tokens[:, :boundary], qa), 'b_diversity': neff(tokens[:, boundary:], qb),
            'paired_diversity': neff(tokens, good),
            'true_support': joint_support(tokens, boundary), 'shuffled_support': joint_support(null, boundary),
            'actual_sequence_changed_fraction': float(changed.any(1).mean()),
            'actual_token_changed_fraction': float(changed.mean()),
            'row_index_changed_fraction': null_meta['changed_fraction'], 'null_levels': null_meta['level_counts'],
            'taxonomic_family_count': len(groups), 'taxonomic_family_effective_number': float(np.exp(-(p * np.log(p)).sum())),
            'unknown_taxonomic_family_rows': groups.get('__unknown__', 0)}
