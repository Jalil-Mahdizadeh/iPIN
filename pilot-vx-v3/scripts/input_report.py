"""Descriptive manipulation checks in the frozen populations, never selection."""
import numpy as np


def distribution(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(a):
        return {'pairs': 0}
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite input diagnostic')
    return {'pairs': len(a), 'mean': float(a.mean()), 'min': float(a.min()),
            'q25': float(np.quantile(a, .25)), 'median': float(np.median(a)),
            'q75': float(np.quantile(a, .75)), 'max': float(a.max())}


def aggregate(audits, sample, pairs, masks):
    eligible = {p['uid'] for p in pairs if p['available']}
    if set(audits) != eligible:
        raise ValueError('Incomplete or unexpected manipulation diagnostics')
    for p in pairs:
        if not p['available']:
            continue
        a = audits[p['uid']]
        if not a['exact_column_histograms'] or not a['query_and_PAD_unchanged'] or a['labels_used']:
            raise ValueError('Failed column-shuffle invariant')
        if a['depth_including_query'] != p['paired_depth']:
            raise ValueError('Diagnostic depth differs from prepared input')
    groups = {}
    for name, mask in masks.items():
        selected = [audits[r['uid']] for r, keep in zip(sample, mask) if keep and r['uid'] in audits]
        g = {'eligible_pairs': len(selected), 'depth_including_query': distribution([a['depth_including_query'] for a in selected]),
             'token_changes': {}, 'row_structure': {}, 'categorical_covariance_energy': {}}
        for region in ('all', 'A', 'B'):
            g['token_changes'][region] = {key: distribution([a['changes'][region][key] for a in selected])
                for key in ('actual_changed_fraction', 'uniform_permutation_expected_changed_fraction', 'variable_columns', 'valid_columns')}
            g['row_structure'][region] = {arm: {key: distribution([a['row_structure'][region][arm][key] for a in selected])
                for key in ('query_match_mean_including_gaps', 'query_match_std', 'row_gap_fraction_mean', 'row_gap_fraction_std')}
                for arm in ('true', 'shuffled', 'C')}
        for kind in ('within_A', 'within_B', 'cross_chain'):
            cc = [a['categorical_covariance_energy'][kind] for a in selected]
            g['categorical_covariance_energy'][kind] = {
                **{arm: distribution([c[arm] for c in cc]) for arm in ('true', 'shuffled', 'C')},
                'C_minus_true': distribution([c['C'] - c['true'] for c in cc if c['C'] is not None]),
                'C_minus_shuffled': distribution([c['C'] - c['shuffled'] for c in cc if c['C'] is not None]),
                'sampled_column_pairs': distribution([c['sampled_column_pairs'] for c in cc])}
        groups[name] = g
    return {'groups': groups, 'pair_weighting': 'equal weight per eligible pair', 'labels_used_for_perturbation': False,
            'finite_depth_covariance_is_not_expected_to_be_zero': True, 'used_for_model_selection': False, 'R_evaluated': False}
