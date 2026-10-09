"""Frozen C sufficiency criteria; noninferiority, equivalence and mechanism differ."""
def decide(groups, cfg):
    m = groups['assessment']; pts = m['contrasts']; ci = m['ap_intervals']; f = cfg['fusion']
    def positive(name):
        return ci[name].get('low', float('-inf')) > 0
    def material(name):
        return pts[name] >= f['continuation_delta_ap'] and positive(name)
    eq = {name: c.get('low', float('-inf')) >= -f['equivalence_margin'] and
                c.get('high', float('inf')) <= f['equivalence_margin'] for name, c in ci.items()}
    auroc = m['metrics']['C']['auroc'] >= m['metrics']['baseline']['auroc'] - f['maximum_auroc_decline']
    useful = material('C_minus_baseline') and auroc
    ni = {name: ci[name].get('low', float('-inf')) > -f['noninferiority_margin'] for name in ('C_minus_shuffled', 'C_minus_true')}
    recovered = useful and all(ni.values()); beyond_q = material('C_minus_Q')
    ec = groups.get('assessment/eligible', {}).get('ap_intervals', {})
    eligible_ni = bool(ec) and all(ec.get(name, {}).get('low', float('-inf')) > -f['noninferiority_margin'] for name in ni)
    loss = material('shuffled_minus_C')
    if recovered and beyond_q:
        status = 'column_profiles_recover_gain_within_tolerance_beyond_Q'
    elif recovered:
        status = 'column_profiles_recover_within_tolerance_Q_contrast_uncertain'
    elif useful and loss:
        status = 'partial_gain_but_column_shuffle_loses_to_row_shuffle'
    elif loss:
        status = 'column_shuffle_loses_practical_gain'
    elif useful:
        status = 'C_has_useful_gain_recovery_inconclusive'
    else:
        status = 'no_go_or_inconclusive'
    return {'status': status, 'primary_point': pts['C_minus_shuffled'], 'primary_interval': ci['C_minus_shuffled'],
        'useful_C_gain': useful, 'auroc_noninferiority': auroc, 'noninferiority': ni,
        'profile_recovery_within_margin': recovered, 'material_C_gain_over_Q': beyond_q,
        'profiles_recover_gain_beyond_Q': recovered and beyond_q, 'eligible_only_noninferiority': eligible_ni,
        'equivalence': eq, 'all_C_true_shuffled_equivalent': all(eq[n] for n in ('C_minus_true', 'C_minus_shuffled', 'true_minus_shuffled')),
        'material_loss_under_column_shuffle': loss, 'true_materially_exceeds_C': material('true_minus_C'),
        'C_exceeds_quality_interval': positive('C_minus_quality'), 'pairing_material_advantage': material('true_minus_shuffled'),
        'family_breadth_established': False, 'assessment_previously_inspected': True,
        'production_authorized': False, 'test_accessed': False, 'R_evaluated': False,
        'interpretation_scope': 'Fixed assessment, original MSA masks/gate, one C permutation. Synthetic-row, gap-pattern, weighting and depth explanations are not uniquely separated.'}
