"""Frozen I sufficiency criteria; noninferiority, equivalence and mechanism differ."""
def decide(groups, cfg):
    m = groups['assessment']; pts = m['contrasts']; ci = m['ap_intervals']; f = cfg['fusion']
    def positive(name):
        return ci[name].get('low', float('-inf')) > 0
    def material(name):
        return pts[name] >= f['continuation_delta_ap'] and positive(name)
    eq = {name: c.get('low', float('-inf')) >= -f['equivalence_margin'] and
                c.get('high', float('inf')) <= f['equivalence_margin'] for name, c in ci.items()}
    auroc = m['metrics']['I']['auroc'] >= m['metrics']['baseline']['auroc'] - f['maximum_auroc_decline']
    useful = material('I_minus_baseline') and auroc
    ni = {name: ci[name].get('low', float('-inf')) > -f['noninferiority_margin'] for name in ('I_minus_shuffled', 'I_minus_true')}
    recovered = useful and all(ni.values()); beyond_q = material('I_minus_Q')
    ec = groups.get('assessment/eligible', {}).get('ap_intervals', {})
    eligible_ni = bool(ec) and all(ec.get(name, {}).get('low', float('-inf')) > -f['noninferiority_margin'] for name in ni)
    loss = material('shuffled_minus_I')
    if recovered and beyond_q:
        status = 'independent_homologs_recover_gain_within_tolerance_beyond_Q'
    elif recovered:
        status = 'independent_homologs_recover_within_tolerance_Q_contrast_uncertain'
    elif useful and loss:
        status = 'partial_gain_but_independent_sampling_loses_to_row_shuffle'
    elif loss:
        status = 'independent_sampling_loses_practical_gain'
    elif useful:
        status = 'I_has_useful_gain_recovery_inconclusive'
    else:
        status = 'no_go_or_inconclusive'
    return {'status': status, 'primary_point': pts['I_minus_shuffled'], 'primary_interval': ci['I_minus_shuffled'],
        'useful_I_gain': useful, 'auroc_noninferiority': auroc, 'noninferiority': ni,
        'independent_recovery_within_margin': recovered, 'material_I_gain_over_Q': beyond_q,
        'independent_homologs_recover_gain_beyond_Q': recovered and beyond_q, 'eligible_only_noninferiority': eligible_ni,
        'equivalence': eq, 'all_I_true_shuffled_equivalent': all(eq[n] for n in ('I_minus_true', 'I_minus_shuffled', 'true_minus_shuffled')),
        'material_loss_under_independent_sampling': loss, 'true_materially_exceeds_I': material('true_minus_I'),
        'I_exceeds_quality_interval': positive('I_minus_quality'), 'pairing_material_advantage': material('true_minus_shuffled'),
        'material_I_gain_over_C': material('I_minus_C'), 'material_I_gain_over_shuffled': material('I_minus_shuffled'), 'family_breadth_established': False, 'assessment_previously_inspected': True,
        'production_authorized': False, 'test_accessed': False, 'R_evaluated': False,
        'interpretation_scope': 'Fixed assessment and original masks/gate/depth; one independent draw. Changes in homolog composition, diversity and matching are not causally separated.'}
