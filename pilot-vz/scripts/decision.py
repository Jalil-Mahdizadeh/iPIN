"""Prespecified practical gain, noninferiority, and equivalence are distinct."""
def decide(groups, cfg):
    m = groups['assessment']; pts = m['contrasts']; ci = m['ap_intervals']; f = cfg['fusion']
    def positive(name):
        return ci[name].get('low', float('-inf')) > 0
    def material(name):
        return pts[name] >= f['continuation_delta_ap'] and positive(name)
    eq = {name: c.get('low', float('-inf')) >= -f['equivalence_margin'] and
                c.get('high', float('inf')) <= f['equivalence_margin'] for name, c in ci.items()}
    auroc = m['metrics']['Q']['auroc'] >= m['metrics']['baseline']['auroc'] - f['maximum_auroc_decline']
    useful = material('Q_minus_baseline') and auroc
    ni = {name: ci[name].get('low', float('-inf')) > -f['noninferiority_margin'] for name in ('Q_minus_shuffled', 'Q_minus_true')}
    recovered = useful and all(ni.values())
    ec = groups.get('assessment/eligible', {}).get('ap_intervals', {})
    eligible_ni = bool(ec) and all(ec.get(name, {}).get('low', float('-inf')) > -f['noninferiority_margin'] for name in ni)
    rows_help = material('shuffled_minus_Q') and material('true_minus_Q')
    if recovered:
        status = 'query_only_sufficiency_within_declared_margin'
    elif rows_help and eq['Q_minus_baseline']:
        status = 'homolog_rows_benefit_Q_no_practical_gain'
    elif rows_help:
        status = 'homolog_rows_add_practical_gain'
    elif useful:
        status = 'Q_has_useful_gain_attribution_inconclusive'
    else:
        status = 'no_go_or_inconclusive'
    return {'status': status, 'primary_point': pts['Q_minus_shuffled'], 'primary_interval': ci['Q_minus_shuffled'],
        'useful_Q_gain': useful, 'auroc_noninferiority': auroc, 'noninferiority': ni,
        'query_only_recovery_within_margin': recovered, 'eligible_only_noninferiority': eligible_ni,
        'equivalence': eq, 'all_Q_true_shuffled_equivalent': all(eq[n] for n in ('Q_minus_true', 'Q_minus_shuffled', 'true_minus_shuffled')),
        'both_homolog_arms_materially_exceed_Q': rows_help, 'Q_exceeds_quality_interval': positive('Q_minus_quality'),
        'pairing_material_advantage': material('true_minus_shuffled'), 'family_breadth_established': False,
        'assessment_previously_inspected': True, 'production_authorized': False, 'test_accessed': False,
        'interpretation_scope': 'Original full fixed assessment with MSA-derived masks, eligibility and gate; no unique biological mechanism identified.'}
