"""Prespecified material depth benefit and separate exploratory pairing contrasts."""
def decide(groups,cfg):
    m=groups['assessment'];f=cfg['fusion'];point=m['contrasts']['T256_minus_true'];ci=m['ap_intervals']['T256_minus_true']
    margin=f['meaningful_depth_gain'];auroc=m['metrics']['T256']['auroc']>=m['metrics']['true']['auroc']-f['maximum_auroc_decline']
    gain=point>=margin and ci.get('low',float('-inf'))>0 and auroc
    ruled_out=ci.get('high',float('inf'))<margin
    loss=point<=-margin and ci.get('high',float('inf'))<0
    equivalent=ci.get('low',float('-inf'))>=-f['equivalence_margin'] and ci.get('high',float('inf'))<=f['equivalence_margin']
    def material(name):return m['contrasts'][name]>=margin and m['ap_intervals'][name].get('low',float('-inf'))>0
    pairing=material('T256_minus_S256') and m['ap_intervals']['depth_pairing_interaction'].get('low',float('-inf'))>0
    status='meaningful_depth_256_gain' if gain else 'depth_256_material_loss' if loss else 'material_depth_gain_not_supported_within_interval' if ruled_out else 'depth_comparison_inconclusive'
    return {'status':status,'primary_point':point,'primary_interval':ci,'meaningful_depth_gain':gain,
        'auroc_noninferiority_to_true128':auroc,'material_gain_ruled_out_within_interval':ruled_out,
        'material_depth_loss':loss,'depths_equivalent_within_margin':equivalent,
        'T256_material_gain_over_native':material('T256_minus_baseline'),
        'S256_material_gain_over_shuffled128':material('S256_minus_shuffled'),
        'P256_material_gain_over_quality128':material('P256_minus_quality'),
        'T256_material_gain_over_P256':material('T256_minus_P256'),
        'pairing_sensitivity_increased_exploratory':pairing,
        'assessment_previously_inspected':True,'test_accessed':False,'R_evaluated':False,'production_authorized':False}
