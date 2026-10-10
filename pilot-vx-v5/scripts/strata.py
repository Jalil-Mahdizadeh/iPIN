"""Original, frozen TRAIN-derived strata and family witnesses."""
import numpy as np
from study import ROOT,read

def strata(sample, pairs, cfg):
    y = np.array([r['label'] for r in sample]); available = np.array([p['available'] for p in pairs])
    train = np.array([r['split'] == 'train' for r in sample]); dev = ~train
    ass = np.array([r['role'] == 'assessment' for r in sample]); length = np.array([r['length'] for r in sample])
    masks = {'assessment': ass, 'fixed_dev_sample': dev, 'calibration': np.array([r['role'] == 'calibration' for r in sample]),
             'crossing_descriptive': np.array([r['role'] == 'crossing' for r in sample])}
    for name, mask in [('assessment', ass), ('fixed_dev_sample', dev)]:
        masks[name + '/eligible'] = mask & available; masks[name + '/fallback'] = mask & ~available
        for label, group in [('le512', length <= 512), ('513_1024', (length > 512) & (length <= 1024)),
                             ('1025_1536', (length > 1024) & (length <= 1536)), ('over1536', length > 1536)]:
            masks[name + '/length_' + label] = mask & group
    cuts = read(ROOT / 'data/v2_metrics.json')['diagnostic_stratum_cuts_from_eligible_train']
    diagnostics = read(ROOT / 'data/diagnostics.json')
    for name, cut in cuts.items():
        values = np.array([d[name] if d is not None else np.nan for d in diagnostics])
        expected = np.unique(np.quantile(values[train & available], [.25, .5, .75]))
        np.testing.assert_array_equal(expected, cut)
        bins = np.searchsorted(cut, values, side='right')
        for i in range(len(cut) + 1):
            masks[f'assessment/{name}/train_quantile_bin_{i}'] = ass & available & (bins == i)
    ann = read(ROOT / 'data/dev_family_witnesses.json'); families = []; shared = np.full(len(sample), -1)
    adequate = np.zeros(len(sample), dtype=bool)
    for i, r in enumerate(sample):
        a, b = ann.get(str(r['a'])), ann.get(str(r['b']))
        if a is not None and b is not None:
            shared[i] = int(a['shared_family'] == 'True') + int(b['shared_family'] == 'True')
            adequate[i] = a.get('adequate') == 'True' and b.get('adequate') == 'True'
        families.append({v['family'] for v in (a, b) if v is not None and v['family']})
    for i in (-1, 0, 1, 2):
        masks[f'assessment/family_witness_count_{i}'] = ass & available & (shared == i)
    masks['assessment/both_family_annotations_adequate'] = ass & available & adequate
    return masks, cuts, families

